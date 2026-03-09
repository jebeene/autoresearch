"""
Trading price prediction training script. Single-GPU, single-file.
Adapted from autoresearch for financial time-series prediction.

Usage: uv run trading_train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trading_prepare import (
    SEQ_LEN, TIME_BUDGET, EVAL_STEPS,
    load_dataset, make_trading_dataloader, evaluate_trading,
    walk_forward_split, compute_features, compute_targets,
)

# ---------------------------------------------------------------------------
# Time-Series Transformer Model
# ---------------------------------------------------------------------------

@dataclass
class TSConfig:
    seq_len: int = 256       # lookback window
    n_features: int = 50     # number of input features (set dynamically)
    n_layer: int = 4         # transformer depth
    n_head: int = 4          # attention heads
    n_embd: int = 128        # embedding dimension
    dropout: float = 0.1     # dropout rate
    output_dim: int = 1      # prediction output (1 = single return forecast)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class TemporalAttention(nn.Module):
    """Causal self-attention for time-series (no future leakage)."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = TemporalAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for time-series."""

    def __init__(self, seq_len, n_embd):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, n_embd) * 0.02)

    def forward(self, x):
        return x + self.pos_emb[:, :x.size(1)]


class PricePredictor(nn.Module):
    """
    Transformer-based price prediction model.

    Input: (batch, seq_len, n_features) - technical features over lookback window
    Output: (batch, output_dim) - predicted forward returns
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input projection: features -> embedding dim
        self.input_proj = nn.Linear(config.n_features, config.n_embd, bias=False)
        self.input_norm = nn.LayerNorm(config.n_embd)
        self.input_dropout = nn.Dropout(config.dropout)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.seq_len, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Output head: pool over time -> predict return
        self.output_norm = nn.LayerNorm(config.n_embd)
        self.output_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, config.output_dim),
        )

    def forward(self, x):
        """
        x: (B, T, n_features) -> (B, output_dim)
        """
        # Project input features to embedding space
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Use last timestep as summary (causal: has seen all history)
        x = x[:, -1, :]
        x = self.output_norm(x)
        out = self.output_head(x)
        return out

    def setup_optimizer(self, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999)):
        """Setup AdamW optimizer with weight decay only on 2D+ params."""
        decay_params = []
        no_decay_params = []
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=1e-8)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(pred, target):
    """Standard MSE loss for return prediction."""
    return F.mse_loss(pred.squeeze(-1), target)


def directional_loss(pred, target, alpha=0.5):
    """
    Combined loss: MSE for magnitude + BCE for direction.
    alpha controls the mix (0 = pure MSE, 1 = pure direction).
    """
    pred_flat = pred.squeeze(-1)
    mse = F.mse_loss(pred_flat, target)

    # Direction loss
    pred_dir = torch.sigmoid(pred_flat * 10)  # sharpen predictions
    true_dir = (target > 0).float()
    bce = F.binary_cross_entropy(pred_dir, true_dir)

    return (1 - alpha) * mse + alpha * bce


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Data
SYMBOL = "BTC-USD"
INTERVAL = "1h"

# Model architecture
DEPTH = 4               # number of transformer layers
N_HEAD = 4              # attention heads
N_EMBD = 128            # embedding dimension
DROPOUT = 0.1           # dropout rate

# Optimization
BATCH_SIZE = 64          # batch size
LEARNING_RATE = 3e-4     # peak learning rate
WEIGHT_DECAY = 0.01      # weight decay
ADAM_BETAS = (0.9, 0.999)
WARMUP_RATIO = 0.05      # fraction of time for LR warmup
WARMDOWN_RATIO = 0.3     # fraction of time for LR cooldown
FINAL_LR_FRAC = 0.1      # final LR as fraction of peak

# Loss
LOSS_FN = "mse"          # "mse" or "directional"
DIRECTION_ALPHA = 0.3    # mix ratio for directional loss

# Gradient clipping
MAX_GRAD_NORM = 1.0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

# Load data
print(f"Loading data: {SYMBOL} ({INTERVAL})...")
features_df, targets_dict, raw_df = load_dataset(SYMBOL, INTERVAL)
n_samples = len(raw_df)
n_features = len(features_df.columns)
print(f"  Samples: {n_samples}, Features: {n_features}")
print(f"  Date range: {raw_df['timestamp'].iloc[0]} to {raw_df['timestamp'].iloc[-1]}")

# Convert to numpy
features_np = features_df.values.astype(np.float32)
targets_np = targets_dict["forward_return"].astype(np.float32)

# Normalize features (fit on training set only)
train_idx, val_idx, test_idx = walk_forward_split(n_samples)
train_mean = features_np[train_idx].mean(axis=0)
train_std = features_np[train_idx].std(axis=0)
train_std = np.where(train_std > 1e-8, train_std, 1.0)  # avoid div by zero
features_np = (features_np - train_mean) / train_std

print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# Build model
config = TSConfig(
    seq_len=SEQ_LEN,
    n_features=n_features,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_embd=N_EMBD,
    dropout=DROPOUT,
)
print(f"Model config: {asdict(config)}")

model = PricePredictor(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

optimizer = model.setup_optimizer(
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS
)

# Select loss function
if LOSS_FN == "directional":
    loss_fn = lambda pred, target: directional_loss(pred, target, DIRECTION_ALPHA)
else:
    loss_fn = mse_loss

# Data loaders
train_loader = make_trading_dataloader(
    features_np, targets_np, train_idx, BATCH_SIZE, seq_len=SEQ_LEN, shuffle=True
)

# LR schedule
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


print(f"Time budget: {TIME_BUDGET}s")
print(f"Loss function: {LOSS_FN}")
print()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
best_val_sharpe = -float("inf")

while True:
    t0 = time.time()

    model.train()
    X, y = next(train_loader)

    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE * lrm

    optimizer.step()

    train_loss_f = loss.item()

    # Fast fail
    if train_loss_f > 100 or math.isnan(train_loss_f):
        print("FAIL: loss exploded")
        exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    # Logging
    ema_beta = 0.95
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    if step % 50 == 0:
        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {LEARNING_RATE * lrm:.2e} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()

    step += 1

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * BATCH_SIZE * SEQ_LEN

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

print("\nEvaluating on validation set...")
model.eval()
val_metrics = evaluate_trading(
    model, features_np, targets_np, val_idx, BATCH_SIZE,
    seq_len=SEQ_LEN, num_steps=EVAL_STEPS,
)

# Also evaluate on test set
print("Evaluating on test set...")
test_metrics = evaluate_trading(
    model, features_np, targets_np, test_idx, BATCH_SIZE,
    seq_len=SEQ_LEN, num_steps=EVAL_STEPS,
)

# Final summary
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("\n---")
print(f"val_mse:              {val_metrics['mse']:.6f}")
print(f"val_mae:              {val_metrics['mae']:.6f}")
print(f"val_direction_acc:    {val_metrics['direction_accuracy']:.4f}")
print(f"val_ic:               {val_metrics['ic']:.4f}")
print(f"val_sharpe:           {val_metrics['sharpe']:.4f}")
print(f"val_sharpe_annual:    {val_metrics['sharpe_annual']:.4f}")
print(f"test_direction_acc:   {test_metrics['direction_accuracy']:.4f}")
print(f"test_ic:              {test_metrics['ic']:.4f}")
print(f"test_sharpe_annual:   {test_metrics['sharpe_annual']:.4f}")
print(f"training_seconds:     {total_training_time:.1f}")
print(f"total_seconds:        {t_end - t_start:.1f}")
print(f"peak_vram_mb:         {peak_vram_mb:.1f}")
print(f"num_steps:            {step}")
print(f"num_params:           {num_params:,}")
print(f"depth:                {DEPTH}")
print(f"symbol:               {SYMBOL}")

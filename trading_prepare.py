"""
Data preparation for trading price prediction experiments.

Downloads OHLCV data, computes technical features, and provides
walk-forward data loaders for time-series model training.

Usage:
    python trading_prepare.py                        # default: BTC-USD, 1h bars
    python trading_prepare.py --symbol ETH-USD       # different asset
    python trading_prepare.py --interval 1d          # daily bars
    python trading_prepare.py --symbols BTC-USD,ETH-USD,SOL-USD  # multiple assets

Data is stored in ~/.cache/autoresearch/trading/.
"""

import os
import sys
import time
import math
import json
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

SEQ_LEN = 256           # lookback window for model input
FORECAST_HORIZON = 1    # predict N steps ahead
TIME_BUDGET = 300       # training time budget in seconds (5 minutes)
EVAL_STEPS = 200        # number of batches for validation eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "trading")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# Yahoo Finance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
DEFAULT_SYMBOL = "BTC-USD"
DEFAULT_INTERVAL = "1h"

# Walk-forward split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Data download (Yahoo Finance)
# ---------------------------------------------------------------------------

def _yahoo_download(symbol, interval="1h", period="2y"):
    """Download OHLCV data from Yahoo Finance. Returns a pandas DataFrame."""
    # Map period to seconds for the range param
    period_map = {
        "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730,
        "5y": 1825, "10y": 3650, "max": 10000,
    }
    days = period_map.get(period, 730)
    end_ts = int(time.time())
    start_ts = end_ts - days * 86400

    url = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": interval,
        "includePrePost": "false",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                url.format(symbol=symbol), params=params,
                headers=headers, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"],
            })
            df = df.dropna().reset_index(drop=True)
            return df

        except (requests.RequestException, KeyError, IndexError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {symbol}: {e}")
            if attempt < max_attempts:
                time.sleep(2 ** attempt)

    print(f"  ERROR: Failed to download {symbol} after {max_attempts} attempts")
    return None


def download_data(symbols, interval="1h", period="2y"):
    """Download OHLCV data for all symbols."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for symbol in symbols:
        filepath = os.path.join(DATA_DIR, f"{symbol}_{interval}.parquet")
        if os.path.exists(filepath):
            print(f"  {symbol}: already downloaded at {filepath}")
            continue

        print(f"  Downloading {symbol} ({interval}, {period})...")
        df = _yahoo_download(symbol, interval=interval, period=period)
        if df is not None:
            df.to_parquet(filepath)
            print(f"  {symbol}: {len(df)} bars saved to {filepath}")
        else:
            print(f"  {symbol}: FAILED")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(df):
    """
    Compute technical features from OHLCV data.
    Returns DataFrame with feature columns. All features are normalized
    as returns/ratios to be scale-invariant.
    """
    feat = pd.DataFrame(index=df.index)

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)

    # --- Price returns at multiple horizons ---
    for period in [1, 2, 4, 8, 16, 32, 64]:
        ret = np.zeros_like(close)
        ret[period:] = close[period:] / close[:-period] - 1
        feat[f"return_{period}"] = ret

    # --- Volatility (rolling std of returns) ---
    ret1 = np.zeros_like(close)
    ret1[1:] = close[1:] / close[:-1] - 1
    for window in [8, 16, 32, 64]:
        vol = pd.Series(ret1).rolling(window, min_periods=1).std().values
        feat[f"volatility_{window}"] = vol

    # --- Moving average ratios (price relative to MA) ---
    for window in [8, 16, 32, 64, 128]:
        ma = pd.Series(close).rolling(window, min_periods=1).mean().values
        feat[f"ma_ratio_{window}"] = close / np.where(ma > 0, ma, 1) - 1

    # --- RSI (Relative Strength Index) ---
    for period in [14, 28]:
        delta = np.zeros_like(close)
        delta[1:] = close[1:] - close[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
        rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
        feat[f"rsi_{period}"] = rs / (1 + rs) - 0.5  # center at 0

    # --- MACD ---
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd = (ema12 - ema26) / np.where(close > 0, close, 1)
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    feat["macd"] = macd
    feat["macd_signal"] = signal
    feat["macd_hist"] = macd - signal

    # --- Bollinger Band position ---
    for window in [20, 40]:
        ma = pd.Series(close).rolling(window, min_periods=1).mean().values
        std = pd.Series(close).rolling(window, min_periods=1).std().values
        std = np.where(std > 0, std, 1e-10)
        feat[f"bb_position_{window}"] = (close - ma) / (2 * std)

    # --- Volume features ---
    vol_ma = pd.Series(volume).rolling(20, min_periods=1).mean().values
    feat["volume_ratio"] = volume / np.where(vol_ma > 0, vol_ma, 1)
    feat["volume_change"] = np.concatenate([[0], volume[1:] / np.where(volume[:-1] > 0, volume[:-1], 1) - 1])

    # --- Candle features ---
    body = close - open_
    range_ = high - low
    range_ = np.where(range_ > 0, range_, 1e-10)
    feat["candle_body"] = body / range_
    feat["candle_upper_shadow"] = (high - np.maximum(close, open_)) / range_
    feat["candle_lower_shadow"] = (np.minimum(close, open_) - low) / range_

    # --- High/Low channel position ---
    for window in [16, 32, 64]:
        rolling_high = pd.Series(high).rolling(window, min_periods=1).max().values
        rolling_low = pd.Series(low).rolling(window, min_periods=1).min().values
        channel_range = rolling_high - rolling_low
        channel_range = np.where(channel_range > 0, channel_range, 1e-10)
        feat[f"channel_position_{window}"] = (close - rolling_low) / channel_range - 0.5

    # Clean up any NaN/inf
    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)

    return feat


# ---------------------------------------------------------------------------
# Target computation
# ---------------------------------------------------------------------------

def compute_targets(df, horizon=FORECAST_HORIZON):
    """
    Compute prediction targets: forward returns.
    Returns dict of target arrays.
    """
    close = df["close"].values.astype(np.float64)

    targets = {}

    # Forward return (regression target)
    fwd_return = np.zeros_like(close)
    fwd_return[:-horizon] = close[horizon:] / close[:-horizon] - 1
    targets["forward_return"] = fwd_return

    # Direction (classification target): 1 = up, 0 = down
    targets["direction"] = (fwd_return > 0).astype(np.float64)

    return targets


# ---------------------------------------------------------------------------
# Walk-forward data splitting
# ---------------------------------------------------------------------------

def walk_forward_split(n_samples, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Time-series aware split. No shuffling — respects temporal order.
    Returns (train_indices, val_indices, test_indices).
    """
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n_samples))

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Runtime utilities (imported by trading_train.py)
# ---------------------------------------------------------------------------

def load_dataset(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL):
    """Load preprocessed dataset. Returns (features, targets, df) tensors."""
    filepath = os.path.join(DATA_DIR, f"{symbol}_{interval}.parquet")
    if not os.path.exists(filepath):
        print(f"Data not found for {symbol}. Run trading_prepare.py first.")
        sys.exit(1)

    df = pd.read_parquet(filepath)
    features = compute_features(df)
    targets = compute_targets(df)

    return features, targets, df


def make_trading_dataloader(features_np, targets_np, indices, batch_size, seq_len=SEQ_LEN, shuffle=True):
    """
    Creates a generator yielding (X, y) batches for time-series prediction.
    X shape: (batch_size, seq_len, n_features)
    y shape: (batch_size,)

    Each sample is a window of `seq_len` consecutive feature rows,
    with the target being the forward return at the last timestep.
    """
    # Filter indices that have enough lookback
    valid_indices = [i for i in indices if i >= seq_len]

    features_t = torch.tensor(features_np, dtype=torch.float32)
    targets_t = torch.tensor(targets_np, dtype=torch.float32)

    while True:
        if shuffle:
            order = np.random.permutation(len(valid_indices))
        else:
            order = np.arange(len(valid_indices))

        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch_idx = order[start:start + batch_size]
            X_batch = torch.stack([
                features_t[valid_indices[j] - seq_len:valid_indices[j]]
                for j in batch_idx
            ])
            y_batch = torch.stack([
                targets_t[valid_indices[j]]
                for j in batch_idx
            ])
            yield X_batch.cuda(), y_batch.cuda()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_trading(model, features_np, targets_np, val_indices, batch_size,
                     seq_len=SEQ_LEN, num_steps=EVAL_STEPS):
    """
    Evaluate trading model on validation set.
    Returns dict with multiple metrics:
    - mse: Mean squared error of return predictions
    - mae: Mean absolute error
    - direction_accuracy: % of times direction predicted correctly
    - sharpe_approx: Approximate Sharpe ratio of predictions
    - ic: Information coefficient (correlation of predicted vs actual returns)
    """
    model.eval()
    loader = make_trading_dataloader(
        features_np, targets_np, val_indices, batch_size,
        seq_len=seq_len, shuffle=False
    )

    all_preds = []
    all_targets = []
    total_mse = 0.0
    total_mae = 0.0
    count = 0

    for step_i in range(num_steps):
        X, y = next(loader)
        pred = model(X).squeeze(-1)

        mse = (pred - y).pow(2).mean().item()
        mae = (pred - y).abs().mean().item()
        total_mse += mse
        total_mae += mae
        count += 1

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Direction accuracy
    pred_dir = (all_preds > 0).float()
    true_dir = (all_targets > 0).float()
    direction_acc = (pred_dir == true_dir).float().mean().item()

    # Information coefficient (Pearson correlation)
    p_mean = all_preds.mean()
    t_mean = all_targets.mean()
    p_std = all_preds.std()
    t_std = all_targets.std()
    if p_std > 1e-8 and t_std > 1e-8:
        ic = ((all_preds - p_mean) * (all_targets - t_mean)).mean() / (p_std * t_std)
        ic = ic.item()
    else:
        ic = 0.0

    # Approximate Sharpe: mean(pred * actual) / std(pred * actual)
    # This simulates a simple long/short strategy based on predictions
    strategy_returns = all_preds.sign() * all_targets
    if strategy_returns.std() > 1e-8:
        sharpe = (strategy_returns.mean() / strategy_returns.std()).item()
        # Annualize (rough: assume hourly data, ~8760 hours/year)
        sharpe_annual = sharpe * math.sqrt(8760)
    else:
        sharpe = 0.0
        sharpe_annual = 0.0

    model.train()

    return {
        "mse": total_mse / max(count, 1),
        "mae": total_mae / max(count, 1),
        "direction_accuracy": direction_acc,
        "ic": ic,
        "sharpe": sharpe,
        "sharpe_annual": sharpe_annual,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare trading data")
    parser.add_argument("--symbols", type=str, default=DEFAULT_SYMBOL,
                        help="Comma-separated list of symbols (e.g. BTC-USD,ETH-USD)")
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL,
                        help="Bar interval (1h, 1d, etc.)")
    parser.add_argument("--period", type=str, default="2y",
                        help="How far back to download (1y, 2y, 5y, max)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    print("Downloading market data...")
    download_data(symbols, interval=args.interval, period=args.period)
    print()

    # Step 2: Verify features
    for symbol in symbols:
        print(f"Verifying features for {symbol}...")
        features, targets, df = load_dataset(symbol, args.interval)
        print(f"  Bars: {len(df)}")
        print(f"  Features: {len(features.columns)} ({', '.join(features.columns[:5])}...)")
        print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        train_idx, val_idx, test_idx = walk_forward_split(len(df))
        print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        print()

    print("Done! Ready to train.")

# autoresearch — trading price prediction

Autonomous experimentation for financial price prediction models.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9-trading`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `trading_prepare.py` — fixed: data download, feature engineering, evaluation metrics, data loaders. Do not modify.
   - `trading_train.py` — the file you modify. Model architecture, optimizer, hyperparameters, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/trading/data/` contains parquet files. If not, tell the human to run `uv run trading_prepare.py`.
5. **Initialize results.tsv**: Create `trading_results.tsv` with just the header row.
6. **Confirm and go**.

## Experimentation

Each experiment runs on a single GPU for a **fixed time budget of 5 minutes**. Launch: `uv run trading_train.py`.

**What you CAN modify** (`trading_train.py`):
- Model architecture (transformer depth, heads, embedding dim, attention type)
- Add new architectures (LSTM, temporal CNN, hybrid models)
- Optimizer choice and hyperparameters (LR, weight decay, betas, scheduler)
- Loss function (MSE, directional, custom combinations)
- Batch size, sequence length usage, dropout
- Feature selection/weighting within the model
- Ensemble or multi-head prediction strategies

**What you CANNOT modify**:
- `trading_prepare.py` — data pipeline, features, and evaluation are fixed
- Cannot install new packages
- Cannot modify evaluation metrics

**The goal: maximize val_sharpe_annual** (annualized Sharpe ratio on validation set). Secondary metrics to track:
- `val_direction_acc` — directional accuracy (>0.5 = better than random)
- `val_ic` — information coefficient (correlation of predictions with actual returns)
- `val_mse` — prediction error magnitude

**Key differences from LLM pretraining experiments:**
- **Overfitting is the enemy**: Financial data is noisy. More parameters ≠ better. Regularization matters.
- **Direction > magnitude**: Predicting whether price goes up/down is often more valuable than predicting exact returns.
- **Stationarity**: Market regimes change. Models that memorize past patterns may fail on future data.
- **Test set matters**: Always check test metrics for generalization. A model that does well on val but poorly on test is overfit.

## Output format

The script prints a summary like:

```
---
val_mse:              0.000123
val_mae:              0.008456
val_direction_acc:    0.5234
val_ic:               0.0456
val_sharpe:           0.0123
val_sharpe_annual:    1.1500
test_direction_acc:   0.5189
test_ic:              0.0312
test_sharpe_annual:   0.9800
training_seconds:     300.1
total_seconds:        305.2
peak_vram_mb:         1234.5
num_steps:            12345
num_params:           456,789
depth:                4
symbol:               BTC-USD
```

Extract key metrics: `grep "^val_sharpe_annual:\|^val_direction_acc:\|^test_sharpe_annual:" run.log`

## Logging results

Log to `trading_results.tsv` (tab-separated):

```
commit	val_sharpe	val_dir_acc	val_ic	test_sharpe	memory_gb	status	description
```

1. git commit hash (7 chars)
2. val_sharpe_annual (e.g. 1.1500) — use 0.0000 for crashes
3. val_direction_acc (e.g. 0.5234) — use 0.0000 for crashes
4. val_ic (e.g. 0.0456) — use 0.0000 for crashes
5. test_sharpe_annual (e.g. 0.9800) — use 0.0000 for crashes
6. peak memory in GB (divide peak_vram_mb by 1024) — use 0.0 for crashes
7. status: `keep`, `discard`, or `crash`
8. short text description

## The experiment loop

LOOP FOREVER:

1. Check git state
2. Modify `trading_train.py` with an experimental idea
3. git commit
4. Run: `uv run trading_train.py > run.log 2>&1`
5. Read results: `grep "^val_sharpe_annual:\|^val_direction_acc:\|^test_sharpe_annual:\|^peak_vram_mb:" run.log`
6. If empty, run crashed. `tail -n 50 run.log` to debug.
7. Record in trading_results.tsv
8. If val_sharpe_annual improved → keep commit
9. If worse → git reset

**Experiment ideas to try (roughly in priority order):**

1. **Baseline first** — always run unmodified to establish baseline
2. **Learning rate sweep** — try 1e-4, 5e-4, 1e-3, 3e-3
3. **Directional loss** — switch to directional loss, tune alpha
4. **Model size** — try smaller (2 layers, 64 dim) and larger (6 layers, 256 dim)
5. **Dropout tuning** — try 0.05, 0.2, 0.3
6. **LSTM baseline** — replace transformer with bidirectional LSTM
7. **Temporal CNN** — try 1D convolutions with dilated receptive fields
8. **Hybrid models** — CNN for local patterns + transformer for long-range
9. **Attention variants** — relative position encoding, local attention windows
10. **Feature attention** — learn which features matter most (feature-wise attention)
11. **Multi-scale** — process input at multiple time resolutions
12. **Label smoothing / target clipping** — reduce impact of outlier returns
13. **Sequence length** — try 64, 128, 512 (within the model, SEQ_LEN in prepare is fixed)

**NEVER STOP**: Run indefinitely until manually interrupted. If stuck, try more radical ideas.

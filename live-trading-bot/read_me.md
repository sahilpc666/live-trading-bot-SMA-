# 📈 Real-Time Trading & Quantitative Research Platform

An end-to-end system for **live algorithmic trading**, **historical data ingestion**, **feature engineering**, **ML model training**, and **rigorous walk-forward backtesting** on US equities.

The platform streams live market quotes from Alpaca into Kafka, generates trading signals, executes paper trades with risk controls, and separately supports a full quantitative research pipeline — from raw daily/intraday bars to cost-aware, out-of-sample strategy validation.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Live Trading Pipeline](#live-trading-pipeline)
- [Data Ingestion](#data-ingestion)
- [Feature Engineering](#feature-engineering)
- [ML Modeling](#ml-modeling)
- [Backtesting & Research](#backtesting--research)
- [Database Schema](#database-schema)
- [Project Structure](#project-structure)
- [Setup & Running](#setup--running)
- [Configuration](#configuration)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)

---


Two independent tracks share one PostgreSQL database:

1. **Live trading track** — streaming quotes → Kafka → signal engine → paper orders → position/signal logging.
2. **Research track** — historical bars → features → ML models → walk-forward and multi-window holdout backtests.

---

## Live Trading Pipeline

| Component | File | Role |
|---|---|---|
| Quote producer | `producer.py` | Streams live Alpaca quotes to Kafka topic `stock-quotes` |
| Mock producer | `mock_producer.py` | Replays static mock quotes for offline testing |
| Signal engine | `signal_engine.py` | Consumes Kafka, computes SMA crossover, applies regime gate, submits orders |
| Risk manager | `risk_manager.py` | Enforces position sizing, max open positions, daily loss limits |
| Order executor | `order_executor.py` | Thin Alpaca paper-trading REST wrapper |
| DB logger | `db_logger.py` | Persists signals, bot positions, and feature lookups |
| Mock signal test | `mock_signal_test.py` | Drives synthetic price ramps to exercise the signal engine end-to-end |

### Strategy: SMA Crossover + Regime Gate

- **Short window**: 100 ticks
- **Long window**: 300 ticks
- **Entry threshold**: short/long diff > +0.3% → go long
- **Exit threshold**: diff < −0.3% → go flat
- **Cooldown**: 60 seconds between signals per symbol
- **Position size**: 1 share (configurable via `TRADE_QTY`)

**Regime gate (optional):** a gradient-boosted classifier predicts whether a symbol is entering a high-volatility regime (`forward_volatility_5d > volatility_20d`). If high-vol is predicted, **new long entries are suppressed** — exits are never blocked, and a missing model file fails open (gate disabled, trading unaffected).

**Order lifecycle:** orders are submitted as market orders, then polled for status. `bot_positions` is only updated on confirmed fills; a background reconciler (`reconcile_pending_orders`) catches orders that were pending at submission and fill later.

---

## Data Ingestion

| Script | Purpose |
|---|---|
| `fetch_historical_bars.py` | Daily bars for 12 symbols since 2018, `feed=sip`, `adjustment=all` |
| `fetch_intraday_bars.py` | 1-minute intraday bars into `intraday_bars` |
| `historical_fetch.py` | 5-minute RTH-only bars into `historical_bars_intraday` (used by ORB/VWAP) |
| `stream_quotes.py` | Live quote streaming helper + `mock_fetch_data()` fixture |

All fetchers share the same pattern:
- Paginated requests with `next_page_token` handling
- Exponential-backoff retries on 429/5xx/network errors
- OHLC + volume validation before insert
- Idempotent `ON CONFLICT (symbol, timestamp) DO NOTHING` upserts
- Regular-session filtering (9:30–16:00 ET) for intraday sets

**Symbols tracked:** `AAPL, MSFT, GOOGL, JPM, BAC, JNJ, PFE, XOM, CVX, TSLA, AMZN, WMT`

---

## Feature Engineering

### Daily features — `feature_engineering.py`

Writes to `historical_features`. All rolling/EWM calculations are **trailing only** (no lookahead), except the explicitly labeled forward-looking targets.

**Feature families:**
- Returns: `return_1d`, `return_5d`, `return_20d`
- Volatility: `volatility_10d/20d/60d`
- Moving averages: `sma_20/50/100/200`, price-to-SMA ratios
- MACD: `ema_12`, `ema_26`, `macd`, `macd_signal`, `macd_hist`, `macd_hist_norm`
- RSI: `rsi_14`
- Bollinger: `bb_upper`, `bb_lower`, `bb_pct`
- Volume: `volume_sma_20`, `relative_volume`
- Bar structure: `high_low_range`, `close_open_range`

**Labels (never model inputs):**
- `forward_return_5d`
- `forward_volatility_5d` (built via `rolling().std().shift(-h)`, i.e. genuinely forward)

Rows before `MAX_LOOKBACK_DAYS = 200` are dropped so no partially-warm rows enter the table. Label columns may be NULL for recent rows and are inserted as SQL NULL rather than filtered out.

### Intraday features — `intraday_feature_engineering.py`

Writes to `intraday_features`. **Regular session only.**

- Returns: `return_1m/3m/5m/10m/30m`
- Volatility: `volatility_5m/15m/30m`
- Volume: `volume_5m`, `relative_volume`, **`relative_volume_tod`**, **`volume_zscore_tod`**, `volume_acceleration`
- Bar structure: `high_low_range`, `close_open_range`, `close_position_in_bar`
- VWAP: `session_vwap`, `price_vs_vwap`, `vwap_return`
- Activity: `trade_count_5m`, `trade_intensity`
- Trend: `price_vs_sma_20/50`, `sma_20_slope`
- Session: `minutes_since_open`, `minutes_to_close`

The **time-of-day volume** features are the notable ones: `relative_volume_tod` and `volume_zscore_tod` compare the current minute's volume against the *same minute on previous trading days* (with `shift(1)` so the current day is excluded from its own baseline). This captures the intraday U-shape that a simple rolling mean misses.

### Intraday targets — `intraday_target_engineering.py`

Adds `future_return_1m/5m/15m` to `intraday_features`, computed **within each symbol × trading date** so a target never crosses an overnight boundary.

---

## ML Modeling

### Daily models

| Script | Target | Models |
|---|---|---|
| `train_model.py` | `forward_return_5d > 0` | Logistic regression, HistGradientBoosting |
| `train_model_regime.py` | `forward_volatility_5d > volatility_20d` | Logistic regression, HistGradientBoosting |
| `cross_sectional.py` | Outperform daily cross-sectional median | Logistic regression, HistGradientBoosting |
| `backtest_ml_regime.py` | — | Backtests SMA crossover with/without regime gate |

All daily models:
- Use **scale-free features only** (nothing in price units) so they transfer across symbols.
- Split **chronologically** with a **5-day embargo** on each side of the boundary (matching the target horizon) to prevent label overlap leakage.
- Compare against the **always-up baseline** (which is ~55% for equities) — a model must beat that, not 50%.

The regime model (`train_model_regime.py`) is deliberately a *different, easier* target than direction: volatility clusters, so predicting "choppier than recent baseline" is far more learnable than predicting up/down. It also compares against a **persistence baseline** (`volatility_20d > volatility_60d → predict high-vol continues`), not just majority class.

### Intraday models

| Script | Target | Models |
|---|---|---|
| `train_intraday_model.py` | `future_return_5m > 0` | Logistic, HistGradientBoosting |
| `train_intraday_return_model.py` | `future_return_5m` (regression) | Ridge, HistGradientBoostingRegressor |
| `intraday_feature_ablation.py` | `future_return_5m` (regression) | HistGradientBoostingRegressor across 5 feature groups |

The ablation script is the most useful one: it trains the same model on progressively larger feature sets (`A_momentum_only` → `E_all_without_symbol`) and reports decile spreads, top-10% returns, and net-of-cost P&L for each. This isolates *which feature families actually add signal* rather than assuming more features is better.

All intraday models:
- Split 2026-01-01 → 2026-05-31 (train), 2026-06-01 → 2026-07-31 (val), 2026-08-01 → 2026-09-01 (test).
- Apply a **5-minute embargo** at each split boundary.
- Report **net-of-cost** per-trade returns using a 6 bps round-trip assumption (1 bp fee + 2 bps slippage per side).

---

## Backtesting & Research

| Script | What it does |
|---|---|
| `backtest_sma.py` | Simple per-symbol SMA(100/300) crossover backtest on daily bars |
| `orb_vwap_backtest.py` | Opening-range breakout + VWAP intraday strategy with stop-loss, dev/holdout split |
| `backtest_ml_regime.py` | SMA crossover with vs. without the regime gate |
| `backtest_regime_gate1.py` | True-holdings walk-forward backtest across momentum variants |
| `holdout_test.py` | Two-phase selection → holdout discipline on 2020–2025 |
| `mullti_window_holdout.py` | Four independent expanding-window holdout folds (2022–2025) |

### Backtest accounting rules (consistent across the research scripts)

- **Signal at previous close → execute at current open.** No same-bar lookahead.
- **Actual shares are held**, not notional weights that get silently rebalanced daily. Weights drift naturally between scheduled rebalances.
- **Transaction costs** applied at execution using actual L1 weight turnover: `equity × turnover × (fee + slippage)`.
- **Final evaluation session included through its close**, with a final liquidation cost applied.
- **Buy-and-hold benchmark** uses the same share-based accounting and the same cost model, so comparisons are apples-to-apples.
- **Walk-forward windows reset capital**, so window results are diagnostics — not a single continuous live equity curve.

### Selection discipline (`holdout_test.py`, `mullti_window_holdout.py`)

The selection rule is committed *before* seeing any holdout data:

1. Rank all `(strategy, rebalance_every, top_n)` combos by **Sharpe over the selection period**.
2. Among the top 10 by Sharpe, pick the one with the **smallest (least negative) max drawdown**.

This rewards risk-adjusted performance first and uses drawdown as a tiebreaker — rather than ranking by raw Sharpe alone (which can reward lucky avoidance of one bad stretch) or by raw return (which the full-grid run showed can be inflated by concentrated, high-turnover positions).

`mullti_window_holdout.py` applies this rule **four times independently** on expanding windows (2022, 2023, 2024, 2025), where each holdout year is never part of any fold's selection period. The summary output explicitly reports:

- Folds where the strategy beat buy-and-hold on Sharpe
- Folds where it beat on return
- Mean Sharpe/return edge over buy-and-hold
- **Number of distinct configs selected across folds** — if this is close to the number of folds, the grid is picking a different "winner" each time, which is itself evidence the selection step is fitting noise.

### Backtest outputs

| File | Contents |
|---|---|
| `backtest_results.csv` | Walk-forward + full-eval grid results |
| `backtest_summary.csv` | Per-symbol ungated vs. gated regime comparison |
| `holdout_selection_grid.csv` | Selection-period grid for the 2024–2025 holdout |
| `holdout_result.csv` | Locked config's selection vs. holdout performance + buy-and-hold |
| `multi_window_summary.csv` | Per-fold results across the four expanding-window folds |
| `orb_vwap_development_trades.csv` | Trade-level detail (dev period) |
| `orb_vwap_holdout_trades.csv` | Trade-level detail (holdout period) |

---

## Database Schema

PostgreSQL database `fin`, all tables keyed on `(symbol, timestamp)` for idempotent upserts.

| Table | Written by | Notes |
|---|---|---|
| `historical_bars` | `fetch_historical_bars.py` | Daily bars, `feed=sip`, `adjustment=all` |
| `historical_bars_intraday` | `historical_fetch.py` | 5-min RTH-only, includes `trading_date` |
| `intraday_bars` | `fetch_intraday_bars.py` | 1-min intraday |
| `historical_features` | `feature_engineering.py` | Daily features + forward labels |
| `intraday_features` | `intraday_feature_engineering.py`, `intraday_target_engineering.py` | Intraday features + forward returns |
| `signals` | `db_logger.py` | Every signal, order submission, fill, block |
| `bot_positions` | `db_logger.py` | Open positions *this bot* holds |
| `stock_quotes` | `insert_records.py` | Raw quote archive (schema in `fin`) |

`bot_positions` deliberately tracks only positions **this bot** opened — not every position in the Alpaca account. The risk manager uses that count for `MAX_OPEN_POSITIONS`, so unrelated manual trades or other strategies in the same account can't cause the bot to block its own new trades.

---


---

## Setup & Running

### Prerequisites

- Docker + Docker Compose
- Alpaca API key (paper trading is sufficient for everything here)
- Python 3.10+ if running scripts outside Docker

### 1. Environment variables

Create a `.env` file alongside `docker-compose.yaml`:

```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
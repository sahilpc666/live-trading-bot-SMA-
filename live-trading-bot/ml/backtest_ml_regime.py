import os
import numpy as np
import pandas as pd
import psycopg2
import joblib

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/ml/models")
REGIME_MODEL_PATH = os.path.join(MODEL_DIR, "high_vol_regime.joblib")

OUTPUT_DIR = os.environ.get("BACKTEST_OUTPUT_DIR", "/app/ml/backtest_output")

# Must match signal_engine.py's strategy settings.
SHORT_WINDOW = 100
LONG_WINDOW = 300
THRESHOLD_PCT = 0.003

TRADING_DAYS_PER_YEAR = 252

# Must match train_model_regime.py's FEATURES list.
FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "volatility_10d", "volatility_20d", "volatility_60d",
    "price_to_sma_20", "price_to_sma_50", "price_to_sma_200",
    "rsi_14",
    "bb_pct",
    "relative_volume",
    "high_low_range", "close_open_range",
    "macd_hist_norm",
]

# ============================================================
# IMPORTANT CAVEATS — read before trusting these numbers
#
# 1. This backtests the crossover STRATEGY on DAILY closes, not
#    the live tick-driven signal_engine.py. The live system reacts
#    to bid/ask midpoints intraday; this can only approximate that
#    with what historical data exists (daily bars). Directionally
#    informative, not a byte-for-byte replay of production.
#
# 2. Positions are sized as "fully invested (1x) when long, 0 when
#    flat," not literal TRADE_QTY=1 shares. A fixed 1-share
#    position means wildly different dollar exposure across a
#    ~$40 stock and a ~$400 stock, which makes cross-symbol
#    Sharpe/drawdown comparisons meaningless. Normalizing to
#    percent-of-capital is the standard fix and is what's needed
#    to judge whether the GATE helps, which is the actual question
#    here.
#
# 3. A trade's return is computed close-to-close across the
#    holding period; the day-by-day equity curve instead lags
#    position by one day (yesterday's signal earns today's return)
#    to avoid crediting a signal with the same day's return that
#    produced it. The two are consistent but computed slightly
#    differently — trade list is for interpretability, equity
#    curve is for Sharpe/drawdown.
#
# 4. No slippage, spread, or commission modeled.
# ============================================================


def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def load_symbol_data(conn, symbol):
    """
    Full daily close history from historical_bars (used for the
    SMA crossover, since it isn't truncated by feature warm-up),
    left-joined with historical_features (used only for regime
    scoring — rows before the feature warm-up period are simply
    unscoreable, which the simulator treats as "unknown regime,
    don't gate," matching signal_engine.py's fail-open behavior.
    """
    bars = pd.read_sql(
        """
        SELECT timestamp, close
        FROM historical_bars
        WHERE symbol = %s
        ORDER BY timestamp ASC
        """,
        conn, params=(symbol,),
    )
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["close"] = bars["close"].astype(float)

    feat_cols = ", ".join(FEATURES)
    features = pd.read_sql(
        f"""
        SELECT timestamp, {feat_cols}
        FROM historical_features
        WHERE symbol = %s
        ORDER BY timestamp ASC
        """,
        conn, params=(symbol,),
    )
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    for col in FEATURES:
        features[col] = features[col].astype(float)

    df = bars.merge(features, on="timestamp", how="left")
    return df


def score_regime(df, model):
    """
    Predicts the high-vol-regime flag for every row that has a
    complete feature set. Rows with any missing feature (pre
    warm-up, or a gap) get NaN -> treated as "unknown" by the
    simulator, never as a false 0 or 1.
    """
    complete = df[FEATURES].notna().all(axis=1)
    df["regime_pred"] = np.nan
    if complete.any():
        df.loc[complete, "regime_pred"] = model.predict(df.loc[complete, FEATURES])
    return df


def simulate(df, gate_enabled):
    """
    Reproduces signal_engine.py's state machine on daily closes:
    only act when the SMA diff crosses +-THRESHOLD_PCT AND the
    resulting state actually differs from the current one. When
    gate_enabled, a would-be new long entry is suppressed (state
    stays flat) if regime_pred == 1 at that row; unknown regime
    (NaN) never gates, matching the live fail-open default.
    """
    close = df["close"]
    short_avg = close.rolling(SHORT_WINDOW).mean()
    long_avg = close.rolling(LONG_WINDOW).mean()
    diff_pct = (short_avg - long_avg) / long_avg

    positions = np.zeros(len(df), dtype=int)
    trades = []          # (entry_idx, exit_idx, entry_price, exit_price)
    suppressed = 0
    state = 0
    entry_idx = None

    regime = df["regime_pred"].values if gate_enabled else None

    for i in range(len(df)):
        d = diff_pct.iloc[i]
        if pd.isna(d):
            positions[i] = state
            continue

        if d > THRESHOLD_PCT:
            desired = 1
        elif d < -THRESHOLD_PCT:
            desired = 0
        else:
            desired = state

        if desired != state:
            if desired == 1 and gate_enabled and regime[i] == 1:
                suppressed += 1
                desired = state  # blocked: stay flat
            else:
                if desired == 1:
                    entry_idx = i
                elif desired == 0 and entry_idx is not None:
                    trades.append((entry_idx, i, close.iloc[entry_idx], close.iloc[i]))
                    entry_idx = None
                state = desired

        positions[i] = state

    # Close out an open position at the end of the window so it
    # counts toward trade stats (marked, not a real exit signal).
    if entry_idx is not None:
        trades.append((entry_idx, len(df) - 1, close.iloc[entry_idx], close.iloc[-1]))

    df = df.copy()
    df["position"] = positions
    # Lag by one day: yesterday's established position earns
    # today's return, so today's signal never earns today's move.
    applied_position = pd.Series(positions, index=df.index).shift(1).fillna(0)
    daily_return = close.pct_change().fillna(0)
    df["strategy_return"] = applied_position * daily_return

    trade_returns = [exit_p / entry_p - 1 for _, _, entry_p, exit_p in trades]

    return df, trade_returns, suppressed


def sharpe(daily_returns):
    std = daily_returns.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return (daily_returns.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def summarize(name, df, trade_returns, suppressed):
    equity = (1 + df["strategy_return"]).cumprod()
    total_return = equity.iloc[-1] - 1
    win_rate = (
        sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        if trade_returns else float("nan")
    )
    return {
        "run": name,
        "total_return": total_return,
        "sharpe": sharpe(df["strategy_return"]),
        "max_drawdown": max_drawdown(equity),
        "num_trades": len(trade_returns),
        "win_rate": win_rate,
        "avg_trade_return": np.mean(trade_returns) if trade_returns else float("nan"),
        "entries_suppressed": suppressed,
    }, equity


def get_symbols(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM historical_bars ORDER BY symbol")
        return [row[0] for row in cur.fetchall()]


def main():
    print()
    print("=" * 70)
    print("REGIME GATE BACKTEST — SMA crossover, ungated vs. gated")
    print("=" * 70)

    try:
        model_bundle = joblib.load(REGIME_MODEL_PATH)
        model = model_bundle["model"]
    except Exception as e:
        print(f"FATAL: could not load regime model at {REGIME_MODEL_PATH}: {e}")
        print("Run train_model_regime.py first.")
        return

    conn = get_connection()
    symbols = get_symbols(conn)
    print(f"\nSymbols: {len(symbols)}\n")

    per_symbol_rows = []
    portfolio_returns = {"ungated": [], "gated": []}

    for symbol in symbols:
        df = load_symbol_data(conn, symbol)
        if len(df) < LONG_WINDOW + 10:
            print(f"{symbol}: SKIPPED, only {len(df)} bars")
            continue

        df = score_regime(df, model)

        ungated_df, ungated_trades, _ = simulate(df, gate_enabled=False)
        gated_df, gated_trades, suppressed = simulate(df, gate_enabled=True)

        u_stats, u_equity = summarize("ungated", ungated_df, ungated_trades, 0)
        g_stats, g_equity = summarize("gated", gated_df, gated_trades, suppressed)

        u_stats["symbol"] = symbol
        g_stats["symbol"] = symbol
        per_symbol_rows.append(u_stats)
        per_symbol_rows.append(g_stats)

        portfolio_returns["ungated"].append(
            ungated_df.set_index("timestamp")["strategy_return"]
        )
        portfolio_returns["gated"].append(
            gated_df.set_index("timestamp")["strategy_return"]
        )

        print(
            f"{symbol:6s} | ungated: return {u_stats['total_return']:+7.2%}  "
            f"sharpe {u_stats['sharpe']:5.2f}  maxDD {u_stats['max_drawdown']:7.2%}  "
            f"trades {u_stats['num_trades']:3d}"
        )
        print(
            f"       | gated  : return {g_stats['total_return']:+7.2%}  "
            f"sharpe {g_stats['sharpe']:5.2f}  maxDD {g_stats['max_drawdown']:7.2%}  "
            f"trades {g_stats['num_trades']:3d}  (suppressed {suppressed})"
        )

    conn.close()

    # Equal-weighted portfolio: average daily return across symbols
    # for each day (missing days for a symbol contribute 0, i.e.
    # flat — consistent with not being tradeable yet).
    print()
    print("=" * 70)
    print("PORTFOLIO (equal-weighted across symbols)")
    print("=" * 70)

    for name in ("ungated", "gated"):
        combined = pd.concat(portfolio_returns[name], axis=1).fillna(0)
        port_daily_return = combined.mean(axis=1)
        equity = (1 + port_daily_return).cumprod()
        print(
            f"{name:8s} | return {equity.iloc[-1] - 1:+7.2%}  "
            f"sharpe {sharpe(port_daily_return):5.2f}  "
            f"maxDD {max_drawdown(equity):7.2%}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "backtest_summary.csv")
    pd.DataFrame(per_symbol_rows).to_csv(out_path, index=False)
    print(f"\nPer-symbol detail saved: {out_path}")


if __name__ == "__main__":
    main()
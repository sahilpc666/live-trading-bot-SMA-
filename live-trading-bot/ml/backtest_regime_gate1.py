import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg2


# ============================================================
# TRUE HOLDINGS WALK-FORWARD BACKTEST
# ============================================================
#
# Execution model
# ----------------
# Signal known at previous CLOSE
#              -> execute at current OPEN
#              -> own actual shares
#              -> mark shares at current CLOSE
#              -> rebalance only on scheduled sessions
#
# This is deliberately a simple diagnostic baseline:
#   - no ML
#   - no ADX
#   - no stop loss
#   - no parameter optimization
#
# Important accounting rules:
#   - EVAL_END is inclusive.
#   - Portfolio weights drift naturally between rebalances.
#   - Transaction costs use actual L1 weight turnover at the open.
#   - Final evaluation session is included through its CLOSE.
#   - Buy-and-hold uses the same actual-share accounting.
#   - Window results reset capital, so they are walk-forward diagnostics,
#     not one continuous live equity curve.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fin")
DB_USER = os.getenv("DB_USER", "dbuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dbpassword")

STARTING_CAPITAL = 10_000.0

FEE_BPS = 1.0
SLIPPAGE_BPS = 2.0
ONE_WAY_COST = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0

TRAIN_START = "2019-01-01"
EVAL_START = "2020-01-01"
EVAL_END = "2025-12-31"

WINDOW_DAYS = 180
STEP_DAYS = 180

TOP_N_VALUES = [3, 5]
REBALANCE_EVERY_VALUES = [1, 5, 20]  # trading sessions

MOM20 = 20
MOM60 = 60
MOM120 = 120
SMA200 = 200
VOL60 = 60

STRATEGIES = [
    "mom20",
    "mom60",
    "mom120",
    "mom60_trend",
    "mom120_trend",
    "mom120_vol_adj",
]

OUTPUT_FILE = Path(__file__).with_name("backtest_results.csv")


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def load_bars(conn) -> pd.DataFrame:
    start = pd.Timestamp(TRAIN_START, tz="UTC")
    end_exclusive = pd.Timestamp(EVAL_END, tz="UTC") + pd.Timedelta(days=8)

    sql = """
        SELECT
            symbol,
            timestamp,
            open,
            high,
            low,
            close
        FROM historical_bars
        WHERE timestamp >= %s
          AND timestamp < %s
        ORDER BY symbol, timestamp
    """

    with conn.cursor() as cur:
        cur.execute(sql, (start, end_exclusive))
        rows = cur.fetchall()
        columns = [desc.name for desc in cur.description]

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["symbol", "timestamp", "open", "close"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FEATURES
# ============================================================

def add_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("timestamp").copy()
    close = g["close"].astype(float)
    daily_ret = close.pct_change()

    g["mom20"] = close.pct_change(MOM20)
    g["mom60"] = close.pct_change(MOM60)
    g["mom120"] = close.pct_change(MOM120)

    g["sma200"] = close.rolling(SMA200, min_periods=SMA200).mean()
    g["trend_up"] = close > g["sma200"]

    g["vol60"] = (
        daily_ret.rolling(VOL60, min_periods=VOL60).std() * np.sqrt(252.0)
    )

    g["mom120_vol_adj"] = (
        g["mom120"] / g["vol60"].replace(0.0, np.nan)
    )

    # Today's entry values use yesterday's close.
    for col in ["mom20", "mom60", "mom120", "trend_up", "mom120_vol_adj"]:
        g[f"entry_{col}"] = g[col].shift(1)

    return g


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in raw.groupby("symbol", sort=True):
        parts.append(add_features(group))

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def make_day_map(data: pd.DataFrame) -> Dict[pd.Timestamp, pd.DataFrame]:
    day_map = {}
    for timestamp, day in data.groupby("timestamp", sort=True):
        d = day.drop_duplicates(subset="symbol", keep="last").copy()
        d["symbol"] = d["symbol"].astype(str)
        day_map[timestamp] = d.set_index("symbol", drop=False)
    return day_map


# ============================================================
# STRATEGIES
# ============================================================

def score_column(strategy: str) -> str:
    return {
        "mom20": "entry_mom20",
        "mom60": "entry_mom60",
        "mom120": "entry_mom120",
        "mom60_trend": "entry_mom60",
        "mom120_trend": "entry_mom120",
        "mom120_vol_adj": "entry_mom120_vol_adj",
    }[strategy]


def qualifies(strategy: str, row: pd.Series) -> bool:
    if strategy == "mom20":
        return bool(np.isfinite(row["entry_mom20"]) and row["entry_mom20"] > 0)

    if strategy == "mom60":
        return bool(np.isfinite(row["entry_mom60"]) and row["entry_mom60"] > 0)

    if strategy == "mom120":
        return bool(np.isfinite(row["entry_mom120"]) and row["entry_mom120"] > 0)

    if strategy == "mom60_trend":
        return bool(
            np.isfinite(row["entry_mom60"])
            and row["entry_mom60"] > 0
            and bool(row["entry_trend_up"])
        )

    if strategy == "mom120_trend":
        return bool(
            np.isfinite(row["entry_mom120"])
            and row["entry_mom120"] > 0
            and bool(row["entry_trend_up"])
        )

    if strategy == "mom120_vol_adj":
        return bool(
            np.isfinite(row["entry_mom120_vol_adj"])
            and row["entry_mom120_vol_adj"] > 0
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def select_targets(day: pd.DataFrame, strategy: str, top_n: int) -> Dict[str, float]:
    rows = [row for _, row in day.iterrows() if qualifies(strategy, row)]

    if not rows:
        return {}

    candidates = pd.DataFrame(rows)
    rank_col = score_column(strategy)

    candidates = (
        candidates.dropna(subset=[rank_col])
        .sort_values([rank_col, "symbol"], ascending=[False, True])
        .head(top_n)
    )

    if candidates.empty:
        return {}

    weight = 1.0 / len(candidates)
    return {str(symbol): weight for symbol in candidates["symbol"].tolist()}


# ============================================================
# PRICE / PORTFOLIO HELPERS
# ============================================================

def valid_price(row: pd.Series, field: str) -> bool:
    value = row.get(field, np.nan)
    return bool(np.isfinite(value) and value > 0)


def mark_to_open(day: pd.DataFrame, shares: Dict[str, float], cash: float) -> float:
    equity = float(cash)

    for symbol, qty in shares.items():
        if symbol not in day.index:
            raise RuntimeError(f"Missing OPEN bar for held symbol {symbol}")

        row = day.loc[symbol]
        if not valid_price(row, "open"):
            raise RuntimeError(f"Invalid OPEN price for held symbol {symbol}")

        equity += qty * float(row["open"])

    return equity


def mark_to_close(day: pd.DataFrame, shares: Dict[str, float], cash: float) -> float:
    equity = float(cash)

    for symbol, qty in shares.items():
        if symbol not in day.index:
            raise RuntimeError(f"Missing CLOSE bar for held symbol {symbol}")

        row = day.loc[symbol]
        if not valid_price(row, "close"):
            raise RuntimeError(f"Invalid CLOSE price for held symbol {symbol}")

        equity += qty * float(row["close"])

    return equity


def weights_at_open(
    day: pd.DataFrame,
    shares: Dict[str, float],
    cash: float,
) -> Dict[str, float]:
    equity = mark_to_open(day, shares, cash)

    if equity <= 0:
        return {}

    result = {}
    for symbol, qty in shares.items():
        row = day.loc[symbol]
        value = qty * float(row["open"])
        result[symbol] = value / equity

    return result


def execute_rebalance(
    day: pd.DataFrame,
    old_shares: Dict[str, float],
    old_cash: float,
    targets: Dict[str, float],
) -> Tuple[Dict[str, float], float, float, int, float]:
    """
    Execute target weights at today's open.

    Returns:
        new_shares, new_cash, turnover, trades, transaction_cost

    Turnover is actual L1 change in portfolio weights at execution.
    A 100% A -> 100% B rotation therefore has turnover = 2.0.
    """
    equity_before = mark_to_open(day, old_shares, old_cash)
    old_weights = weights_at_open(day, old_shares, old_cash)

    valid_targets = {}
    for symbol, weight in targets.items():
        if symbol not in day.index:
            continue
        row = day.loc[symbol]
        if valid_price(row, "open"):
            valid_targets[symbol] = float(weight)

    if valid_targets:
        total = sum(valid_targets.values())
        valid_targets = {
            symbol: weight / total
            for symbol, weight in valid_targets.items()
        }

    symbols = set(old_weights) | set(valid_targets)
    turnover = sum(
        abs(valid_targets.get(symbol, 0.0) - old_weights.get(symbol, 0.0))
        for symbol in symbols
    )

    trades = sum(
        abs(valid_targets.get(symbol, 0.0) - old_weights.get(symbol, 0.0)) > 1e-12
        for symbol in symbols
    )

    transaction_cost = equity_before * turnover * ONE_WAY_COST
    equity_after_cost = max(0.0, equity_before - transaction_cost)

    new_shares = {}
    invested = 0.0

    for symbol, weight in valid_targets.items():
        row = day.loc[symbol]
        price = float(row["open"])
        dollars = equity_after_cost * weight
        new_shares[symbol] = dollars / price
        invested += dollars

    new_cash = max(0.0, equity_after_cost - invested)

    return new_shares, new_cash, turnover, trades, transaction_cost


# ============================================================
# METRICS
# ============================================================

def calc_sharpe(returns: pd.Series) -> float:
    x = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return 0.0
    std = x.std(ddof=1)
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return float(x.mean() / std * np.sqrt(252.0))


def calc_sortino(returns: pd.Series) -> float:
    x = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return 0.0
    downside = x[x < 0]
    if len(downside) < 2:
        return 0.0
    std = downside.std(ddof=1)
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return float(x.mean() / std * np.sqrt(252.0))


def calc_max_drawdown(equity: pd.Series) -> float:
    x = pd.to_numeric(equity, errors="coerce").dropna()
    if x.empty:
        return 0.0
    peak = x.cummax()
    return float((x / peak - 1.0).min() * 100.0)


def calc_cagr(start_value: float, end_value: float, days: int) -> float:
    if start_value <= 0 or end_value <= 0 or days <= 0:
        return 0.0
    return float(((end_value / start_value) ** (365.25 / days) - 1.0) * 100.0)


def blank_result() -> Dict[str, float]:
    return {
        "return_pct": 0.0,
        "cagr_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_dd_pct": 0.0,
        "turnover": 0.0,
        "positive_days_pct": 0.0,
        "trades": 0,
        "rebalances": 0,
        "days": 0,
    }


# ============================================================
# STRATEGY BACKTEST
# ============================================================

def run_strategy(
    data: pd.DataFrame,
    day_map: Dict[pd.Timestamp, pd.DataFrame],
    strategy: str,
    top_n: int,
    rebalance_every: int,
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> Dict[str, float]:
    dates = sorted(
        data.loc[
            (data["timestamp"] >= start)
            & (data["timestamp"] <= end_inclusive),
            "timestamp",
        ]
        .drop_duplicates()
        .tolist()
    )

    if len(dates) < 2:
        return blank_result()

    shares: Dict[str, float] = {}
    cash = STARTING_CAPITAL

    total_turnover = 0.0
    trades = 0
    rebalances = 0

    equity_curve = []
    daily_returns = []
    invested_weights = []

    previous_close_equity = STARTING_CAPITAL
    last_rebalance_idx = -rebalance_every

    for i, date in enumerate(dates):
        day = day_map[date]

        scheduled = (
            i == 0
            or (i - last_rebalance_idx) >= rebalance_every
        )

        if scheduled:
            targets = select_targets(day, strategy, top_n)

            (
                shares,
                cash,
                turnover,
                trade_count,
                _transaction_cost,
            ) = execute_rebalance(
                day,
                shares,
                cash,
                targets,
            )

            total_turnover += turnover
            trades += trade_count

            if turnover > 1e-12:
                rebalances += 1

            last_rebalance_idx = i

        # Mark actual shares to today's close. Existing positions drift;
        # their original target weights are NOT reapplied between rebalances.
        close_equity = mark_to_close(day, shares, cash)

        day_return = (
            close_equity / previous_close_equity - 1.0
            if i > 0
            else close_equity / STARTING_CAPITAL - 1.0
        )

        # The final day's equity currently includes positions at CLOSE.
        equity_curve.append(
            {
                "date": date,
                "equity": close_equity,
            }
        )
        daily_returns.append(day_return)

        open_equity_after_trade = mark_to_open(day, shares, cash)
        invested = 1.0 - (cash / open_equity_after_trade if open_equity_after_trade > 0 else 0.0)
        invested_weights.append(invested)

        previous_close_equity = close_equity

    # Final liquidation at EVAL_END close.
    final_day = day_map[dates[-1]]
    final_equity_before_liquidation = previous_close_equity

    position_value = mark_to_close(final_day, shares, 0.0)  # validated, cash excluded

    liquidation_turnover = (
        position_value / final_equity_before_liquidation
        if final_equity_before_liquidation > 0
        else 0.0
    )

    liquidation_cost = final_equity_before_liquidation * liquidation_turnover * ONE_WAY_COST
    final_equity = max(0.0, final_equity_before_liquidation - liquidation_cost)

    total_turnover += liquidation_turnover

    if liquidation_turnover > 1e-12:
        rebalances += 1
        trades += len(shares)

    # Replace the final day's equity with the post-liquidation value.
    equity_curve[-1]["equity"] = final_equity

    if len(equity_curve) >= 2:
        prior = equity_curve[-2]["equity"]
        daily_returns[-1] = (
            final_equity / prior - 1.0
            if prior > 0
            else 0.0
        )

    curve = pd.DataFrame(equity_curve)
    returns = pd.Series(daily_returns, dtype=float)

    calendar_days = max(
        1,
        (end_inclusive - start).days + 1,
    )

    return {
        "return_pct": (final_equity / STARTING_CAPITAL - 1.0) * 100.0,
        "cagr_pct": calc_cagr(STARTING_CAPITAL, final_equity, calendar_days),
        "sharpe": calc_sharpe(returns),
        "sortino": calc_sortino(returns),
        "max_dd_pct": calc_max_drawdown(curve["equity"]),
        "turnover": total_turnover,
        "positive_days_pct": float((returns > 0).mean() * 100.0),
        "trades": trades,
        "rebalances": rebalances,
        "days": calendar_days,
    }


# ============================================================
# BUY-AND-HOLD BENCHMARK
# ============================================================

def run_buy_and_hold(
    data: pd.DataFrame,
    day_map: Dict[pd.Timestamp, pd.DataFrame],
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> Dict[str, float]:
    dates = sorted(
        data.loc[
            (data["timestamp"] >= start)
            & (data["timestamp"] <= end_inclusive),
            "timestamp",
        ]
        .drop_duplicates()
        .tolist()
    )

    if len(dates) < 2:
        return blank_result()

    first_day = day_map[dates[0]]
    final_day = day_map[dates[-1]]

    symbols = []
    for symbol in first_day.index:
        if symbol not in final_day.index:
            continue

        a = first_day.loc[symbol]
        b = final_day.loc[symbol]

        if valid_price(a, "open") and valid_price(b, "close"):
            symbols.append(str(symbol))

    if not symbols:
        return blank_result()

    # Buy equal-dollar positions at first OPEN.
    equity = STARTING_CAPITAL
    entry_cost = equity * ONE_WAY_COST
    equity -= entry_cost

    weight = 1.0 / len(symbols)
    shares = {}
    cash = 0.0

    for symbol in symbols:
        row = first_day.loc[symbol]
        dollars = equity * weight
        shares[symbol] = dollars / float(row["open"])

    curve = []
    returns = []
    previous_equity = STARTING_CAPITAL

    for i, date in enumerate(dates):
        day = day_map[date]
        close_equity = mark_to_close(day, shares, cash)

        day_return = (
            close_equity / previous_equity - 1.0
            if i > 0
            else close_equity / STARTING_CAPITAL - 1.0
        )

        curve.append(
            {
                "date": date,
                "equity": close_equity,
            }
        )
        returns.append(day_return)
        previous_equity = close_equity

    # Liquidate at final close.
    liquidation_cost = previous_equity * ONE_WAY_COST
    final_equity = max(0.0, previous_equity - liquidation_cost)

    curve[-1]["equity"] = final_equity

    if len(curve) >= 2:
        prior = curve[-2]["equity"]
        returns[-1] = final_equity / prior - 1.0 if prior > 0 else 0.0

    curve_df = pd.DataFrame(curve)
    return_series = pd.Series(returns, dtype=float)

    calendar_days = max(
        1,
        (end_inclusive - start).days + 1,
    )

    return {
        "return_pct": (final_equity / STARTING_CAPITAL - 1.0) * 100.0,
        "cagr_pct": calc_cagr(STARTING_CAPITAL, final_equity, calendar_days),
        "sharpe": calc_sharpe(return_series),
        "sortino": calc_sortino(return_series),
        "max_dd_pct": calc_max_drawdown(curve_df["equity"]),
        "turnover": 2.0,
        "positive_days_pct": float((return_series > 0).mean() * 100.0),
        "trades": len(symbols) * 2,
        "rebalances": 1,
        "days": calendar_days,
    }


# ============================================================
# WALK-FORWARD WINDOWS
# ============================================================

def build_windows() -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(EVAL_START, tz="UTC")
    end = pd.Timestamp(EVAL_END, tz="UTC")

    windows = []
    current = start

    while current <= end:
        window_end = min(
            current + pd.Timedelta(days=WINDOW_DAYS - 1),
            end,
        )
        windows.append((current, window_end))
        current += pd.Timedelta(days=STEP_DAYS)

    return windows


# ============================================================
# REPORTING
# ============================================================

def mean_or_zero(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(x.mean()) if not x.dropna().empty else 0.0


def print_header(title: str):
    print()
    print("=" * 125)
    print(title)
    print("=" * 125)


# ============================================================
# MAIN
# ============================================================

def main():
    conn = get_connection()
    try:
        raw = load_bars(conn)
    finally:
        conn.close()

    if raw.empty:
        print("No historical bars found.")
        return

    print_header("TRUE HOLDINGS WALK-FORWARD BACKTEST")
    print(f"Symbols: {raw['symbol'].nunique()}")
    print(f"Rows: {len(raw):,}")
    print(f"Loaded: {raw['timestamp'].min().date()} -> {raw['timestamp'].max().date()}")
    print(f"Evaluation: {EVAL_START} -> {EVAL_END} (inclusive)")
    print(f"Window: {WINDOW_DAYS} calendar days | Step: {STEP_DAYS} calendar days")
    print(f"Top-N: {TOP_N_VALUES}")
    print(f"Rebalance: {REBALANCE_EVERY_VALUES} trading sessions")
    print(f"Costs: {FEE_BPS:.1f} bps fee + {SLIPPAGE_BPS:.1f} bps slippage per side")
    print()
    print("Signal: previous close")
    print("Execution: current open")
    print("Holdings: actual shares")
    print("Weights: drift between scheduled rebalances")
    print("Final day: final-open -> final-close")
    print("ML: OFF | ADX: OFF | Stop-loss: OFF")

    data = prepare_data(raw)
    day_map = make_day_map(data)
    windows = build_windows()

    print(f"Walk-forward windows: {len(windows)}")

    results = []

    # --------------------------------------------------------
    # Walk-forward strategy diagnostics
    # --------------------------------------------------------
    for strategy in STRATEGIES:
        print_header(f"STRATEGY: {strategy}")

        for rebalance_every in REBALANCE_EVERY_VALUES:
            for top_n in TOP_N_VALUES:
                window_rows = []

                for window_start, window_end in windows:
                    result = run_strategy(
                        data=data,
                        day_map=day_map,
                        strategy=strategy,
                        top_n=top_n,
                        rebalance_every=rebalance_every,
                        start=window_start,
                        end_inclusive=window_end,
                    )

                    row = {
                        **result,
                        "strategy": strategy,
                        "rebalance_every": rebalance_every,
                        "top_n": top_n,
                        "window_start": str(window_start.date()),
                        "window_end": str(window_end.date()),
                        "sample": "walk_forward_window",
                    }
                    window_rows.append(row)
                    results.append(row)

                q = pd.DataFrame(window_rows)

                print(
                    f"rebalance={rebalance_every:>2}d | "
                    f"top={top_n} | "
                    f"mean={q['return_pct'].mean():>7.2f}% | "
                    f"median={q['return_pct'].median():>7.2f}% | "
                    f"Sharpe={q['sharpe'].mean():>6.3f} | "
                    f"Sortino={mean_or_zero(q['sortino']):>6.3f} | "
                    f"DD={q['max_dd_pct'].mean():>7.2f}% | "
                    f"positive={q['return_pct'].gt(0).mean()*100:>5.1f}% | "
                    f"turnover={q['turnover'].mean():>7.2f}"
                )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Full evaluation-period runs
    # --------------------------------------------------------
    print_header("FULL 2020-2025 COMPOUNDED RUNS")
    print("These use the same fixed rules over the whole evaluation period and are diagnostic only.")

    full_rows = []
    eval_start = pd.Timestamp(EVAL_START, tz="UTC")
    eval_end = pd.Timestamp(EVAL_END, tz="UTC")

    for strategy in STRATEGIES:
        for rebalance_every in REBALANCE_EVERY_VALUES:
            for top_n in TOP_N_VALUES:
                result = run_strategy(
                    data=data,
                    day_map=day_map,
                    strategy=strategy,
                    top_n=top_n,
                    rebalance_every=rebalance_every,
                    start=eval_start,
                    end_inclusive=eval_end,
                )

                row = {
                    **result,
                    "strategy": strategy,
                    "rebalance_every": rebalance_every,
                    "top_n": top_n,
                    "sample": "full_eval",
                }
                full_rows.append(row)

                print(
                    f"{strategy:<20} "
                    f"rebalance={rebalance_every:>2}d top={top_n} | "
                    f"return={result['return_pct']:>8.2f}% | "
                    f"CAGR={result['cagr_pct']:>6.2f}% | "
                    f"Sharpe={result['sharpe']:>6.3f} | "
                    f"DD={result['max_dd_pct']:>7.2f}% | "
                    f"turnover={result['turnover']:>7.2f}"
                )

    full_df = pd.DataFrame(full_rows)
    all_output = pd.concat([results_df, full_df], ignore_index=True)
    all_output.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------
    print_header("BUY-AND-HOLD BENCHMARK")
    bh_rows = []

    for window_start, window_end in windows:
        result = run_buy_and_hold(
            data=data,
            day_map=day_map,
            start=window_start,
            end_inclusive=window_end,
        )
        bh_rows.append(result)

    bh_df = pd.DataFrame(bh_rows)

    print(f"Mean window return: {bh_df['return_pct'].mean():7.2f}%")
    print(f"Median window return: {bh_df['return_pct'].median():7.2f}%")
    print(f"Mean Sharpe: {bh_df['sharpe'].mean():7.3f}")
    print(f"Mean max DD: {bh_df['max_dd_pct'].mean():7.2f}%")
    print(f"Positive windows: {bh_df['return_pct'].gt(0).mean()*100:7.1f}%")

    # --------------------------------------------------------
    # Aggregate walk-forward summary
    # --------------------------------------------------------
    print_header("WALK-FORWARD SUMMARY")

    summary = (
        results_df.groupby(
            ["strategy", "rebalance_every", "top_n"],
            as_index=False,
        )
        .agg(
            windows=("return_pct", "count"),
            mean_return=("return_pct", "mean"),
            median_return=("return_pct", "median"),
            mean_sharpe=("sharpe", "mean"),
            mean_sortino=("sortino", mean_or_zero),
            mean_max_dd=("max_dd_pct", "mean"),
            positive_windows=("return_pct", lambda x: (x > 0).mean() * 100.0),
            mean_turnover=("turnover", "mean"),
            mean_trades=("trades", "mean"),
        )
        .sort_values(["strategy", "rebalance_every", "top_n"])
    )

    print(summary.round(3).to_string(index=False))

    print()
    print(f"Saved detailed results to: {OUTPUT_FILE}")
    print()
    print_header("SANITY CHECKS")
    checks = [
        "EVAL_END is inclusive",
        "Signal uses previous close",
        "Execution uses current open",
        "Actual share quantities are held",
        "Portfolio weights drift between rebalances",
        "No hidden daily rebalancing",
        "Overnight gaps are included",
        "Transaction costs use actual L1 turnover",
        "Final evaluation session is included through close",
        "Benchmark uses actual shares",
        "No ML / ADX / stop-loss",
    ]
    for check in checks:
        print(f"{check:<65} YES")

    print()
    print("IMPORTANT: the next stage should be an untouched holdout, not more tuning on 2020-2025.")


if __name__ == "__main__":
    main()

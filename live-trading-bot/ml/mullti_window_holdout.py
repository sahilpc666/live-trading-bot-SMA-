import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg2


# ============================================================
# MULTI-WINDOW HOLDOUT (expanding-window walk-forward)
#
# holdout_test.py answered "does the selected config beat buy-and-
# hold on ONE 2024-2025 holdout" — and the answer was no. One
# holdout year is one data point; it could be an unlucky year for
# momentum rather than proof momentum never works here.
#
# This script repeats the same selection -> lock -> holdout process
# across FOUR independent holdout years (2022, 2023, 2024, 2025),
# each time selecting only on years strictly before that holdout
# year (an expanding window — 2022's selection never sees 2022's
# data, 2023's selection never sees 2023's data, etc). No holdout
# year is ever reused as a selection year in a later fold, so all
# four holdout results are genuinely out-of-sample.
#
# The selection rule (select_config) is IDENTICAL to holdout_test.py
# and is not touched here — same discipline, applied four times
# instead of once.
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

# Extra lookback before the earliest selection window so the
# 120/200-day features are already warm on day one.
LOAD_START = "2019-01-01"
LOAD_END = "2025-12-31"

TOP_N_VALUES = [3, 5]
REBALANCE_EVERY_VALUES = [1, 5, 20]

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

SELECTION_TOP_K = 10

# (selection_start, selection_end, holdout_start, holdout_end)
# Expanding window: each fold's selection period is everything from
# 2020-01-01 up to (but not including) its holdout year. Each
# holdout year appears exactly once, in exactly one fold, and is
# never part of any fold's selection period.
FOLDS = [
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]

OUTPUT_DIR = Path(__file__).parent
FOLD_DETAIL_FILE = OUTPUT_DIR / "multi_window_fold_detail.csv"
FOLD_SUMMARY_FILE = OUTPUT_DIR / "multi_window_summary.csv"


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def load_bars(conn) -> pd.DataFrame:
    start = pd.Timestamp(LOAD_START, tz="UTC")
    end_exclusive = pd.Timestamp(LOAD_END, tz="UTC") + pd.Timedelta(days=8)

    sql = """
        SELECT symbol, timestamp, open, high, low, close
        FROM historical_bars
        WHERE timestamp >= %s AND timestamp < %s
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

    return (
        df.dropna(subset=["symbol", "timestamp", "open", "close"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


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

    g["vol60"] = daily_ret.rolling(VOL60, min_periods=VOL60).std() * np.sqrt(252.0)
    g["mom120_vol_adj"] = g["mom120"] / g["vol60"].replace(0.0, np.nan)

    for col in ["mom20", "mom60", "mom120", "trend_up", "mom120_vol_adj"]:
        g[f"entry_{col}"] = g[col].shift(1)

    return g


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    parts = [add_features(group) for _, group in raw.groupby("symbol", sort=True)]
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
            np.isfinite(row["entry_mom60"]) and row["entry_mom60"] > 0
            and bool(row["entry_trend_up"])
        )
    if strategy == "mom120_trend":
        return bool(
            np.isfinite(row["entry_mom120"]) and row["entry_mom120"] > 0
            and bool(row["entry_trend_up"])
        )
    if strategy == "mom120_vol_adj":
        return bool(
            np.isfinite(row["entry_mom120_vol_adj"]) and row["entry_mom120_vol_adj"] > 0
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


def weights_at_open(day: pd.DataFrame, shares: Dict[str, float], cash: float) -> Dict[str, float]:
    equity = mark_to_open(day, shares, cash)
    if equity <= 0:
        return {}
    result = {}
    for symbol, qty in shares.items():
        row = day.loc[symbol]
        result[symbol] = (qty * float(row["open"])) / equity
    return result


def execute_rebalance(
    day: pd.DataFrame,
    old_shares: Dict[str, float],
    old_cash: float,
    targets: Dict[str, float],
) -> Tuple[Dict[str, float], float, float, int, float]:
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
        valid_targets = {s: w / total for s, w in valid_targets.items()}

    symbols = set(old_weights) | set(valid_targets)
    turnover = sum(
        abs(valid_targets.get(s, 0.0) - old_weights.get(s, 0.0)) for s in symbols
    )
    trades = sum(
        abs(valid_targets.get(s, 0.0) - old_weights.get(s, 0.0)) > 1e-12 for s in symbols
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
        "return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "max_dd_pct": 0.0, "turnover": 0.0, "positive_days_pct": 0.0,
        "trades": 0, "rebalances": 0, "days": 0,
    }


# ============================================================
# STRATEGY BACKTEST (single fixed-period run — liquidation fix applied)
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
            (data["timestamp"] >= start) & (data["timestamp"] <= end_inclusive),
            "timestamp",
        ].drop_duplicates().tolist()
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
    previous_close_equity = STARTING_CAPITAL
    last_rebalance_idx = -rebalance_every

    for i, date in enumerate(dates):
        day = day_map[date]
        scheduled = i == 0 or (i - last_rebalance_idx) >= rebalance_every

        if scheduled:
            targets = select_targets(day, strategy, top_n)
            shares, cash, turnover, trade_count, _cost = execute_rebalance(
                day, shares, cash, targets
            )
            total_turnover += turnover
            trades += trade_count
            if turnover > 1e-12:
                rebalances += 1
            last_rebalance_idx = i

        close_equity = mark_to_close(day, shares, cash)
        day_return = (
            close_equity / previous_close_equity - 1.0
            if i > 0 else close_equity / STARTING_CAPITAL - 1.0
        )
        equity_curve.append({"date": date, "equity": close_equity})
        daily_returns.append(day_return)
        previous_close_equity = close_equity

    final_day = day_map[dates[-1]]
    final_equity_before_liquidation = previous_close_equity
    position_value = mark_to_close(final_day, shares, 0.0)

    liquidation_turnover = (
        position_value / final_equity_before_liquidation
        if final_equity_before_liquidation > 0 else 0.0
    )
    liquidation_cost = final_equity_before_liquidation * liquidation_turnover * ONE_WAY_COST
    final_equity = max(0.0, final_equity_before_liquidation - liquidation_cost)
    total_turnover += liquidation_turnover

    if liquidation_turnover > 1e-12:
        rebalances += 1
        trades += len(shares)

    equity_curve[-1]["equity"] = final_equity
    if len(equity_curve) >= 2:
        prior = equity_curve[-2]["equity"]
        daily_returns[-1] = final_equity / prior - 1.0 if prior > 0 else 0.0

    curve = pd.DataFrame(equity_curve)
    returns = pd.Series(daily_returns, dtype=float)
    calendar_days = max(1, (end_inclusive - start).days + 1)

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
            (data["timestamp"] >= start) & (data["timestamp"] <= end_inclusive),
            "timestamp",
        ].drop_duplicates().tolist()
    )
    if len(dates) < 2:
        return blank_result()

    first_day = day_map[dates[0]]
    final_day = day_map[dates[-1]]

    symbols = []
    for symbol in first_day.index:
        if symbol not in final_day.index:
            continue
        a, b = first_day.loc[symbol], final_day.loc[symbol]
        if valid_price(a, "open") and valid_price(b, "close"):
            symbols.append(str(symbol))
    if not symbols:
        return blank_result()

    equity = STARTING_CAPITAL
    equity -= equity * ONE_WAY_COST
    weight = 1.0 / len(symbols)
    shares = {}
    cash = 0.0
    for symbol in symbols:
        row = first_day.loc[symbol]
        shares[symbol] = (equity * weight) / float(row["open"])

    curve, returns = [], []
    previous_equity = STARTING_CAPITAL
    for i, date in enumerate(dates):
        day = day_map[date]
        close_equity = mark_to_close(day, shares, cash)
        day_return = (
            close_equity / previous_equity - 1.0
            if i > 0 else close_equity / STARTING_CAPITAL - 1.0
        )
        curve.append({"date": date, "equity": close_equity})
        returns.append(day_return)
        previous_equity = close_equity

    liquidation_cost = previous_equity * ONE_WAY_COST
    final_equity = max(0.0, previous_equity - liquidation_cost)
    curve[-1]["equity"] = final_equity
    if len(curve) >= 2:
        prior = curve[-2]["equity"]
        returns[-1] = final_equity / prior - 1.0 if prior > 0 else 0.0

    curve_df = pd.DataFrame(curve)
    return_series = pd.Series(returns, dtype=float)
    calendar_days = max(1, (end_inclusive - start).days + 1)

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
# SELECTION RULE — identical to holdout_test.py, unchanged
# ============================================================

def select_config(selection_df: pd.DataFrame) -> pd.Series:
    """
    Same rule as holdout_test.py: rank by Sharpe over the selection
    period, take the top SELECTION_TOP_K, then pick the smallest
    (least negative) max_dd_pct among those. Applied independently
    within each fold, using only that fold's selection-period data.
    """
    ranked = selection_df.sort_values("sharpe", ascending=False).head(SELECTION_TOP_K)
    return ranked.sort_values("max_dd_pct", ascending=False).iloc[0]


def run_grid(data, day_map, start, end_inclusive) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        for rebalance_every in REBALANCE_EVERY_VALUES:
            for top_n in TOP_N_VALUES:
                result = run_strategy(
                    data=data, day_map=day_map, strategy=strategy,
                    top_n=top_n, rebalance_every=rebalance_every,
                    start=start, end_inclusive=end_inclusive,
                )
                rows.append({**result, "strategy": strategy,
                             "rebalance_every": rebalance_every, "top_n": top_n})
    return pd.DataFrame(rows)


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

    data = prepare_data(raw)
    day_map = make_day_map(data)

    print("=" * 100)
    print(f"MULTI-WINDOW HOLDOUT — {len(FOLDS)} independent folds")
    print("=" * 100)

    fold_rows = []
    detail_rows = []

    for fold_idx, (sel_start, sel_end, hold_start, hold_end) in enumerate(FOLDS, start=1):
        sel_start_ts = pd.Timestamp(sel_start, tz="UTC")
        sel_end_ts = pd.Timestamp(sel_end, tz="UTC")
        hold_start_ts = pd.Timestamp(hold_start, tz="UTC")
        hold_end_ts = pd.Timestamp(hold_end, tz="UTC")

        print(f"\n--- Fold {fold_idx}: select {sel_start} -> {sel_end}, "
              f"holdout {hold_start} -> {hold_end} ---")

        selection_df = run_grid(data, day_map, sel_start_ts, sel_end_ts)
        chosen = select_config(selection_df)

        holdout_result = run_strategy(
            data=data, day_map=day_map,
            strategy=chosen["strategy"], top_n=int(chosen["top_n"]),
            rebalance_every=int(chosen["rebalance_every"]),
            start=hold_start_ts, end_inclusive=hold_end_ts,
        )
        bh_holdout = run_buy_and_hold(data, day_map, hold_start_ts, hold_end_ts)

        beat_sharpe = holdout_result["sharpe"] > bh_holdout["sharpe"]
        beat_return = holdout_result["return_pct"] > bh_holdout["return_pct"]

        print(
            f"  chosen: {chosen['strategy']}  reb={int(chosen['rebalance_every'])}d  "
            f"top={int(chosen['top_n'])}  "
            f"(selection Sharpe {chosen['sharpe']:.3f})"
        )
        print(
            f"  holdout -> strategy: return {holdout_result['return_pct']:+7.2f}%  "
            f"sharpe {holdout_result['sharpe']:6.3f}  DD {holdout_result['max_dd_pct']:7.2f}%"
        )
        print(
            f"  holdout -> buy&hold: return {bh_holdout['return_pct']:+7.2f}%  "
            f"sharpe {bh_holdout['sharpe']:6.3f}  DD {bh_holdout['max_dd_pct']:7.2f}%"
        )
        print(f"  beat buy-and-hold on Sharpe: {beat_sharpe}   on return: {beat_return}")

        fold_rows.append({
            "fold": fold_idx,
            "selection_period": f"{sel_start} -> {sel_end}",
            "holdout_period": f"{hold_start} -> {hold_end}",
            "chosen_strategy": chosen["strategy"],
            "chosen_rebalance_every": int(chosen["rebalance_every"]),
            "chosen_top_n": int(chosen["top_n"]),
            "selection_sharpe": chosen["sharpe"],
            "holdout_return_pct": holdout_result["return_pct"],
            "holdout_sharpe": holdout_result["sharpe"],
            "holdout_max_dd_pct": holdout_result["max_dd_pct"],
            "bh_holdout_return_pct": bh_holdout["return_pct"],
            "bh_holdout_sharpe": bh_holdout["sharpe"],
            "bh_holdout_max_dd_pct": bh_holdout["max_dd_pct"],
            "beat_bh_sharpe": beat_sharpe,
            "beat_bh_return": beat_return,
        })

        detail_rows.append({"fold": fold_idx, **selection_df.assign(fold=fold_idx).to_dict()})

    summary_df = pd.DataFrame(fold_rows)
    summary_df.to_csv(FOLD_SUMMARY_FILE, index=False)

    print()
    print("=" * 100)
    print("SUMMARY ACROSS ALL FOLDS")
    print("=" * 100)
    print(
        summary_df[[
            "fold", "chosen_strategy", "chosen_rebalance_every", "chosen_top_n",
            "holdout_return_pct", "holdout_sharpe", "bh_holdout_return_pct",
            "bh_holdout_sharpe", "beat_bh_sharpe", "beat_bh_return",
        ]].round(3).to_string(index=False)
    )

    n_folds = len(summary_df)
    n_beat_sharpe = int(summary_df["beat_bh_sharpe"].sum())
    n_beat_return = int(summary_df["beat_bh_return"].sum())
    mean_sharpe_edge = (summary_df["holdout_sharpe"] - summary_df["bh_holdout_sharpe"]).mean()
    mean_return_edge = (summary_df["holdout_return_pct"] - summary_df["bh_holdout_return_pct"]).mean()
    unique_configs = summary_df[
        ["chosen_strategy", "chosen_rebalance_every", "chosen_top_n"]
    ].drop_duplicates().shape[0]

    print()
    print(f"Folds where the strategy beat buy-and-hold on Sharpe : {n_beat_sharpe} / {n_folds}")
    print(f"Folds where the strategy beat buy-and-hold on return : {n_beat_return} / {n_folds}")
    print(f"Mean Sharpe edge over buy-and-hold (can be negative) : {mean_sharpe_edge:+.3f}")
    print(f"Mean return edge over buy-and-hold (can be negative) : {mean_return_edge:+.2f} pct points")
    print(f"Distinct configs selected across {n_folds} folds       : {unique_configs}")
    print(
        "  (if this is close to n_folds, the grid is picking a different 'winner' "
        "each time rather than converging on one robust config — itself evidence "
        "the selection step is fitting noise, not finding a stable edge.)"
    )

    print(f"\nPer-fold summary saved: {FOLD_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
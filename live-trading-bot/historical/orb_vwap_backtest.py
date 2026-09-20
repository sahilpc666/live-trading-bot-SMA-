import os
import numpy as np
import pandas as pd
import psycopg2

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")

# ============================================================
# CONFIGURATION
# ============================================================

# Fixed baseline parameters. Do NOT optimize these on the holdout.
OPENING_RANGE_MINUTES = 15       # 3 x 5-minute bars
BAR_MINUTES = 5

STOP_LOSS_PCT = 0.02
FEE_BPS = 1.0
SLIPPAGE_BPS = 2.0
POSITION_SIZE_PCT = 0.20

STARTING_CASH = 10_000.0

# Keep this period untouched until the development period passes.
HOLDOUT_START = "2026-06-01"

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


def get_symbols(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT symbol
            FROM historical_bars_intraday
            ORDER BY symbol
            """
        )
        return [row[0] for row in cur.fetchall()]


def load_bars(conn, symbol):
    query = """
        SELECT timestamp, trading_date, open, high, low, close, volume
        FROM historical_bars_intraday
        WHERE symbol = %s
        ORDER BY timestamp ASC
    """

    df = pd.read_sql(query, conn, params=(symbol,))

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["timestamp", "trading_date", "open", "high", "low", "close", "volume"]
    ).copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# COST / EXECUTION HELPERS
# ============================================================

def apply_entry_cost(price):
    """
    Buy execution:
        market price + slippage + fee
    """
    return price * (
        1.0
        + SLIPPAGE_BPS / 10_000.0
        + FEE_BPS / 10_000.0
    )


def apply_exit_cost(price):
    """
    Sell execution:
        market price - slippage - fee
    """
    return price * (
        1.0
        - SLIPPAGE_BPS / 10_000.0
        - FEE_BPS / 10_000.0
    )


# ============================================================
# METRICS
# ============================================================

def calculate_max_drawdown(equity_series):
    if equity_series.empty:
        return 0.0

    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak

    return float(drawdown.min())


def calculate_profit_factor(trades_df):
    if trades_df.empty:
        return 0.0

    gross_profit = trades_df.loc[
        trades_df["net_pnl"] > 0, "net_pnl"
    ].sum()

    gross_loss = -trades_df.loc[
        trades_df["net_pnl"] < 0, "net_pnl"
    ].sum()

    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calculate_trade_metrics(trades_df):
    if trades_df.empty:
        return {
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "avg_trade": 0.0,
            "median_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_hold_minutes": 0.0,
            "avg_mae_pct": 0.0,
            "avg_mfe_pct": 0.0,
        }

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] < 0]

    return {
        "trade_count": len(trades_df),
        "win_rate_pct": float((len(wins) / len(trades_df)) * 100.0),
        "avg_trade": float(trades_df["net_pnl"].mean()),
        "median_trade": float(trades_df["net_pnl"].median()),
        "avg_win": float(wins["net_pnl"].mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses["net_pnl"].mean()) if not losses.empty else 0.0,
        "profit_factor": calculate_profit_factor(trades_df),
        "expectancy": float(trades_df["net_pnl"].mean()),
        "avg_hold_minutes": float(trades_df["holding_minutes"].mean()),
        "avg_mae_pct": float(trades_df["mae_pct"].mean()),
        "avg_mfe_pct": float(trades_df["mfe_pct"].mean()),
    }


# ============================================================
# SINGLE-DAY SIMULATION
# ============================================================

def simulate_day(day_bars, starting_cash, symbol):
    """
    Simulate one symbol for one trading day.

    Strategy:
      1. First 15 minutes define the opening-range high.
      2. Session VWAP starts at the first regular-session bar.
      3. A completed 5-minute bar after the opening range generates
         a long signal when:
             close > opening-range high
             AND
             close > session VWAP
      4. The signal is executed at the NEXT 5-minute bar OPEN.
      5. Only one entry per symbol/day.
      6. Exit on:
             - stop-loss hit, or
             - completed bar closes below VWAP, or
             - final bar of session.
      7. Close-based exits are executed at the NEXT bar OPEN.
      8. Stop-loss is evaluated intrabar. With OHLC data, if the
         stop is touched, we conservatively assume the stop price.
      9. Always flat by the end of the session.

    Returns:
        ending_cash, trades, trade_records
    """

    day_bars = day_bars.sort_values("timestamp").reset_index(drop=True).copy()

    n_range_bars = OPENING_RANGE_MINUTES // BAR_MINUTES

    # Need enough bars for opening range + at least one bar after it.
    if len(day_bars) <= n_range_bars:
        return starting_cash, 0, []

    # Opening range uses only the first 15 minutes.
    opening_range = day_bars.iloc[:n_range_bars]
    opening_range_high = float(opening_range["high"].max())

    # Session VWAP using close * volume, starting at session open.
    volume = day_bars["volume"].astype(float)
    close = day_bars["close"].astype(float)

    cumulative_volume = volume.cumsum()
    cumulative_pv = (close * volume).cumsum()

    day_bars["session_vwap"] = np.where(
        cumulative_volume > 0,
        cumulative_pv / cumulative_volume,
        day_bars["close"],
    )

    cash = float(starting_cash)

    shares = 0
    entry_price = None
    entry_timestamp = None

    entry_reference_price = None
    entry_allocation = None

    trade_records = []

    # Used to calculate MFE/MAE after entry.
    max_favorable_price = None
    min_adverse_price = None

    entry_bar_index = None

    # Entry signal is created on one bar and executed on the next.
    pending_entry = False

    # Exit signal is created on one bar and executed on the next.
    pending_exit = False
    pending_exit_reason = None

    for i in range(len(day_bars)):
        row = day_bars.iloc[i]

        current_open = float(row["open"])
        current_high = float(row["high"])
        current_low = float(row["low"])
        current_close = float(row["close"])
        current_vwap = float(row["session_vwap"])

        current_timestamp = row["timestamp"]

        is_last_bar = i == len(day_bars) - 1

        # --------------------------------------------------------
        # 1. Execute pending entry at CURRENT bar OPEN.
        # --------------------------------------------------------
        if pending_entry and shares == 0:
            allocation = cash * POSITION_SIZE_PCT

            executed_entry_price = apply_entry_cost(current_open)

            new_shares = int(allocation // executed_entry_price)

            if new_shares > 0:
                shares = new_shares
                entry_price = executed_entry_price
                entry_reference_price = current_open
                entry_timestamp = current_timestamp
                entry_bar_index = i
                entry_allocation = shares * executed_entry_price

                cash -= entry_allocation

                max_favorable_price = current_high
                min_adverse_price = current_low

            pending_entry = False

        # --------------------------------------------------------
        # 2. Execute pending close-based exit at CURRENT bar OPEN.
        # --------------------------------------------------------
        if pending_exit and shares > 0:
            exit_reference_price = current_open
            executed_exit_price = apply_exit_cost(exit_reference_price)

            proceeds = shares * executed_exit_price
            cash += proceeds

            gross_pnl = shares * (
                exit_reference_price - entry_reference_price
            )

            net_pnl = proceeds - entry_allocation

            holding_minutes = (
                current_timestamp - entry_timestamp
            ).total_seconds() / 60.0

            mfe_pct = (
                (max_favorable_price - entry_reference_price)
                / entry_reference_price
                * 100.0
            )

            mae_pct = (
                (min_adverse_price - entry_reference_price)
                / entry_reference_price
                * 100.0
            )

            trade_records.append(
                {
                    "symbol": symbol,
                    "entry_timestamp": entry_timestamp,
                    "entry_price": entry_price,
                    "exit_timestamp": current_timestamp,
                    "exit_price": executed_exit_price,
                    "shares": shares,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "holding_minutes": holding_minutes,
                    "exit_reason": pending_exit_reason,
                    "mae_pct": mae_pct,
                    "mfe_pct": mfe_pct,
                }
            )

            shares = 0
            entry_price = None
            entry_timestamp = None
            entry_reference_price = None
            entry_allocation = None
            entry_bar_index = None
            max_favorable_price = None
            min_adverse_price = None

            pending_exit = False
            pending_exit_reason = None

        # --------------------------------------------------------
        # 3. If holding a position, update MFE/MAE.
        # --------------------------------------------------------
        if shares > 0:
            max_favorable_price = max(
                max_favorable_price,
                current_high,
            )

            min_adverse_price = min(
                min_adverse_price,
                current_low,
            )

            stop_price = entry_reference_price * (
                1.0 - STOP_LOSS_PCT
            )

            # ----------------------------------------------------
            # Stop-loss is an intrabar event.
            # ----------------------------------------------------
            if current_low <= stop_price:
                executed_exit_price = apply_exit_cost(stop_price)

                proceeds = shares * executed_exit_price
                cash += proceeds

                gross_pnl = shares * (
                    stop_price - entry_reference_price
                )

                net_pnl = proceeds - entry_allocation

                holding_minutes = (
                    current_timestamp - entry_timestamp
                ).total_seconds() / 60.0

                mfe_pct = (
                    (max_favorable_price - entry_reference_price)
                    / entry_reference_price
                    * 100.0
                )

                mae_pct = (
                    (min_adverse_price - entry_reference_price)
                    / entry_reference_price
                    * 100.0
                )

                trade_records.append(
                    {
                        "symbol": symbol,
                        "entry_timestamp": entry_timestamp,
                        "entry_price": entry_price,
                        "exit_timestamp": current_timestamp,
                        "exit_price": executed_exit_price,
                        "shares": shares,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "holding_minutes": holding_minutes,
                        "exit_reason": "stop_loss",
                        "mae_pct": mae_pct,
                        "mfe_pct": mfe_pct,
                    }
                )

                shares = 0
                entry_price = None
                entry_timestamp = None
                entry_reference_price = None
                entry_allocation = None
                entry_bar_index = None
                max_favorable_price = None
                min_adverse_price = None

                # No entry signal should be generated from this same bar.
                pending_entry = False
                pending_exit = False
                pending_exit_reason = None

                continue

            # ----------------------------------------------------
            # Final bar: force exit at final close.
            #
            # This is unavoidable because there is no next bar.
            # ----------------------------------------------------
            if is_last_bar:
                executed_exit_price = apply_exit_cost(current_close)

                proceeds = shares * executed_exit_price
                cash += proceeds

                gross_pnl = shares * (
                    current_close - entry_reference_price
                )

                net_pnl = proceeds - entry_allocation

                holding_minutes = (
                    current_timestamp - entry_timestamp
                ).total_seconds() / 60.0

                mfe_pct = (
                    (max_favorable_price - entry_reference_price)
                    / entry_reference_price
                    * 100.0
                )

                mae_pct = (
                    (min_adverse_price - entry_reference_price)
                    / entry_reference_price
                    * 100.0
                )

                trade_records.append(
                    {
                        "symbol": symbol,
                        "entry_timestamp": entry_timestamp,
                        "entry_price": entry_price,
                        "exit_timestamp": current_timestamp,
                        "exit_price": executed_exit_price,
                        "shares": shares,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "holding_minutes": holding_minutes,
                        "exit_reason": "end_of_day",
                        "mae_pct": mae_pct,
                        "mfe_pct": mfe_pct,
                    }
                )

                shares = 0
                entry_price = None
                entry_timestamp = None
                entry_reference_price = None
                entry_allocation = None
                entry_bar_index = None
                max_favorable_price = None
                min_adverse_price = None

                continue

            # ----------------------------------------------------
            # Close below VWAP generates an exit for NEXT bar.
            # ----------------------------------------------------
            if current_close < current_vwap:
                pending_exit = True
                pending_exit_reason = "vwap_exit"

        # --------------------------------------------------------
        # 4. Generate entry signal at completed bar close.
        #
        # Signal can only be generated after opening range.
        # It executes on the NEXT bar's open.
        # --------------------------------------------------------
        if (
            shares == 0
            and not pending_entry
            and not pending_exit
            and i >= n_range_bars
            and not is_last_bar
        ):
            breakout = current_close > opening_range_high
            above_vwap = current_close > current_vwap

            if breakout and above_vwap:
                pending_entry = True

    # Safety: never carry overnight.
    if shares > 0:
        final_close = float(day_bars.iloc[-1]["close"])
        executed_exit_price = apply_exit_cost(final_close)

        proceeds = shares * executed_exit_price
        cash += proceeds

        gross_pnl = shares * (
            final_close - entry_reference_price
        )

        net_pnl = proceeds - entry_allocation

        holding_minutes = (
            day_bars.iloc[-1]["timestamp"] - entry_timestamp
        ).total_seconds() / 60.0

        mfe_pct = (
            (max_favorable_price - entry_reference_price)
            / entry_reference_price
            * 100.0
        )

        mae_pct = (
            (min_adverse_price - entry_reference_price)
            / entry_reference_price
            * 100.0
        )

        trade_records.append(
            {
                "symbol": symbol,
                "entry_timestamp": entry_timestamp,
                "entry_price": entry_price,
                "exit_timestamp": day_bars.iloc[-1]["timestamp"],
                "exit_price": executed_exit_price,
                "shares": shares,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "holding_minutes": holding_minutes,
                "exit_reason": "safety_end_of_day",
                "mae_pct": mae_pct,
                "mfe_pct": mfe_pct,
            }
        )

        shares = 0

    return cash, len(trade_records), trade_records


# ============================================================
# SINGLE-SYMBOL BACKTEST
# ============================================================

def backtest_symbol(df, period_label, symbol):
    if df.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    cash = STARTING_CASH

    daily_results = []
    all_trades = []

    for trading_date, day_bars in df.groupby("trading_date", sort=True):
        day_start_cash = cash

        cash, _, day_trades = simulate_day(
            day_bars,
            cash,
            symbol,
        )

        all_trades.extend(day_trades)

        daily_return = (
            (cash - day_start_cash) / day_start_cash
            if day_start_cash > 0
            else 0.0
        )

        daily_results.append(
            {
                "date": trading_date,
                "return": daily_return,
                "equity": cash,
                "trades": len(day_trades),
            }
        )

    if not daily_results:
        return None, pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(daily_results)

    trades_df = pd.DataFrame(all_trades)

    # Daily Sharpe.
    daily_std = results_df["return"].std()

    sharpe = (
        results_df["return"].mean() / daily_std * np.sqrt(252)
        if daily_std and daily_std > 0
        else 0.0
    )

    max_drawdown_pct = (
        calculate_max_drawdown(results_df["equity"]) * 100.0
    )

    trade_metrics = calculate_trade_metrics(trades_df)

    total_return_pct = (
        (cash - STARTING_CASH) / STARTING_CASH * 100.0
    )

    # Buy-and-hold is shown only as a contextual reference.
    # It is NOT used as a pass/fail criterion because this strategy
    # is intraday and flat overnight.
    first_close = float(df.iloc[0]["close"])
    last_close = float(df.iloc[-1]["close"])

    bh_return_pct = (
        (last_close - first_close) / first_close * 100.0
    )

    summary = {
        "symbol": symbol,
        "period": period_label,
        "return_pct": total_return_pct,
        "sharpe": float(sharpe),
        "max_drawdown_pct": max_drawdown_pct,
        "bh_return_pct": bh_return_pct,
        "trades": trade_metrics["trade_count"],
        "win_rate_pct": trade_metrics["win_rate_pct"],
        "avg_trade": trade_metrics["avg_trade"],
        "median_trade": trade_metrics["median_trade"],
        "avg_win": trade_metrics["avg_win"],
        "avg_loss": trade_metrics["avg_loss"],
        "profit_factor": trade_metrics["profit_factor"],
        "expectancy": trade_metrics["expectancy"],
        "avg_hold_minutes": trade_metrics["avg_hold_minutes"],
        "avg_mae_pct": trade_metrics["avg_mae_pct"],
        "avg_mfe_pct": trade_metrics["avg_mfe_pct"],
        "trading_days": len(results_df),
    }

    return summary, results_df, trades_df


# ============================================================
# REPORTING
# ============================================================

def print_results(results_df, title):
    print("\n" + "=" * 140)
    print(title)
    print("=" * 140)

    if results_df.empty:
        print("No results.")
        return

    columns = [
        "symbol",
        "return_pct",
        "sharpe",
        "max_drawdown_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "expectancy",
        "avg_hold_minutes",
        "avg_mae_pct",
        "avg_mfe_pct",
        "bh_return_pct",
        "trading_days",
    ]

    display_df = results_df[columns].copy()

    print(display_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def print_aggregate(results_df):
    if results_df.empty:
        return

    print("\n" + "-" * 100)
    print("DEVELOPMENT AGGREGATE")
    print("-" * 100)

    print(
        f"Average return:        {results_df['return_pct'].mean():.2f}%"
    )
    print(
        f"Median return:         {results_df['return_pct'].median():.2f}%"
    )
    print(
        f"Average Sharpe:        {results_df['sharpe'].mean():.3f}"
    )
    print(
        f"Median Sharpe:         {results_df['sharpe'].median():.3f}"
    )
    print(
        f"Average max drawdown:  {results_df['max_drawdown_pct'].mean():.2f}%"
    )
    print(
        f"Average win rate:      {results_df['win_rate_pct'].mean():.2f}%"
    )
    print(
        f"Average profit factor: {results_df['profit_factor'].replace(np.inf, np.nan).mean():.3f}"
    )
    print(
        f"Average expectancy:    ${results_df['expectancy'].mean():.4f}"
    )
    print(
        f"Total round trips:     {int(results_df['trades'].sum())}"
    )

    positive_symbols = int((results_df["return_pct"] > 0).sum())
    negative_symbols = int((results_df["return_pct"] < 0).sum())

    print(
        f"Positive symbols:      {positive_symbols}/{len(results_df)}"
    )
    print(
        f"Negative symbols:      {negative_symbols}/{len(results_df)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        f"Opening range: {OPENING_RANGE_MINUTES} min | "
        f"Stop-loss: {STOP_LOSS_PCT:.1%}"
    )
    print(
        f"Costs: {FEE_BPS}bps fee + "
        f"{SLIPPAGE_BPS}bps slippage per side"
    )
    print(
        "Execution: signal on completed 5-min bar -> "
        "next 5-min bar OPEN"
    )
    print(
        f"Position size: {POSITION_SIZE_PCT:.0%} of available cash"
    )
    print(
        f"Holdout: {HOLDOUT_START} onward (not used for development decisions)\n"
    )

    conn = get_connection()

    try:
        symbols = get_symbols(conn)

        if not symbols:
            print("No symbols found in historical_bars_intraday.")
            return

        holdout_start = pd.Timestamp(HOLDOUT_START).date()

        dev_results = []
        holdout_results = []

        # Store detailed trades in memory so we can optionally export them.
        dev_trades_all = []
        holdout_trades_all = []

        for symbol in symbols:
            print(f"Processing {symbol}...")

            df = load_bars(conn, symbol)

            if df.empty:
                print(f"  No data for {symbol}.")
                continue

            dev_df = df[df["trading_date"] < holdout_start].copy()
            holdout_df = df[df["trading_date"] >= holdout_start].copy()

            dev_summary, _, dev_trades = backtest_symbol(
                dev_df,
                "development",
                symbol,
            )

            if dev_summary:
                dev_results.append(dev_summary)

                if not dev_trades.empty:
                    dev_trades_all.append(dev_trades)

            holdout_summary, _, holdout_trades = backtest_symbol(
                holdout_df,
                "holdout",
                symbol,
            )

            if holdout_summary:
                holdout_results.append(holdout_summary)

                if not holdout_trades.empty:
                    holdout_trades_all.append(holdout_trades)

    finally:
        conn.close()

    dev_results_df = pd.DataFrame(dev_results)

    print_results(
        dev_results_df,
        "DEVELOPMENT PERIOD — BEFORE HOLDOUT",
    )

    print_aggregate(dev_results_df)

    # --------------------------------------------------------
    # Development gate.
    #
    # This is intentionally diagnostic rather than comparing
    # an intraday strategy directly against buy-and-hold.
    # --------------------------------------------------------
    if dev_results_df.empty:
        print("\nRESULT: No development results.")
        return

    avg_return = dev_results_df["return_pct"].mean()
    avg_sharpe = dev_results_df["sharpe"].mean()
    avg_expectancy = dev_results_df["expectancy"].mean()
    avg_profit_factor = (
        dev_results_df["profit_factor"]
        .replace(np.inf, np.nan)
        .mean()
    )

    positive_symbol_fraction = (
        (dev_results_df["return_pct"] > 0).mean()
    )

    print("\n" + "=" * 100)
    print("DEVELOPMENT DIAGNOSTIC")
    print("=" * 100)

    print(
        f"Average return:             {avg_return:.2f}%"
    )
    print(
        f"Average Sharpe:             {avg_sharpe:.3f}"
    )
    print(
        f"Average expectancy/trade:   ${avg_expectancy:.4f}"
    )
    print(
        f"Average profit factor:      {avg_profit_factor:.3f}"
    )
    print(
        f"Positive-symbol fraction:   {positive_symbol_fraction:.1%}"
    )

    # Do NOT use buy-and-hold as the gate.
    #
    # The baseline gate is deliberately simple:
    #   - positive aggregate expectancy
    #   - positive aggregate Sharpe
    #   - majority of symbols not losing
    #
    # These are research gates, not claims that the strategy is
    # deployable if it passes them.
    dev_pass = (
        avg_expectancy > 0
        and avg_sharpe > 0
        and positive_symbol_fraction >= 0.50
    )

    print(
        f"\n[{'PASS' if avg_expectancy > 0 else 'FAIL'}] "
        "Average expectancy > 0"
    )
    print(
        f"[{'PASS' if avg_sharpe > 0 else 'FAIL'}] "
        "Average Sharpe > 0"
    )
    print(
        f"[{'PASS' if positive_symbol_fraction >= 0.50 else 'FAIL'}] "
        "At least 50% of symbols have positive return"
    )

    if not dev_pass:
        print(
            "\nRESULT: ORB+VWAP does not clear the development diagnostic."
        )
        print(
            "Holdout remains untouched and is NOT evaluated."
        )
        return

    # --------------------------------------------------------
    # Holdout is only viewed after development passes.
    # --------------------------------------------------------
    holdout_results_df = pd.DataFrame(holdout_results)

    print_results(
        holdout_results_df,
        "HOLDOUT PERIOD — FIRST UNTOUCHED CHECK",
    )

    if holdout_results_df.empty:
        print("\nRESULT: No holdout data available.")
        return

    print("\n" + "-" * 100)
    print("HOLDOUT SUMMARY")
    print("-" * 100)

    print(
        f"Average return:        {holdout_results_df['return_pct'].mean():.2f}%"
    )
    print(
        f"Average Sharpe:        {holdout_results_df['sharpe'].mean():.3f}"
    )
    print(
        f"Average expectancy:    ${holdout_results_df['expectancy'].mean():.4f}"
    )
    print(
        f"Positive-symbol share: "
        f"{(holdout_results_df['return_pct'] > 0).mean():.1%}"
    )

    holdout_pass = (
        holdout_results_df["expectancy"].mean() > 0
        and holdout_results_df["sharpe"].mean() > 0
        and (holdout_results_df["return_pct"] > 0).mean() >= 0.50
    )

    if holdout_pass:
        print(
            "\nRESULT: Development edge also appeared in the untouched holdout."
        )
        print(
            "This is evidence for further research, NOT automatic deployment."
        )
    else:
        print(
            "\nRESULT: Development result did not generalize to the holdout."
        )
        print(
            "Do not deploy this baseline as-is."
        )

    # --------------------------------------------------------
    # Optional detailed trade exports.
    # --------------------------------------------------------
    if dev_trades_all:
        dev_trade_df = pd.concat(
            dev_trades_all,
            ignore_index=True,
        )
        dev_trade_df.to_csv(
            "orb_vwap_development_trades.csv",
            index=False,
        )
        print(
            "\nSaved development trades: "
            "orb_vwap_development_trades.csv"
        )

    if holdout_trades_all:
        holdout_trade_df = pd.concat(
            holdout_trades_all,
            ignore_index=True,
        )
        holdout_trade_df.to_csv(
            "orb_vwap_holdout_trades.csv",
            index=False,
        )
        print(
            "Saved holdout trades: "
            "orb_vwap_holdout_trades.csv"
        )


if __name__ == "__main__":
    main()

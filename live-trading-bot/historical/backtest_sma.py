import os
import pandas as pd
import psycopg2


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")


# ============================================================
# STRATEGY CONFIG
# Same values as your current signal_engine.py
# ============================================================

SHORT_WINDOW = 100
LONG_WINDOW = 300
THRESHOLD_PCT = 0.003

STARTING_CASH = 10_000.00

# Set to True if you want to print every trade.
# False = only summary + last few trades.
SHOW_ALL_TRADES = False


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


def load_symbols(conn):
    query = """
        SELECT DISTINCT symbol
        FROM historical_bars
        ORDER BY symbol;
    """

    df = pd.read_sql_query(query, conn)
    return df["symbol"].tolist()


def load_symbol_data(conn, symbol):
    query = """
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM historical_bars
        WHERE symbol = %s
        ORDER BY timestamp;
    """

    df = pd.read_sql_query(query, conn, params=(symbol,))

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])

    return df


# ============================================================
# SMA SIGNAL
# ============================================================

def apply_sma_strategy(df):
    df = df.copy()

    # Same SMA windows as live strategy
    df["sma_short"] = (
        df["close"]
        .rolling(window=SHORT_WINDOW)
        .mean()
    )

    df["sma_long"] = (
        df["close"]
        .rolling(window=LONG_WINDOW)
        .mean()
    )

    # Difference between short SMA and long SMA
    df["diff_pct"] = (
        (df["sma_short"] - df["sma_long"])
        / df["sma_long"]
    )

    # Desired position
    #
    #  1 = long
    #  0 = flat
    #
    # This matches the live strategy:
    #
    # diff > +0.3%  -> long
    # diff < -0.3%  -> flat
    #
    # Between the thresholds we HOLD the current position.

    df["desired_position"] = None

    df.loc[
        df["diff_pct"] > THRESHOLD_PCT,
        "desired_position"
    ] = 1

    df.loc[
        df["diff_pct"] < -THRESHOLD_PCT,
        "desired_position"
    ] = 0

    return df


# ============================================================
# BACKTEST
# ============================================================

def backtest_symbol(df, symbol, starting_cash=STARTING_CASH):

    cash = starting_cash
    shares = 0

    trades = []

    entry_price = None
    entry_date = None

    equity_curve = []

    for _, row in df.iterrows():

        date = row["timestamp"]
        close = row["close"]
        desired_position = row["desired_position"]

        # Cannot trade until SMA values exist
        if pd.isna(desired_position):
            equity = cash + shares * close
            equity_curve.append(
                {
                    "timestamp": date,
                    "equity": equity,
                }
            )
            continue

        # ====================================================
        # BUY
        # ====================================================

        if desired_position == 1 and shares == 0:

            shares = int(cash // close)

            if shares > 0:

                cost = shares * close
                cash -= cost

                entry_price = close
                entry_date = date

                trades.append(
                    {
                        "date": date,
                        "type": "BUY",
                        "price": close,
                        "shares": shares,
                        "pnl": None,
                    }
                )

        # ====================================================
        # SELL
        # ====================================================

        elif desired_position == 0 and shares > 0:

            sell_value = shares * close
            cash += sell_value

            pnl = (close - entry_price) * shares

            trades.append(
                {
                    "date": date,
                    "type": "SELL",
                    "price": close,
                    "shares": shares,
                    "pnl": pnl,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                }
            )

            shares = 0
            entry_price = None
            entry_date = None

        # ====================================================
        # EQUITY
        # ====================================================

        equity = cash + shares * close

        equity_curve.append(
            {
                "timestamp": date,
                "equity": equity,
            }
        )

    # ========================================================
    # CLOSE ANY OPEN POSITION AT END OF DATA
    # ========================================================

    if shares > 0:

        final_row = df.iloc[-1]

        final_price = final_row["close"]
        final_date = final_row["timestamp"]

        cash += shares * final_price

        pnl = (final_price - entry_price) * shares

        trades.append(
            {
                "date": final_date,
                "type": "FINAL_SELL",
                "price": final_price,
                "shares": shares,
                "pnl": pnl,
                "entry_date": entry_date,
                "entry_price": entry_price,
            }
        )

        shares = 0

    final_value = cash

    # ========================================================
    # TRADE ANALYSIS
    # ========================================================

    completed_trades = [
        t for t in trades
        if t["type"] in ("SELL", "FINAL_SELL")
    ]

    winning_trades = [
        t for t in completed_trades
        if t["pnl"] > 0
    ]

    losing_trades = [
        t for t in completed_trades
        if t["pnl"] < 0
    ]

    total_trades = len(completed_trades)

    win_rate = (
        len(winning_trades) / total_trades * 100
        if total_trades > 0
        else 0
    )

    gross_profit = sum(
        t["pnl"]
        for t in winning_trades
    )

    gross_loss = abs(
        sum(
            t["pnl"]
            for t in losing_trades
        )
    )

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0

    total_pnl = final_value - starting_cash

    total_return = (
        total_pnl / starting_cash * 100
    )

    # ========================================================
    # MAX DRAWDOWN
    # ========================================================

    equity_df = pd.DataFrame(equity_curve)

    if not equity_df.empty:

        equity_df["peak"] = (
            equity_df["equity"]
            .cummax()
        )

        equity_df["drawdown"] = (
            equity_df["equity"]
            - equity_df["peak"]
        ) / equity_df["peak"]

        max_drawdown = (
            equity_df["drawdown"].min() * 100
        )

    else:
        max_drawdown = 0

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "symbol": symbol,
        "starting_cash": starting_cash,
        "final_value": final_value,
        "total_pnl": total_pnl,
        "return_pct": total_return,
        "trades": total_trades,
        "wins": len(winning_trades),
        "losses": len(losing_trades),
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown,
        "trade_list": trades,
    }

    return result


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print("\n" + "=" * 75)
    print(f" {result['symbol']} SMA BACKTEST")
    print("=" * 75)

    print(
        f"Starting cash:       "
        f"${result['starting_cash']:,.2f}"
    )

    print(
        f"Final value:         "
        f"${result['final_value']:,.2f}"
    )

    print(
        f"Total P&L:           "
        f"${result['total_pnl']:,.2f}"
    )

    print(
        f"Total return:        "
        f"{result['return_pct']:.2f}%"
    )

    print(
        f"Completed trades:    "
        f"{result['trades']}"
    )

    print(
        f"Winning trades:      "
        f"{result['wins']}"
    )

    print(
        f"Losing trades:       "
        f"{result['losses']}"
    )

    print(
        f"Win rate:            "
        f"{result['win_rate']:.2f}%"
    )

    print(
        f"Gross profit:        "
        f"${result['gross_profit']:,.2f}"
    )

    print(
        f"Gross loss:          "
        f"${result['gross_loss']:,.2f}"
    )

    if result["profit_factor"] == float("inf"):
        print("Profit factor:       ∞")
    else:
        print(
            f"Profit factor:      "
            f"{result['profit_factor']:.2f}"
        )

    print(
        f"Max drawdown:        "
        f"{result['max_drawdown_pct']:.2f}%"
    )

    trades = result["trade_list"]

    if trades:

        print("\nLast 10 trade events:")

        for trade in trades[-10:]:

            if trade["type"] in ("SELL", "FINAL_SELL"):

                print(
                    f"{trade['date'].date()}  "
                    f"{trade['type']:10} "
                    f"${trade['price']:.2f}  "
                    f"shares={trade['shares']}  "
                    f"P&L=${trade['pnl']:+.2f}"
                )

            else:

                print(
                    f"{trade['date'].date()}  "
                    f"{trade['type']:10} "
                    f"${trade['price']:.2f}  "
                    f"shares={trade['shares']}"
                )

    if SHOW_ALL_TRADES and trades:

        print("\nAll trades:")

        for trade in trades:
            print(trade)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SMA STRATEGY BACKTEST")
    print("=" * 75)

    print(f"Short SMA:       {SHORT_WINDOW}")
    print(f"Long SMA:        {LONG_WINDOW}")
    print(
        f"Threshold:       "
        f"{THRESHOLD_PCT * 100:.2f}%"
    )
    print(
        f"Starting cash:   "
        f"${STARTING_CASH:,.2f} per symbol"
    )

    print("\nConnecting to PostgreSQL...")

    conn = get_connection()

    symbols = load_symbols(conn)

    print(
        f"Found {len(symbols)} symbols: "
        f"{', '.join(symbols)}"
    )

    results = []

    # ========================================================
    # RUN EACH SYMBOL
    # ========================================================

    for symbol in symbols:

        print(
            f"\nProcessing {symbol}..."
        )

        df = load_symbol_data(
            conn,
            symbol
        )

        if len(df) < LONG_WINDOW:

            print(
                f"Skipping {symbol}: "
                f"only {len(df)} rows"
            )

            continue

        df = apply_sma_strategy(df)

        result = backtest_symbol(
            df,
            symbol
        )

        results.append(result)

        print_result(result)

    conn.close()

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    if not results:
        print("\nNo results.")
        return

    summary = pd.DataFrame(
        [
            {
                "Symbol": r["symbol"],
                "Trades": r["trades"],
                "Wins": r["wins"],
                "Losses": r["losses"],
                "Win Rate %": r["win_rate"],
                "P&L": r["total_pnl"],
                "Return %": r["return_pct"],
                "Profit Factor": (
                    r["profit_factor"]
                    if r["profit_factor"] != float("inf")
                    else 999.0
                ),
                "Max DD %": r["max_drawdown_pct"],
            }
            for r in results
        ]
    )

    print("\n\n")
    print("=" * 110)
    print("OVERALL SMA BACKTEST SUMMARY")
    print("=" * 110)

    print(
        summary.to_string(
            index=False,
            formatters={
                "Win Rate %": "{:.2f}".format,
                "P&L": "${:,.2f}".format,
                "Return %": "{:.2f}".format,
                "Profit Factor": "{:.2f}".format,
                "Max DD %": "{:.2f}".format,
            }
        )
    )

    # ========================================================
    # PORTFOLIO-STYLE TOTAL
    # ========================================================

    total_starting = sum(
        r["starting_cash"]
        for r in results
    )

    total_final = sum(
        r["final_value"]
        for r in results
    )

    total_pnl = total_final - total_starting

    total_return = (
        total_pnl / total_starting * 100
    )

    total_trades = sum(
        r["trades"]
        for r in results
    )

    total_wins = sum(
        r["wins"]
        for r in results
    )

    total_losses = sum(
        r["losses"]
        for r in results
    )

    total_gross_profit = sum(
        r["gross_profit"]
        for r in results
    )

    total_gross_loss = sum(
        r["gross_loss"]
        for r in results
    )

    overall_profit_factor = (
        total_gross_profit / total_gross_loss
        if total_gross_loss > 0
        else float("inf")
    )

    overall_win_rate = (
        total_wins / total_trades * 100
        if total_trades > 0
        else 0
    )

    print("\n" + "=" * 75)
    print("PORTFOLIO-STYLE TOTAL")
    print("=" * 75)

    print(
        f"Starting capital:   "
        f"${total_starting:,.2f}"
    )

    print(
        f"Final value:        "
        f"${total_final:,.2f}"
    )

    print(
        f"Total P&L:          "
        f"${total_pnl:,.2f}"
    )

    print(
        f"Total return:       "
        f"{total_return:.2f}%"
    )

    print(
        f"Total trades:       "
        f"{total_trades}"
    )

    print(
        f"Overall win rate:   "
        f"{overall_win_rate:.2f}%"
    )

    if overall_profit_factor == float("inf"):
        print("Overall profit factor: ∞")
    else:
        print(
            f"Overall profit factor: "
            f"{overall_profit_factor:.2f}"
        )


if __name__ == "__main__":
    main()
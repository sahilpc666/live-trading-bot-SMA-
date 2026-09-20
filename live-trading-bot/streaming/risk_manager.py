from order_executor import (
    get_position,
    get_account,
)

from db_logger import get_bot_open_position_count


# ============================================================
# RISK LIMITS
# ============================================================

MAX_POSITION_QTY = 5
MAX_OPEN_POSITIONS = 3
MAX_DAILY_LOSS_PCT = 0.02


# ============================================================
# TRADE RISK CHECK
# ============================================================

def check_trade_allowed(symbol, side, qty):

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if qty <= 0:
        print(f"RISK BLOCKED: Invalid quantity {qty} for {symbol}")
        return False

    # --------------------------------------------------------
    # Currently only BUY orders need opening-position checks.
    # SELL is handled by the signal engine only when a position
    # already exists.
    # --------------------------------------------------------

    if side.lower() != "buy":
        return True

    # --------------------------------------------------------
    # 1. Verify account
    # --------------------------------------------------------

    account = get_account()

    if account is None:
        print(
            f"RISK BLOCKED: Could not verify account for {symbol}"
        )
        return False

    try:
        equity = float(account["equity"])
        last_equity = float(account["last_equity"])
    except (KeyError, TypeError, ValueError):
        print(
            f"RISK BLOCKED: Invalid account data for {symbol}"
        )
        return False

    # --------------------------------------------------------
    # 2. Daily loss protection
    # --------------------------------------------------------

    if last_equity > 0:

        daily_pnl_pct = (
            (equity - last_equity)
            / last_equity
        )

        if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:

            print(
                f"RISK BLOCKED: Daily loss limit hit "
                f"({daily_pnl_pct:.2%}) — "
                f"no new buys until next session"
            )

            return False

    # --------------------------------------------------------
    # 3. Verify current position for this symbol
    # --------------------------------------------------------

    current_qty = get_position(symbol)

    if current_qty is None:

        print(
            f"RISK BLOCKED: Could not verify position "
            f"for {symbol}"
        )

        return False

    # --------------------------------------------------------
    # 4. Maximum position size
    # --------------------------------------------------------

    if current_qty + qty > MAX_POSITION_QTY:

        print(
            f"RISK BLOCKED: {symbol} would exceed "
            f"max position "
            f"({current_qty + qty} > {MAX_POSITION_QTY})"
        )

        return False

    # --------------------------------------------------------
    # 5. Maximum number of open positions
    #
    # This counts positions THIS BOT opened (tracked in
    # bot_positions in Postgres), not every position in the
    # Alpaca account. Using the broker-wide count would cause
    # the bot to block its own new trades because of unrelated
    # positions (manual trades, other strategies, leftover
    # test positions) sitting in the same account.
    # --------------------------------------------------------

    if current_qty == 0:

        bot_open_position_count = get_bot_open_position_count()

        if bot_open_position_count is None:

            print(
                f"RISK BLOCKED: Could not verify bot's open "
                f"positions for {symbol}"
            )

            return False

        if bot_open_position_count >= MAX_OPEN_POSITIONS:

            print(
                f"RISK BLOCKED: Max open positions reached "
                f"({bot_open_position_count}/{MAX_OPEN_POSITIONS}) "
                f"— can't open {symbol}"
            )

            return False

    # --------------------------------------------------------
    # All checks passed
    # --------------------------------------------------------

    print(
        f"RISK APPROVED: {symbol} "
        f"{side.upper()} qty={qty}"
    )

    return True
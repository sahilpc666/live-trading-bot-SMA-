import json
import os
import time
from collections import deque, defaultdict

import joblib
from confluent_kafka import Consumer

from order_executor import (
    place_order,
    get_position,
    get_open_orders,
    get_order_status,
)
from risk_manager import check_trade_allowed
from db_logger import (
    ensure_schema,
    log_signal,
    record_position_opened,
    record_position_closed,
    get_pending_orders,
    update_signal_status,
    get_latest_features,
)


# ============================================================
# STRATEGY SETTINGS
# ============================================================

SHORT_WINDOW = 100
LONG_WINDOW = 300

THRESHOLD_PCT = 0.003  # 0.3%
COOLDOWN_SECONDS = 60

TRADE_QTY = 1  # Keep small while testing

RECONCILE_INTERVAL_SECONDS = 30
last_reconcile_time = 0

# ============================================================
# REGIME GATE (volatility-regime model)
#
# This model predicts whether a symbol is about to enter a
# HIGHER-volatility regime than its recent trailing baseline
# (forward_volatility_5d > volatility_20d, see train_model_regime.py).
# It does NOT predict direction and it is never used to pick a
# side. It only suppresses NEW long entries when a choppier regime
# is predicted -- it never blocks exits, and if the model file or
# a symbol's features aren't available, the gate is simply skipped
# and the crossover strategy trades exactly as it did before this
# existed.
# ============================================================

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/ml/models")
REGIME_MODEL_PATH = os.path.join(MODEL_DIR, "high_vol_regime.joblib")
REGIME_REFRESH_INTERVAL_SECONDS = 6 * 60 * 60  # underlying features are daily

_regime_model = None
_regime_features = None
try:
    _bundle = joblib.load(REGIME_MODEL_PATH)
    _regime_model = _bundle["model"]
    _regime_features = _bundle["features"]
    print(f"REGIME MODEL LOADED: {REGIME_MODEL_PATH}")
except Exception as e:
    print(f"REGIME MODEL NOT LOADED (gate disabled, trading unaffected): {e}")

# symbol -> 1 (high-vol regime predicted, suppress new entries)
#        -> 0 (normal regime)
#        -> absent (unknown / not yet scored -- do NOT gate)
regime_state = {}
last_regime_refresh_time = 0

# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = Consumer({
    "bootstrap.servers": "kafka:9092",
    "group.id": "signal-engine",
    "auto.offset.reset": "latest",
})

# Verify that the topic exists before subscribing.
metadata = consumer.list_topics("stock-quotes", timeout=10)
if "stock-quotes" not in metadata.topics:
    raise RuntimeError("Kafka topic stock-quotes does not exist")

print("Kafka topic found:", metadata.topics["stock-quotes"].partitions)
consumer.subscribe(["stock-quotes"])


# ============================================================
# DATABASE
# ============================================================

ensure_schema()


# ============================================================
# IN-MEMORY STATE
# ============================================================

price_history = defaultdict(lambda: deque(maxlen=LONG_WINDOW))

# Desired strategy state:
#     1 = long
#     0 = flat
#
# None means we have not synchronized this symbol yet.
last_position = defaultdict(lambda: None)

# Time of last successfully submitted order.
last_signal_time = defaultdict(lambda: 0)


# ============================================================
# HELPERS
# ============================================================

def midpoint(quote):
    """Calculate bid/ask midpoint. Returns a float."""
    bid = float(quote["bid_price"])
    ask = float(quote["ask_price"])

    if bid <= 0 or ask <= 0:
        raise ValueError(f"Invalid bid/ask prices: bid={bid}, ask={ask}")

    return (bid + ask) / 2.0


def sync_strategy_state(symbol, current_qty):
    """
    Synchronize the strategy's internal state with the actual
    Alpaca position.

    Long-only strategy:
        qty > 0  -> long
        qty <= 0 -> flat
    """
    if last_position[symbol] is None:
        actual_state = 1 if current_qty > 0 else 0
        last_position[symbol] = actual_state

        print(
            f"STATE SYNC: {symbol} | Alpaca qty={current_qty} | "
            f"strategy_state={actual_state}"
        )


def refresh_regime_predictions():
    """
    Runs on a timer, not per-quote -- the underlying
    historical_features rows are daily, so scoring more often than
    that buys nothing. Pulls the latest row per known symbol and
    scores it with the regime model. A missing model, a missing
    feature row, or a prediction error just leaves regime_state
    as-is (or unset), which means check_signal() gates nothing.
    """
    global last_regime_refresh_time

    if _regime_model is None:
        return

    now = time.time()
    if now - last_regime_refresh_time < REGIME_REFRESH_INTERVAL_SECONDS:
        return
    last_regime_refresh_time = now

    for symbol in list(price_history.keys()):
        latest = get_latest_features(symbol)
        if latest is None:
            continue

        try:
            row = [[latest[f] for f in _regime_features]]
            prediction = int(_regime_model.predict(row)[0])
        except (KeyError, ValueError) as e:
            print(f"REGIME PREDICT FAILED: {symbol} | {e}")
            continue

        if regime_state.get(symbol) != prediction:
            label = "HIGH-VOL" if prediction == 1 else "normal"
            print(f"REGIME UPDATE: {symbol} -> {label} regime")
        regime_state[symbol] = prediction


def check_signal(symbol):
    """Calculate the moving-average signal and execute the strategy
    when the threshold is crossed."""
    prices = price_history[symbol]

    # Need at least LONG_WINDOW observations.
    if len(prices) < LONG_WINDOW:
        return

    price_list = list(prices)
    short_prices = price_list[-SHORT_WINDOW:]

    short_avg = sum(short_prices) / SHORT_WINDOW
    long_avg = sum(price_list) / LONG_WINDOW

    if long_avg <= 0:
        print(f"SIGNAL SKIPPED: {symbol} | Invalid long average={long_avg}")
        return

    diff_pct = (short_avg - long_avg) / long_avg

    # Determine desired strategy state.
    if diff_pct > THRESHOLD_PCT:
        desired_position = 1
    elif diff_pct < -THRESHOLD_PCT:
        desired_position = 0
    else:
        return

    # Current strategy state. We only ignore duplicates after the
    # initial state has been synchronized with Alpaca.
    previous = last_position[symbol]
    if previous is not None and desired_position == previous:
        return

    # Cooldown.
    now = time.time()
    if now - last_signal_time[symbol] < COOLDOWN_SECONDS:
        print(f"COOLDOWN: {symbol} | diff={diff_pct:.4%}")
        return

    signal_type = "BUY" if desired_position == 1 else "SELL"

    print(
        f"{signal_type} SIGNAL: {symbol} | short_avg={short_avg:.4f} | "
        f"long_avg={long_avg:.4f} | diff={diff_pct:.4%}"
    )

    def log(status, order_id=None, detail=None):
        log_signal(
            symbol,
            signal_type,
            short_avg,
            long_avg,
            diff_pct,
            status,
            order_id=order_id,
            detail=detail,
        )

    # --------------------------------------------------------
    # Verify actual Alpaca position
    # --------------------------------------------------------

    current_qty = get_position(symbol)
    if current_qty is None:
        print(f"TRADE BLOCKED: Could not verify position for {symbol}")
        log("BLOCKED_VERIFY_FAILED", detail="position check failed")
        return

    # Synchronize internal state with broker.
    sync_strategy_state(symbol, current_qty)

    # --------------------------------------------------------
    # Check open orders
    # --------------------------------------------------------

    open_orders = get_open_orders(symbol)
    if open_orders is None:
        print(f"TRADE BLOCKED: Could not verify open orders for {symbol}")
        log("BLOCKED_VERIFY_FAILED", detail="open orders check failed")
        return

    if len(open_orders) > 0:
        print(f"TRADE BLOCKED: {symbol} already has {len(open_orders)} open order(s)")
        log("BLOCKED_OPEN_ORDER", detail=f"{len(open_orders)} open order(s)")
        return

    # ==========================================================
    # BUY
    # ==========================================================

    if desired_position == 1:
        # Already holding shares.
        if current_qty > 0:
            print(f"BUY SKIPPED: Already holding {current_qty} {symbol}")
            last_position[symbol] = 1
            log("SKIPPED_ALREADY_LONG", detail=f"holding {current_qty}")
            return

        # Regime gate: only ever suppresses NEW entries, never exits.
        if regime_state.get(symbol) == 1:
            print(f"TRADE BLOCKED BY REGIME GATE: {symbol} BUY (high-vol regime predicted)")
            log("BLOCKED_REGIME", detail="high-vol regime predicted")
            return

        if not check_trade_allowed(symbol, "buy", TRADE_QTY):
            print(f"TRADE BLOCKED BY RISK MANAGER: {symbol} BUY")
            log("BLOCKED_RISK")
            return

        order_qty = TRADE_QTY
        result = place_order(symbol, order_qty, "buy")

    # ==========================================================
    # SELL
    # ==========================================================

    else:
        # Already flat.
        if current_qty <= 0:
            print(f"SELL SKIPPED: No {symbol} position")
            last_position[symbol] = 0
            log("SKIPPED_ALREADY_FLAT")
            return

        # NOTE: sell_qty may be LESS than current_qty (e.g. if
        # current_qty > TRADE_QTY because of a manual trade, a
        # config change, or a previous partial fill). A partial
        # sell must NOT be recorded as "flat" below, or the bot
        # will believe it holds nothing when the broker still
        # shows an open position.
        order_qty = min(TRADE_QTY, current_qty)

        if not check_trade_allowed(symbol, "sell", order_qty):
            print(f"TRADE BLOCKED BY RISK MANAGER: {symbol} SELL")
            log("BLOCKED_RISK")
            return

        result = place_order(symbol, order_qty, "sell")

    # --------------------------------------------------------
    # Order submission result
    # --------------------------------------------------------

    if result is None:
        print(f"TRADE NOT ACCEPTED: {symbol} {signal_type}")
        log("NOT_ACCEPTED")
        return

    order_id = result["id"]
    order_status = result.get("status", "unknown")
    last_signal_time[symbol] = now

    print(f"ORDER SUBMITTED: {symbol} {signal_type} | order_id={order_id} | status={order_status}")
    log("ORDER_SUBMITTED", order_id=order_id, detail=f"Alpaca status={order_status}")

    # IMPORTANT: Do NOT immediately update bot_positions. The order
    # may be accepted but not filled yet.

    order_info = get_order_status(order_id)

    if order_info is None:
        print(f"ORDER STATUS UNKNOWN: {symbol} | order_id={order_id}")
        log("FILL_STATUS_UNKNOWN", order_id=order_id)
        return

    final_status = order_info["status"]
    filled_price = order_info.get("filled_avg_price")
    filled_price = float(filled_price) if filled_price else None

    print(f"ORDER STATUS: {symbol} | order_id={order_id} | status={final_status}")

    # --------------------------------------------------------
    # Filled
    # --------------------------------------------------------

    if final_status == "filled":
        if desired_position == 1:
            record_position_opened(symbol, order_qty)
            log("FILLED", order_id=order_id, detail="BUY order filled", filled_price=filled_price)
            last_position[symbol] = 1
        else:
            remaining_qty = current_qty - order_qty
            if remaining_qty <= 0:
                record_position_closed(symbol)
                log("FILLED", order_id=order_id, detail="SELL order filled (flat)", filled_price=filled_price)
                last_position[symbol] = 0
            else:
                confirmed_qty = get_position(symbol)
                if confirmed_qty is None:
                    log("FILLED_VERIFY_FAILED", order_id=order_id,
                        detail=f"sold {order_qty}, could not confirm remaining position")
                else:
                    record_position_opened(symbol, confirmed_qty)
                    log("FILLED", order_id=order_id,
                        detail=f"SELL order partially filled (sold {order_qty}, {confirmed_qty} remaining)",
                        filled_price=filled_price)
                    last_position[symbol] = 1 if confirmed_qty > 0 else 0
        print(f"ORDER FILLED: {symbol} | {signal_type} | price={filled_price}")
        return
    # --------------------------------------------------------
    # Rejected / canceled / expired
    # --------------------------------------------------------

    if final_status in ("rejected", "canceled", "expired"):
        print(f"ORDER NOT FILLED: {symbol} | status={final_status}")
        log("NOT_FILLED", order_id=order_id, detail=f"status={final_status}")
        return

    # --------------------------------------------------------
    # Still pending
    # --------------------------------------------------------

    print(f"ORDER STILL OPEN: {symbol} | order_id={order_id} | status={final_status}")
    log("PENDING", order_id=order_id, detail=f"status={final_status}")

def reconcile_pending_orders():
    """
    Runs on a timer, not per-quote. Checks every signal still
    logged as PENDING and updates bot_positions once the broker
    reports a final status. This is what catches orders that
    were 'new'/'pending_new' at submission time and filled a
    few seconds later, after check_signal() had already moved on.
    """
    global last_reconcile_time

    now = time.time()
    if now - last_reconcile_time < RECONCILE_INTERVAL_SECONDS:
        return
    last_reconcile_time = now

    pending = get_pending_orders()
    if pending is None:
        print("RECONCILE: could not fetch pending signals")
        return
    if not pending:
        return

    print(f"RECONCILE: checking {len(pending)} pending order(s)")
    for signal_id, symbol, signal_type, order_id in pending:
        order_info = get_order_status(order_id)

        if order_info is None:
            print(f"RECONCILE: could not check status | order_id={order_id}")
            continue

        status = order_info["status"]
        filled_price = order_info.get("filled_avg_price")
        filled_price = float(filled_price) if filled_price else None

        if status == "filled":
            current_qty = get_position(symbol)
            if current_qty is None:
                print(f"RECONCILE: could not verify position for {symbol} after fill")
                continue

            if signal_type == "BUY":
                record_position_opened(symbol, current_qty)
                update_signal_status(signal_id, "FILLED", detail="reconciled: BUY filled", filled_price=filled_price)
                last_position[symbol] = 1
            else:
                if current_qty <= 0:
                    record_position_closed(symbol)
                    update_signal_status(signal_id, "FILLED", detail="reconciled: SELL filled (flat)", filled_price=filled_price)
                    last_position[symbol] = 0
                else:
                    record_position_opened(symbol, current_qty)
                    update_signal_status(signal_id, "FILLED", detail="reconciled: SELL partially filled", filled_price=filled_price)
                    last_position[symbol] = 1

            print(f"RECONCILE: {symbol} order_id={order_id} -> FILLED @ {filled_price}")
            
        elif status in ("rejected", "canceled", "expired"):
            update_signal_status(signal_id, "NOT_FILLED", detail=f"reconciled: status={status}")
            print(f"RECONCILE: {symbol} order_id={order_id} -> {status}")

        else:
            print(f"RECONCILE: {symbol} order_id={order_id} still {status}")

# ============================================================
# MAIN CONSUMER LOOP
# ============================================================

print("Signal engine listening...")

try:
    while True:
        try:
            reconcile_pending_orders()
            refresh_regime_predictions()
            msg = consumer.poll(1.0)
            if msg is None:
                continue

            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            # Decode Kafka message.
            try:
                quote = json.loads(msg.value().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"INVALID KAFKA MESSAGE: {e}")
                continue

            # Validate required fields.
            required_fields = ("symbol", "bid_price", "ask_price")
            if not all(field in quote for field in required_fields):
                print(f"INVALID QUOTE: Missing fields | {quote}")
                continue

            symbol = quote["symbol"]

            # Calculate midpoint.
            try:
                price = midpoint(quote)
            except (KeyError, TypeError, ValueError) as e:
                print(f"INVALID PRICE: {symbol} | {e}")
                continue

            # Store price and check strategy.
            price_history[symbol].append(price)
            check_signal(symbol)

        except Exception as e:
            print(f"Unexpected error processing message: {e}")
            continue

except KeyboardInterrupt:
    print("Stopping signal engine...")

finally:
    consumer.close()
    print("Signal engine stopped.")
import os
import requests

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

TRADING_URL = "https://paper-api.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}

TIMEOUT = 10


def get_position(symbol):
    """
    Returns current position quantity.

    Returns:
        float: position quantity
        0.0: no position
        None: unable to check position
    """
    try:
        response = requests.get(
            f"{TRADING_URL}/positions/{symbol}",
            headers=HEADERS,
            timeout=TIMEOUT,
        )

        if response.status_code == 404:
            return 0.0

        response.raise_for_status()
        return float(response.json()["qty"])

    except requests.exceptions.RequestException as e:
        print(f"POSITION CHECK FAILED: {symbol} | {e}")
        return None


def get_all_positions():
    """
    Returns all open positions across every symbol.

    Returns:
        list: open positions (empty list if flat everywhere)
        None: unable to check positions
    """
    try:
        response = requests.get(
            f"{TRADING_URL}/positions",
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"ALL POSITIONS CHECK FAILED: {e}")
        return None


def get_account():
    """
    Returns account details (equity, last_equity, cash, buying_power, etc.)

    Returns:
        dict: account info
        None: unable to check account
    """
    try:
        response = requests.get(
            f"{TRADING_URL}/account",
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"ACCOUNT CHECK FAILED: {e}")
        return None


def get_open_orders(symbol):
    """Returns open/pending orders for the symbol."""
    try:
        response = requests.get(
            f"{TRADING_URL}/orders",
            headers=HEADERS,
            params={"status": "open", "symbols": symbol},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"OPEN ORDER CHECK FAILED: {symbol} | {e}")
        return None

def place_order(symbol, qty, side):
    """
    Places a paper market order.

    Assumes the caller (signal_engine.py) has already verified
    position and open-order state — this function only validates
    inputs and submits.

    Returns:
        dict: Alpaca order response if successful
        None: if order failed
    """
    if side not in ("buy", "sell"):
        print(f"INVALID ORDER SIDE: {side}")
        return None

    if qty <= 0:
        print(f"INVALID ORDER QUANTITY: {qty}")
        return None

    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }

    try:
        response = requests.post(
            f"{TRADING_URL}/orders",
            headers=HEADERS,
            json=order,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()

        print(
            f"ORDER SUBMITTED: {side.upper()} {qty} {symbol} | "
            f"order_id={result['id']} | status={result['status']}"
        )
        return result

    except requests.exceptions.RequestException as e:
        print(f"ORDER FAILED: {side.upper()} {qty} {symbol} | {e}")
        return None

def get_order_status(order_id):
    """
    Returns the current status of a specific order.

    Returns:
        str: order status (e.g. "filled", "pending_new", "rejected")
        None: unable to check order status
    """
    try:
        response = requests.get(
            f"{TRADING_URL}/orders/{order_id}",
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"ORDER STATUS CHECK FAILED: {order_id} | {e}")
        return None
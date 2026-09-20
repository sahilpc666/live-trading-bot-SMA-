import os
import psycopg2

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")


def get_connection():
    """
    Opens a new connection to Postgres.

    Returns:
        psycopg2 connection, or None if the connection failed
    """
    try:
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    except psycopg2.OperationalError as e:
        print(f"DB CONNECTION FAILED: {e}")
        return None


def ensure_schema():
    """Creates the signals table if it doesn't already exist. Call once at startup."""
    conn = get_connection()
    if conn is None:
        print("DB SCHEMA SETUP SKIPPED: no connection")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        short_avg NUMERIC NOT NULL,
                        long_avg NUMERIC NOT NULL,
                        diff_pct NUMERIC NOT NULL,
                        status TEXT NOT NULL,
                        order_id TEXT,
                        detail TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bot_positions (
                        symbol TEXT PRIMARY KEY,
                        qty NUMERIC NOT NULL,
                        opened_at TIMESTAMPTZ NOT NULL DEFAULT now()
                     )
                """)
        print("DB SCHEMA READY: signals table")
    except Exception as e:
        print(f"DB SCHEMA SETUP FAILED: {e}")
    finally:
        conn.close()

def record_position_opened(symbol, qty):
    conn = get_connection()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bot_positions (symbol, qty)
                    VALUES (%s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET qty = %s
                """, (symbol, qty, qty))
    finally:
        conn.close()

def record_position_closed(symbol):
    conn = get_connection()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM bot_positions WHERE symbol = %s", (symbol,))
    finally:
        conn.close()

def get_bot_open_position_count():
    conn = get_connection()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bot_positions")
            return cur.fetchone()[0]
    finally:
        conn.close()

def log_signal(symbol, signal_type, short_avg, long_avg, diff_pct, status,
                order_id=None, detail=None, filled_price=None):
    conn = get_connection()
    if conn is None:
        print(f"SIGNAL LOG SKIPPED: {symbol} {signal_type} (no DB connection)")
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO signals
                        (symbol, signal_type, short_avg, long_avg, diff_pct, status, order_id, detail, filled_price)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (symbol, signal_type, short_avg, long_avg, diff_pct, status, order_id, detail, filled_price),
                )
    except Exception as e:
        print(f"SIGNAL LOG FAILED: {symbol} {signal_type} | {e}")
    finally:
        conn.close()

def get_pending_orders():
    """
    Returns signals still awaiting fill confirmation.

    Returns:
        list of (id, symbol, signal_type, order_id) tuples
        None if the query failed
    """
    conn = get_connection()
    if conn is None:
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, signal_type, order_id
                FROM signals
                WHERE status = 'PENDING' AND order_id IS NOT NULL
                ORDER BY ts ASC
            """)
            return cur.fetchall()
    except Exception as e:
        print(f"PENDING ORDERS QUERY FAILED: {e}")
        return None
    finally:
        conn.close()

def update_signal_status(signal_id, status, detail=None, filled_price=None):
    conn = get_connection()
    if conn is None:
        print(f"SIGNAL UPDATE SKIPPED: id={signal_id} (no DB connection)")
        return
    try:
        with conn:
            with conn.cursor() as cur:
                if filled_price is not None:
                    cur.execute(
                        "UPDATE signals SET status = %s, detail = %s, filled_price = %s WHERE id = %s",
                        (status, detail, filled_price, signal_id),
                    )
                else:
                    cur.execute(
                        "UPDATE signals SET status = %s, detail = %s WHERE id = %s",
                        (status, detail, signal_id),
                    )
    except Exception as e:
        print(f"SIGNAL UPDATE FAILED: id={signal_id} | {e}")
    finally:
        conn.close()

def get_latest_features(symbol):
    """
    Returns the most recent historical_features row for a symbol,
    as a dict of feature_name -> float, for the signal_engine.py
    regime gate. NOT used by feature_engineering.py or
    train_model_regime.py — they read historical_features directly
    with pandas for full-history / bulk training queries.

    Returns:
        dict: latest feature values (see keys below)
        None: no connection, no rows yet, or a value was missing/NULL
              (e.g. still warming up) — callers should treat this as
              "regime unknown" and skip gating rather than guess.
    """
    conn = get_connection()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT return_1d, return_5d, return_20d,
                       volatility_10d, volatility_20d, volatility_60d,
                       price_to_sma_20, price_to_sma_50, price_to_sma_200,
                       rsi_14, bb_pct, relative_volume,
                       high_low_range, close_open_range,
                       macd_hist, close, timestamp
                FROM historical_features
                WHERE symbol = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            row = cur.fetchone()
            if row is None:
                return None

            (return_1d, return_5d, return_20d,
             volatility_10d, volatility_20d, volatility_60d,
             price_to_sma_20, price_to_sma_50, price_to_sma_200,
             rsi_14, bb_pct, relative_volume,
             high_low_range, close_open_range,
             macd_hist, close, timestamp) = row

            # Any NULL here (not enough trailing history yet, or a
            # divide-by-zero guard upstream) means "can't score this
            # symbol right now" -- return None rather than a partial
            # dict, so the caller doesn't silently score on zeros.
            return {
                "return_1d": float(return_1d),
                "return_5d": float(return_5d),
                "return_20d": float(return_20d),
                "volatility_10d": float(volatility_10d),
                "volatility_20d": float(volatility_20d),
                "volatility_60d": float(volatility_60d),
                "price_to_sma_20": float(price_to_sma_20),
                "price_to_sma_50": float(price_to_sma_50),
                "price_to_sma_200": float(price_to_sma_200),
                "rsi_14": float(rsi_14),
                "bb_pct": float(bb_pct),
                "relative_volume": float(relative_volume),
                "high_low_range": float(high_low_range),
                "close_open_range": float(close_open_range),
                "macd_hist_norm": float(macd_hist) / float(close),
                "timestamp": timestamp,
            }
    except Exception as e:
        print(f"LATEST FEATURES QUERY FAILED: {symbol} | {e}")
        return None
    finally:
        conn.close()
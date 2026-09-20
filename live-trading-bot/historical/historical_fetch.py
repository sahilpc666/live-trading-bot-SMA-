import os
import time
import requests
import psycopg2
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from psycopg2.extras import execute_values


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "JPM", "BAC", "JNJ",
    "PFE", "XOM", "CVX", "TSLA", "AMZN", "WMT",
]

# One year is enough to see multiple volatility regimes without
# an unmanageable fetch. Widen later only if ORB+VWAP looks
# promising and you want more confidence.
START_DATE = "2025-01-01"
END_DATE = "2026-09-01"

TIMEFRAME = "5Min"
FEED = "sip"
ADJUSTMENT = "all"

API_LIMIT = 10000
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
RETRY_DELAY = 2
DB_BATCH_SIZE = 1000

# Regular trading hours only (Eastern). Alpaca's bars endpoint
# returns pre/post-market bars too; ORB and VWAP are both defined
# relative to the regular session, so anything outside this
# window is dropped rather than silently included.
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
EASTERN = ZoneInfo("America/New_York")


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS historical_bars_intraday (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                trading_date DATE NOT NULL,

                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,

                volume BIGINT,
                trade_count INTEGER,
                vwap NUMERIC,

                PRIMARY KEY (symbol, timestamp)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_intraday_symbol_date
            ON historical_bars_intraday (symbol, trading_date)
        """)
    conn.commit()
    print("Schema ready: historical_bars_intraday")


# ============================================================
# HTTP REQUEST WITH RETRY (same pattern as the daily fetch)
# ============================================================

def request_with_retry(session, url, params):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"Network error: {e}. Retry {attempt}/{MAX_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"Rate limited (429). Retry {attempt}/{MAX_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)
            continue

        if response.status_code in (500, 502, 503, 504):
            wait_time = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"Server error ({response.status_code}). Retry {attempt}/{MAX_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)
            continue

        response.raise_for_status()

    raise RuntimeError("Request failed after maximum retries.")


# ============================================================
# VALIDATION
# ============================================================

def validate_bar(bar):
    required_fields = ["t", "o", "h", "l", "c", "v"]
    for field in required_fields:
        if field not in bar:
            return False
    try:
        o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])
        v = int(bar["v"])
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return False
        if h < l or h < o or h < c or l > o or l > c:
            return False
        if v < 0:
            return False
        return True
    except (ValueError, TypeError):
        return False


def is_regular_session(bar_utc_str):
    """
    Alpaca timestamps are UTC. Convert to US Eastern and check
    it falls inside 9:30-16:00 -- otherwise this is a pre-market
    or after-hours bar and doesn't belong in an ORB/VWAP dataset.
    """
    ts_utc = datetime.fromisoformat(bar_utc_str.replace("Z", "+00:00"))
    ts_eastern = ts_utc.astimezone(EASTERN)
    return MARKET_OPEN <= ts_eastern.time() < MARKET_CLOSE


# ============================================================
# FETCH
# ============================================================

def fetch_bars(session, symbol, start, end):
    print()
    print("=" * 60)
    print(f"Fetching {TIMEFRAME} bars for {symbol}")
    print(f"Period: {start} -> {end}")
    print("=" * 60)

    url = f"{DATA_URL}/stocks/{symbol}/bars"
    params = {
        "start": start,
        "end": end,
        "timeframe": TIMEFRAME,
        "feed": FEED,
        "limit": API_LIMIT,
        "adjustment": ADJUSTMENT,
    }

    all_bars = []
    page_number = 1

    while True:
        print(f"Requesting page {page_number}...")
        response = request_with_retry(session, url, params)
        data = response.json()
        bars = data.get("bars", [])
        all_bars.extend(bars)
        print(f"  Page {page_number}: {len(bars)} bars")

        next_page_token = data.get("next_page_token")
        if not next_page_token:
            break
        params["page_token"] = next_page_token
        page_number += 1
        time.sleep(0.2)

    print(f"Total raw bars fetched for {symbol}: {len(all_bars)}")
    return all_bars


def clean_bars(symbol, bars):
    """Validate, drop pre/post-market bars, sort chronologically."""
    valid_bars = []
    invalid_count = 0
    off_session_count = 0

    for bar in bars:
        if not validate_bar(bar):
            invalid_count += 1
            continue
        if not is_regular_session(bar["t"]):
            off_session_count += 1
            continue
        valid_bars.append(bar)

    if invalid_count > 0:
        print(f"WARNING: {symbol}: {invalid_count} invalid bars removed.")
    if off_session_count > 0:
        print(f"  {symbol}: {off_session_count} pre/post-market bars excluded.")

    valid_bars.sort(key=lambda x: x["t"])
    return valid_bars


# ============================================================
# INSERT
# ============================================================

def insert_bars(conn, symbol, bars):
    if not bars:
        print(f"No valid bars to insert for {symbol}.")
        return 0

    rows = []
    for bar in bars:
        ts_utc = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        trading_date = ts_utc.astimezone(EASTERN).date()
        rows.append((
            symbol, bar["t"], trading_date,
            bar["o"], bar["h"], bar["l"], bar["c"],
            bar["v"], bar.get("n"), bar.get("vw"),
        ))

    inserted_count = 0
    try:
        with conn.cursor() as cur:
            for i in range(0, len(rows), DB_BATCH_SIZE):
                batch = rows[i:i + DB_BATCH_SIZE]
                execute_values(
                    cur,
                    """
                    INSERT INTO historical_bars_intraday (
                        symbol, timestamp, trading_date,
                        open, high, low, close,
                        volume, trade_count, vwap
                    )
                    VALUES %s
                    ON CONFLICT (symbol, timestamp) DO NOTHING
                    """,
                    batch,
                    page_size=len(batch),
                )
                inserted_count += cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print(f"  -> {inserted_count} new rows inserted for {symbol}")
    skipped = len(rows) - inserted_count
    if skipped > 0:
        print(f"  -> {skipped} existing rows skipped")
    return inserted_count


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("INTRADAY HISTORICAL DATA DOWNLOADER")
    print(f"Timeframe: {TIMEFRAME} | Session: {MARKET_OPEN}-{MARKET_CLOSE} ET only")
    print("=" * 60)

    if not API_KEY or not SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")

    conn = None
    try:
        conn = get_connection()
        print("PostgreSQL connection successful.")
        ensure_schema(conn)

        with requests.Session() as session:
            total_fetched = 0
            total_inserted = 0

            for symbol in SYMBOLS:
                try:
                    bars = fetch_bars(session, symbol, START_DATE, END_DATE)
                    total_fetched += len(bars)

                    bars = clean_bars(symbol, bars)
                    inserted = insert_bars(conn, symbol, bars)
                    total_inserted += inserted

                except Exception as e:
                    print(f"\nFAILED for {symbol}: {e}")

                time.sleep(0.5)

            print()
            print("=" * 60)
            print("DOWNLOAD COMPLETE")
            print(f"Symbols:          {len(SYMBOLS)}")
            print(f"Total fetched:    {total_fetched}")
            print(f"Total new rows:   {total_inserted}")
            print("=" * 60)

    finally:
        if conn is not None:
            conn.close()
            print("PostgreSQL connection closed.")


if __name__ == "__main__":
    main()
import os
import time
import requests
import psycopg2
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


# ============================================================
# POSTGRESQL
# ============================================================

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")


# ============================================================
# INTRADAY DATA
# ============================================================

SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "JPM",
    "BAC",
    "JNJ",
    "PFE",
    "XOM",
    "CVX",
    "TSLA",
    "AMZN",
    "WMT",
]


# IMPORTANT:
# Start with a SHORT period first.
# Once everything is verified, expand the range.
START_DATE = "2026-01-01"
END_DATE = "2026-09-01"

TIMEFRAME = "1Min"
FEED = "sip"
ADJUSTMENT = "all"

API_LIMIT = 10000
REQUEST_TIMEOUT = 30

MAX_RETRIES = 5
RETRY_DELAY = 2

DB_BATCH_SIZE = 5000


# ============================================================
# VALIDATION
# ============================================================

def validate_environment():
    if not API_KEY:
        raise RuntimeError(
            "ALPACA_API_KEY environment variable is missing."
        )

    if not SECRET_KEY:
        raise RuntimeError(
            "ALPACA_SECRET_KEY environment variable is missing."
        )

    print("Environment validation passed.")


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


def ensure_schema(conn):
    """
    Create a separate table for intraday bars.

    We intentionally do NOT modify historical_bars.
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS intraday_bars (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,

                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,

                volume BIGINT,
                trade_count INTEGER,
                vwap NUMERIC,

                PRIMARY KEY (symbol, timestamp)
            )
            """
        )

        # Useful for time-series queries.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intraday_bars_symbol_timestamp
            ON intraday_bars (symbol, timestamp)
            """
        )

    conn.commit()

    print("Schema ready: intraday_bars")


# ============================================================
# HTTP REQUEST
# ============================================================

def request_with_retry(session, url, params):

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as e:

            if attempt == MAX_RETRIES:
                raise

            wait_time = RETRY_DELAY * (2 ** (attempt - 1))

            print(
                f"Network error: {e}. "
                f"Retry {attempt}/{MAX_RETRIES} "
                f"in {wait_time}s..."
            )

            time.sleep(wait_time)
            continue

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:
            return response

        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        if response.status_code == 429:

            wait_time = RETRY_DELAY * (2 ** (attempt - 1))

            print(
                f"Rate limited (429). "
                f"Retry {attempt}/{MAX_RETRIES} "
                f"in {wait_time}s..."
            )

            time.sleep(wait_time)
            continue

        # ----------------------------------------------------
        # SERVER ERRORS
        # ----------------------------------------------------

        if response.status_code in (500, 502, 503, 504):

            wait_time = RETRY_DELAY * (2 ** (attempt - 1))

            print(
                f"Server error ({response.status_code}). "
                f"Retry {attempt}/{MAX_RETRIES} "
                f"in {wait_time}s..."
            )

            time.sleep(wait_time)
            continue

        # ----------------------------------------------------
        # NON-RETRYABLE ERROR
        # ----------------------------------------------------

        response.raise_for_status()

    raise RuntimeError(
        "Request failed after maximum retries."
    )


# ============================================================
# BAR VALIDATION
# ============================================================

def validate_bar(bar):

    required_fields = [
        "t",
        "o",
        "h",
        "l",
        "c",
        "v",
    ]

    for field in required_fields:
        if field not in bar:
            return False

    try:

        open_price = float(bar["o"])
        high_price = float(bar["h"])
        low_price = float(bar["l"])
        close_price = float(bar["c"])
        volume = int(bar["v"])

        # Positive prices
        if (
            open_price <= 0
            or high_price <= 0
            or low_price <= 0
            or close_price <= 0
        ):
            return False

        # OHLC consistency
        if high_price < low_price:
            return False

        if high_price < open_price:
            return False

        if high_price < close_price:
            return False

        if low_price > open_price:
            return False

        if low_price > close_price:
            return False

        # Volume
        if volume < 0:
            return False

        return True

    except (ValueError, TypeError):

        return False


# ============================================================
# FETCH INTRADAY BARS
# ============================================================

def fetch_bars(
    session,
    symbol,
    start,
    end,
    timeframe=TIMEFRAME,
    feed=FEED,
    limit=API_LIMIT,
):

    print()
    print("=" * 60)
    print(f"Fetching {timeframe} intraday bars for {symbol}")
    print(f"Period: {start} -> {end}")
    print("=" * 60)

    url = f"{DATA_URL}/stocks/{symbol}/bars"

    params = {
        "start": start,
        "end": end,
        "timeframe": timeframe,
        "feed": feed,
        "limit": limit,
        "adjustment": ADJUSTMENT,
    }

    all_bars = []

    page_number = 1

    while True:

        print(f"Requesting page {page_number}...")

        response = request_with_retry(
            session,
            url,
            params,
        )

        data = response.json()

        bars = data.get("bars", [])

        all_bars.extend(bars)

        print(
            f"  Page {page_number}: "
            f"{len(bars)} bars"
        )

        next_page_token = data.get(
            "next_page_token"
        )

        if not next_page_token:
            break

        params["page_token"] = next_page_token

        page_number += 1

        time.sleep(0.2)

    print(
        f"Total bars fetched for {symbol}: "
        f"{len(all_bars)}"
    )

    return all_bars


# ============================================================
# CLEAN DATA
# ============================================================

def clean_bars(symbol, bars):

    valid_bars = []

    invalid_count = 0

    for bar in bars:

        if validate_bar(bar):
            valid_bars.append(bar)

        else:
            invalid_count += 1

    if invalid_count > 0:

        print(
            f"WARNING: {symbol}: "
            f"{invalid_count} invalid bars removed."
        )

    valid_bars.sort(
        key=lambda x: x["t"]
    )

    return valid_bars


# ============================================================
# INSERT
# ============================================================

def insert_bars(conn, symbol, bars):

    if not bars:

        print(
            f"No valid bars to insert "
            f"for {symbol}."
        )

        return 0

    rows = [

        (
            symbol,
            bar["t"],
            bar["o"],
            bar["h"],
            bar["l"],
            bar["c"],
            bar["v"],
            bar.get("n"),
            bar.get("vw"),
        )

        for bar in bars
    ]

    inserted_count = 0

    try:

        with conn.cursor() as cur:

            for i in range(
                0,
                len(rows),
                DB_BATCH_SIZE,
            ):

                batch = rows[
                    i:i + DB_BATCH_SIZE
                ]

                execute_values(
                    cur,
                    """
                    INSERT INTO intraday_bars (
                        symbol,
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        trade_count,
                        vwap
                    )
                    VALUES %s

                    ON CONFLICT (
                        symbol,
                        timestamp
                    )
                    DO NOTHING
                    """,
                    batch,
                    page_size=len(batch),
                )

                inserted_count += cur.rowcount

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    print(
        f"  -> {inserted_count} "
        f"new rows inserted for {symbol}"
    )

    skipped = (
        len(rows) - inserted_count
    )

    if skipped > 0:

        print(
            f"  -> {skipped} "
            f"existing rows skipped"
        )

    return inserted_count


# ============================================================
# DATA SUMMARY
# ============================================================

def print_data_summary(
    symbol,
    bars,
):

    if not bars:
        return

    first_timestamp = bars[0]["t"]
    last_timestamp = bars[-1]["t"]

    closes = [
        float(bar["c"])
        for bar in bars
        if bar.get("c") is not None
    ]

    volumes = [
        int(bar["v"])
        for bar in bars
        if bar.get("v") is not None
    ]

    print()
    print(
        f"INTRADAY DATA SUMMARY — {symbol}"
    )
    print("-" * 45)

    print(
        f"Bars:          {len(bars)}"
    )

    print(
        f"First timestamp: {first_timestamp}"
    )

    print(
        f"Last timestamp:  {last_timestamp}"
    )

    if closes:

        print(
            f"First close:   {closes[0]}"
        )

        print(
            f"Last close:    {closes[-1]}"
        )

    if volumes:

        print(
            f"Max volume:    {max(volumes)}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("HISTORICAL INTRADAY DATA DOWNLOADER")
    print("=" * 60)
    print()

    validate_environment()

    conn = None

    try:

        print(
            "Connecting to PostgreSQL..."
        )

        conn = get_connection()

        print(
            "PostgreSQL connection successful."
        )

        ensure_schema(conn)

        with requests.Session() as session:

            total_fetched = 0
            total_inserted = 0

            for symbol in SYMBOLS:

                try:

                    bars = fetch_bars(
                        session=session,
                        symbol=symbol,
                        start=START_DATE,
                        end=END_DATE,
                        timeframe=TIMEFRAME,
                        feed=FEED,
                        limit=API_LIMIT,
                    )

                    total_fetched += len(bars)

                    bars = clean_bars(
                        symbol,
                        bars,
                    )

                    print_data_summary(
                        symbol,
                        bars,
                    )

                    inserted = insert_bars(
                        conn,
                        symbol,
                        bars,
                    )

                    total_inserted += inserted

                except Exception as e:

                    print()
                    print(
                        f"FAILED for {symbol}: {e}"
                    )

                time.sleep(0.5)

            print()
            print("=" * 60)
            print("INTRADAY DOWNLOAD COMPLETE")
            print("=" * 60)

            print(
                f"Symbols:       {len(SYMBOLS)}"
            )

            print(
                f"Total fetched: {total_fetched}"
            )

            print(
                f"Total new rows: {total_inserted}"
            )

            print("=" * 60)

    except Exception as e:

        print()
        print(
            f"FATAL ERROR: {e}"
        )

        raise

    finally:

        if conn is not None:

            conn.close()

            print(
                "PostgreSQL connection closed."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
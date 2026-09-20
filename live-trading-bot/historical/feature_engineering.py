import os
import psycopg2
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values


# ============================================================
# CONFIGURATION
# ============================================================

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")

# Forward-looking label horizon (trading days). Used ONLY as a
# training target — never as a feature. Rows near the end of a
# symbol's history won't have a target and are inserted with
# forward_return_5d / forward_volatility_5d = NULL rather than
# dropped, so the feature rows themselves are still usable at
# inference time.
TARGET_HORIZON_DAYS = 5

# Longest rolling window used below. Rows before a symbol has
# this many prior bars can't have every feature computed and are
# dropped, so the feature table never contains partially-warm
# (silently NaN-padded) rows.
MAX_LOOKBACK_DAYS = 200

DB_BATCH_SIZE = 1000


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
    """Create historical_features table if it does not exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS historical_features (
                symbol TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,

                close NUMERIC,

                return_1d NUMERIC,
                return_5d NUMERIC,
                return_20d NUMERIC,

                volatility_10d NUMERIC,
                volatility_20d NUMERIC,
                volatility_60d NUMERIC,

                sma_20 NUMERIC,
                sma_50 NUMERIC,
                sma_100 NUMERIC,
                sma_200 NUMERIC,

                price_to_sma_20 NUMERIC,
                price_to_sma_50 NUMERIC,
                price_to_sma_200 NUMERIC,

                ema_12 NUMERIC,
                ema_26 NUMERIC,
                macd NUMERIC,
                macd_signal NUMERIC,
                macd_hist NUMERIC,
                macd_hist_norm NUMERIC,

                rsi_14 NUMERIC,

                bb_upper NUMERIC,
                bb_lower NUMERIC,
                bb_pct NUMERIC,

                volume_sma_20 NUMERIC,
                relative_volume NUMERIC,

                high_low_range NUMERIC,
                close_open_range NUMERIC,

                -- Labels only. Exclude from model inputs.
                forward_return_5d NUMERIC,
                forward_volatility_5d NUMERIC,

                PRIMARY KEY (symbol, timestamp)
            )
        """)
    conn.commit()
    print("Schema ready: historical_features")


def load_bars(conn, symbol):
    """Load all historical bars for one symbol, oldest first."""
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM historical_bars
        WHERE symbol = %s
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df


def get_symbols(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM historical_bars ORDER BY symbol")
        return [row[0] for row in cur.fetchall()]


# ============================================================
# FEATURE CALCULATIONS
#
# Every rolling/ewm calculation below only ever looks backward
# from the current row (pandas .rolling()/.ewm() are trailing by
# default, never centered), so none of these leak future
# information into the same row. The exceptions are
# forward_return_5d and forward_volatility_5d, which are
# intentionally forward-looking and are documented as
# label-only, never features.
# ============================================================

def compute_returns(df):
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    return df


def compute_volatility(df):
    daily_return = df["close"].pct_change(1)
    df["volatility_10d"] = daily_return.rolling(10).std()
    df["volatility_20d"] = daily_return.rolling(20).std()
    df["volatility_60d"] = daily_return.rolling(60).std()
    return df


def compute_moving_averages(df):
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_100"] = df["close"].rolling(100).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    df["price_to_sma_20"] = df["close"] / df["sma_20"] - 1
    df["price_to_sma_50"] = df["close"] / df["sma_50"] - 1
    df["price_to_sma_200"] = df["close"] / df["sma_200"] - 1
    return df


def compute_macd(df):
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_norm"] = df["macd_hist"] / df["close"]
    return df


def compute_rsi(df, period=14):
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss is exactly 0 (pure uptrend over the window),
    # RSI is defined as 100 rather than left NaN from the divide.
    rsi = rsi.where(avg_loss != 0, 100.0)

    df["rsi_14"] = rsi
    return df


def compute_bollinger_bands(df, window=20, num_std=2):
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    df["bb_upper"] = upper
    df["bb_lower"] = lower
    # Position of price within the band, 0 = lower band, 1 = upper band.
    band_width = (upper - lower).replace(0, np.nan)
    df["bb_pct"] = (df["close"] - lower) / band_width
    return df


def compute_volume_features(df):
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["relative_volume"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
    return df


def compute_price_range_features(df):
    df["high_low_range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    df["close_open_range"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    return df


def compute_target(df):
    """Forward N-day return. Label only — never a model input.
    NaN for the last TARGET_HORIZON_DAYS rows of each symbol,
    since the future close isn't known yet."""
    df["forward_return_5d"] = (
        df["close"].shift(-TARGET_HORIZON_DAYS) / df["close"] - 1
    )
    return df


def compute_forward_volatility(df):
    """
    Realized volatility of the NEXT TARGET_HORIZON_DAYS daily
    returns, aligned back to the current row.

    rolling(w).std() at row i is trailing: std of returns
    (i-w+1 .. i). Shifting the whole series by -w moves that
    value back w rows, so the value landing on row t is the std
    of returns (t+1 .. t+w) — i.e. genuinely forward-looking.

    Label only, like forward_return_5d — never a model input.
    Used downstream to build a volatility-REGIME target
    (forward_volatility_5d vs. the trailing volatility_20d
    baseline), which is a materially easier, more autocorrelated
    target than predicting return direction.
    """
    daily_return = df["close"].pct_change(1)
    df["forward_volatility_5d"] = (
        daily_return.rolling(TARGET_HORIZON_DAYS).std().shift(-TARGET_HORIZON_DAYS)
    )
    return df


def build_features(df):
    df = compute_returns(df)
    df = compute_volatility(df)
    df = compute_moving_averages(df)
    df = compute_macd(df)
    df = compute_rsi(df)
    df = compute_bollinger_bands(df)
    df = compute_volume_features(df)
    df = compute_price_range_features(df)
    df = compute_target(df)
    df = compute_forward_volatility(df)
    return df


# ============================================================
# INSERT INTO POSTGRESQL
# ============================================================

FEATURE_COLUMNS = [
    "close",
    "return_1d", "return_5d", "return_20d",
    "volatility_10d", "volatility_20d", "volatility_60d",
    "sma_20", "sma_50", "sma_100", "sma_200",
    "price_to_sma_20", "price_to_sma_50", "price_to_sma_200",
    "ema_12", "ema_26", "macd", "macd_signal",
    "macd_hist", "macd_hist_norm",
    "rsi_14",
    "bb_upper", "bb_lower", "bb_pct",
    "volume_sma_20", "relative_volume",
    "high_low_range", "close_open_range",
    "forward_return_5d",
    "forward_volatility_5d",
]

# Both labels are forward-looking and are allowed to be NaN near
# the end of a symbol's history — they must NOT be required for
# a row to be considered "fully warmed."
LABEL_COLUMNS = ("forward_return_5d", "forward_volatility_5d")


def insert_features(conn, symbol, df):
    """Insert feature rows into PostgreSQL in batches.

    Rows where any NON-label feature is still NaN (not enough
    lookback history yet, e.g. the first ~200 bars of a symbol)
    are dropped. The label columns are allowed to be NaN (recent
    rows without a known future outcome yet) — converted to SQL
    NULL rather than filtered out, since the row's features are
    still valid for inference even without a label.
    """
    feature_only_cols = [c for c in FEATURE_COLUMNS if c not in LABEL_COLUMNS]
    clean_df = df.dropna(subset=feature_only_cols)

    if clean_df.empty:
        print(f"No fully-warmed feature rows to insert for {symbol}.")
        return 0

    rows = []
    for _, row in clean_df.iterrows():
        values = [symbol, row["timestamp"]]
        for col in FEATURE_COLUMNS:
            val = row[col]
            values.append(None if pd.isna(val) else float(val))
        rows.append(tuple(values))

    columns_sql = ", ".join(["symbol", "timestamp"] + FEATURE_COLUMNS)
    update_sql = ", ".join(f"{col} = EXCLUDED.{col}" for col in FEATURE_COLUMNS)

    inserted_count = 0

    try:
        with conn.cursor() as cur:
            for i in range(0, len(rows), DB_BATCH_SIZE):
                batch = rows[i:i + DB_BATCH_SIZE]

                execute_values(
                    cur,
                    f"""
                    INSERT INTO historical_features ({columns_sql})
                    VALUES %s
                    ON CONFLICT (symbol, timestamp)
                    DO UPDATE SET {update_sql}
                    """,
                    batch,
                    page_size=len(batch),
                )
                inserted_count += len(batch)

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    print(f"  -> {inserted_count} feature rows upserted for {symbol}")
    dropped = len(df) - len(clean_df)
    if dropped > 0:
        print(f"  -> {dropped} rows dropped (insufficient lookback history)")

    return inserted_count


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print()

    conn = None

    try:
        conn = get_connection()
        print("PostgreSQL connection successful.")

        ensure_schema(conn)

        symbols = get_symbols(conn)
        print(f"Found {len(symbols)} symbols in historical_bars.")

        total_inserted = 0

        for symbol in symbols:
            print()
            print(f"Processing {symbol}...")

            df = load_bars(conn, symbol)

            if len(df) < MAX_LOOKBACK_DAYS:
                print(
                    f"  SKIPPED: only {len(df)} bars available, "
                    f"need at least {MAX_LOOKBACK_DAYS}."
                )
                continue

            df = build_features(df)
            inserted = insert_features(conn, symbol, df)
            total_inserted += inserted

        print()
        print("=" * 60)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 60)
        print(f"Symbols processed: {len(symbols)}")
        print(f"Total feature rows written: {total_inserted}")
        print("=" * 60)

    except Exception as e:
        print()
        print(f"FATAL ERROR: {e}")
        raise

    finally:
        if conn is not None:
            conn.close()
            print("PostgreSQL connection closed.")


if __name__ == "__main__":
    main()
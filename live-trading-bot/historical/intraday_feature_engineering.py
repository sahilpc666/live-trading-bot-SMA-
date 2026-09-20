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

SOURCE_TABLE = "intraday_bars"
TARGET_TABLE = "intraday_features"

TIMEZONE = "America/New_York"

# Only create ML observations during regular market hours
REGULAR_START = "09:30"
REGULAR_END = "16:00"

# Number of previous trading days used for the
# time-of-day volume baseline.
TOD_VOLUME_LOOKBACK_DAYS = 20


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


# ============================================================
# CREATE / UPDATE TARGET TABLE
# ============================================================

def create_table(conn):

    sql = f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (

        symbol TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,

        -- Session
        session TEXT NOT NULL,
        minutes_since_open INTEGER,
        minutes_to_close INTEGER,

        -- Raw bar information
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume BIGINT,
        trade_count INTEGER,

        -- Price returns
        return_1m DOUBLE PRECISION,
        return_3m DOUBLE PRECISION,
        return_5m DOUBLE PRECISION,
        return_10m DOUBLE PRECISION,
        return_30m DOUBLE PRECISION,

        -- Volatility
        volatility_5m DOUBLE PRECISION,
        volatility_15m DOUBLE PRECISION,
        volatility_30m DOUBLE PRECISION,

        -- Volume
        volume_5m DOUBLE PRECISION,
        relative_volume DOUBLE PRECISION,
        relative_volume_tod DOUBLE PRECISION,
        volume_zscore_tod DOUBLE PRECISION,
        volume_acceleration DOUBLE PRECISION,

        -- Bar structure
        high_low_range DOUBLE PRECISION,
        close_open_range DOUBLE PRECISION,
        close_position_in_bar DOUBLE PRECISION,

        -- VWAP
        session_vwap DOUBLE PRECISION,
        price_vs_vwap DOUBLE PRECISION,
        vwap_return DOUBLE PRECISION,

        -- Trading activity
        trade_count_5m DOUBLE PRECISION,
        trade_intensity DOUBLE PRECISION,

        -- Trend context
        price_vs_sma_20 DOUBLE PRECISION,
        price_vs_sma_50 DOUBLE PRECISION,
        sma_20_slope DOUBLE PRECISION,

        PRIMARY KEY (symbol, timestamp)
    );

    -- Add new columns if the table already existed.
    ALTER TABLE {TARGET_TABLE}
        ADD COLUMN IF NOT EXISTS relative_volume_tod DOUBLE PRECISION;

    ALTER TABLE {TARGET_TABLE}
        ADD COLUMN IF NOT EXISTS volume_zscore_tod DOUBLE PRECISION;

    CREATE INDEX IF NOT EXISTS idx_intraday_features_symbol_timestamp
    ON {TARGET_TABLE} (symbol, timestamp);

    CREATE INDEX IF NOT EXISTS idx_intraday_features_session
    ON {TARGET_TABLE} (session);
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()

    print(f"FEATURE TABLE READY: {TARGET_TABLE}")


# ============================================================
# LOAD DATA
# ============================================================

def load_symbol_data(conn, symbol):

    query = f"""
        SELECT
            symbol,
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            trade_count,
            vwap
        FROM {SOURCE_TABLE}
        WHERE symbol = %s
        ORDER BY timestamp
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(symbol,)
    )

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # Convert UTC -> New York time.
    # Automatically handles EST/EDT.
    df["timestamp_ny"] = (
        df["timestamp"]
        .dt.tz_convert(TIMEZONE)
    )

    return df


# ============================================================
# SESSION CLASSIFICATION
# ============================================================

def classify_session(df):

    local_time = df["timestamp_ny"].dt.time

    premarket_start = pd.Timestamp("04:00").time()
    regular_start = pd.Timestamp("09:30").time()
    regular_end = pd.Timestamp("16:00").time()
    afterhours_end = pd.Timestamp("20:00").time()

    conditions = [
        (local_time >= premarket_start) &
        (local_time < regular_start),

        (local_time >= regular_start) &
        (local_time < regular_end),

        (local_time >= regular_end) &
        (local_time <= afterhours_end),
    ]

    choices = [
        "PREMARKET",
        "REGULAR",
        "AFTERHOURS",
    ]

    df["session"] = np.select(
        conditions,
        choices,
        default="OVERNIGHT"
    )

    return df


# ============================================================
# REGULAR SESSION FEATURES
# ============================================================

def build_features(df):

    if df.empty:
        return df

    df = classify_session(df)

    # --------------------------------------------------------
    # Trading date
    # --------------------------------------------------------

    df["session_date"] = (
        df["timestamp_ny"].dt.date
    )

    # --------------------------------------------------------
    # Only regular-session rows become ML observations.
    # --------------------------------------------------------

    regular = df[
        df["session"] == "REGULAR"
    ].copy()

    if regular.empty:
        return regular

    regular = regular.sort_values(
        "timestamp_ny"
    )

    # --------------------------------------------------------
    # Session-relative time
    # --------------------------------------------------------

    regular["minutes_since_open"] = (
        (
            regular["timestamp_ny"].dt.hour * 60
            + regular["timestamp_ny"].dt.minute
        )
        - (9 * 60 + 30)
    )

    regular["minutes_to_close"] = (
        (16 * 60)
        -
        (
            regular["timestamp_ny"].dt.hour * 60
            + regular["timestamp_ny"].dt.minute
        )
    )

    # --------------------------------------------------------
    # Features are calculated independently within
    # each trading session.
    # --------------------------------------------------------

    grouped = regular.groupby(
        "session_date",
        group_keys=False
    )

    # ========================================================
    # PRICE RETURNS
    # ========================================================

    regular["return_1m"] = grouped["close"].transform(
        lambda x: x.pct_change(1)
    )

    regular["return_3m"] = grouped["close"].transform(
        lambda x: x.pct_change(3)
    )

    regular["return_5m"] = grouped["close"].transform(
        lambda x: x.pct_change(5)
    )

    regular["return_10m"] = grouped["close"].transform(
        lambda x: x.pct_change(10)
    )

    regular["return_30m"] = grouped["close"].transform(
        lambda x: x.pct_change(30)
    )

    # ========================================================
    # VOLATILITY
    # ========================================================

    regular["volatility_5m"] = grouped[
        "return_1m"
    ].transform(
        lambda x: x.rolling(5).std()
    )

    regular["volatility_15m"] = grouped[
        "return_1m"
    ].transform(
        lambda x: x.rolling(15).std()
    )

    regular["volatility_30m"] = grouped[
        "return_1m"
    ].transform(
        lambda x: x.rolling(30).std()
    )

    # ========================================================
    # VOLUME
    # ========================================================

    regular["volume_5m"] = grouped[
        "volume"
    ].transform(
        lambda x: x.rolling(5).sum()
    )

    # --------------------------------------------------------
    # Existing relative volume
    #
    # Current bar / average of previous 30 bars.
    #
    # shift(1) is important:
    # the current bar must not be part of its own baseline.
    # --------------------------------------------------------

    volume_baseline = grouped[
        "volume"
    ].transform(
        lambda x:
        x.shift(1).rolling(30).mean()
    )

    regular["relative_volume"] = (
        regular["volume"] /
        volume_baseline.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # NEW:
    # TIME-OF-DAY NORMALIZED VOLUME
    #
    # Compare this minute's volume with the same minute
    # of previous trading days.
    #
    # Example:
    #
    # 09:30 today
    #     compared with
    # 09:30 from previous days
    #
    # 14:15 today
    #     compared with
    # 14:15 from previous days
    #
    # IMPORTANT:
    # shift(1) means the current day's observation is NOT
    # included in its own historical baseline.
    # --------------------------------------------------------

    tod_group = regular.groupby(
        "minutes_since_open",
        group_keys=False
    )["volume"]

    tod_mean = tod_group.transform(
        lambda x:
        x.shift(1).rolling(
            TOD_VOLUME_LOOKBACK_DAYS,
            min_periods=5
        ).mean()
    )

    tod_std = tod_group.transform(
        lambda x:
        x.shift(1).rolling(
            TOD_VOLUME_LOOKBACK_DAYS,
            min_periods=5
        ).std()
    )

    regular["relative_volume_tod"] = (
        regular["volume"] /
        tod_mean.replace(0, np.nan)
    )

    regular["volume_zscore_tod"] = (
        (
            regular["volume"] -
            tod_mean
        ) /
        tod_std.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Recent volume acceleration
    # --------------------------------------------------------

    volume_5m_avg = grouped[
        "volume"
    ].transform(
        lambda x: x.rolling(5).mean()
    )

    volume_previous_5m = grouped[
        "volume"
    ].transform(
        lambda x:
        x.shift(5).rolling(5).mean()
    )

    regular["volume_acceleration"] = (
        volume_5m_avg /
        volume_previous_5m.replace(0, np.nan)
    )

    # ========================================================
    # BAR STRUCTURE
    # ========================================================

    regular["high_low_range"] = (
        (regular["high"] - regular["low"]) /
        regular["close"]
    )

    regular["close_open_range"] = (
        (regular["close"] - regular["open"]) /
        regular["open"]
    )

    bar_range = (
        regular["high"] -
        regular["low"]
    )

    regular["close_position_in_bar"] = np.where(
        bar_range > 0,
        (
            regular["close"] -
            regular["low"]
        ) / bar_range,
        0.5
    )

    # ========================================================
    # SESSION VWAP
    # ========================================================

    typical_price = (
        regular["high"] +
        regular["low"] +
        regular["close"]
    ) / 3.0

    regular["_cum_dollar_volume"] = grouped[
        "volume"
    ].transform(
        lambda x: (
            (
                typical_price.loc[x.index] *
                x
            ).cumsum()
        )
    )

    regular["_cum_volume"] = grouped[
        "volume"
    ].transform(
        lambda x: x.cumsum()
    )

    regular["session_vwap"] = (
        regular["_cum_dollar_volume"] /
        regular["_cum_volume"].replace(0, np.nan)
    )

    regular["price_vs_vwap"] = (
        regular["close"] /
        regular["session_vwap"]
    ) - 1.0

    regular["vwap_return"] = grouped[
        "session_vwap"
    ].transform(
        lambda x: x.pct_change(5)
    )

    # ========================================================
    # TRADE ACTIVITY
    # ========================================================

    regular["trade_count_5m"] = grouped[
        "trade_count"
    ].transform(
        lambda x: x.rolling(5).sum()
    )

    regular["trade_intensity"] = (
        regular["trade_count"] /
        regular["volume"].replace(0, np.nan)
    )

    # ========================================================
    # SMA CONTEXT
    # ========================================================

    regular["sma_20"] = grouped[
        "close"
    ].transform(
        lambda x: x.rolling(20).mean()
    )

    regular["sma_50"] = grouped[
        "close"
    ].transform(
        lambda x: x.rolling(50).mean()
    )

    regular["price_vs_sma_20"] = (
        regular["close"] /
        regular["sma_20"]
    ) - 1.0

    regular["price_vs_sma_50"] = (
        regular["close"] /
        regular["sma_50"]
    ) - 1.0

    regular["sma_20_slope"] = grouped[
        "sma_20"
    ].transform(
        lambda x: x.pct_change(5)
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    regular.drop(
        columns=[
            "timestamp_ny",
            "session_date",
            "sma_20",
            "sma_50",
            "_cum_dollar_volume",
            "_cum_volume",
            "vwap",
        ],
        inplace=True,
        errors="ignore"
    )

    regular["timestamp"] = pd.to_datetime(
        regular["timestamp"],
        utc=True
    )

    return regular


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(conn, df):

    if df.empty:
        return 0

    columns = [
        "symbol",
        "timestamp",
        "session",
        "minutes_since_open",
        "minutes_to_close",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",

        "return_1m",
        "return_3m",
        "return_5m",
        "return_10m",
        "return_30m",

        "volatility_5m",
        "volatility_15m",
        "volatility_30m",

        "volume_5m",
        "relative_volume",
        "relative_volume_tod",
        "volume_zscore_tod",
        "volume_acceleration",

        "high_low_range",
        "close_open_range",
        "close_position_in_bar",

        "session_vwap",
        "price_vs_vwap",
        "vwap_return",

        "trade_count_5m",
        "trade_intensity",

        "price_vs_sma_20",
        "price_vs_sma_50",
        "sma_20_slope",
    ]

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    values = []

    for row in df[columns].itertuples(
        index=False,
        name=None
    ):

        cleaned = tuple(
            None if pd.isna(value) else value
            for value in row
        )

        values.append(cleaned)

    sql = f"""
        INSERT INTO {TARGET_TABLE}
        ({",".join(columns)})
        VALUES %s
        ON CONFLICT (symbol, timestamp)
        DO UPDATE SET

            session = EXCLUDED.session,
            minutes_since_open = EXCLUDED.minutes_since_open,
            minutes_to_close = EXCLUDED.minutes_to_close,

            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            trade_count = EXCLUDED.trade_count,

            return_1m = EXCLUDED.return_1m,
            return_3m = EXCLUDED.return_3m,
            return_5m = EXCLUDED.return_5m,
            return_10m = EXCLUDED.return_10m,
            return_30m = EXCLUDED.return_30m,

            volatility_5m = EXCLUDED.volatility_5m,
            volatility_15m = EXCLUDED.volatility_15m,
            volatility_30m = EXCLUDED.volatility_30m,

            volume_5m = EXCLUDED.volume_5m,
            relative_volume = EXCLUDED.relative_volume,
            relative_volume_tod = EXCLUDED.relative_volume_tod,
            volume_zscore_tod = EXCLUDED.volume_zscore_tod,
            volume_acceleration = EXCLUDED.volume_acceleration,

            high_low_range = EXCLUDED.high_low_range,
            close_open_range = EXCLUDED.close_open_range,
            close_position_in_bar = EXCLUDED.close_position_in_bar,

            session_vwap = EXCLUDED.session_vwap,
            price_vs_vwap = EXCLUDED.price_vs_vwap,
            vwap_return = EXCLUDED.vwap_return,

            trade_count_5m = EXCLUDED.trade_count_5m,
            trade_intensity = EXCLUDED.trade_intensity,

            price_vs_sma_20 = EXCLUDED.price_vs_sma_20,
            price_vs_sma_50 = EXCLUDED.price_vs_sma_50,
            sma_20_slope = EXCLUDED.sma_20_slope
    """

    with conn.cursor() as cur:

        execute_values(
            cur,
            sql,
            values,
            page_size=5000
        )

    conn.commit()

    return len(values)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INTRADAY FEATURE ENGINEERING")
    print("=" * 70)

    conn = get_connection()

    create_table(conn)

    with conn.cursor() as cur:

        cur.execute(f"""
            SELECT DISTINCT symbol
            FROM {SOURCE_TABLE}
            ORDER BY symbol
        """)

        symbols = [
            row[0]
            for row in cur.fetchall()
        ]

    print(f"Symbols found: {symbols}")
    print(
        f"Time-of-day volume lookback: "
        f"{TOD_VOLUME_LOOKBACK_DAYS} previous trading days"
    )

    total_rows = 0

    for symbol in symbols:

        print()
        print("-" * 70)
        print(f"Processing {symbol}")

        df = load_symbol_data(
            conn,
            symbol
        )

        print(f"Raw rows: {len(df):,}")

        features = build_features(df)

        print(
            f"Regular-session feature rows: "
            f"{len(features):,}"
        )

        if not features.empty:

            inserted = save_features(
                conn,
                features
            )

            total_rows += inserted

            print(
                f"Saved: {inserted:,}"
            )

    conn.close()

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print(f"Total rows processed: {total_rows:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
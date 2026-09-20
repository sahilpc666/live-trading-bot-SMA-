import os
import psycopg2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")

SOURCE_TABLE = "intraday_features"
TARGET_TABLE = "intraday_features"

HORIZONS = [1, 5, 15]


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
# ADD TARGET COLUMNS
# ============================================================

def add_target_columns(conn):

    sql = f"""
    ALTER TABLE {TARGET_TABLE}

    ADD COLUMN IF NOT EXISTS future_return_1m
        DOUBLE PRECISION,

    ADD COLUMN IF NOT EXISTS future_return_5m
        DOUBLE PRECISION,

    ADD COLUMN IF NOT EXISTS future_return_15m
        DOUBLE PRECISION;
    """

    with conn.cursor() as cur:
        cur.execute(sql)

    conn.commit()

    print("TARGET COLUMNS READY")


# ============================================================
# LOAD DATA
# ============================================================

def load_data(conn):

    query = f"""
        SELECT
            symbol,
            timestamp,
            close
        FROM {SOURCE_TABLE}
        ORDER BY symbol, timestamp
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    return df


# ============================================================
# BUILD TARGETS
# ============================================================

def build_targets(df):

    df = df.copy()

    # IMPORTANT:
    # We must never allow a target to cross from one
    # trading session into another.
    #
    # Therefore calculate future returns within each
    # symbol + trading date.

    df["trading_date"] = (
        df["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    grouped = df.groupby(
        ["symbol", "trading_date"]
    )

    for horizon in HORIZONS:

        future_close = grouped["close"].shift(
            -horizon
        )

        df[f"future_return_{horizon}m"] = (
            future_close / df["close"]
        ) - 1.0

    return df


# ============================================================
# UPDATE DATABASE
# ============================================================

def update_database(conn, df):

    columns = [
        "future_return_1m",
        "future_return_5m",
        "future_return_15m",
    ]

    with conn.cursor() as cur:

        rows = df[
            [
                "symbol",
                "timestamp",
            ] + columns
        ].replace(
            [np.inf, -np.inf],
            np.nan
        )

        update_sql = f"""
            UPDATE {TARGET_TABLE}
            SET
                future_return_1m = %s,
                future_return_5m = %s,
                future_return_15m = %s
            WHERE symbol = %s
              AND timestamp = %s
        """

        data = []

        for row in rows.itertuples(index=False):

            symbol = row[0]
            timestamp = row[1]

            future_1m = None if pd.isna(row[2]) else row[2]
            future_5m = None if pd.isna(row[3]) else row[3]
            future_15m = None if pd.isna(row[4]) else row[4]

            data.append(
                (
                    future_1m,
                    future_5m,
                    future_15m,
                    symbol,
                    timestamp,
                )
            )

        cur.executemany(
            update_sql,
            data
        )

    conn.commit()

    return len(data)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_distribution(df):

    print()
    print("=" * 70)
    print("FUTURE RETURN DISTRIBUTIONS")
    print("=" * 70)

    for horizon in HORIZONS:

        column = f"future_return_{horizon}m"

        series = df[column].dropna()

        print()
        print(f"{column}")
        print("-" * 50)

        print(
            series.describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INTRADAY TARGET ENGINEERING")
    print("=" * 70)

    conn = get_connection()

    add_target_columns(conn)

    print("Loading feature data...")

    df = load_data(conn)

    print(
        f"Loaded {len(df):,} rows"
    )

    print("Building future-return targets...")

    df = build_targets(df)

    print_distribution(df)

    print()
    print("Updating PostgreSQL...")

    updated = update_database(
        conn,
        df
    )

    print(
        f"Updated {updated:,} rows"
    )

    conn.close()

    print()
    print("=" * 70)
    print("TARGET ENGINEERING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
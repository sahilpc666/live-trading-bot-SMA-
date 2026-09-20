import os
import numpy as np
import pandas as pd
import psycopg2
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/ml/models")

FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "volatility_10d", "volatility_20d", "volatility_60d",
    "price_to_sma_20", "price_to_sma_50", "price_to_sma_200",
    "rsi_14",
    "bb_pct",
    "relative_volume",
    "high_low_range", "close_open_range",
    "macd_hist_norm",
]

TARGET_HORIZON_DAYS = 5
EMBARGO_DAYS = TARGET_HORIZON_DAYS

TRAIN_END = "2024-01-01"
VAL_END = "2025-07-01"

# Pre-committed. Do not move this after seeing results.
SUCCESS_AUC = 0.53


def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def load_data(conn):
    query = """
        SELECT
            symbol, timestamp,
            return_1d, return_5d, return_20d,
            volatility_10d, volatility_20d, volatility_60d,
            price_to_sma_20, price_to_sma_50, price_to_sma_200,
            rsi_14, bb_pct, relative_volume,
            high_low_range, close_open_range,
            macd_hist / NULLIF(close, 0) AS macd_hist_norm,
            forward_return_5d
        FROM historical_features
        WHERE forward_return_5d IS NOT NULL
        ORDER BY timestamp, symbol
    """
    df = pd.read_sql(query, conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in FEATURES + ["forward_return_5d"]:
        df[col] = df[col].astype(float)
    return df.dropna(subset=FEATURES)


def build_cross_sectional_target(df):
    """
    For each date, compare every symbol's forward return to that
    date's median across all symbols. This removes the common
    market move that dominated the direction-only label — a day
    where everything falls 3% now produces both winners and
    losers instead of 12 identical 'down' labels.

    Dates where fewer than half the universe has data are
    dropped: the median isn't meaningful over 2-3 symbols.
    """
    min_symbols = df["symbol"].nunique() // 2

    counts = df.groupby("timestamp")["symbol"].transform("count")
    df = df[counts >= min_symbols].copy()

    daily_median = df.groupby("timestamp")["forward_return_5d"].transform("median")
    df["target"] = (df["forward_return_5d"] > daily_median).astype(int)
    return df


def time_split(df):
    train_end = pd.Timestamp(TRAIN_END, tz="UTC")
    val_end = pd.Timestamp(VAL_END, tz="UTC")
    gap = pd.Timedelta(days=EMBARGO_DAYS * 2)

    train = df[df["timestamp"] < train_end - gap]
    val = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end - gap)]
    test = df[df["timestamp"] >= val_end]
    return train, val, test


def evaluate(name, model, X, y, split_name):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    majority = max(y.mean(), 1 - y.mean())
    auc = roc_auc_score(y, proba)

    print(f"\n  {name} — {split_name}")
    print(f"    rows              : {len(y)}")
    print(f"    class balance     : {y.mean():.4f}")
    print(f"    accuracy          : {accuracy_score(y, pred):.4f}")
    print(f"    precision         : {precision_score(y, pred, zero_division=0):.4f}")
    print(f"    recall            : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"    roc auc           : {auc:.4f}")
    print(f"    edge vs baseline  : {accuracy_score(y, pred) - majority:+.4f}")

    return auc


def main():
    conn = get_connection()
    print("Loading features...")
    df = load_data(conn)
    conn.close()

    df = build_cross_sectional_target(df)

    print(f"Loaded {len(df)} rows across {df['symbol'].nunique()} symbols")
    print(f"Date range: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    print(f"Target balance (share 'outperform'): {df['target'].mean():.4f}")
    print(f"Pre-committed success threshold: validation AUC >= {SUCCESS_AUC}")

    train, val, test = time_split(df)
    print(f"\nTrain: {len(train)} rows")
    print(f"Val  : {len(val)} rows")
    print(f"Test : {len(test)} rows")

    X_train, y_train = train[FEATURES], train["target"]
    X_val, y_val = val[FEATURES], val["target"]
    X_test, y_test = test[FEATURES], test["target"]

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=0.1),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            l2_regularization=1.0, random_state=42,
        ),
    }

    val_aucs = {}
    for name, model in models.items():
        print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
        model.fit(X_train, y_train)
        evaluate(name, model, X_train, y_train, "TRAIN")
        val_auc = evaluate(name, model, X_val, y_val, "VALIDATION")
        val_aucs[name] = (model, val_auc)

    best_name = max(val_aucs, key=lambda k: val_aucs[k][1])
    best_model, best_val_auc = val_aucs[best_name]

    print(f"\n{'=' * 60}")
    print(f"Best on validation: {best_name} (AUC {best_val_auc:.4f})")
    print(f"{'=' * 60}")

    if best_val_auc < SUCCESS_AUC:
        print(f"\nRESULT: validation AUC {best_val_auc:.4f} < threshold {SUCCESS_AUC}.")
        print("Per the pre-committed criteria, this closes the technical-ML question.")
        print("Not evaluating on test — no point spending the holdout on a rejected model.")
        return

    print(f"\nValidation cleared the threshold. Confirming on test (touched once):")
    test_auc = evaluate(best_name, best_model, X_test, y_test, "TEST")

    if test_auc < SUCCESS_AUC:
        print(f"\nRESULT: test AUC {test_auc:.4f} did not confirm validation. Treat as not working.")
        return

    print(f"\nRESULT: confirmed on both validation and test. Saving model.")
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"cross_sectional_{best_name}.joblib")
    joblib.dump({"model": best_model, "features": FEATURES}, path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
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

# Scale-free only — see note above. Anything in price units is excluded
# so the model transfers to symbols it has never seen.
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

TARGET_HORIZON_DAYS = 5      # must match feature_engineering.py
EMBARGO_DAYS = TARGET_HORIZON_DAYS

TRAIN_END = "2024-01-01"
VAL_END = "2025-07-01"


def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def load_data(conn):
    """Load features, normalising the one price-unit feature we keep."""
    query = """
        SELECT
            symbol,
            timestamp,
            return_1d, return_5d, return_20d,
            volatility_10d, volatility_20d, volatility_60d,
            price_to_sma_20, price_to_sma_50, price_to_sma_200,
            rsi_14,
            bb_pct,
            relative_volume,
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

    # Label: 1 if the stock is higher in 5 days, 0 otherwise.
    df["target"] = (df["forward_return_5d"] > 0).astype(int)
    return df.dropna(subset=FEATURES)


def time_split(df):
    """
    Split by date, never randomly, and embargo EMBARGO_DAYS either side
    of each boundary. Without the embargo, the 5-day forward label of a
    late training row overlaps days that appear in validation.
    """
    train_end = pd.Timestamp(TRAIN_END, tz="UTC")
    val_end = pd.Timestamp(VAL_END, tz="UTC")
    gap = pd.Timedelta(days=EMBARGO_DAYS * 2)   # calendar days, generous

    train = df[df["timestamp"] < train_end - gap]
    val = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end - gap)]
    test = df[df["timestamp"] >= val_end]

    return train, val, test


def evaluate(name, model, X, y, split_name):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    # The number that actually matters: a model must beat the naive
    # "always predict up" rule, which is already right ~55% of the time
    # because stocks drift upward.
    majority = max(y.mean(), 1 - y.mean())

    print(f"\n  {name} — {split_name}")
    print(f"    rows            : {len(y)}")
    print(f"    always-up baseline: {majority:.4f}")
    print(f"    accuracy        : {accuracy_score(y, pred):.4f}")
    print(f"    precision (up)  : {precision_score(y, pred, zero_division=0):.4f}")
    print(f"    recall (up)     : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"    roc auc         : {roc_auc_score(y, proba):.4f}")
    print(f"    edge vs baseline: {accuracy_score(y, pred) - majority:+.4f}")

    return accuracy_score(y, pred) - majority


def main():
    conn = get_connection()
    print("Loading features...")
    df = load_data(conn)
    conn.close()

    print(f"Loaded {len(df)} rows across {df['symbol'].nunique()} symbols")
    print(f"Date range: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    print(f"Target balance (share up): {df['target'].mean():.4f}")

    train, val, test = time_split(df)
    print(f"\nTrain: {len(train)} rows ({train['timestamp'].min().date()} -> {train['timestamp'].max().date()})")
    print(f"Val  : {len(val)} rows ({val['timestamp'].min().date()} -> {val['timestamp'].max().date()})")
    print(f"Test : {len(test)} rows ({test['timestamp'].min().date()} -> {test['timestamp'].max().date()})")

    X_train, y_train = train[FEATURES], train["target"]
    X_val, y_val = val[FEATURES], val["target"]
    X_test, y_test = test[FEATURES], test["target"]

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.1),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=4,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
        model.fit(X_train, y_train)
        evaluate(name, model, X_train, y_train, "TRAIN")
        val_edge = evaluate(name, model, X_val, y_val, "VALIDATION")
        results[name] = (model, val_edge)

    # Pick on validation, then touch test exactly once.
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_edge = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"Best on validation: {best_name} (edge {best_edge:+.4f})")
    print(f"{'=' * 60}")
    evaluate(best_name, best_model, X_test, y_test, "TEST (untouched until now)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{best_name}.joblib")
    joblib.dump({"model": best_model, "features": FEATURES}, path)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()

    
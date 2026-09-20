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

# ============================================================
# TARGET
#
# Instead of predicting return direction (tried in train_model.py,
# scored below the always-up baseline on val/test), this predicts
# a REGIME: will realized volatility over the next 5 days come in
# above the symbol's trailing 20-day volatility baseline?
#
# This is a materially easier target because volatility clusters
# (autocorrelates) in a way returns don't — a high-vol day is
# followed by more high-vol days far more reliably than an up day
# is followed by another up day. volatility_20d is deliberately
# kept IN the feature set: using a known, trailing value to help
# predict a forward value is not leakage, it's the whole point.
# ============================================================

# Scale-free only, matching train_model.py — nothing in price units,
# so the model transfers across symbols at different price levels.
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
            forward_volatility_5d
        FROM historical_features
        WHERE forward_volatility_5d IS NOT NULL
        ORDER BY timestamp, symbol
    """
    df = pd.read_sql(query, conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in FEATURES + ["forward_volatility_5d"]:
        df[col] = df[col].astype(float)

    df = df.dropna(subset=FEATURES)

    # Label: 1 if realized vol over the next 5 days exceeds the
    # trailing 20-day baseline (entering a choppier regime), 0
    # otherwise. volatility_20d is one of FEATURES, so this is
    # comparing a known trailing value to an unknown forward one —
    # not comparing two unknowns.
    df["target"] = (df["forward_volatility_5d"] > df["volatility_20d"]).astype(int)
    return df


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


def evaluate(name, model, X, y, split_name, persistence_pred=None):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    # Weakest baseline: always predict the majority class.
    majority = max(y.mean(), 1 - y.mean())

    print(f"\n  {name} — {split_name}")
    print(f"    rows                 : {len(y)}")
    print(f"    majority-class baseline: {majority:.4f}")

    if persistence_pred is not None:
        # Stronger, more honest baseline for a volatility target:
        # "the regime that's already showing in the trailing window
        # keeps going" (volatility_20d > volatility_60d => predict
        # high-vol continues). A model has to beat THIS, not just
        # majority class, or it isn't adding anything over what you
        # could read off the existing trailing features directly.
        persistence_acc = accuracy_score(y, persistence_pred)
        print(f"    persistence baseline : {persistence_acc:.4f}")

    acc = accuracy_score(y, pred)
    print(f"    accuracy             : {acc:.4f}")
    print(f"    precision (high-vol) : {precision_score(y, pred, zero_division=0):.4f}")
    print(f"    recall (high-vol)    : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"    roc auc              : {roc_auc_score(y, proba):.4f}")
    print(f"    edge vs majority     : {acc - majority:+.4f}")
    if persistence_pred is not None:
        print(f"    edge vs persistence  : {acc - persistence_acc:+.4f}")

    return acc - persistence_acc if persistence_pred is not None else acc - majority


def main():
    conn = get_connection()
    print("Loading features...")
    df = load_data(conn)
    conn.close()

    print(f"Loaded {len(df)} rows across {df['symbol'].nunique()} symbols")
    print(f"Date range: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    print(f"Target balance (share high-vol): {df['target'].mean():.4f}")

    train, val, test = time_split(df)
    print(f"\nTrain: {len(train)} rows ({train['timestamp'].min().date()} -> {train['timestamp'].max().date()})")
    print(f"Val  : {len(val)} rows ({val['timestamp'].min().date()} -> {val['timestamp'].max().date()})")
    print(f"Test : {len(test)} rows ({test['timestamp'].min().date()} -> {test['timestamp'].max().date()})")

    X_train, y_train = train[FEATURES], train["target"]
    X_val, y_val = val[FEATURES], val["target"]
    X_test, y_test = test[FEATURES], test["target"]

    # Persistence baseline predictions per split (regime already
    # elevated in the trailing window => predict it stays elevated).
    persist_train = (train["volatility_20d"] > train["volatility_60d"]).astype(int)
    persist_val = (val["volatility_20d"] > val["volatility_60d"]).astype(int)
    persist_test = (test["volatility_20d"] > test["volatility_60d"]).astype(int)

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
        evaluate(name, model, X_train, y_train, "TRAIN", persist_train)
        val_edge = evaluate(name, model, X_val, y_val, "VALIDATION", persist_val)
        results[name] = (model, val_edge)

    # Pick on validation (edge vs. the persistence baseline, not just
    # majority class), then touch test exactly once.
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_edge = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"Best on validation: {best_name} (edge vs. persistence {best_edge:+.4f})")
    print(f"{'=' * 60}")
    evaluate(best_name, best_model, X_test, y_test, "TEST (untouched until now)", persist_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "high_vol_regime.joblib")
    joblib.dump({"model": best_model, "features": FEATURES}, path)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
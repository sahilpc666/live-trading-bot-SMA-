import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import psycopg2

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")

TABLE = "intraday_features"

MODEL_DIR = "/app/ml/models"

# ------------------------------------------------------------
# Chronological split
#
# Train:
#   2026-01-01 -> 2026-05-31
#
# Validation:
#   2026-06-01 -> 2026-07-31
#
# Test:
#   2026-08-01 -> 2026-09-01
#
# These dates are intentionally chronological.
# ------------------------------------------------------------

TRAIN_START = "2026-01-01"
TRAIN_END = "2026-05-31"

VAL_START = "2026-06-01"
VAL_END = "2026-07-31"

TEST_START = "2026-08-01"
TEST_END = "2026-09-01"

# Because target = t -> t+5 minutes,
# leave a 5-minute embargo between datasets.
EMBARGO_MINUTES = 5

TARGET_HORIZON = 5

# Probability thresholds for trading-oriented analysis.
PROBABILITY_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

# Approximate round-trip cost assumptions.
#
# These are NOT used to train the model.
# They are only used to show whether the predicted
# returns are remotely large enough to matter.
#
# 1 bp fee + 2 bp slippage per side
# = 6 bps round trip.
FEE_PER_SIDE = 0.0001
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = (
    2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
)


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [

    # Price momentum
    "return_1m",
    "return_3m",
    "return_5m",
    "return_10m",
    "return_30m",

    # Volatility
    "volatility_5m",
    "volatility_15m",
    "volatility_30m",

    # Volume
    "volume_5m",
    "relative_volume",
    "relative_volume_tod",
    "volume_zscore_tod",
    "volume_acceleration",

    # Bar structure
    "high_low_range",
    "close_open_range",
    "close_position_in_bar",

    # VWAP
    "price_vs_vwap",
    "vwap_return",

    # Trading activity
    "trade_count_5m",
    "trade_intensity",

    # Trend context
    "price_vs_sma_20",
    "price_vs_sma_50",
    "sma_20_slope",

    # Session timing
    "minutes_since_open",
    "minutes_to_close",
]

CATEGORICAL_FEATURES = [
    "symbol",
]


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
# LOAD DATA
# ============================================================

def load_data(conn):

    print("=" * 70)
    print("LOADING INTRADAY DATA")
    print("=" * 70)

    columns = (
        ["symbol", "timestamp", "future_return_5m"]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    # Remove duplicates while preserving order.
    columns = list(dict.fromkeys(columns))

    query = f"""
        SELECT
            {", ".join(columns)}
        FROM {TABLE}
        WHERE timestamp >= %s
          AND timestamp < %s
        ORDER BY timestamp, symbol
    """

    end_date = pd.Timestamp(TEST_END) + pd.Timedelta(days=1)

    df = pd.read_sql_query(
        query,
        conn,
        params=(
            TRAIN_START,
            end_date.strftime("%Y-%m-%d"),
        ),
    )

    if df.empty:
        raise RuntimeError(
            "No intraday data found."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['timestamp'].min()} -> "
        f"{df['timestamp'].max()}"
    )

    print(
        f"Symbols: "
        f"{sorted(df['symbol'].unique())}"
    )

    return df


# ============================================================
# TARGET
# ============================================================

def create_target(df):

    # --------------------------------------------------------
    # Development label only.
    #
    # 1 = price is higher 5 minutes later
    # 0 = price is not higher 5 minutes later
    #
    # This is NOT yet our final trading label.
    # --------------------------------------------------------

    df = df.copy()

    df["target"] = (
        df["future_return_5m"] > 0
    ).astype(int)

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_data(df):

    train_start = pd.Timestamp(
        TRAIN_START,
        tz="UTC",
    )

    train_end = pd.Timestamp(
        TRAIN_END,
        tz="UTC",
    ) + pd.Timedelta(days=1)

    val_start = pd.Timestamp(
        VAL_START,
        tz="UTC",
    )

    val_end = pd.Timestamp(
        VAL_END,
        tz="UTC",
    ) + pd.Timedelta(days=1)

    test_start = pd.Timestamp(
        TEST_START,
        tz="UTC",
    )

    test_end = pd.Timestamp(
        TEST_END,
        tz="UTC",
    ) + pd.Timedelta(days=1)

    # --------------------------------------------------------
    # Embargo:
    #
    # Train ends before validation begins.
    # Validation starts after train end + embargo.
    #
    # Same for validation -> test.
    # --------------------------------------------------------

    train_cutoff = val_start - pd.Timedelta(
        minutes=EMBARGO_MINUTES
    )

    val_cutoff_start = val_start

    val_cutoff_end = test_start - pd.Timedelta(
        minutes=EMBARGO_MINUTES
    )

    test_cutoff_start = test_start

    train = df[
        (df["timestamp"] >= train_start) &
        (df["timestamp"] < train_cutoff)
    ].copy()

    val = df[
        (df["timestamp"] >= val_cutoff_start) &
        (df["timestamp"] < val_cutoff_end)
    ].copy()

    test = df[
        (df["timestamp"] >= test_cutoff_start) &
        (df["timestamp"] < test_end)
    ].copy()

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    print(
        f"Train:      {len(train):,} rows"
    )
    print(
        f"Validation: {len(val):,} rows"
    )
    print(
        f"Test:       {len(test):,} rows"
    )

    for name, data in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test),
    ]:

        if data.empty:
            raise RuntimeError(
                f"{name} dataset is empty."
            )

        print(
            f"{name}: "
            f"{data['timestamp'].min()} -> "
            f"{data['timestamp'].max()}"
        )

        print(
            f"{name} target share-up: "
            f"{data['target'].mean():.4f}"
        )

    return train, val, test


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_xy(df):

    X = df[
        NUMERIC_FEATURES +
        CATEGORICAL_FEATURES
    ].copy()

    y = df["target"].copy()

    return X, y


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ============================================================
# BUILD MODELS
# ============================================================

def build_logistic_model():

    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def build_hgb_model():

    # HistGradientBoosting cannot directly handle the
    # sparse output used by OneHotEncoder.
    #
    # We therefore use ordinal encoding for symbol.
    #
    # The numeric features still receive median imputation.

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=100,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


# ============================================================
# BASIC MODEL METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


# ============================================================
# TRADING-ORIENTED ANALYSIS
# ============================================================

def evaluate_thresholds(
    df,
    probabilities,
    dataset_name,
):

    results = []

    actual_returns = (
        df["future_return_5m"]
        .to_numpy()
    )

    print()
    print("=" * 70)
    print(
        f"{dataset_name} THRESHOLD ANALYSIS"
    )
    print("=" * 70)

    print(
        f"Round-trip estimated cost: "
        f"{ROUND_TRIP_COST * 100:.3f}%"
    )

    print()

    header = (
        f"{'Threshold':>10} "
        f"{'Signals':>10} "
        f"{'Signal %':>10} "
        f"{'Win Rate':>10} "
        f"{'Mean Ret':>12} "
        f"{'Net/Trade':>12}"
    )

    print(header)
    print("-" * len(header))

    for threshold in PROBABILITY_THRESHOLDS:

        mask = probabilities >= threshold

        signal_count = int(
            mask.sum()
        )

        if signal_count == 0:

            print(
                f"{threshold:>10.2f} "
                f"{0:>10} "
                f"{0.0:>9.2f}% "
                f"{0.0:>9.2f}% "
                f"{0.0:>11.5f}% "
                f"{0.0:>11.5f}%"
            )

            continue

        selected_returns = (
            actual_returns[mask]
        )

        win_rate = (
            selected_returns > 0
        ).mean()

        mean_return = (
            selected_returns.mean()
        )

        net_per_trade = (
            mean_return -
            ROUND_TRIP_COST
        )

        signal_pct = (
            signal_count /
            len(df)
        )

        print(
            f"{threshold:>10.2f} "
            f"{signal_count:>10,} "
            f"{signal_pct * 100:>9.2f}% "
            f"{win_rate * 100:>9.2f}% "
            f"{mean_return * 100:>11.5f}% "
            f"{net_per_trade * 100:>11.5f}%"
        )

        results.append({
            "threshold": threshold,
            "signals": signal_count,
            "signal_pct": signal_pct,
            "win_rate": float(win_rate),
            "mean_return": float(mean_return),
            "net_per_trade": float(
                net_per_trade
            ),
        })

    return results


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    model,
    model_name,
):

    print()
    print("=" * 70)
    print(
        f"{model_name} FEATURE INFORMATION"
    )
    print("=" * 70)

    try:

        preprocessor = (
            model.named_steps["preprocessor"]
        )

        classifier = (
            model.named_steps["classifier"]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            classifier,
            "coef_"
        ):

            coefficients = (
                classifier.coef_[0]
            )

            importance = pd.DataFrame({
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(
                    coefficients
                ),
            })

            importance = (
                importance
                .sort_values(
                    "abs_coefficient",
                    ascending=False,
                )
                .head(20)
            )

            print(
                importance[
                    [
                        "feature",
                        "coefficient",
                    ]
                ].to_string(
                    index=False
                )
            )

        elif hasattr(
            classifier,
            "feature_importances_"
        ):

            importance = pd.DataFrame({
                "feature": feature_names,
                "importance":
                    classifier.feature_importances_,
            })

            importance = (
                importance
                .sort_values(
                    "importance",
                    ascending=False,
                )
                .head(20)
            )

            print(
                importance.to_string(
                    index=False
                )
            )

    except Exception as exc:

        print(
            f"Feature importance unavailable: "
            f"{exc}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    model_name,
    metrics,
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.joblib",
    )

    metrics_path = os.path.join(
        MODEL_DIR,
        f"{model_name}_metrics.json",
    )

    joblib.dump(
        model,
        model_path,
    )

    with open(
        metrics_path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print()
    print(
        f"Model saved: {model_path}"
    )

    print(
        f"Metrics saved: {metrics_path}"
    )


# ============================================================
# TRAIN + EVALUATE MODEL
# ============================================================

def run_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    val_df,
    X_test,
    y_test,
    test_df,
):

    print()
    print("=" * 70)
    print(
        f"TRAINING {model_name}"
    )
    print("=" * 70)

    print(
        f"Training rows: {len(X_train):,}"
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    val_metrics = calculate_metrics(
        y_val,
        val_probabilities,
    )

    print()
    print("VALIDATION")
    print(
        f"AUC:       "
        f"{val_metrics['auc']:.4f}"
    )

    print(
        f"Accuracy:  "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{val_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{val_metrics['recall']:.4f}"
    )

    val_threshold_results = (
        evaluate_thresholds(
            val_df,
            val_probabilities,
            "VALIDATION",
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
    )

    print()
    print("TEST")
    print(
        f"AUC:       "
        f"{test_metrics['auc']:.4f}"
    )

    print(
        f"Accuracy:  "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{test_metrics['recall']:.4f}"
    )

    test_threshold_results = (
        evaluate_thresholds(
            test_df,
            test_probabilities,
            "TEST",
        )
    )

    # --------------------------------------------------------
    # Feature information
    # --------------------------------------------------------

    print_feature_importance(
        model,
        model_name,
    )

    # --------------------------------------------------------
    # Save everything needed to reproduce the experiment.
    # --------------------------------------------------------

    metrics = {
        "model": model_name,

        "target": (
            "future_return_5m > 0"
        ),

        "target_horizon_minutes":
            TARGET_HORIZON,

        "train_start": TRAIN_START,
        "train_end": TRAIN_END,

        "validation_start": VAL_START,
        "validation_end": VAL_END,

        "test_start": TEST_START,
        "test_end": TEST_END,

        "embargo_minutes":
            EMBARGO_MINUTES,

        "round_trip_cost":
            ROUND_TRIP_COST,

        "validation": {
            **val_metrics,
            "thresholds":
                val_threshold_results,
        },

        "test": {
            **test_metrics,
            "thresholds":
                test_threshold_results,
        },
    }

    save_model(
        model,
        model_name,
        metrics,
    )

    return {
        "model": model,
        "metrics": metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INTRADAY ML TRAINING")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    conn = get_connection()

    try:

        df = load_data(conn)

    finally:

        conn.close()

    # --------------------------------------------------------
    # Create target.
    # --------------------------------------------------------

    df = create_target(df)

    # --------------------------------------------------------
    # Remove rows where target does not exist.
    #
    # The last 5 minutes of each session naturally have
    # missing future_return_5m.
    # --------------------------------------------------------

    before = len(df)

    df = df[
        df["future_return_5m"].notna()
    ].copy()

    print()
    print(
        f"Removed rows without 5-minute target: "
        f"{before - len(df):,}"
    )

    # --------------------------------------------------------
    # Split chronologically.
    # --------------------------------------------------------

    train, val, test = split_data(df)

    # --------------------------------------------------------
    # Prepare X / y.
    # --------------------------------------------------------

    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(val)
    X_test, y_test = prepare_xy(test)

    print()
    print("=" * 70)
    print("FEATURE SET")
    print("=" * 70)

    print(
        f"Numeric features: "
        f"{len(NUMERIC_FEATURES)}"
    )

    for feature in NUMERIC_FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        f"Categorical features: "
        f"{CATEGORICAL_FEATURES}"
    )

    # --------------------------------------------------------
    # Train models.
    # --------------------------------------------------------

    models = [
        (
            "intraday_logistic_regression",
            build_logistic_model(),
        ),
        (
            "intraday_hist_gradient_boosting",
            build_hgb_model(),
        ),
    ]

    results = {}

    for model_name, model in models:

        result = run_model(
            model_name,
            model,

            X_train,
            y_train,

            X_val,
            y_val,
            val,

            X_test,
            y_test,
            test,
        )

        results[model_name] = (
            result["metrics"]
        )

    # --------------------------------------------------------
    # Final comparison.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<40} "
        f"{'Val AUC':>10} "
        f"{'Test AUC':>10}"
    )

    print("-" * 65)

    for model_name, metrics in results.items():

        print(
            f"{model_name:<40} "
            f"{metrics['validation']['auc']:>10.4f} "
            f"{metrics['test']['auc']:>10.4f}"
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "IMPORTANT:"
    )

    print(
        "Do not deploy either model yet."
    )

    print(
        "The next step is a proper cost-aware "
        "out-of-sample trading backtest."
    )


if __name__ == "__main__":
    main()
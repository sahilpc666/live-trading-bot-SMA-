import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import psycopg2

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

TRAIN_START = "2026-01-01"
TRAIN_END = "2026-05-31"

VAL_START = "2026-06-01"
VAL_END = "2026-07-31"

TEST_START = "2026-08-01"
TEST_END = "2026-09-01"

EMBARGO_MINUTES = 5

TARGET_HORIZON = 5


# ============================================================
# COST ASSUMPTIONS
# ============================================================

# 1 bp fee + 2 bp slippage per side
#
# Total:
# 3 bps entry
# 3 bps exit
# = 6 bps round trip

FEE_PER_SIDE = 0.0001
SLIPPAGE_PER_SIDE = 0.0002

ROUND_TRIP_COST = (
    2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)
)


# ============================================================
# PREDICTED RETURN THRESHOLDS
# ============================================================

# These are NOT optimized thresholds.
#
# They are simply a diagnostic ladder showing whether
# increasingly positive model predictions correspond
# to increasingly attractive realized returns.

RETURN_THRESHOLDS = [
    0.0000,   # 0.00%
    0.0002,   # 0.02%
    0.0004,   # 0.04%
    0.0006,   # 0.06%
    0.0008,   # 0.08%
    0.0010,   # 0.10%
    0.0015,   # 0.15%
    0.0020,   # 0.20%
]


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [

    # --------------------------------------------------------
    # Price momentum
    # --------------------------------------------------------

    "return_1m",
    "return_3m",
    "return_5m",
    "return_10m",
    "return_30m",

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    "volatility_5m",
    "volatility_15m",
    "volatility_30m",

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    "volume_5m",
    "relative_volume",
    "relative_volume_tod",
    "volume_zscore_tod",
    "volume_acceleration",

    # --------------------------------------------------------
    # Bar structure
    # --------------------------------------------------------

    "high_low_range",
    "close_open_range",
    "close_position_in_bar",

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    "price_vs_vwap",
    "vwap_return",

    # --------------------------------------------------------
    # Trading activity
    # --------------------------------------------------------

    "trade_count_5m",
    "trade_intensity",

    # --------------------------------------------------------
    # Trend context
    # --------------------------------------------------------

    "price_vs_sma_20",
    "price_vs_sma_50",
    "sma_20_slope",

    # --------------------------------------------------------
    # Session timing
    # --------------------------------------------------------

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

    end_date = (
        pd.Timestamp(TEST_END)
        + pd.Timedelta(days=1)
    )

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

    df = df.copy()

    # --------------------------------------------------------
    # Regression target:
    #
    # Actual percentage return over the next 5 minutes.
    #
    # Example:
    #
    # +0.0010 = +0.10%
    # -0.0010 = -0.10%
    # --------------------------------------------------------

    df["target"] = (
        df["future_return_5m"]
    )

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_data(df):

    train_start = pd.Timestamp(
        TRAIN_START,
        tz="UTC",
    )

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
    # Embargo
    # --------------------------------------------------------

    train_cutoff = (
        val_start
        - pd.Timedelta(
            minutes=EMBARGO_MINUTES
        )
    )

    val_cutoff_end = (
        test_start
        - pd.Timedelta(
            minutes=EMBARGO_MINUTES
        )
    )

    train = df[
        (df["timestamp"] >= train_start)
        &
        (df["timestamp"] < train_cutoff)
    ].copy()

    val = df[
        (df["timestamp"] >= val_start)
        &
        (df["timestamp"] < val_cutoff_end)
    ].copy()

    test = df[
        (df["timestamp"] >= test_start)
        &
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
            f"{name} mean target: "
            f"{data['target'].mean() * 100:.6f}%"
        )

        print(
            f"{name} median target: "
            f"{data['target'].median() * 100:.6f}%"
        )

    return train, val, test


# ============================================================
# PREPARE X / Y
# ============================================================

def prepare_xy(df):

    X = df[
        NUMERIC_FEATURES
        +
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

    return ColumnTransformer(
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


# ============================================================
# MODELS
# ============================================================

def build_ridge_model():

    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                Ridge(
                    alpha=10.0,
                ),
            ),
        ]
    )

    return model


def build_hgb_model():

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
                "regressor",
                HistGradientBoostingRegressor(
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
# BASIC REGRESSION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    correlation = np.corrcoef(
        y_true,
        predictions,
    )[0, 1]

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                predictions,
            )
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions,
                )
            )
        ),

        "r2": float(
            r2_score(
                y_true,
                predictions,
            )
        ),

        "correlation": float(
            correlation
        ),
    }


# ============================================================
# DECILE ANALYSIS
# ============================================================

def evaluate_deciles(
    df,
    predictions,
    dataset_name,
):

    actual_returns = (
        df["future_return_5m"]
        .to_numpy()
    )

    predictions = np.asarray(
        predictions
    )

    analysis = pd.DataFrame({
        "prediction": predictions,
        "actual_return": actual_returns,
    })

    # --------------------------------------------------------
    # Equal-sized prediction buckets.
    # --------------------------------------------------------

    analysis["decile"] = pd.qcut(
        analysis["prediction"],
        q=10,
        labels=False,
        duplicates="drop",
    ) + 1

    print()
    print("=" * 70)
    print(
        f"{dataset_name} PREDICTED-RETURN DECILES"
    )
    print("=" * 70)

    header = (
        f"{'Decile':>8} "
        f"{'Trades':>10} "
        f"{'Avg Pred':>12} "
        f"{'Avg Actual':>14} "
        f"{'Median Actual':>16} "
        f"{'Win Rate':>11} "
        f"{'Net/Trade':>13}"
    )

    print(header)
    print("-" * len(header))

    results = []

    for decile, group in analysis.groupby(
        "decile",
        observed=True,
    ):

        actual = group[
            "actual_return"
        ]

        prediction = group[
            "prediction"
        ]

        mean_actual = actual.mean()

        net_return = (
            mean_actual
            - ROUND_TRIP_COST
        )

        win_rate = (
            actual > 0
        ).mean()

        print(
            f"{int(decile):>8} "
            f"{len(group):>10,} "
            f"{prediction.mean() * 100:>11.5f}% "
            f"{mean_actual * 100:>13.5f}% "
            f"{actual.median() * 100:>15.5f}% "
            f"{win_rate * 100:>10.2f}% "
            f"{net_return * 100:>12.5f}%"
        )

        results.append({
            "decile": int(decile),
            "trades": int(len(group)),
            "avg_prediction": float(
                prediction.mean()
            ),
            "avg_actual_return": float(
                mean_actual
            ),
            "median_actual_return": float(
                actual.median()
            ),
            "win_rate": float(
                win_rate
            ),
            "net_per_trade": float(
                net_return
            ),
        })

    # --------------------------------------------------------
    # Monotonicity diagnostic.
    #
    # Compare highest predicted decile against lowest.
    # --------------------------------------------------------

    grouped = (
        analysis
        .groupby(
            "decile",
            observed=True,
        )["actual_return"]
        .mean()
    )

    if len(grouped) >= 2:

        lowest = grouped.iloc[0]
        highest = grouped.iloc[-1]

        spread = highest - lowest

        print()
        print(
            f"Lowest-decile actual return: "
            f"{lowest * 100:.5f}%"
        )

        print(
            f"Highest-decile actual return: "
            f"{highest * 100:.5f}%"
        )

        print(
            f"Top-bottom spread: "
            f"{spread * 100:.5f}%"
        )

    return results


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def evaluate_thresholds(
    df,
    predictions,
    dataset_name,
):

    actual_returns = (
        df["future_return_5m"]
        .to_numpy()
    )

    predictions = np.asarray(
        predictions
    )

    print()
    print("=" * 70)
    print(
        f"{dataset_name} RETURN THRESHOLD ANALYSIS"
    )
    print("=" * 70)

    print(
        f"Round-trip estimated cost: "
        f"{ROUND_TRIP_COST * 100:.3f}%"
    )

    print()

    header = (
        f"{'Pred >':>10} "
        f"{'Signals':>10} "
        f"{'Signal %':>10} "
        f"{'Win Rate':>10} "
        f"{'Avg Pred':>12} "
        f"{'Mean Ret':>12} "
        f"{'Net/Trade':>13}"
    )

    print(header)
    print("-" * len(header))

    results = []

    for threshold in RETURN_THRESHOLDS:

        mask = (
            predictions >= threshold
        )

        signal_count = int(
            mask.sum()
        )

        if signal_count == 0:

            print(
                f"{threshold * 100:>9.3f}% "
                f"{0:>10} "
                f"{0.0:>9.2f}% "
                f"{0.0:>9.2f}% "
                f"{0.0:>11.5f}% "
                f"{0.0:>11.5f}% "
                f"{0.0:>12.5f}%"
            )

            continue

        selected_returns = (
            actual_returns[mask]
        )

        selected_predictions = (
            predictions[mask]
        )

        win_rate = (
            selected_returns > 0
        ).mean()

        mean_return = (
            selected_returns.mean()
        )

        mean_prediction = (
            selected_predictions.mean()
        )

        net_per_trade = (
            mean_return
            - ROUND_TRIP_COST
        )

        signal_pct = (
            signal_count
            /
            len(df)
        )

        print(
            f"{threshold * 100:>9.3f}% "
            f"{signal_count:>10,} "
            f"{signal_pct * 100:>9.2f}% "
            f"{win_rate * 100:>9.2f}% "
            f"{mean_prediction * 100:>11.5f}% "
            f"{mean_return * 100:>11.5f}% "
            f"{net_per_trade * 100:>12.5f}%"
        )

        results.append({
            "threshold": float(
                threshold
            ),
            "signals": signal_count,
            "signal_pct": float(
                signal_pct
            ),
            "win_rate": float(
                win_rate
            ),
            "avg_prediction": float(
                mean_prediction
            ),
            "mean_return": float(
                mean_return
            ),
            "net_per_trade": float(
                net_per_trade
            ),
        })

    return results


# ============================================================
# LONG-ONLY SIMPLE P&L DIAGNOSTIC
# ============================================================

def evaluate_top_bucket_strategy(
    df,
    predictions,
    dataset_name,
):

    actual_returns = (
        df["future_return_5m"]
        .to_numpy()
    )

    predictions = np.asarray(
        predictions
    )

    # --------------------------------------------------------
    # Only the top 10% of predictions.
    #
    # This is deliberately NOT presented as a final strategy.
    # It is a diagnostic for whether the model can rank
    # opportunities.
    # --------------------------------------------------------

    cutoff = np.quantile(
        predictions,
        0.90,
    )

    mask = (
        predictions >= cutoff
    )

    selected = (
        actual_returns[mask]
    )

    if len(selected) == 0:
        return {}

    gross_mean = (
        selected.mean()
    )

    net_mean = (
        gross_mean
        - ROUND_TRIP_COST
    )

    print()
    print("=" * 70)
    print(
        f"{dataset_name} TOP 10% DIAGNOSTIC"
    )
    print("=" * 70)

    print(
        f"Prediction cutoff: "
        f"{cutoff * 100:.5f}%"
    )

    print(
        f"Signals: "
        f"{len(selected):,}"
    )

    print(
        f"Average realized return: "
        f"{gross_mean * 100:.5f}%"
    )

    print(
        f"Estimated net/trade: "
        f"{net_mean * 100:.5f}%"
    )

    print(
        f"Win rate: "
        f"{(selected > 0).mean() * 100:.2f}%"
    )

    return {
        "prediction_cutoff": float(
            cutoff
        ),
        "signals": int(
            len(selected)
        ),
        "avg_return": float(
            gross_mean
        ),
        "net_per_trade": float(
            net_mean
        ),
        "win_rate": float(
            (selected > 0).mean()
        ),
    }


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
            model.named_steps[
                "preprocessor"
            ]
        )

        regressor = (
            model.named_steps[
                "regressor"
            ]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        # ----------------------------------------------------
        # Ridge
        # ----------------------------------------------------

        if hasattr(
            regressor,
            "coef_",
        ):

            coefficients = (
                regressor.coef_
            )

            importance = pd.DataFrame({
                "feature":
                    feature_names,

                "coefficient":
                    coefficients,

                "abs_coefficient":
                    np.abs(
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

        # ----------------------------------------------------
        # Gradient boosting
        # ----------------------------------------------------

        elif hasattr(
            regressor,
            "feature_importances_",
        ):

            importance = pd.DataFrame({
                "feature":
                    feature_names,

                "importance":
                    regressor
                    .feature_importances_,
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
            f"Feature information unavailable: "
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
        f"Training rows: "
        f"{len(X_train):,}"
    )

    model.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_predictions = (
        model.predict(
            X_val
        )
    )

    val_metrics = calculate_metrics(
        y_val,
        val_predictions,
    )

    print()
    print("VALIDATION")

    print(
        f"MAE:         "
        f"{val_metrics['mae'] * 100:.6f}%"
    )

    print(
        f"RMSE:        "
        f"{val_metrics['rmse'] * 100:.6f}%"
    )

    print(
        f"R²:          "
        f"{val_metrics['r2']:.6f}"
    )

    print(
        f"Correlation: "
        f"{val_metrics['correlation']:.6f}"
    )

    val_deciles = evaluate_deciles(
        val_df,
        val_predictions,
        "VALIDATION",
    )

    val_thresholds = (
        evaluate_thresholds(
            val_df,
            val_predictions,
            "VALIDATION",
        )
    )

    val_top10 = (
        evaluate_top_bucket_strategy(
            val_df,
            val_predictions,
            "VALIDATION",
        )
    )

    # ========================================================
    # TEST
    # ========================================================

    test_predictions = (
        model.predict(
            X_test
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print()
    print("TEST")

    print(
        f"MAE:         "
        f"{test_metrics['mae'] * 100:.6f}%"
    )

    print(
        f"RMSE:        "
        f"{test_metrics['rmse'] * 100:.6f}%"
    )

    print(
        f"R²:          "
        f"{test_metrics['r2']:.6f}"
    )

    print(
        f"Correlation: "
        f"{test_metrics['correlation']:.6f}"
    )

    test_deciles = evaluate_deciles(
        test_df,
        test_predictions,
        "TEST",
    )

    test_thresholds = (
        evaluate_thresholds(
            test_df,
            test_predictions,
            "TEST",
        )
    )

    test_top10 = (
        evaluate_top_bucket_strategy(
            test_df,
            test_predictions,
            "TEST",
        )
    )

    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    print_feature_importance(
        model,
        model_name,
    )

    # ========================================================
    # SAVE EXPERIMENT
    # ========================================================

    metrics = {

        "model": model_name,

        "target":
            "future_return_5m",

        "target_horizon_minutes":
            TARGET_HORIZON,

        "train_start":
            TRAIN_START,

        "train_end":
            TRAIN_END,

        "validation_start":
            VAL_START,

        "validation_end":
            VAL_END,

        "test_start":
            TEST_START,

        "test_end":
            TEST_END,

        "embargo_minutes":
            EMBARGO_MINUTES,

        "round_trip_cost":
            ROUND_TRIP_COST,

        "validation": {

            **val_metrics,

            "deciles":
                val_deciles,

            "thresholds":
                val_thresholds,

            "top_10_percent":
                val_top10,
        },

        "test": {

            **test_metrics,

            "deciles":
                test_deciles,

            "thresholds":
                test_thresholds,

            "top_10_percent":
                test_top10,
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
    print("INTRADAY RETURN REGRESSION")
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
    # Create regression target.
    # --------------------------------------------------------

    df = create_target(df)

    # --------------------------------------------------------
    # Remove missing target rows.
    # --------------------------------------------------------

    before = len(df)

    df = df[
        df["target"].notna()
    ].copy()

    print()
    print(
        f"Removed rows without "
        f"5-minute target: "
        f"{before - len(df):,}"
    )

    # --------------------------------------------------------
    # Target distribution.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    print(
        df["target"]
        .describe(
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
        .to_string()
    )

    # --------------------------------------------------------
    # Split.
    # --------------------------------------------------------

    train, val, test = split_data(
        df
    )

    # --------------------------------------------------------
    # Prepare X / y.
    # --------------------------------------------------------

    X_train, y_train = prepare_xy(
        train
    )

    X_val, y_val = prepare_xy(
        val
    )

    X_test, y_test = prepare_xy(
        test
    )

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
    # Models.
    # --------------------------------------------------------

    models = [

        (
            "intraday_return_ridge",
            build_ridge_model(),
        ),

        (
            "intraday_return_hist_gradient_boosting",
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

        results[
            model_name
        ] = result["metrics"]

    # --------------------------------------------------------
    # Final comparison.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<45} "
        f"{'Val Corr':>10} "
        f"{'Test Corr':>10} "
        f"{'Val R²':>10} "
        f"{'Test R²':>10}"
    )

    print("-" * 90)

    for model_name, metrics in results.items():

        print(
            f"{model_name:<45} "
            f"{metrics['validation']['correlation']:>10.4f} "
            f"{metrics['test']['correlation']:>10.4f} "
            f"{metrics['validation']['r2']:>10.4f} "
            f"{metrics['test']['r2']:>10.4f}"
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "IMPORTANT:"
    )

    print(
        "These models are diagnostic only."
    )

    print(
        "Do not deploy them yet."
    )

    print(
        "The most important result is whether "
        "predicted-return deciles show a stable "
        "relationship with realized returns."
    )


if __name__ == "__main__":
    main()
import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import psycopg2

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "fin")
DB_USER = os.environ.get("DB_USER", "dbuser")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "dbpassword")

TABLE_NAME = "intraday_features"

TARGET = "future_return_5m"

# Chronological split
TRAIN_START = "2026-01-01"
TRAIN_END = "2026-05-31"

VAL_START = "2026-06-01"
VAL_END = "2026-07-31"

TEST_START = "2026-08-01"
TEST_END = "2026-09-01"

# Target leakage embargo
EMBARGO_MINUTES = 5

# Estimated round-trip trading cost
ROUND_TRIP_COST = 0.0006  # 0.06%

MODEL_DIR = "/app/ml/models"

RANDOM_STATE = 42


# ============================================================
# FEATURE GROUPS
# ============================================================

MOMENTUM_FEATURES = [
    "return_1m",
    "return_3m",
    "return_5m",
    "return_10m",
    "return_30m",
]

VOLUME_FEATURES = [
    "volume_5m",
    "relative_volume",
    "relative_volume_tod",
    "volume_zscore_tod",
    "volume_acceleration",
    "trade_count_5m",
    "trade_intensity",
]

VWAP_FEATURES = [
    "session_vwap",
    "price_vs_vwap",
    "vwap_return",
]

VOLATILITY_FEATURES = [
    "volatility_5m",
    "volatility_15m",
    "volatility_30m",
    "high_low_range",
    "close_open_range",
    "close_position_in_bar",
]

TREND_FEATURES = [
    "price_vs_sma_20",
    "price_vs_sma_50",
    "sma_20_slope",
]

TIME_FEATURES = [
    "minutes_since_open",
    "minutes_to_close",
]

SYMBOL_FEATURE = [
    "symbol",
]


# Four main ablation groups
FEATURE_GROUPS = {
    "A_momentum_only": (
        MOMENTUM_FEATURES
        + SYMBOL_FEATURE
    ),

    "B_momentum_volume": (
        MOMENTUM_FEATURES
        + VOLUME_FEATURES
        + SYMBOL_FEATURE
    ),

    "C_momentum_volume_vwap": (
        MOMENTUM_FEATURES
        + VOLUME_FEATURES
        + VWAP_FEATURES
        + SYMBOL_FEATURE
    ),

    "D_all_features": (
        MOMENTUM_FEATURES
        + VOLUME_FEATURES
        + VWAP_FEATURES
        + VOLATILITY_FEATURES
        + TREND_FEATURES
        + TIME_FEATURES
        + SYMBOL_FEATURE
    ),
}


# Additional diagnostic:
# all numeric features WITHOUT symbol.
FEATURE_GROUPS["E_all_without_symbol"] = (
    MOMENTUM_FEATURES
    + VOLUME_FEATURES
    + VWAP_FEATURES
    + VOLATILITY_FEATURES
    + TREND_FEATURES
    + TIME_FEATURES
)


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

def load_data():

    all_features = sorted(
        set(
            feature
            for features in FEATURE_GROUPS.values()
            for feature in features
        )
    )

    # trading_date is NOT stored in intraday_features.
    # We derive it from timestamp after loading.
    columns = [
        "symbol",
        "timestamp",
        TARGET,
    ] + [
        c for c in all_features
        if c != "symbol"
    ]

    columns = list(dict.fromkeys(columns))

    query = f"""
        SELECT
            {", ".join(columns)}
        FROM {TABLE_NAME}
        WHERE timestamp >= %s
          AND timestamp < %s
        ORDER BY timestamp
    """

    print("Loading data...")

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            query,
            conn,
            params=(
                f"{TRAIN_START} 00:00:00",
                f"{TEST_END} 23:59:59",
            ),
        )
    finally:
        conn.close()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    # --------------------------------------------------------
    # Derive trading date from New York session timezone
    # --------------------------------------------------------

    df["trading_date"] = (
        df["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    df["trading_date"] = pd.to_datetime(
        df["trading_date"]
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['timestamp'].min()} -> "
        f"{df['timestamp'].max()}"
    )

    return df

# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    before = len(df)

    df = df.dropna(subset=[TARGET]).copy()

    print(
        f"Removed {before - len(df):,} rows "
        f"without target"
    )

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_data(df):

    train_mask = (
        (df["trading_date"] >= TRAIN_START)
        & (df["trading_date"] <= TRAIN_END)
    )

    val_mask = (
        (df["trading_date"] >= VAL_START)
        & (df["trading_date"] <= VAL_END)
    )

    test_mask = (
        (df["trading_date"] >= TEST_START)
        & (df["trading_date"] <= TEST_END)
    )

    train = df.loc[train_mask].copy()
    val = df.loc[val_mask].copy()
    test = df.loc[test_mask].copy()

    # --------------------------------------------------------
    # Embargo
    # --------------------------------------------------------
    #
    # The 5-minute target overlaps the boundary.
    # Remove the last 5 minutes of training and validation.
    #

    if len(train) > 0:
        train_max = train["timestamp"].max()
        cutoff = train_max - pd.Timedelta(
            minutes=EMBARGO_MINUTES
        )

        train = train[
            train["timestamp"] <= cutoff
        ]

    if len(val) > 0:
        val_max = val["timestamp"].max()
        cutoff = val_max - pd.Timedelta(
            minutes=EMBARGO_MINUTES
        )

        val = val[
            val["timestamp"] <= cutoff
        ]

    print("\nSplits:")
    print(
        f"Train: {len(train):,} | "
        f"{train['timestamp'].min()} -> "
        f"{train['timestamp'].max()}"
    )

    print(
        f"Val:   {len(val):,} | "
        f"{val['timestamp'].min()} -> "
        f"{val['timestamp'].max()}"
    )

    print(
        f"Test:  {len(test):,} | "
        f"{test['timestamp'].min()} -> "
        f"{test['timestamp'].max()}"
    )

    return train, val, test


# ============================================================
# MODEL
# ============================================================

def build_model():

    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


# ============================================================
# METRICS
# ============================================================

def correlation(y_true, y_pred):

    if np.std(y_pred) == 0:
        return 0.0

    return float(
        np.corrcoef(y_true, y_pred)[0, 1]
    )


def decile_analysis(y_true, y_pred):

    temp = pd.DataFrame({
        "actual": y_true,
        "pred": y_pred,
    })

    # rank avoids qcut failure when many predictions are identical
    temp["rank"] = temp["pred"].rank(
        method="first"
    )

    temp["decile"] = pd.qcut(
        temp["rank"],
        10,
        labels=False,
    ) + 1

    result = (
        temp
        .groupby("decile")
        .agg(
            count=("actual", "size"),
            avg_prediction=("pred", "mean"),
            avg_actual=("actual", "mean"),
            win_rate=("actual", lambda x: (x > 0).mean()),
        )
    )

    result["net_after_cost"] = (
        result["avg_actual"]
        - ROUND_TRIP_COST
    )

    return result


def top_10_analysis(y_true, y_pred):

    temp = pd.DataFrame({
        "actual": y_true,
        "pred": y_pred,
    })

    cutoff = temp["pred"].quantile(0.90)

    top = temp[temp["pred"] >= cutoff]

    if len(top) == 0:
        return {}

    avg_return = top["actual"].mean()

    return {
        "cutoff": float(cutoff),
        "signals": int(len(top)),
        "avg_actual_return": float(avg_return),
        "net_after_cost": float(
            avg_return - ROUND_TRIP_COST
        ),
        "win_rate": float(
            (top["actual"] > 0).mean()
        ),
    }


def threshold_analysis(y_true, y_pred):

    thresholds = [
        0.0000,
        0.0002,
        0.0004,
        0.0006,
    ]

    rows = []

    for threshold in thresholds:

        mask = y_pred > threshold

        if mask.sum() == 0:
            continue

        actual = y_true[mask]

        mean_return = actual.mean()

        rows.append({
            "threshold": threshold,
            "signals": int(mask.sum()),
            "signal_fraction": float(mask.mean()),
            "avg_actual_return": float(mean_return),
            "net_after_cost": float(
                mean_return - ROUND_TRIP_COST
            ),
            "win_rate": float(
                (actual > 0).mean()
            ),
        })

    return rows


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    name,
    features,
    train,
    val,
    test,
):

    print("\n")
    print("=" * 80)
    print(f"FEATURE GROUP: {name}")
    print("=" * 80)

    print("Features:")

    for feature in features:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Prepare X
    # --------------------------------------------------------

    train_x = train[features].copy()
    val_x = val[features].copy()
    test_x = test[features].copy()

    train_y = train[TARGET].values
    val_y = val[TARGET].values
    test_y = test[TARGET].values

    # --------------------------------------------------------
    # Categorical symbol
    # --------------------------------------------------------

    if "symbol" in features:

        train_x = pd.get_dummies(
            train_x,
            columns=["symbol"],
            dtype=float,
        )

        val_x = pd.get_dummies(
            val_x,
            columns=["symbol"],
            dtype=float,
        )

        test_x = pd.get_dummies(
            test_x,
            columns=["symbol"],
            dtype=float,
        )

        # Ensure exactly same columns
        val_x = val_x.reindex(
            columns=train_x.columns,
            fill_value=0,
        )

        test_x = test_x.reindex(
            columns=train_x.columns,
            fill_value=0,
        )

    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    train_x = train_x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    val_x = val_x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    test_x = test_x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # HGB can handle NaN values
    # directly.

    print(
        f"\nTraining rows: {len(train_x):,}"
    )

    print(
        f"Feature count: {train_x.shape[1]}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = build_model()

    model.fit(
        train_x,
        train_y,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def evaluate(
        split_name,
        y,
        pred,
    ):

        mae = mean_absolute_error(
            y,
            pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y,
                pred,
            )
        )

        r2 = r2_score(
            y,
            pred,
        )

        corr = correlation(
            y,
            pred,
        )

        print(
            f"\n{split_name}"
        )

        print(
            f"MAE:         {mae:.6%}"
        )

        print(
            f"RMSE:        {rmse:.6%}"
        )

        print(
            f"R²:          {r2:.6f}"
        )

        print(
            f"Correlation: {corr:.6f}"
        )

        print(
            f"Mean actual: {np.mean(y):.6%}"
        )

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "correlation": float(corr),
            "mean_actual": float(np.mean(y)),
        }

    val_metrics = evaluate(
        "VALIDATION",
        val_y,
        val_pred,
    )

    test_metrics = evaluate(
        "TEST",
        test_y,
        test_pred,
    )

    # --------------------------------------------------------
    # Deciles
    # --------------------------------------------------------

    print("\nValidation deciles:")

    val_deciles = decile_analysis(
        val_y,
        val_pred,
    )

    print(
        val_deciles.to_string(
            float_format=lambda x: f"{x:.6%}"
            if abs(x) < 1
            else f"{x:.4f}"
        )
    )

    print("\nTest deciles:")

    test_deciles = decile_analysis(
        test_y,
        test_pred,
    )

    print(
        test_deciles.to_string(
            float_format=lambda x: f"{x:.6%}"
            if abs(x) < 1
            else f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Top 10%
    # --------------------------------------------------------

    val_top10 = top_10_analysis(
        val_y,
        val_pred,
    )

    test_top10 = top_10_analysis(
        test_y,
        test_pred,
    )

    print("\nTop 10% validation:")
    print(val_top10)

    print("\nTop 10% test:")
    print(test_top10)

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    print("\nValidation thresholds:")

    val_thresholds = threshold_analysis(
        val_y,
        val_pred,
    )

    for row in val_thresholds:
        print(row)

    print("\nTest thresholds:")

    test_thresholds = threshold_analysis(
        test_y,
        test_pred,
    )

    for row in test_thresholds:
        print(row)

    # --------------------------------------------------------
    # Top-bottom spread
    # --------------------------------------------------------

    val_top_bottom = (
        val_deciles.iloc[-1]["avg_actual"]
        - val_deciles.iloc[0]["avg_actual"]
    )

    test_top_bottom = (
        test_deciles.iloc[-1]["avg_actual"]
        - test_deciles.iloc[0]["avg_actual"]
    )

    print(
        f"\nValidation top-bottom spread: "
        f"{val_top_bottom:.6%}"
    )

    print(
        f"Test top-bottom spread: "
        f"{test_top_bottom:.6%}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"intraday_ablation_{name}.joblib",
    )

    joblib.dump(
        model,
        model_path,
    )

    # --------------------------------------------------------
    # Return summary
    # --------------------------------------------------------

    return {
        "feature_group": name,
        "feature_count": int(
            train_x.shape[1]
        ),

        "validation": val_metrics,
        "test": test_metrics,

        "validation_top10": val_top10,
        "test_top10": test_top10,

        "validation_thresholds":
            val_thresholds,

        "test_thresholds":
            test_thresholds,

        "validation_top_bottom_spread":
            float(val_top_bottom),

        "test_top_bottom_spread":
            float(test_top_bottom),

        "model_path": model_path,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("INTRADAY FEATURE ABLATION")
    print("=" * 80)

    print(
        f"\nTarget: {TARGET}"
    )

    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST:.4%}"
    )

    df = load_data()

    df = prepare_data(df)

    train, val, test = split_data(df)

    results = []

    for name, features in FEATURE_GROUPS.items():

        result = run_experiment(
            name,
            features,
            train,
            val,
            test,
        )

        results.append(result)

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL ABLATION COMPARISON")
    print("=" * 80)

    comparison_rows = []

    for result in results:

        comparison_rows.append({
            "feature_group":
                result["feature_group"],

            "features":
                result["feature_count"],

            "val_corr":
                result["validation"]["correlation"],

            "test_corr":
                result["test"]["correlation"],

            "val_r2":
                result["validation"]["r2"],

            "test_r2":
                result["test"]["r2"],

            "val_top10_return":
                result["validation_top10"].get(
                    "avg_actual_return",
                    np.nan,
                ),

            "test_top10_return":
                result["test_top10"].get(
                    "avg_actual_return",
                    np.nan,
                ),

            "val_top10_net":
                result["validation_top10"].get(
                    "net_after_cost",
                    np.nan,
                ),

            "test_top10_net":
                result["test_top10"].get(
                    "net_after_cost",
                    np.nan,
                ),

            "val_top_bottom":
                result[
                    "validation_top_bottom_spread"
                ],

            "test_top_bottom":
                result[
                    "test_top_bottom_spread"
                ],
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6%}"
                if abs(x) < 1
                else f"{x:.4f}"
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = os.path.join(
        MODEL_DIR,
        "intraday_feature_ablation_results.json",
    )

    with open(
        results_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to:"
        f"\n{results_path}"
    )


if __name__ == "__main__":
    main()
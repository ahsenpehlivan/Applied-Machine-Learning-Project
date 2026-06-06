from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "df_train_ready.parquet"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

TARGET = "personal_best_lap_ms"
OFFSET_COLUMN = "best_quali_ms"
RANDOM_STATE = 42
RESIDUAL_CLIP_QUANTILES = (0.01, 0.99)

# These values are only known during or after the race. Using them for a
# pre-race prediction would leak future information into the model.
POST_RACE_LEAKAGE_COLUMNS = {
    TARGET,
    "gap_to_fastest_ms",
    "is_podium",
    "pit_stop_count",
    "avg_pit_dur_ms",
}

# Human-readable categorical columns carry the same information more safely.
REDUNDANT_IDENTIFIER_COLUMNS = {
    "raceId",
    "driverId",
    "constructorId",
    "circuitId",
}

LIGHTGBM_CANDIDATES = [
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 20,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.025,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_lambda": 2.0,
    },
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        frame = pd.read_parquet(path)

    required = {TARGET, OFFSET_COLUMN, "date", "year", "raceId", "driverId"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame[TARGET].isna().any():
        raise ValueError(
            "The regression target contains missing values. Targets must not be "
            "imputed; remove rows without a recorded fastest lap upstream."
        )

    return frame.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)


def split_chronologically(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["year"] <= 2018].copy()
    validation = frame.loc[frame["year"].between(2019, 2021)].copy()
    test = frame.loc[frame["year"] >= 2022].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("The chronological split produced an empty dataset.")
    return train, validation, test


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = (
        POST_RACE_LEAKAGE_COLUMNS
        | REDUNDANT_IDENTIFIER_COLUMNS
        | {"date"}
    )
    return [column for column in frame.columns if column not in excluded]


def regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    return {
        "mae_ms": float(mean_absolute_error(y_true, y_pred)),
        "rmse_ms": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    # Keep a deterministic alphabetical order. This also makes seeded feature
    # subsampling reproducible across pandas versions and source-column order.
    numerical = frame.columns.difference(categorical).tolist()

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            ("numerical", "passthrough", numerical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(frame: pd.DataFrame, model: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(frame)),
            ("model", model),
        ]
    )


def fit_residual_model(
    pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
) -> Pipeline:
    residual = target - features[OFFSET_COLUMN]
    lower, upper = residual.quantile(RESIDUAL_CLIP_QUANTILES).to_numpy()

    # Only the training target is clipped. Validation and test targets remain
    # untouched so evaluation still reflects all real observations.
    pipeline.fit(features, residual.clip(lower=lower, upper=upper))
    return pipeline


def predict_absolute_lap(
    pipeline: Pipeline, features: pd.DataFrame
) -> np.ndarray:
    predicted_residual = pipeline.predict(features)
    return predicted_residual + features[OFFSET_COLUMN].to_numpy()


def circuit_median_baseline(
    history: pd.DataFrame, evaluation: pd.DataFrame
) -> np.ndarray:
    circuit_medians = history.groupby("circuit_name", observed=True)[TARGET].median()
    global_median = float(history[TARGET].median())
    return (
        evaluation["circuit_name"]
        .map(circuit_medians)
        .fillna(global_median)
        .to_numpy()
    )


def save_feature_importance(
    pipeline: Pipeline, output_path: Path
) -> pd.DataFrame:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return importance_frame


def save_plots(
    test_metrics: list[dict[str, object]],
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    model_names = [str(row["model"]) for row in test_metrics]
    mae_seconds = [float(row["mae_ms"]) / 1000 for row in test_metrics]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(model_names, mae_seconds, color="#2f69a1")
    axis.bar_label(bars, fmt="%.2f s", padding=3)
    axis.set_title("Fastest-Lap Regression: Test MAE")
    axis.set_ylabel("Mean absolute error (seconds)")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(DOCS_DIR / "regression_model_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.scatter(
        predictions["actual_lap_ms"],
        predictions["predicted_lap_ms"],
        alpha=0.55,
        s=24,
        color="#2f69a1",
    )
    lower = min(
        predictions["actual_lap_ms"].min(),
        predictions["predicted_lap_ms"].min(),
    )
    upper = max(
        predictions["actual_lap_ms"].max(),
        predictions["predicted_lap_ms"].max(),
    )
    axis.plot([lower, upper], [lower, upper], "--", color="#b63b3b")
    axis.set_title("Actual vs Predicted Fastest-Lap Times")
    axis.set_xlabel("Actual lap time (ms)")
    axis.set_ylabel("Predicted lap time (ms)")
    figure.tight_layout()
    figure.savefig(DOCS_DIR / "regression_actual_vs_predicted.png", dpi=180)
    plt.close(figure)

    residuals = (
        predictions["actual_lap_ms"] - predictions["predicted_lap_ms"]
    )
    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.scatter(
        predictions["predicted_lap_ms"],
        residuals,
        alpha=0.55,
        s=24,
        color="#2f69a1",
    )
    axis.axhline(0, linestyle="--", color="#b63b3b")
    axis.set_title("Residual Analysis on the Test Set")
    axis.set_xlabel("Predicted lap time (ms)")
    axis.set_ylabel("Residual: actual - predicted (ms)")
    figure.tight_layout()
    figure.savefig(DOCS_DIR / "regression_residuals.png", dpi=180)
    plt.close(figure)

    top_features = feature_importance.head(20).sort_values("importance")
    figure, axis = plt.subplots(figsize=(8.5, 7))
    axis.barh(
        top_features["feature"],
        top_features["importance"],
        color="#2f69a1",
    )
    axis.set_title("Top 20 LightGBM Feature Importances")
    axis.set_xlabel("Split importance")
    figure.tight_layout()
    figure.savefig(DOCS_DIR / "regression_feature_importance.png", dpi=180)
    plt.close(figure)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_data()
    train, validation, test = split_chronologically(frame)
    feature_columns = select_feature_columns(frame)

    train_validation = pd.concat([train, validation], ignore_index=True)
    x_train = train[feature_columns]
    y_train = train[TARGET]
    x_validation = validation[feature_columns]
    y_validation = validation[TARGET]
    x_train_validation = train_validation[feature_columns]
    y_train_validation = train_validation[TARGET]
    x_test = test[feature_columns]
    y_test = test[TARGET]

    validation_results: list[dict[str, object]] = []

    random_forest = build_pipeline(
        x_train,
        RandomForestRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    )
    fit_residual_model(random_forest, x_train, y_train)
    random_forest_validation_prediction = predict_absolute_lap(
        random_forest, x_validation
    )
    random_forest_validation_metrics = regression_metrics(
        y_validation, random_forest_validation_prediction
    )
    validation_results.append(
        {
            "model": "Random Forest",
            **random_forest_validation_metrics,
        }
    )

    lightgbm_candidates: list[
        tuple[float, dict[str, object], Pipeline]
    ] = []
    for index, parameters in enumerate(LIGHTGBM_CANDIDATES, start=1):
        candidate = build_pipeline(
            x_train,
            lgb.LGBMRegressor(
                objective="regression_l1",
                random_state=RANDOM_STATE,
                n_jobs=2,
                verbosity=-1,
                **parameters,
            ),
        )
        fit_residual_model(candidate, x_train, y_train)
        prediction = predict_absolute_lap(candidate, x_validation)
        metrics = regression_metrics(y_validation, prediction)
        validation_results.append(
            {
                "model": f"LightGBM candidate {index}",
                **metrics,
            }
        )
        lightgbm_candidates.append((metrics["mae_ms"], parameters, candidate))

    best_lightgbm_mae, best_lightgbm_parameters, _ = min(
        lightgbm_candidates, key=lambda result: result[0]
    )
    selected_model = (
        "Random Forest"
        if random_forest_validation_metrics["mae_ms"] <= best_lightgbm_mae
        else "LightGBM"
    )

    global_median = DummyRegressor(strategy="median").fit(
        x_train_validation, y_train_validation
    )
    global_median_prediction = global_median.predict(x_test)

    circuit_median_prediction = circuit_median_baseline(
        train_validation, test
    )

    final_random_forest = build_pipeline(
        x_train_validation,
        RandomForestRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    )
    fit_residual_model(
        final_random_forest, x_train_validation, y_train_validation
    )
    random_forest_test_prediction = predict_absolute_lap(
        final_random_forest, x_test
    )

    final_lightgbm = build_pipeline(
        x_train_validation,
        lgb.LGBMRegressor(
            objective="regression_l1",
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbosity=-1,
            **best_lightgbm_parameters,
        ),
    )
    fit_residual_model(final_lightgbm, x_train_validation, y_train_validation)
    lightgbm_test_prediction = predict_absolute_lap(final_lightgbm, x_test)

    test_results = [
        {
            "model": "Global median",
            **regression_metrics(y_test, global_median_prediction),
        },
        {
            "model": "Circuit median",
            **regression_metrics(y_test, circuit_median_prediction),
        },
        {
            "model": "Random Forest",
            **regression_metrics(y_test, random_forest_test_prediction),
        },
        {
            "model": "LightGBM",
            **regression_metrics(y_test, lightgbm_test_prediction),
        },
    ]

    selected_pipeline = (
        final_random_forest
        if selected_model == "Random Forest"
        else final_lightgbm
    )
    selected_prediction = (
        random_forest_test_prediction
        if selected_model == "Random Forest"
        else lightgbm_test_prediction
    )

    predictions = test[
        [
            "date",
            "raceId",
            "year",
            "round",
            "circuit_name",
            "driver_name",
            "constructor_name",
        ]
    ].copy()
    predictions["actual_lap_ms"] = y_test.to_numpy()
    predictions["predicted_lap_ms"] = selected_prediction
    predictions["absolute_error_ms"] = np.abs(
        predictions["actual_lap_ms"] - predictions["predicted_lap_ms"]
    )
    predictions.to_csv(
        DATA_DIR / "regression_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    yearly_results = []
    for year, group in predictions.groupby("year"):
        yearly_results.append(
            {
                "year": int(year),
                **regression_metrics(
                    group["actual_lap_ms"], group["predicted_lap_ms"]
                ),
            }
        )
    pd.DataFrame(yearly_results).to_csv(
        DATA_DIR / "regression_metrics_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_importance = save_feature_importance(
        selected_pipeline,
        DATA_DIR / "regression_feature_importance.csv",
    )

    metrics_payload = {
        "objective": "Predict each driver's personal fastest race lap",
        "prediction_time": "After qualifying and before the race",
        "target": TARGET,
        "target_strategy": (
            f"Predict {TARGET} - {OFFSET_COLUMN}, then add {OFFSET_COLUMN}"
        ),
        "training_residual_clip_quantiles": list(
            RESIDUAL_CLIP_QUANTILES
        ),
        "selected_model": selected_model,
        "selection_metric": "Validation MAE",
        "best_lightgbm_parameters": best_lightgbm_parameters,
        "split": {
            "train": {"years": "2003-2018", "rows": int(len(train))},
            "validation": {
                "years": "2019-2021",
                "rows": int(len(validation)),
            },
            "test": {"years": "2022-2024", "rows": int(len(test))},
        },
        "excluded_post_race_columns": sorted(
            POST_RACE_LEAKAGE_COLUMNS - {TARGET}
        ),
        "validation_results": validation_results,
        "test_results": test_results,
        "yearly_test_results": yearly_results,
    }
    (DATA_DIR / "regression_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    joblib.dump(
        {
            "pipeline": selected_pipeline,
            "feature_columns": feature_columns,
            "target": TARGET,
            "offset_column": OFFSET_COLUMN,
            "target_strategy": metrics_payload["target_strategy"],
        },
        DATA_DIR / "regression_fastest_lap_model.joblib",
    )

    save_plots(test_results, predictions, feature_importance)

    print("=== Fastest-Lap Regression Report ===")
    print(
        "Train / validation / test rows: "
        f"{len(train)} / {len(validation)} / {len(test)}"
    )
    print(f"Selected model: {selected_model}")
    print("\nTest results:")
    print(pd.DataFrame(test_results).round(4).to_string(index=False))
    print("\nYearly test results:")
    print(pd.DataFrame(yearly_results).round(4).to_string(index=False))
    print("\nTop 15 feature importances:")
    print(feature_importance.head(15).to_string(index=False))
    print("\nRegression artifacts were written to data/ and docs/.")


if __name__ == "__main__":
    main()

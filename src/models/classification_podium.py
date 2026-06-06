from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = Path("df_train_ready.parquet")
TARGET = "is_podium"
RANDOM_STATE = 42

# Race sonrasinda olusan kolonlari siniflandirmada kullanmiyoruz.
LEAKAGE_COLUMNS = [
    TARGET,
    "personal_best_lap_ms",
    "pit_stop_count",
    "avg_pit_dur_ms",
    "gap_to_fastest_ms",
    "raceId",
    "date",
]

# ID kolonlari sayisal dursa da anlamsal olarak kategoriktir.
FORCED_CATEGORICAL_COLUMNS = ["driverId", "constructorId", "circuitId"]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {path.resolve()}")
    return pd.read_parquet(path).sort_values(["date", "raceId", "grid"]).reset_index(drop=True)


def split_by_race(
    df: pd.DataFrame, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    races = (
        df[["raceId", "date"]]
        .drop_duplicates()
        .sort_values(["date", "raceId"])
        .reset_index(drop=True)
    )

    n_races = len(races)
    n_test = max(1, int(round(n_races * test_ratio)))
    n_val = max(1, int(round(n_races * val_ratio)))
    n_train = n_races - n_val - n_test

    if n_train < 1:
        raise ValueError("Train set bos kaldi. Oranlari azaltin.")

    train_races = set(races.iloc[:n_train]["raceId"])
    val_races = set(races.iloc[n_train : n_train + n_val]["raceId"])
    test_races = set(races.iloc[n_train + n_val :]["raceId"])

    train_df = df[df["raceId"].isin(train_races)].copy()
    val_df = df[df["raceId"].isin(val_races)].copy()
    test_df = df[df["raceId"].isin(test_races)].copy()
    return train_df, val_df, test_df


def build_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    feature_columns = [col for col in train_df.columns if col not in LEAKAGE_COLUMNS]

    categorical_columns = list(train_df[feature_columns].select_dtypes(include=["object"]).columns)
    for column in FORCED_CATEGORICAL_COLUMNS:
        if column in feature_columns and column not in categorical_columns:
            categorical_columns.append(column)

    numerical_columns = [col for col in feature_columns if col not in categorical_columns]

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numeric", "passthrough", numerical_columns),
        ]
    )

    x_train = train_df[feature_columns].copy()
    x_val = val_df[feature_columns].copy()
    x_test = test_df[feature_columns].copy()

    for column in categorical_columns:
        x_train[column] = x_train[column].astype(str)
        x_val[column] = x_val[column].astype(str)
        x_test[column] = x_test[column].astype(str)

    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    x_train_t = transformer.fit_transform(x_train)
    x_val_t = transformer.transform(x_val)
    x_test_t = transformer.transform(x_test)
    return x_train_t, x_val_t, x_test_t, y_train, y_val, y_test, transformer


def select_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.10, 0.91, 0.01):
        preds = (proba >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))
    return best_threshold, float(best_f1)


def race_top3_metrics(df: pd.DataFrame, proba: np.ndarray) -> dict[str, float]:
    scored = df[["raceId", TARGET, "driver_name", "constructor_name", "grid"]].copy()
    scored["pred_proba"] = proba

    precision_scores = []
    exact_match_count = 0
    total_hits = 0
    total_actual = 0

    for _, race_df in scored.groupby("raceId"):
        top3_pred = race_df.nlargest(3, "pred_proba")
        top3_actual = race_df[race_df[TARGET] == 1]

        pred_drivers = set(top3_pred["driver_name"])
        actual_drivers = set(top3_actual["driver_name"])

        hits = len(pred_drivers & actual_drivers)
        precision_scores.append(hits / 3.0)
        total_hits += hits
        total_actual += max(1, len(actual_drivers))
        if pred_drivers == actual_drivers:
            exact_match_count += 1

    return {
        "precision_at_3": float(np.mean(precision_scores)),
        "recall_at_3": float(total_hits / total_actual),
        "exact_top3_race_rate": float(exact_match_count / scored["raceId"].nunique()),
    }


def save_feature_importance(
    transformer: ColumnTransformer, model: XGBClassifier, output_path: Path
) -> pd.DataFrame:
    feature_names = transformer.get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return importance_df


def main() -> None:
    df = load_data(DATA_PATH)
    train_df, val_df, test_df = split_by_race(df)

    x_train, x_val, x_test, y_train, y_val, y_test, transformer = build_features(
        train_df, val_df, test_df
    )

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / max(positives, 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_proba = model.predict_proba(x_val)[:, 1]
    best_threshold, best_val_f1 = select_best_threshold(y_val, val_proba)

    test_proba = model.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)

    metrics = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_races": int(train_df["raceId"].nunique()),
        "val_races": int(val_df["raceId"].nunique()),
        "test_races": int(test_df["raceId"].nunique()),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "selected_threshold": best_threshold,
        "validation_f1_best_threshold": best_val_f1,
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }
    metrics.update(race_top3_metrics(test_df, test_proba))

    predictions = test_df[
        [
            "date",
            "raceId",
            "year",
            "round",
            "driver_name",
            "constructor_name",
            "grid",
            TARGET,
        ]
    ].copy()
    predictions["pred_proba"] = test_proba
    predictions["pred_label"] = test_pred
    predictions = predictions.sort_values(["date", "raceId", "pred_proba"], ascending=[True, True, False])

    importance_df = save_feature_importance(
        transformer, model, Path("classification_feature_importance.csv")
    )

    Path("classification_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions.to_csv("classification_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("=== Podium Classification Report ===")
    print(f"Train / Val / Test race count: {metrics['train_races']} / {metrics['val_races']} / {metrics['test_races']}")
    print(f"Best threshold (validation): {best_threshold:.2f}")
    print(f"Test F1: {metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"Precision@3: {metrics['precision_at_3']:.4f}")
    print(f"Recall@3: {metrics['recall_at_3']:.4f}")
    print(f"Exact Top-3 Race Rate: {metrics['exact_top3_race_rate']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, digits=4, zero_division=0))
    print("\nTop 15 Feature Importance:")
    print(importance_df.head(15).to_string(index=False))
    print("\nFiles written:")
    print(" - classification_metrics.json")
    print(" - classification_test_predictions.csv")
    print(" - classification_feature_importance.csv")


if __name__ == "__main__":
    main()

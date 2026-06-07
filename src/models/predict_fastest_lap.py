from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "regression_fastest_lap_model.joblib"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "df_train_ready.parquet"


def format_lap_time(milliseconds: float) -> str:
    minutes = int(milliseconds // 60_000)
    seconds = (milliseconds % 60_000) / 1_000
    return f"{minutes}:{seconds:06.3f}"


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input must be a CSV or Parquet file.")


def validate_features(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Input data is missing required model features:\n- "
            + "\n- ".join(missing)
        )
    if frame[feature_columns].isna().any().any():
        columns = frame[feature_columns].columns[
            frame[feature_columns].isna().any()
        ].tolist()
        raise ValueError(f"Input features contain missing values: {columns}")


def predict(frame: pd.DataFrame, bundle: dict[str, object]) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    offset_column = bundle["offset_column"]
    validate_features(frame, feature_columns)

    features = frame[feature_columns]
    predicted_residual = bundle["pipeline"].predict(features)
    predicted_lap_ms = predicted_residual + features[offset_column].to_numpy()

    result_columns = [
        column
        for column in [
            "date",
            "raceId",
            "year",
            "round",
            "driver_name",
            "constructor_name",
            "circuit_name",
        ]
        if column in frame.columns
    ]
    result = frame[result_columns].copy()
    result["predicted_lap_ms"] = predicted_lap_ms
    result["predicted_lap_time"] = [
        format_lap_time(value) for value in predicted_lap_ms
    ]

    target = bundle["target"]
    if target in frame.columns:
        result["actual_lap_ms"] = frame[target].to_numpy()
        result["actual_lap_time"] = [
            format_lap_time(value) for value in frame[target]
        ]
        result["absolute_error_ms"] = (
            result["actual_lap_ms"] - result["predicted_lap_ms"]
        ).abs()

    return result


def historical_prediction(
    args: argparse.Namespace, bundle: dict[str, object]
) -> pd.DataFrame:
    history = load_table(args.data)
    matches = history.loc[
        (history["driver_name"].str.casefold() == args.driver.casefold())
        & (history["year"] == args.year)
        & (history["circuit_name"].str.casefold() == args.circuit.casefold())
    ].copy()

    if args.race_id is not None:
        matches = matches.loc[matches["raceId"] == args.race_id]

    if matches.empty:
        raise ValueError(
            "No matching driver-race was found. Check --driver, --year, "
            "--circuit, and optional --race-id."
        )
    if len(matches) > 1:
        race_ids = matches["raceId"].tolist()
        raise ValueError(
            f"More than one race matched (raceIds: {race_ids}). "
            "Provide --race-id to select one."
        )

    return predict(matches, bundle)


def file_prediction(
    args: argparse.Namespace, bundle: dict[str, object]
) -> pd.DataFrame:
    return predict(load_table(args.input), bundle)


def write_template(args: argparse.Namespace, bundle: dict[str, object]) -> None:
    template = pd.DataFrame(columns=bundle["feature_columns"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.output, index=False)
    print(f"Prediction template written to: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the trained fastest-lap regression model without retraining it."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained joblib model bundle.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    historical = subparsers.add_parser(
        "historical",
        help="Predict one historical driver-race from the prepared dataset.",
    )
    historical.add_argument("--driver", required=True)
    historical.add_argument("--year", required=True, type=int)
    historical.add_argument("--circuit", required=True)
    historical.add_argument("--race-id", type=int)
    historical.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Prepared dataset containing the selected historical race.",
    )
    historical.add_argument("--output", type=Path)

    file_command = subparsers.add_parser(
        "file",
        help="Predict all rows in a CSV or Parquet feature file.",
    )
    file_command.add_argument("--input", required=True, type=Path)
    file_command.add_argument("--output", type=Path)

    template = subparsers.add_parser(
        "template",
        help="Create an empty CSV with every feature required by the model.",
    )
    template.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "regression_prediction_template.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Trained model not found: {args.model}")

    bundle = joblib.load(args.model)

    if args.command == "template":
        write_template(args, bundle)
        return

    result = (
        historical_prediction(args, bundle)
        if args.command == "historical"
        else file_prediction(args, bundle)
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Predictions written to: {args.output}")

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()


# Example terminal commands (run from the repository root):
#
# Predict Max Verstappen's 2024 Monaco fastest lap:
# python src/models/predict_fastest_lap.py historical \
#   --driver "Max Verstappen" \
#   --year 2024 \
#   --circuit "Circuit de Monaco"
#
# Predict Lewis Hamilton's 2024 Silverstone fastest lap:
# python src/models/predict_fastest_lap.py historical \
#   --driver "Lewis Hamilton" \
#   --year 2024 \
#   --circuit "Silverstone Circuit"
#
# Save a historical prediction to CSV:
# python src/models/predict_fastest_lap.py historical \
#   --driver "Charles Leclerc" \
#   --year 2024 \
#   --circuit "Circuit de Monaco" \
#   --output data/leclerc_monaco_prediction.csv
#
# Create a 29-feature CSV template for future-race predictions:
# python src/models/predict_fastest_lap.py template \
#   --output data/new_race_features.csv
#
# Predict every completed row in a CSV file:
# python src/models/predict_fastest_lap.py file \
#   --input data/new_race_features.csv \
#   --output data/new_race_predictions.csv

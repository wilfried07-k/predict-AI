"""Script pour lancer une inference."""

from pathlib import Path
import sys
import argparse
import joblib
import yaml

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_csv
from src.models.versioning import get_latest_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to a CSV file. If omitted, uses config data.path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to a model.joblib. If omitted, uses latest run in models/runs",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=-1,
        help="Row index to predict (-1 for last row).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    data_path = Path(args.input) if args.input else Path(config["data"]["path"])
    model_output_dir = Path(config["model"]["output_dir"])
    feature_cols = config["model"]["feature_cols"]
    target_cols = config["model"]["target_cols"]

    if args.model:
        model_path = Path(args.model)
    else:
        latest_run = get_latest_run_dir(model_output_dir)
        if latest_run is None:
            raise FileNotFoundError("No trained models found. Run training first.")
        model_path = latest_run / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}.")

    model = joblib.load(model_path)
    print("Model used:", model_path)
    df = load_csv(data_path)
    X = df[feature_cols]

    if args.row == -1:
        row = X.tail(1)
        row_label = "last row"
    else:
        if args.row < 0 or args.row >= len(X):
            raise ValueError(f"Row index out of range: {args.row}")
        row = X.iloc[[args.row]]
        row_label = f"row {args.row}"

    preds = model.predict(row)[0]

    print(f"Inference ({row_label}):")
    for target, value in zip(target_cols, preds):
        print(f"- {target}: {value:.3f}")


if __name__ == "__main__":
    main()

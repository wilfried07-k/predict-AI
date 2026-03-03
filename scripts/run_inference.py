"""Script to run inference on a single row of the dataset."""

from pathlib import Path
import sys
import joblib
import yaml

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_csv


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    data_path = Path(config["data"]["path"])
    feature_cols = config["model"]["feature_cols"]
    target_cols = config["model"]["target_cols"]

    model_path = Path("models") / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    model = joblib.load(model_path)
    df = load_csv(data_path)
    X = df[feature_cols]

    row = X.tail(1)
    preds = model.predict(row)[0]

    print("Inference (last row):")
    for target, value in zip(target_cols, preds):
        print(f"- {target}: {value:.3f}")


if __name__ == "__main__":
    main()

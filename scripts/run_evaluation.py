"""Script to evaluate the trained model."""

from pathlib import Path
import sys
import json
import joblib
import yaml

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_csv
from src.models.evaluate import evaluate_multioutput


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    data_path = Path(config["data"]["path"])
    target_cols = config["model"]["target_cols"]
    feature_cols = config["model"]["feature_cols"]

    model_path = Path("models") / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    model = joblib.load(model_path)
    df = load_csv(data_path)
    X = df[feature_cols]
    y = df[target_cols]

    preds = model.predict(X)
    metrics = evaluate_multioutput(y, preds, target_cols)

    Path("reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("reports") / "metrics_eval.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")

    print("Metrics saved to:", out_path)
    print(metrics)


if __name__ == "__main__":
    main()

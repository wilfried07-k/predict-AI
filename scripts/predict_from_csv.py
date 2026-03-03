"""Predict from a CSV and save outputs."""

from pathlib import Path
import json
import joblib
import pandas as pd
import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf8"))

    data_path = root / "data" / "synthetic_poultry_farm.csv"
    feature_cols = config["model"]["feature_cols"]
    target_cols = config["model"]["target_cols"]

    model_path = root / "models" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    df = pd.read_csv(data_path)
    X = df[feature_cols]

    model = joblib.load(model_path)
    preds = model.predict(X)

    out = pd.DataFrame(preds, columns=target_cols)
    out_path = root / "reports" / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Saved predictions to:", out_path)

    # Also save a quick JSON sample
    sample = out.head(5).to_dict(orient="records")
    (root / "reports" / "predictions_sample.json").write_text(
        json.dumps(sample, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()

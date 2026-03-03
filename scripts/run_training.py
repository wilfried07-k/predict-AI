"""Script to run training."""

from pathlib import Path
import sys
import json
import yaml

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train import train_model


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    data_path = Path(config["data"]["path"])
    target_cols = config["model"]["target_cols"]
    feature_cols = config["model"]["feature_cols"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]
    cv_folds = config["model"].get("cv_folds", 0)

    model_path = Path("models") / "model.joblib"

    result = train_model(
        data_path=data_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        test_size=test_size,
        random_state=random_state,
        model_path=model_path,
        cv_folds=cv_folds,
    )

    print("Train size:", result["n_train"], "Test size:", result["n_test"])
    print("Model saved to:", model_path)
    print("Metrics:")
    for target, m in result["metrics"].items():
        print(f"- {target}: MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} R2={m['r2']:.3f}")

    if result["cv_metrics"] is not None:
        Path("reports").mkdir(parents=True, exist_ok=True)
        metrics_path = Path("reports") / "metrics_cv.json"
        metrics_path.write_text(json.dumps(result["cv_metrics"], indent=2), encoding="utf8")
        print("CV metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()

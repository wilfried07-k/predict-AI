"""Script pour lancer l entrainement."""

from pathlib import Path
import sys
import json
import yaml
import os

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train import train_model
from src.models.versioning import make_run_id


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    env_data_path = os.getenv("DATA_PATH")
    data_path = Path(env_data_path) if env_data_path else Path(config["data"]["path"])
    target_cols = config["model"]["target_cols"]
    feature_cols = config["model"]["feature_cols"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]
    cv_folds = config["model"].get("cv_folds", 0)
    cv_type = config["model"].get("cv_type", "kfold")
    model_output_dir = Path(config["model"]["output_dir"])
    reports_output_dir = Path(config["reports"]["output_dir"])

    run_id = make_run_id()
    run_model_dir = model_output_dir / run_id
    run_report_dir = reports_output_dir / run_id
    model_path = run_model_dir / "model.joblib"
    metrics_path = run_report_dir / "metrics_train.json"
    meta_path = run_report_dir / "run_metadata.json"

    result = train_model(
        data_path=data_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        test_size=test_size,
        random_state=random_state,
        model_path=model_path,
        cv_folds=cv_folds,
        cv_type=cv_type,
    )

    print("Run ID:", run_id)
    print("Train size:", result["n_train"], "Test size:", result["n_test"])
    if result["model_path"]:
        print("Model saved to:", result["model_path"])
    print("Metrics:")
    for target, m in result["metrics"].items():
        print(f"- {target}: MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} R2={m['r2']:.3f}")

    run_report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf8")
    print("Metrics saved to:", metrics_path)

    metadata = {
        "run_id": run_id,
        "data_path": str(data_path),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "test_size": test_size,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "cv_type": cv_type,
        "model_path": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print("Run metadata saved to:", meta_path)


if __name__ == "__main__":
    main()

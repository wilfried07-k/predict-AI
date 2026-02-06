"""Script to evaluate a saved model on the dataset."""

from pathlib import Path
import sys
import json
import argparse
import joblib
import yaml
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_csv
from src.models.evaluate import evaluate_multioutput
from src.models.versioning import get_latest_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to a model.joblib. If omitted, uses latest run in models/runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    data_path = Path(config["data"]["path"])
    model_output_dir = Path(config["model"]["output_dir"])
    feature_cols = config["model"]["feature_cols"]
    target_cols = config["model"]["target_cols"]

    if args.model:
        model_path = Path(args.model)
        run_dir = model_path.parent
    else:
        run_dir = get_latest_run_dir(model_output_dir)
        if run_dir is None:
            raise FileNotFoundError("No trained models found. Run training first.")
        model_path = run_dir / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}.")

    model = joblib.load(model_path)
    print("Model used:", model_path)
    df = load_csv(data_path)
    X = df[feature_cols]
    y = df[target_cols]

    preds = model.predict(X)
    metrics = evaluate_multioutput(y, preds, target_cols)

    reports_output_dir = Path(config["reports"]["output_dir"])
    out_dir = reports_output_dir / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_eval.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")

    print("Metrics saved to:", out_path)
    print(metrics)

    # Plot y_true vs y_pred
    fig, axes = plt.subplots(1, len(target_cols), figsize=(6 * len(target_cols), 5))
    if len(target_cols) == 1:
        axes = [axes]
    for idx, target in enumerate(target_cols):
        ax = axes[idx]
        ax.scatter(y.iloc[:, idx], preds[:, idx], alpha=0.7)
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.set_title(f"{target}: y_true vs y_pred")
    fig.tight_layout()
    plot_path = out_dir / "y_true_vs_y_pred.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print("Plot saved to:", plot_path)

    # Linear model coefficients (if available)
    linear_model_path = run_dir / "model_linear.joblib"
    if linear_model_path.exists():
        linear_model = joblib.load(linear_model_path)
        coef = linear_model.named_steps["model"].coef_
        scaler = linear_model.named_steps["scaler"]
        feature_std = scaler.scale_
        target_std = y.std(ddof=0).to_numpy()
        coef_payload = []
        for t_idx, target in enumerate(target_cols):
            for f_idx, feature in enumerate(feature_cols):
                std_coef = float(coef[t_idx, f_idx] * (feature_std[f_idx] / target_std[t_idx]))
                coef_payload.append(
                    {
                        "target": target,
                        "feature": feature,
                        "coefficient": float(coef[t_idx, f_idx]),
                        "std_coefficient": std_coef,
                        "feature_std": float(feature_std[f_idx]),
                        "target_std": float(target_std[t_idx]),
                    }
                )
        coef_path = out_dir / "linear_coefficients.json"
        coef_path.write_text(json.dumps(coef_payload, indent=2), encoding="utf8")
        print("Linear coefficients saved to:", coef_path)

        # Plot standardized coefficients per target
        for target in target_cols:
            rows = [r for r in coef_payload if r["target"] == target]
            rows = sorted(rows, key=lambda r: abs(r["std_coefficient"]), reverse=True)
            names = [r["feature"] for r in rows]
            vals = [r["std_coefficient"] for r in rows]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(names, vals)
            ax.set_title(f"Standardized Coefficients - {target}")
            ax.set_ylabel("Std Coefficient")
            ax.set_xticklabels(names, rotation=45, ha="right")
            fig.tight_layout()
            plot_path = out_dir / f"linear_coefficients_{target}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            print("Linear coefficients plot saved to:", plot_path)

    # Feature importance (permutation)
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=20,
        random_state=42,
        scoring="r2",
    )
    importances = sorted(
        zip(feature_cols, perm.importances_mean, perm.importances_std),
        key=lambda x: x[1],
        reverse=True,
    )
    imp_path = out_dir / "feature_importance.json"
    imp_payload = [
        {"feature": name, "importance_mean": float(mean), "importance_std": float(std)}
        for name, mean, std in importances
    ]
    imp_path.write_text(json.dumps(imp_payload, indent=2), encoding="utf8")
    print("Feature importance saved to:", imp_path)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [x[0] for x in importances]
    means = [x[1] for x in importances]
    ax.bar(names, means)
    ax.set_title("Permutation Feature Importance (R2)")
    ax.set_ylabel("Importance")
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    plot_path = out_dir / "feature_importance.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print("Feature importance plot saved to:", plot_path)


if __name__ == "__main__":
    main()

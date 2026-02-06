"""Update models/latest with the best evaluated model."""

from pathlib import Path
import json
import shutil


def score_from_metrics(metrics: dict) -> float:
    r2s = [v.get("r2", 0.0) for v in metrics.values()]
    return sum(r2s) / len(r2s) if r2s else float("-inf")


def main() -> None:
    runs_dir = Path("reports") / "runs"
    models_dir = Path("models") / "runs"
    latest_dir = Path("models") / "latest"

    if not runs_dir.exists():
        print("No runs found.")
        return

    best_run = None
    best_score = float("-inf")

    for run in runs_dir.iterdir():
        if not run.is_dir():
            continue
        metrics_path = run / "metrics_eval.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf8"))
        score = score_from_metrics(metrics)
        if score > best_score:
            best_score = score
            best_run = run

    if best_run is None:
        print("No evaluated runs found. Run evaluation first.")
        return

    model_path = models_dir / best_run.name / "model.joblib"
    if not model_path.exists():
        print(f"Model file missing for run {best_run.name}: {model_path}")
        return

    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, latest_dir / "model.joblib")
    shutil.copy2(best_run / "metrics_eval.json", latest_dir / "metrics_eval.json")
    meta_path = best_run / "run_metadata.json"
    if meta_path.exists():
        shutil.copy2(meta_path, latest_dir / "run_metadata.json")

    (latest_dir / "LATEST_RUN").write_text(best_run.name, encoding="utf8")

    print("Latest model updated from run:", best_run.name)
    print("Score (avg_r2):", f"{best_score:.3f}")


if __name__ == "__main__":
    main()

"""List available training/evaluation runs."""

from pathlib import Path
import json


def main() -> None:
    runs_dir = Path("reports") / "runs"
    if not runs_dir.exists():
        print("No runs found.")
        return

    runs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    if not runs:
        print("No runs found.")
        return

    print("Runs:")
    for run in runs:
        metrics_path = run / "metrics_eval.json"
        meta_path = run / "run_metadata.json"
        summary = ""
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf8"))
            r2s = [v.get("r2", 0.0) for v in metrics.values()]
            avg_r2 = sum(r2s) / len(r2s) if r2s else 0.0
            summary = f"avg_r2={avg_r2:.3f}"
        elif meta_path.exists():
            summary = "train-only"
        print(f"- {run.name} {summary}")


if __name__ == "__main__":
    main()

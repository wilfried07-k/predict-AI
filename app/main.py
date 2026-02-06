from pathlib import Path
import sys
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.versioning import get_latest_run_dir, make_run_id
from src.models.train import train_model, train_linear_model
from src.models.evaluate import evaluate_multioutput
from src.visualization.plot import plot_timeseries

app = FastAPI(title="predict-AI", version="1.0")
app.state.model = None
app.state.model_path = None
app.state.feature_cols = None
app.state.target_cols = None


class PredictRequest(BaseModel):
    features: dict
    model_path: str | None = None


class PredictResponse(BaseModel):
    predictions: dict
    model_used: str


class PredictBatchRequest(BaseModel):
    rows: list[dict]
    model_path: str | None = None


class PredictBatchResponse(BaseModel):
    predictions: list[dict]
    model_used: str


class TrainRequest(BaseModel):
    data_path: str | None = None
    feature_cols: list[str] | None = None
    target_cols: list[str] | None = None
    test_size: float | None = None
    random_state: int | None = None
    cv_folds: int | None = None
    cv_type: str | None = None
    model_output_dir: str | None = None
    reports_output_dir: str | None = None


class TrainResponse(BaseModel):
    run_id: str
    model_path: str
    linear_model_path: str
    metrics_path: str
    linear_metrics_path: str
    cv_metrics_path: str | None
    metadata_path: str


class EvaluateRequest(BaseModel):
    data_path: str | None = None
    model_path: str | None = None
    run_id: str | None = None
    feature_cols: list[str] | None = None
    target_cols: list[str] | None = None
    reports_output_dir: str | None = None
    n_repeats: int | None = None


class EvaluateResponse(BaseModel):
    model_used: str
    metrics_path: str
    plot_path: str
    feature_importance_path: str
    feature_importance_plot_path: str
    linear_coefficients_path: str | None
    linear_coefficients_plots: list[str]


class PlotRequest(BaseModel):
    data_path: str | None = None
    columns: list[str] | None = None
    output_path: str | None = None


class PlotResponse(BaseModel):
    plot_path: str


class RunsRequest(BaseModel):
    reports_dir: str | None = None


class RunsResponse(BaseModel):
    runs: list[dict]


def load_config() -> dict:
    config_path = ROOT / "configs" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf8"))


def get_data_path(config: dict, override_path: str | None) -> Path:
    if override_path:
        return Path(override_path)
    env_path = os.getenv("DATA_PATH")
    if env_path:
        return Path(env_path)
    return Path(config["data"]["path"])


def resolve_model_path(config: dict, model_path: str | None) -> Path:
    if model_path:
        return Path(model_path)
    model_output_dir = Path(config["model"]["output_dir"])
    latest_run = get_latest_run_dir(model_output_dir)
    if latest_run is None:
        raise FileNotFoundError("No trained models found. Run training first.")
    return latest_run / "model.joblib"


def load_model(model_path: Path):
    return joblib.load(model_path)


@app.on_event("startup")
def startup_event():
    config = load_config()
    app.state.feature_cols = config["model"]["feature_cols"]
    app.state.target_cols = config["model"]["target_cols"]

    try:
        model_path = resolve_model_path(config, None)
        if model_path.exists():
            app.state.model = load_model(model_path)
            app.state.model_path = str(model_path)
    except FileNotFoundError:
        app.state.model = None
        app.state.model_path = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root():
    config = load_config()
    data_path = str(get_data_path(config, None))
    html = """
    <!doctype html>
    <html lang="fr">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>predict-AI</title>
        <style>
          :root { --bg: #0f172a; --card: #111827; --text: #e5e7eb; --muted: #94a3b8; --accent: #22c55e; }
          body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial; background: var(--bg); color: var(--text); }
          .wrap { max-width: 900px; margin: 40px auto; padding: 0 20px; }
          .hero { background: linear-gradient(135deg, #0b1220, #111827); padding: 28px; border-radius: 16px; border: 1px solid #1f2937; }
          h1 { margin: 0 0 8px; font-size: 28px; }
          p { margin: 6px 0; color: var(--muted); }
          .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }
          .card { background: var(--card); border: 1px solid #1f2937; padding: 14px; border-radius: 12px; }
          code { color: var(--accent); }
          a { color: var(--accent); text-decoration: none; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="hero">
            <h1>predict-AI</h1>
            <p>API locale pour la prediction economique (inflation, chomage).</p>
            <p>DATA_PATH: <code>""" + data_path + """</code></p>
            <p>Documentation interactive: <a href="/docs">/docs</a> ou <a href="/redoc">/redoc</a></p>
          </div>
          <div class="grid">
            <div class="card"><strong>Health</strong><br/><code>GET /health</code></div>
            <div class="card"><strong>Predict</strong><br/><code>POST /predict</code></div>
            <div class="card"><strong>Batch</strong><br/><code>POST /predict_batch</code></div>
            <div class="card"><strong>Reload</strong><br/><code>POST /reload</code></div>
            <div class="card"><strong>Train</strong><br/><code>POST /train</code></div>
            <div class="card"><strong>Evaluate</strong><br/><code>POST /evaluate</code></div>
            <div class="card"><strong>Plot</strong><br/><code>POST /plot</code></div>
            <div class="card"><strong>Runs</strong><br/><code>GET /runs</code></div>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/reload")
def reload_model():
    config = load_config()
    try:
        model_path = resolve_model_path(config, None)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        app.state.model = load_model(model_path)
        app.state.model_path = str(model_path)
        app.state.feature_cols = config["model"]["feature_cols"]
        app.state.target_cols = config["model"]["target_cols"]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {"status": "reloaded", "model_used": str(app.state.model_path)}


def get_model(req_model_path: str | None):
    if req_model_path:
        model_path = Path(req_model_path)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        return load_model(model_path), str(model_path)

    if app.state.model is None:
        try:
            config = load_config()
            model_path = resolve_model_path(config, None)
            if not model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
            app.state.model = load_model(model_path)
            app.state.model_path = str(model_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    return app.state.model, app.state.model_path


def ensure_exists(path: Path, label: str):
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"{label} not found: {path}")


def validate_cv_type(cv_type: str):
    if cv_type not in {"kfold", "timeseries"}:
        raise HTTPException(status_code=400, detail=f"Invalid cv_type: {cv_type}")


def validate_test_size(test_size: float):
    if not (0.05 <= test_size <= 0.5):
        raise HTTPException(status_code=400, detail="test_size must be between 0.05 and 0.5")


def validate_cols(cols: list[str], label: str):
    if not cols:
        raise HTTPException(status_code=400, detail=f"{label} must be non-empty")


def validate_cv_folds(cv_folds: int):
    if cv_folds and cv_folds < 2:
        raise HTTPException(status_code=400, detail="cv_folds must be >= 2")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    feature_cols = app.state.feature_cols or load_config()["model"]["feature_cols"]
    target_cols = app.state.target_cols or load_config()["model"]["target_cols"]

    missing = [c for c in feature_cols if c not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    model, model_used = get_model(req.model_path)

    row = {c: req.features[c] for c in feature_cols}
    X = pd.DataFrame([row])
    preds = model.predict(X)[0]

    return PredictResponse(
        predictions={t: float(v) for t, v in zip(target_cols, preds)},
        model_used=str(model_used),
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    feature_cols = app.state.feature_cols or load_config()["model"]["feature_cols"]
    target_cols = app.state.target_cols or load_config()["model"]["target_cols"]

    if not req.rows:
        raise HTTPException(status_code=400, detail="rows is empty")

    for i, row in enumerate(req.rows):
        missing = [c for c in feature_cols if c not in row]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Row {i} missing features: {missing}",
            )

    model, model_used = get_model(req.model_path)

    X = pd.DataFrame([{c: r[c] for c in feature_cols} for r in req.rows])
    preds = model.predict(X)

    pred_rows = []
    for i in range(len(req.rows)):
        pred_rows.append({t: float(v) for t, v in zip(target_cols, preds[i])})

    return PredictBatchResponse(
        predictions=pred_rows,
        model_used=str(model_used),
    )


@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest | None = None):
    config = load_config()
    req = req or TrainRequest()

    data_path = get_data_path(config, req.data_path)
    feature_cols = req.feature_cols or config["model"]["feature_cols"]
    target_cols = req.target_cols or config["model"]["target_cols"]
    test_size = req.test_size if req.test_size is not None else config["model"]["test_size"]
    random_state = req.random_state if req.random_state is not None else config["model"]["random_state"]
    cv_folds = req.cv_folds if req.cv_folds is not None else config["model"].get("cv_folds", 0)
    cv_type = req.cv_type or config["model"].get("cv_type", "kfold")

    model_output_dir = (
        Path(req.model_output_dir) if req.model_output_dir else Path(config["model"]["output_dir"])
    )
    reports_output_dir = (
        Path(req.reports_output_dir) if req.reports_output_dir else Path(config["reports"]["output_dir"])
    )

    ensure_exists(data_path, "data_path")
    validate_cols(feature_cols, "feature_cols")
    validate_cols(target_cols, "target_cols")
    validate_test_size(test_size)
    validate_cv_folds(cv_folds)
    validate_cv_type(cv_type)

    model_output_dir.mkdir(parents=True, exist_ok=True)
    reports_output_dir.mkdir(parents=True, exist_ok=True)

    run_id = make_run_id()
    run_model_dir = model_output_dir / run_id
    run_report_dir = reports_output_dir / run_id
    model_path = run_model_dir / "model.joblib"
    linear_model_path = run_model_dir / "model_linear.joblib"
    metrics_path = run_report_dir / "metrics_train.json"
    linear_metrics_path = run_report_dir / "metrics_train_linear.json"
    cv_metrics_path = run_report_dir / "metrics_cv.json"
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

    linear_result = train_linear_model(
        data_path=data_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        test_size=test_size,
        random_state=random_state,
        model_path=linear_model_path,
    )

    run_report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf8")
    linear_metrics_path.write_text(json.dumps(linear_result["metrics"], indent=2), encoding="utf8")

    cv_metrics_out = None
    if result["cv_metrics"] is not None:
        cv_metrics_path.write_text(json.dumps(result["cv_metrics"], indent=2), encoding="utf8")
        cv_metrics_out = str(cv_metrics_path)

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
        "linear_model_path": str(linear_model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")

    # Refresh in-memory model to the new one
    if model_path.exists():
        app.state.model = load_model(model_path)
        app.state.model_path = str(model_path)
        app.state.feature_cols = feature_cols
        app.state.target_cols = target_cols

    return TrainResponse(
        run_id=run_id,
        model_path=str(model_path),
        linear_model_path=str(linear_model_path),
        metrics_path=str(metrics_path),
        linear_metrics_path=str(linear_metrics_path),
        cv_metrics_path=cv_metrics_out,
        metadata_path=str(meta_path),
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(req: EvaluateRequest | None = None):
    config = load_config()
    req = req or EvaluateRequest()

    data_path = get_data_path(config, req.data_path)
    model_output_dir = Path(config["model"]["output_dir"])
    reports_output_dir = (
        Path(req.reports_output_dir) if req.reports_output_dir else Path(config["reports"]["output_dir"])
    )
    feature_cols = req.feature_cols or config["model"]["feature_cols"]
    target_cols = req.target_cols or config["model"]["target_cols"]

    ensure_exists(data_path, "data_path")
    validate_cols(feature_cols, "feature_cols")
    validate_cols(target_cols, "target_cols")
    reports_output_dir.mkdir(parents=True, exist_ok=True)

    if req.model_path:
        model_path = Path(req.model_path)
        run_dir = model_path.parent
    elif req.run_id:
        run_dir = model_output_dir / req.run_id
        model_path = run_dir / "model.joblib"
    else:
        run_dir = get_latest_run_dir(model_output_dir)
        if run_dir is None:
            raise HTTPException(status_code=404, detail="No trained models found. Run training first.")
        model_path = run_dir / "model.joblib"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    model = load_model(model_path)
    df = pd.read_csv(data_path)
    X = df[feature_cols]
    y = df[target_cols]

    preds = model.predict(X)
    metrics = evaluate_multioutput(y, preds, target_cols)

    out_dir = reports_output_dir / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_eval.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")

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

    # Feature importance (permutation)
    n_repeats = req.n_repeats if req.n_repeats is not None else 20
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
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

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [x[0] for x in importances]
    means = [x[1] for x in importances]
    ax.bar(names, means)
    ax.set_title("Permutation Feature Importance (R2)")
    ax.set_ylabel("Importance")
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    imp_plot_path = out_dir / "feature_importance.png"
    fig.savefig(imp_plot_path)
    plt.close(fig)

    # Linear model coefficients (if available)
    linear_coeff_path = None
    linear_plots = []
    linear_model_path = run_dir / "model_linear.joblib"
    if linear_model_path.exists():
        linear_model = load_model(linear_model_path)
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
        linear_coeff_path = out_dir / "linear_coefficients.json"
        linear_coeff_path.write_text(json.dumps(coef_payload, indent=2), encoding="utf8")

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
            coef_plot_path = out_dir / f"linear_coefficients_{target}.png"
            fig.savefig(coef_plot_path)
            plt.close(fig)
            linear_plots.append(str(coef_plot_path))

    return EvaluateResponse(
        model_used=str(model_path),
        metrics_path=str(metrics_path),
        plot_path=str(plot_path),
        feature_importance_path=str(imp_path),
        feature_importance_plot_path=str(imp_plot_path),
        linear_coefficients_path=str(linear_coeff_path) if linear_coeff_path else None,
        linear_coefficients_plots=linear_plots,
    )


@app.post("/plot", response_model=PlotResponse)
def plot_endpoint(req: PlotRequest | None = None):
    config = load_config()
    req = req or PlotRequest()

    data_path = get_data_path(config, req.data_path)
    df = pd.read_csv(data_path)

    columns = req.columns or ["inflation", "unemployment", "gdp_growth"]
    out_path = Path(req.output_path) if req.output_path else Path("reports") / "timeseries.png"

    ensure_exists(data_path, "data_path")
    validate_cols(columns, "columns")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_timeseries(df, columns, out_path)

    return PlotResponse(plot_path=str(out_path))


@app.get("/runs", response_model=RunsResponse)
def runs_endpoint(req: RunsRequest | None = None):
    req = req or RunsRequest()
    runs_dir = Path(req.reports_dir) if req.reports_dir else Path("reports") / "runs"
    ensure_exists(runs_dir, "reports_dir")
    if not runs_dir.exists():
        return RunsResponse(runs=[])

    runs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    out = []
    for run in runs:
        metrics_path = run / "metrics_eval.json"
        meta_path = run / "run_metadata.json"
        summary = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf8"))
            r2s = [v.get("r2", 0.0) for v in metrics.values()]
            avg_r2 = sum(r2s) / len(r2s) if r2s else 0.0
            summary["avg_r2"] = avg_r2
        if meta_path.exists():
            summary["has_meta"] = True
        out.append({"run_id": run.name, **summary})

    return RunsResponse(runs=out)


from pathlib import Path
import sys
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import yaml

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.db import Base, SessionLocal, engine
from src.data.models import Batch, Prediction

app = FastAPI(title="predict-AI Poultry", version="1.0")
app.state.model = None
app.state.model_path = None
app.state.reports_dir = Path("reports")

FEATURE_LABELS_FR = {
    "age_days": "Age (jours)",
    "chick_weight_g": "Poids poussin (g)",
    "temp_c": "Temperature (C)",
    "humidity_pct": "Humidite (%)",
    "stocking_density": "Densite d'elevage",
    "vaccine_score": "Score vaccinal",
    "breed_index": "Indice de race",
    "housing_quality": "Qualite du batiment",
    "feed_protein_pct": "Proteines aliment (%)",
    "water_quality": "Qualite de l'eau",
    "management_index": "Indice de gestion",
    "flock_size": "Taille du lot",
    "feed_price_usd_kg": "Prix aliment (USD/kg)",
    "sale_price_usd_kg": "Prix de vente (USD/kg)",
    "energy_cost_usd": "Cout energie (USD)",
}

TARGET_LABELS_FR = {
    "final_weight_kg": "Poids final (kg)",
    "mortality_rate_pct": "Taux de mortalite (%)",
    "avg_daily_gain_g": "Gain quotidien (g/j)",
    "feed_intake_kg": "Consommation d'aliment (kg)",
    "fcr": "Indice de conversion (FCR)",
    "annual_revenue_usd": "Revenu annuel (USD)",
}


class PredictRequest(BaseModel):
    features: dict


class PredictResponse(BaseModel):
    predictions: dict


class PredictBatchRequest(BaseModel):
    rows: list[dict]


class PredictBatchResponse(BaseModel):
    predictions: list[dict]


def load_config() -> dict:
    config_path = ROOT / "configs" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf8"))


def get_model_path() -> Path:
    return Path("models") / "model.joblib"


@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)
    model_path = get_model_path()
    if model_path.exists():
        app.state.model = joblib.load(model_path)
        app.state.model_path = str(model_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reload")
def reload_model():
    model_path = get_model_path()
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found. Run training first.")
    app.state.model = joblib.load(model_path)
    app.state.model_path = str(model_path)
    return {"status": "reloaded", "model_used": app.state.model_path}


@app.get("/", response_class=HTMLResponse)
def root():
    config = load_config()
    data_path = config["data"]["path"]
    html = f"""
    <!doctype html>
    <html lang="fr">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>predict-AI Poultry</title>
        <style>
          :root {{ --bg: #0f172a; --card: #111827; --text: #e5e7eb; --muted: #94a3b8; --accent: #22c55e; }}
          body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial; background: var(--bg); color: var(--text); }}
          .wrap {{ max-width: 900px; margin: 40px auto; padding: 0 20px; }}
          .hero {{ background: linear-gradient(135deg, #0b1220, #111827); padding: 28px; border-radius: 16px; border: 1px solid #1f2937; }}
          h1 {{ margin: 0 0 8px; font-size: 28px; }}
          p {{ margin: 6px 0; color: var(--muted); }}
          .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }}
          .card {{ background: var(--card); border: 1px solid #1f2937; padding: 14px; border-radius: 12px; }}
          code {{ color: var(--accent); }}
          a {{ color: var(--accent); text-decoration: none; }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="hero">
            <h1>PREDICT-AI</h1>
            <p>API locale pour la prediction en elevage de poulets (broilers).</p>
            <p>DATA: <code>{data_path}</code></p>
            <p>Documentation: <a href="/docs">/docs</a> ou <a href="/redoc">/redoc</a></p>
          </div>
          <div class="grid">
            <div class="card"><strong>Sante API</strong><br/><code>GET /health</code></div>
            <div class="card"><strong>Predire (1 ligne)</strong><br/><code>POST /predict</code></div>
            <div class="card"><strong>Predire (batch)</strong><br/><code>POST /predict_batch</code></div>
            <div class="card"><strong>Tableaux</strong><br/><code>GET /tables</code></div>
            <div class="card"><strong>PREDICT-AI</strong><br/><code>GET /predict-ai</code></div>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/demo", response_class=HTMLResponse)
@app.get("/predict-ai", response_class=HTMLResponse)
def demo():
    config = load_config()
    feature_cols = config["model"]["feature_cols"]
    target_cols = config["model"]["target_cols"]

    feature_inputs = "\n".join(
        [
            f"<label>{FEATURE_LABELS_FR.get(name, name)}<input type='number' step='any' name='{name}' required /></label>"
            for name in feature_cols
        ]
    )
    target_cards = "\n".join(
        [f"<div class='result-card'><h4>{TARGET_LABELS_FR.get(t, t)}</h4><p id='out-{t}'>-</p></div>" for t in target_cols]
    )
    targets_json = json.dumps(target_cols)

    html = f"""
    <!doctype html>
    <html lang="fr">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>PREDICT-AI</title>
        <style>
          :root {{ --bg:#f4f1e8; --panel:#fffdf6; --ink:#1d1d1b; --muted:#57534e; --accent:#0f766e; --line:#d6d3d1; }}
          * {{ box-sizing: border-box; }}
          body {{ margin:0; font-family: "Segoe UI", Arial, sans-serif; background: radial-gradient(circle at top, #fef9c3 0%, var(--bg) 45%); color:var(--ink); }}
          .wrap {{ max-width:1100px; margin:24px auto; padding:0 16px; }}
          .hero {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:20px; }}
          h1 {{ margin:0 0 8px; font-size:28px; }}
          p {{ color:var(--muted); margin:0; }}
          .grid {{ display:grid; grid-template-columns: 1.25fr 1fr; gap:16px; margin-top:16px; }}
          .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }}
          form {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
          label {{ display:flex; flex-direction:column; font-size:13px; gap:4px; }}
          input {{ border:1px solid var(--line); border-radius:8px; padding:8px; font-size:14px; }}
          button {{ grid-column:1 / -1; border:0; border-radius:8px; padding:10px 12px; background:var(--accent); color:white; cursor:pointer; font-weight:600; }}
          .results {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
          .result-card {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:white; }}
          .result-card h4 {{ margin:0 0 6px; font-size:13px; color:var(--muted); }}
          .result-card p {{ margin:0; font-size:18px; color:var(--ink); font-weight:700; }}
          .status {{ margin-top:10px; color:var(--muted); min-height:20px; }}
          @media (max-width: 900px) {{ .grid {{ grid-template-columns:1fr; }} form {{ grid-template-columns:1fr; }} .results {{ grid-template-columns:1fr; }} }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="hero">
            <h1>PREDICT-AI</h1>
            <p>Saisis les variables d'elevage, lance la prediction, puis lis les resultats instantanement.</p>
          </div>
          <div class="grid">
            <div class="card">
              <h3>Entrees</h3>
              <form id="predict-form">
                {feature_inputs}
                <button type="submit">Lancer la prediction</button>
              </form>
              <div class="status" id="status"></div>
            </div>
            <div class="card">
              <h3>Sorties predites</h3>
              <div class="results">
                {target_cards}
              </div>
            </div>
          </div>
        </div>
        <script>
          const targets = {targets_json};
          const form = document.getElementById("predict-form");
          const status = document.getElementById("status");

          form.addEventListener("submit", async (e) => {{
            e.preventDefault();
            const data = new FormData(form);
            const features = Object.fromEntries([...data.entries()].map(([k, v]) => [k, Number(v)]));

            status.textContent = "Prediction en cours...";
            try {{
              const resp = await fetch("/predict", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ features }})
              }});
              const payload = await resp.json();
              if (!resp.ok) {{
                status.textContent = payload.detail || "Erreur API";
                return;
              }}

              for (const t of targets) {{
                const value = payload.predictions[t];
                const out = document.getElementById("out-" + t);
                out.textContent = Number(value).toFixed(3);
              }}
              status.textContent = "Prediction terminee et enregistree en base.";
            }} catch (err) {{
              status.textContent = "Erreur reseau/API.";
            }}
          }});
        </script>
      </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/tables", response_class=HTMLResponse)
def tables():
    reports = app.state.reports_dir
    preds_csv = reports / "predictions.csv"
    preds_json = reports / "predictions_sample.json"
    metrics_json = reports / "metrics_cv.json"

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>Tables</title>",
        "<style>body{font-family:Arial, sans-serif; margin:20px;} table{border-collapse:collapse;} th,td{border:1px solid #ccc; padding:6px 8px;} h2{margin-top:24px;}</style>",
        "</head><body>",
        "<h1>Reports Tables</h1>",
    ]

    if preds_csv.exists():
        df = pd.read_csv(preds_csv).head(50)
        html_parts.append("<h2>predictions.csv (top 50)</h2>")
        html_parts.append(df.to_html(index=False))
    else:
        html_parts.append("<h2>predictions.csv</h2><p>Not found. Run predict_from_csv.py</p>")

    if preds_json.exists():
        df = pd.read_json(preds_json)
        html_parts.append("<h2>predictions_sample.json</h2>")
        html_parts.append(df.to_html(index=False))
    else:
        html_parts.append("<h2>predictions_sample.json</h2><p>Not found. Run predict_from_csv.py</p>")

    if metrics_json.exists():
        metrics = json.loads(metrics_json.read_text(encoding="utf8"))
        rows = []
        for k, v in metrics.items():
            row = {"target": k}
            row.update(v)
            rows.append(row)
        df = pd.DataFrame(rows)
        html_parts.append("<h2>metrics_cv.json</h2>")
        html_parts.append(df.to_html(index=False))
    else:
        html_parts.append("<h2>metrics_cv.json</h2><p>Not found. Run training to generate CV metrics.</p>")

    html_parts.append("</body></html>")
    return HTMLResponse("\n".join(html_parts))


def save_prediction(feature_cols, target_cols, features, preds):
    db = SessionLocal()
    try:
        batch = Batch(**{c: float(features[c]) for c in feature_cols})
        db.add(batch)
        db.flush()

        pred = Prediction(
            batch_id=batch.id,
            model_version=app.state.model_path,
            **{t: float(v) for t, v in zip(target_cols, preds)},
        )
        db.add(pred)
        db.commit()
    finally:
        db.close()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    config = load_config()
    feature_cols = config["model"]["feature_cols"]
    target_cols = config["model"]["target_cols"]

    if app.state.model is None:
        model_path = get_model_path()
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found. Run training first.")
        app.state.model = joblib.load(model_path)
        app.state.model_path = str(model_path)

    missing = [c for c in feature_cols if c not in req.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # strict numeric validation
    row = {}
    for c in feature_cols:
        try:
            row[c] = float(req.features[c])
        except Exception:
            raise HTTPException(status_code=400, detail=f"Feature {c} must be numeric")
    X = pd.DataFrame([row])
    preds = app.state.model.predict(X)[0]

    save_prediction(feature_cols, target_cols, row, preds)

    return PredictResponse(
        predictions={t: float(v) for t, v in zip(target_cols, preds)}
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    config = load_config()
    feature_cols = config["model"]["feature_cols"]
    target_cols = config["model"]["target_cols"]

    if not req.rows:
        raise HTTPException(status_code=400, detail="rows is empty")

    if app.state.model is None:
        model_path = get_model_path()
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found. Run training first.")
        app.state.model = joblib.load(model_path)
        app.state.model_path = str(model_path)

    pred_rows = []
    for i, row in enumerate(req.rows):
        missing = [c for c in feature_cols if c not in row]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Row {i} missing features: {missing}",
            )

        for c in feature_cols:
            try:
                row[c] = float(row[c])
            except Exception:
                raise HTTPException(status_code=400, detail=f"Row {i} feature {c} must be numeric")

        X = pd.DataFrame([{c: row[c] for c in feature_cols}])
        preds = app.state.model.predict(X)[0]
        pred_rows.append({t: float(v) for t, v in zip(target_cols, preds)})

        save_prediction(feature_cols, target_cols, row, preds)

    return PredictBatchResponse(predictions=pred_rows)

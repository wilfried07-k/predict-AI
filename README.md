# predict-AI
![tests](https://github.com/wilfried07-k/predict-AI/actions/workflows/tests.yml/badge.svg)

Projet Python pour predire des indicateurs en elevage de poulets (broilers).

## Structure
- `data/`: donnees brutes et traitees
- `notebooks/`: notebooks d'exploration
- `src/`: code source
- `src/data/`: chargement et preparation des donnees
- `src/features/`: feature engineering
- `src/models/`: entrainement et evaluation
- `src/visualization/`: graphiques et rapports
- `configs/`: configurations du projet
- `scripts/`: scripts d'execution
- `tests/`: tests

## Utilisation
Installer les dependances:
```powershell
pip install -r requirements.txt
```

Entrainer le modele (versionne):
```powershell
python scripts\run_training.py
```
Cela sauvegarde aussi un modele lineaire dans `models/runs/<run_id>/model_linear.joblib`.

Inference sur le dernier enregistrement du CSV (dernier modele):
```powershell
python scripts\run_inference.py
```

Inference sur un nouveau CSV:
```powershell
python scripts\run_inference.py --input data\ton_fichier.csv --row -1
```

Inference avec un modele precis:
```powershell
python scripts\run_inference.py --model models\runs\YYYYMMDD_HHMMSS\model.joblib
```

Evaluation et sauvegarde des metriques (versionne):
```powershell
python scripts\run_evaluation.py
```

Lister les runs disponibles:
```powershell
python scripts\list_runs.py
```

Mettre a jour le meilleur modele dans `models/latest/`:
```powershell
python scripts\update_latest.py
```

Generer un graphique simple:
```powershell
python scripts\run_visualization.py
```

Importance des features (permutation) et graphique:
```powershell
python scripts\run_evaluation.py
```
Si un modele lineaire existe, les coefficients sont sauvegardes dans `linear_coefficients.json`.

## API FastAPI
Lancer le serveur:
```powershell
uvicorn app.main:app --reload
```

Exemple de requete:
```powershell
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\": {\"gdp_growth\": 3.0, \"exchange_rate\": 600, \"oil_price\": 80, \"imports\": 3.4, \"exports\": 3.5}}"
```

Parametres optionnels (exemples):
```powershell
# Entrainer avec un autre CSV
curl -X POST http://127.0.0.1:8000/train ^
  -H "Content-Type: application/json" ^
  -d "{\"data_path\": \"data/ton_fichier.csv\", \"cv_type\": \"timeseries\", \"cv_folds\": 3}"

# Evaluer un run precis
curl -X POST http://127.0.0.1:8000/evaluate ^
  -H "Content-Type: application/json" ^
  -d "{\"run_id\": \"YYYYMMDD_HHMMSS\", \"n_repeats\": 10}"

# Plot avec colonnes personnalisees
curl -X POST http://127.0.0.1:8000/plot ^
  -H "Content-Type: application/json" ^
  -d "{\"columns\": [\"inflation\", \"unemployment\"]}"
```

## Donnees
Les sorties versionnees sont dans `models/runs/` et `reports/runs/`.
Le fichier `data/synthetic_poultry_farm.csv` contient un dataset synthetique (50 000 lignes).
Cibles: `final_weight_kg`, `mortality_rate_pct`, `avg_daily_gain_g`, `feed_intake_kg`, `fcr`.
Remplace-le par vos donnees reelles si disponible.


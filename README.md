# predict-AI (Poultry)

Projet Python pour la prediction en elevage de poulets (broilers).

## Objectifs
- Poids final (kg)
- Taux de mortalite (%)
- Gain de poids quotidien (g/j)
- Consommation d'aliment (kg)
- FCR (indice de conversion)
- Revenu annuel (USD)

## Structure
- `data/`: donnees
- `notebooks/`: notebooks d'exploration
- `src/`: code source
- `configs/`: configurations
- `scripts/`: scripts d'execution
- `tests/`: tests

## Demarrage rapide
```powershell
pip install -r requirements.txt
python scripts\generate_dataset.py
python scripts\run_training.py
python scripts\run_evaluation.py
```

## API FastAPI
```powershell
uvicorn app.main:app --reload
```

## Donnees
Le fichier `data/synthetic_poultry_farm.csv` est genere automatiquement (50 000 lignes).
Remplace-le par vos donnees reelles si disponible.
## Batch prediction from CSV
```powershell
python scripts\predict_from_csv.py
```
Les sorties sont dans `reports/predictions.csv`.

## Graphiques
```powershell
python scripts\run_visualization.py
```
Sorties dans `reports/figures/`.
## MySQL setup
Créer la base :
```powershell
python scripts\create_db.py
```
Variables (si besoin) : `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`.
Importer un CSV vers MySQL :
```powershell
setx DATABASE_URL "mysql+pymysql://user:password@localhost:3306/poultry_ai"
setx CSV_PATH "data/synthetic_poultry_farm.csv"
python scripts\load_csv_to_mysql.py
```

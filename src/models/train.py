"""Training pipeline."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data.load import load_csv
from src.features.build_features import build_features


def train_model(
    data_path: Path,
    feature_cols: list[str],
    target_cols: list[str],
    test_size: float,
    random_state: int,
    model_path: Path | None = None,
    cv_folds: int = 0,
) -> dict:
    df = load_csv(data_path)
    X, y = build_features(df, feature_cols, target_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    base = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=random_state,
    )
    model = MultiOutputRegressor(base)

    cv_metrics = None
    if cv_folds and cv_folds >= 2:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_preds = cross_val_predict(model, X, y, cv=kf)
        cv_metrics = {}
        for idx, target in enumerate(target_cols):
            cv_metrics[target] = {
                "mae": float(mean_absolute_error(y.iloc[:, idx], cv_preds[:, idx])),
                "rmse": float(mean_squared_error(y.iloc[:, idx], cv_preds[:, idx]) ** 0.5),
                "r2": float(r2_score(y.iloc[:, idx], cv_preds[:, idx])),
            }

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {}
    for idx, target in enumerate(target_cols):
        metrics[target] = {
            "mae": float(mean_absolute_error(y_test.iloc[:, idx], y_pred[:, idx])),
            "rmse": float(mean_squared_error(y_test.iloc[:, idx], y_pred[:, idx]) ** 0.5),
            "r2": float(r2_score(y_test.iloc[:, idx], y_pred[:, idx])),
        }

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    return {
        "model": model,
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_path": str(model_path) if model_path is not None else None,
    }

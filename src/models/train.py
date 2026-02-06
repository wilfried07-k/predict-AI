"""Training pipeline."""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from src.data.load import load_csv
from src.data.validate import validate_columns, validate_no_nulls
from src.features.build_features import build_features


def train_model(
    data_path: Path,
    feature_cols: list[str],
    target_cols: list[str],
    test_size: float,
    random_state: int,
    model_path: Path | None = None,
    cv_folds: int = 0,
    cv_type: str = "kfold",
) -> dict:
    df = load_csv(data_path)
    required_cols = feature_cols + target_cols
    validate_columns(df, required_cols)
    validate_no_nulls(df, required_cols)
    X, y = build_features(df, feature_cols, target_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    base = RandomForestRegressor(n_estimators=300, random_state=random_state)
    estimator = MultiOutputRegressor(base)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )

    cv_metrics = None
    if cv_folds and cv_folds >= 2:
        if cv_type == "timeseries":
            splitter = TimeSeriesSplit(n_splits=cv_folds)
            y_true_all = []
            y_pred_all = []
            for train_idx, test_idx in splitter.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                y_true_all.append(y_te)
                y_pred_all.append(preds)

            y_true_all = pd.concat(y_true_all, axis=0)
            y_pred_all = np.vstack(y_pred_all)
        else:
            splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            y_pred_all = cross_val_predict(model, X, y, cv=splitter)
            y_true_all = y

        cv_metrics = {}
        for idx, target in enumerate(target_cols):
            cv_metrics[target] = {
                "mae": float(mean_absolute_error(y_true_all.iloc[:, idx], y_pred_all[:, idx])),
                "rmse": float(mean_squared_error(y_true_all.iloc[:, idx], y_pred_all[:, idx]) ** 0.5),
                "r2": float(r2_score(y_true_all.iloc[:, idx], y_pred_all[:, idx])),
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
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_path": str(model_path) if model_path is not None else None,
        "cv_metrics": cv_metrics,
    }


def train_linear_model(
    data_path: Path,
    feature_cols: list[str],
    target_cols: list[str],
    test_size: float,
    random_state: int,
    model_path: Path | None = None,
) -> dict:
    df = load_csv(data_path)
    required_cols = feature_cols + target_cols
    validate_columns(df, required_cols)
    validate_no_nulls(df, required_cols)
    X, y = build_features(df, feature_cols, target_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
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
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_path": str(model_path) if model_path is not None else None,
    }

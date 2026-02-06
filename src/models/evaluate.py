"""Evaluation utilities."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd


def evaluate_multioutput(y_true: pd.DataFrame, y_pred, target_cols: list[str]) -> dict:
    metrics = {}
    for idx, target in enumerate(target_cols):
        metrics[target] = {
            "mae": float(mean_absolute_error(y_true.iloc[:, idx], y_pred[:, idx])),
            "rmse": float(mean_squared_error(y_true.iloc[:, idx], y_pred[:, idx]) ** 0.5),
            "r2": float(r2_score(y_true.iloc[:, idx], y_pred[:, idx])),
        }
    return metrics

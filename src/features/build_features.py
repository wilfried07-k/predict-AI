"""Feature engineering."""

import pandas as pd


def build_features(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    return X, y

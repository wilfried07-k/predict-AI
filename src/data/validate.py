"""Data validation utilities."""

import pandas as pd


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def validate_no_nulls(df: pd.DataFrame, cols: list[str]) -> None:
    nulls = df[cols].isnull().sum()
    bad = nulls[nulls > 0]
    if not bad.empty:
        details = ", ".join([f"{k}={v}" for k, v in bad.items()])
        raise ValueError(f"Null values found: {details}")

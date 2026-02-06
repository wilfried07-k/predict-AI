"""Visualization utilities."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_timeseries(df: pd.DataFrame, columns: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for col in columns:
        plt.plot(pd.to_datetime(df["date"]), df[col], label=col)
    plt.legend()
    plt.title("Economic Indicators")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

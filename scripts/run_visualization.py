"""Generate basic exploratory plots for poultry dataset."""

from pathlib import Path
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf8"))

    data_path = root / config["data"]["path"]
    df = pd.read_csv(data_path)

    target_cols = config["model"]["target_cols"]
    feature_cols = config["model"]["feature_cols"]

    out_dir = root / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Distributions of targets
    for col in target_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"dist_{col}.png")
        plt.close()

    # 2) Correlation heatmap
    corr = df[feature_cols + target_cols].corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")
    plt.close()

    # 3) Feature vs target scatter (top 3 features by correlation per target)
    for target in target_cols:
        corr_series = corr[target].drop(labels=[target])
        top_features = corr_series.abs().sort_values(ascending=False).head(3).index.tolist()
        for feat in top_features:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[feat], y=df[target], alpha=0.3, s=10)
            plt.title(f"{feat} vs {target}")
            plt.tight_layout()
            plt.savefig(out_dir / f"scatter_{feat}_vs_{target}.png")
            plt.close()

    # 4) Feature importance (permutation) using a quick model fit
    # This is optional and may take a bit longer.
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance

        X = df[feature_cols]
        y = df[target_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        base = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=200,
            random_state=42,
        )
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)

        perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, scoring="r2")
        importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title("Permutation Feature Importance (avg R2)")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance.png")
        plt.close()
    except Exception as exc:
        print("Skipping feature importance plot:", exc)

    print("Plots saved to:", out_dir)


if __name__ == "__main__":
    main()

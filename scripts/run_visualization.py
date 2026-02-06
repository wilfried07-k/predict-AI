"""Script to generate basic plots."""

from pathlib import Path
import sys
import yaml

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_csv
from src.visualization.plot import plot_timeseries


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf8"))

    data_path = Path(config["data"]["path"])
    df = load_csv(data_path)

    out_path = Path("reports") / "timeseries.png"
    plot_timeseries(df, ["inflation", "unemployment", "gdp_growth"], out_path)
    print("Plot saved to:", out_path)


if __name__ == "__main__":
    main()

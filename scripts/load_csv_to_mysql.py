"""Load CSV data into MySQL batches table."""

import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@localhost:3306/poultry_ai",
)
CSV_PATH = os.getenv("CSV_PATH", "data/synthetic_poultry_farm.csv")


def main() -> None:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    df = pd.read_csv(Path(CSV_PATH))

    feature_cols = [
        "age_days",
        "chick_weight_g",
        "temp_c",
        "humidity_pct",
        "stocking_density",
        "vaccine_score",
        "breed_index",
        "housing_quality",
        "feed_protein_pct",
        "water_quality",
        "management_index",
        "flock_size",
        "feed_price_usd_kg",
        "sale_price_usd_kg",
        "energy_cost_usd",
    ]

    df = df[feature_cols]
    df.to_sql("batches", con=engine, if_exists="append", index=False)
    print(f"Inserted {len(df)} rows into batches")


if __name__ == "__main__":
    main()

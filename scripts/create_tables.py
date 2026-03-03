"""Create MySQL tables for the poultry project."""

from pathlib import Path
import sys

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.db import Base, engine
from src.data.models import Batch, Prediction  # noqa: F401


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("Tables ensured: batches, predictions")


if __name__ == "__main__":
    main()

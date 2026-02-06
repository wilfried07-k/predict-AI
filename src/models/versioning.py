"""Run versioning utilities."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone


def make_run_id() -> str:
    # UTC timestamp to ensure stable ordering
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def get_latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.name)[-1]

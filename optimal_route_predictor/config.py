"""Runtime configuration for the route predictor."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("ORP_DATA_DIR", ROOT_DIR / "data"))
MODEL_DIR = Path(os.getenv("ORP_MODEL_DIR", ROOT_DIR / "model"))
NOTEBOOK_DIR = ROOT_DIR / "notebooks"

TRIPS_PATH = DATA_DIR / "trips.csv"
LOCATIONS_PATH = DATA_DIR / "locations.csv"
DRIVERS_PATH = DATA_DIR / "drivers.csv"
MODEL_PATH = MODEL_DIR / "route_model.json"
GOOGLE_CACHE_PATH = Path(os.getenv("GOOGLE_CACHE_DB", DATA_DIR / "google_cache.sqlite"))

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
GOOGLE_CACHE_TTL_SECONDS = int(os.getenv("ORP_CACHE_TTL_SECONDS", "86400"))

RANDOM_SEED = int(os.getenv("ORP_RANDOM_SEED", "42"))
SYNTHETIC_DRIVER_COUNT = int(os.getenv("ORP_DRIVER_COUNT", "12"))
SYNTHETIC_LOCATION_COUNT = int(os.getenv("ORP_LOCATION_COUNT", "72"))


def ensure_directories() -> None:
    """Create the expected project folders."""

    for path in (DATA_DIR, MODEL_DIR, NOTEBOOK_DIR):
        path.mkdir(parents=True, exist_ok=True)


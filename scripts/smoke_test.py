"""Run a small end-to-end prediction without starting the API server."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimal_route_predictor.service import RoutePredictionService


if __name__ == "__main__":
    service = RoutePredictionService()
    pprint(service.health())
    pprint(service.predict_daily("D01", date(2026, 5, 20), ["A", "B", "C", "D"]))
    pprint(service.predict_weekly("D01", "2026-W20"))

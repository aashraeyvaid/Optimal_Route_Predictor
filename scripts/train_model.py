"""Train the route ETA and pattern model."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimal_route_predictor.data_generator import load_dataset
from optimal_route_predictor.modeling import train_route_model


if __name__ == "__main__":
    trips, locations, drivers = load_dataset()
    report = train_route_model(trips, locations, drivers)
    print(
        f"Trained model on {report.records} records "
        f"(R2={report.r2}, MAE={report.mae_minutes} min). "
        f"Saved to {report.model_path}"
    )

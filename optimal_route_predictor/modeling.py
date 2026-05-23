"""Model training, persistence, and ETA inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MODEL_PATH, ensure_directories
from .features import FEATURE_COLUMNS, build_driver_profiles, build_location_profiles, build_training_frame, prediction_features
from .google_client import DistanceResult


@dataclass
class TrainingReport:
    records: int
    drivers: int
    locations: int
    r2: float
    mae_minutes: float
    model_path: Path


class RouteDurationModel:
    """Ridge-regression ETA model plus learned driver/location patterns."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.feature_means = np.array(artifact["feature_means"], dtype=float)
        self.feature_stds = np.array(artifact["feature_stds"], dtype=float)
        self.coefficients = np.array(artifact["coefficients"], dtype=float)
        self.intercept = float(artifact["intercept"])

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "RouteDurationModel":
        with path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.artifact, handle, indent=2)

    @property
    def driver_profiles(self) -> dict[str, dict[str, Any]]:
        return self.artifact["driver_profiles"]

    @property
    def location_profiles(self) -> dict[str, dict[str, Any]]:
        return self.artifact["location_profiles"]

    @property
    def route_patterns(self) -> dict[str, Any]:
        return self.artifact["route_patterns"]

    @property
    def metrics(self) -> dict[str, float]:
        return self.artifact["metrics"]

    def predict_leg_minutes(
        self,
        *,
        driver_id: str,
        destination: dict[str, Any],
        distance_result: DistanceResult,
        stop_count: int,
        departure_hour: int,
        prediction_date: date,
    ) -> float:
        driver = self._driver_profile(driver_id)
        vector = prediction_features(
            distance_km=distance_result.distance_km,
            stop_count=stop_count,
            departure_hour=departure_hour,
            prediction_date=prediction_date,
            traffic_category=str(destination.get("traffic_category", "medium")),
            store_density=float(destination.get("store_density", 0.5)),
            driver_avg_speed_kmph=float(driver["avg_speed_kmph"]),
            driver_daily_visits=float(driver["daily_visits"]),
            driver_route_efficiency=float(driver["route_efficiency"]),
            location_priority=float(destination.get("priority", 3.0)),
        )
        normalized = (vector - self.feature_means) / self.feature_stds
        model_minutes = float(self.intercept + normalized.dot(self.coefficients))
        blended = (model_minutes * 0.58) + (distance_result.duration_in_traffic_min * 0.42)
        return round(max(1.5, blended), 2)

    def _driver_profile(self, driver_id: str) -> dict[str, Any]:
        if driver_id in self.driver_profiles:
            return self.driver_profiles[driver_id]
        speeds = [float(row["avg_speed_kmph"]) for row in self.driver_profiles.values()]
        visits = [float(row["daily_visits"]) for row in self.driver_profiles.values()]
        return {
            "home_region": "central",
            "hub_latitude": 23.0225,
            "hub_longitude": 72.5714,
            "avg_speed_kmph": float(np.mean(speeds)),
            "daily_visits": float(np.mean(visits)),
            "route_efficiency": 0.72,
            "max_daily_stops": 7.0,
        }


def _ridge_fit(features: pd.DataFrame, target: pd.Series, alpha: float = 0.65) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    x = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds < 1e-6] = 1.0
    x_scaled = (x - means) / stds
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    predictions = design @ weights
    return weights[1:], float(weights[0]), means, stds, predictions


def _build_transition_patterns(trips: pd.DataFrame) -> dict[str, Any]:
    transition_counts: dict[str, int] = {}
    day_location_counts: dict[str, dict[str, dict[str, int]]] = {}
    region_preferences: dict[str, dict[str, int]] = {}
    for (_, _), route in trips.sort_values(["date", "driver_id", "stop_sequence"]).groupby(["driver_id", "date"]):
        previous_id: str | None = None
        for row in route.to_dict("records"):
            driver_id = str(row["driver_id"])
            day = str(int(row["day_of_week"]))
            location_id = str(row["location_id"])
            day_location_counts.setdefault(driver_id, {}).setdefault(day, {})
            day_location_counts[driver_id][day][location_id] = day_location_counts[driver_id][day].get(location_id, 0) + 1
            region_preferences.setdefault(driver_id, {})
            region_preferences[driver_id][str(row["region"])] = region_preferences[driver_id].get(str(row["region"]), 0) + 1
            if previous_id is not None:
                key = f"{previous_id}->{location_id}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
            previous_id = location_id
    max_transition = max(transition_counts.values(), default=1)
    transition_scores = {key: round(value / max_transition, 4) for key, value in transition_counts.items()}
    return {
        "transition_scores": transition_scores,
        "day_location_counts": day_location_counts,
        "region_preferences": region_preferences,
    }


def train_route_model(
    trips: pd.DataFrame,
    locations: pd.DataFrame,
    drivers: pd.DataFrame,
    output_path: Path = MODEL_PATH,
) -> TrainingReport:
    ensure_directories()
    training_frame = build_training_frame(trips, drivers)
    coefficients, intercept, means, stds, predictions = _ridge_fit(
        training_frame[FEATURE_COLUMNS],
        training_frame["target_minutes"],
    )
    y = training_frame["target_minutes"].to_numpy(dtype=float)
    residuals = y - predictions
    mae = float(np.mean(np.abs(residuals)))
    total_variance = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / total_variance if total_variance else 0.0
    artifact = {
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_means": [round(float(value), 8) for value in means],
        "feature_stds": [round(float(value), 8) for value in stds],
        "coefficients": [round(float(value), 8) for value in coefficients],
        "intercept": round(intercept, 8),
        "metrics": {
            "records": int(len(trips)),
            "drivers": int(trips["driver_id"].nunique()),
            "locations": int(trips["location_id"].nunique()),
            "r2": round(r2, 4),
            "mae_minutes": round(mae, 3),
        },
        "driver_profiles": build_driver_profiles(trips, drivers),
        "location_profiles": build_location_profiles(locations, trips),
        "route_patterns": _build_transition_patterns(trips),
    }
    model = RouteDurationModel(artifact)
    model.save(output_path)
    return TrainingReport(
        records=len(trips),
        drivers=int(trips["driver_id"].nunique()),
        locations=int(trips["location_id"].nunique()),
        r2=round(r2, 4),
        mae_minutes=round(mae, 3),
        model_path=output_path,
    )


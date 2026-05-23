"""Feature engineering used for training and prediction."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

TRAFFIC_SCORES = {"low": 0.2, "medium": 0.45, "high": 0.72, "severe": 0.95}

FEATURE_COLUMNS = [
    "distance_km",
    "stop_count",
    "hour",
    "day_of_week",
    "is_weekend",
    "traffic_score",
    "store_density",
    "driver_avg_speed_kmph",
    "driver_daily_visits",
    "driver_route_efficiency",
    "location_priority",
]


def traffic_score(category: str) -> float:
    return TRAFFIC_SCORES.get(str(category).lower(), 0.45)


def parse_hour(value: Any) -> int:
    text = str(value)
    try:
        return int(text.split(":")[0])
    except (ValueError, IndexError):
        return 9


def build_driver_profiles(trips: pd.DataFrame, drivers: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    profile_rows: dict[str, dict[str, float | str]] = {}
    daily_visits = trips.groupby(["driver_id", "date"]).size().groupby("driver_id").mean()
    route_efficiency = trips.groupby("driver_id")["route_efficiency_score"].mean()
    observed_speed = (
        trips.assign(
            observed_speed=lambda frame: np.where(
                frame["travel_duration_from_prev_min"] > 0,
                frame["distance_from_prev_km"] / frame["travel_duration_from_prev_min"] * 60,
                np.nan,
            )
        )
        .groupby("driver_id")["observed_speed"]
        .mean()
    )
    for driver in drivers.to_dict("records"):
        driver_id = str(driver["driver_id"])
        profile_rows[driver_id] = {
            "home_region": str(driver["home_region"]),
            "hub_latitude": float(driver["hub_latitude"]),
            "hub_longitude": float(driver["hub_longitude"]),
            "avg_speed_kmph": float(observed_speed.get(driver_id, driver["avg_speed_kmph"])),
            "daily_visits": float(daily_visits.get(driver_id, driver["max_daily_stops"])),
            "route_efficiency": float(route_efficiency.get(driver_id, 0.72)),
            "max_daily_stops": float(driver["max_daily_stops"]),
        }
    return profile_rows


def build_location_profiles(locations: pd.DataFrame, trips: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    visit_stats = trips.groupby("location_id").agg(
        avg_visit_duration_min=("visit_duration_min", "mean"),
        historical_visits=("location_id", "size"),
        avg_arrival_hour=("visit_time", lambda values: float(np.mean([parse_hour(value) for value in values]))),
    )
    rows: dict[str, dict[str, float | str]] = {}
    for location in locations.to_dict("records"):
        location_id = str(location["location_id"])
        stats = visit_stats.loc[location_id] if location_id in visit_stats.index else None
        rows[location_id] = {
            "location_id": location_id,
            "location_code": str(location["location_code"]),
            "location_name": str(location["location_name"]),
            "place_id": str(location["place_id"]),
            "latitude": float(location["latitude"]),
            "longitude": float(location["longitude"]),
            "region": str(location["region"]),
            "traffic_category": str(location["traffic_category"]),
            "store_density": float(location["store_density"]),
            "priority": float(location["priority"]),
            "avg_visit_duration_min": float(stats["avg_visit_duration_min"]) if stats is not None else float(location["base_visit_duration_min"]),
            "historical_visits": float(stats["historical_visits"]) if stats is not None else 0.0,
            "avg_arrival_hour": float(stats["avg_arrival_hour"]) if stats is not None else 10.0,
        }
    return rows


def build_training_frame(trips: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    driver_profiles = build_driver_profiles(trips, drivers)
    rows: list[dict[str, float]] = []
    for trip in trips.to_dict("records"):
        driver = driver_profiles[str(trip["driver_id"])]
        row = {
            "distance_km": float(trip["distance_from_prev_km"]),
            "stop_count": float(trip["daily_stop_count"]),
            "hour": float(parse_hour(trip["visit_time"])),
            "day_of_week": float(trip["day_of_week"]),
            "is_weekend": 1.0 if int(trip["day_of_week"]) >= 5 else 0.0,
            "traffic_score": traffic_score(str(trip["traffic_category"])),
            "store_density": float(trip["store_density"]),
            "driver_avg_speed_kmph": float(driver["avg_speed_kmph"]),
            "driver_daily_visits": float(driver["daily_visits"]),
            "driver_route_efficiency": float(driver["route_efficiency"]),
            "location_priority": float(trip["priority"]),
            "target_minutes": float(trip["travel_duration_from_prev_min"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_features(
    *,
    distance_km: float,
    stop_count: int,
    departure_hour: int,
    prediction_date: date,
    traffic_category: str,
    store_density: float,
    driver_avg_speed_kmph: float,
    driver_daily_visits: float,
    driver_route_efficiency: float,
    location_priority: float,
) -> np.ndarray:
    values = [
        distance_km,
        float(stop_count),
        float(departure_hour),
        float(prediction_date.weekday()),
        1.0 if prediction_date.weekday() >= 5 else 0.0,
        traffic_score(traffic_category),
        store_density,
        driver_avg_speed_kmph,
        driver_daily_visits,
        driver_route_efficiency,
        location_priority,
    ]
    return np.array(values, dtype=float)


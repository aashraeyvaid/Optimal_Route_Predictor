"""Synthetic data generation for drivers, stores, and historical trips."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    DATA_DIR,
    DRIVERS_PATH,
    LOCATIONS_PATH,
    RANDOM_SEED,
    SYNTHETIC_DRIVER_COUNT,
    SYNTHETIC_LOCATION_COUNT,
    TRIPS_PATH,
    ensure_directories,
)
from .geo import driving_minutes, haversine_km, traffic_multiplier


REGION_CENTERS = {
    "central": (23.0225, 72.5714),
    "north": (23.0790, 72.5850),
    "south": (22.9720, 72.5920),
    "east": (23.0300, 72.6550),
    "west": (23.0400, 72.5000),
    "industrial": (22.9980, 72.6200),
}

REGION_TRAFFIC = {
    "central": "severe",
    "north": "medium",
    "south": "medium",
    "east": "high",
    "west": "low",
    "industrial": "high",
}


@dataclass(frozen=True)
class SyntheticDatasetSummary:
    drivers: int
    locations: int
    trips: int
    output_dir: Path


def _location_code(index: int) -> str:
    """Return spreadsheet-like location codes: A, B, ..., Z, AA."""

    letters = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        letters.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(letters))


def _weighted_choice(rng: random.Random, rows: list[dict], weights: list[float], k: int) -> list[dict]:
    selected: list[dict] = []
    pool = rows[:]
    pool_weights = weights[:]
    for _ in range(min(k, len(pool))):
        total = sum(pool_weights)
        marker = rng.random() * total
        cumulative = 0.0
        pick_index = 0
        for idx, weight in enumerate(pool_weights):
            cumulative += weight
            if cumulative >= marker:
                pick_index = idx
                break
        selected.append(pool.pop(pick_index))
        pool_weights.pop(pick_index)
    return selected


def build_locations(count: int = SYNTHETIC_LOCATION_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    regions = list(REGION_CENTERS)
    rows: list[dict] = []
    for idx in range(count):
        region = regions[idx % len(regions)]
        center_lat, center_lon = REGION_CENTERS[region]
        angle = rng.random() * math.tau
        radius = rng.uniform(0.003, 0.035)
        lat = center_lat + math.sin(angle) * radius
        lon = center_lon + math.cos(angle) * radius
        code = _location_code(idx)
        rows.append(
            {
                "location_id": f"L{idx + 1:03d}",
                "location_code": code,
                "location_name": f"Store {code}",
                "place_id": f"synthetic_place_{idx + 1:03d}",
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "region": region,
                "traffic_category": REGION_TRAFFIC[region],
                "store_density": round(rng.uniform(0.35, 0.95) if region in {"central", "east"} else rng.uniform(0.15, 0.65), 3),
                "priority": rng.randint(1, 5),
                "base_visit_duration_min": rng.randint(12, 38),
            }
        )
    return pd.DataFrame(rows)


def build_drivers(count: int = SYNTHETIC_DRIVER_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed + 7)
    regions = list(REGION_CENTERS)
    rows: list[dict] = []
    for idx in range(count):
        region = regions[idx % len(regions)]
        lat, lon = REGION_CENTERS[region]
        rows.append(
            {
                "driver_id": f"D{idx + 1:02d}",
                "home_region": region,
                "hub_latitude": lat,
                "hub_longitude": lon,
                "avg_speed_kmph": round(rng.uniform(24.0, 38.0), 2),
                "max_daily_stops": rng.randint(7, 10),
                "experience_level": rng.choice(["new", "standard", "senior"]),
            }
        )
    return pd.DataFrame(rows)


def _nearest_neighbor_order(start_lat: float, start_lon: float, stops: list[dict]) -> list[dict]:
    remaining = stops[:]
    ordered: list[dict] = []
    current_lat = start_lat
    current_lon = start_lon
    while remaining:
        next_stop = min(
            remaining,
            key=lambda row: haversine_km(current_lat, current_lon, row["latitude"], row["longitude"]),
        )
        ordered.append(next_stop)
        remaining.remove(next_stop)
        current_lat = next_stop["latitude"]
        current_lon = next_stop["longitude"]
    return ordered


def build_trips(
    drivers: pd.DataFrame,
    locations: pd.DataFrame,
    start_date: date = date(2026, 3, 23),
    end_date: date = date(2026, 5, 22),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = random.Random(seed + 19)
    location_rows = locations.to_dict("records")
    rows: list[dict] = []
    current = start_date
    while current <= end_date:
        if current.weekday() <= 5:
            for driver in drivers.to_dict("records"):
                stop_count = rng.randint(5, int(driver["max_daily_stops"]))
                if current.weekday() == 5:
                    stop_count = max(3, stop_count - 2)
                weights: list[float] = []
                for location in location_rows:
                    same_region = location["region"] == driver["home_region"]
                    weekday_affinity = 1.0 + ((current.weekday() + int(location["priority"])) % 3) * 0.12
                    weights.append((1.8 if same_region else 0.9) * weekday_affinity * (1.0 + location["priority"] / 8))
                picked = _weighted_choice(rng, location_rows, weights, stop_count)
                ordered = _nearest_neighbor_order(driver["hub_latitude"], driver["hub_longitude"], picked)
                trip_id = f"{driver['driver_id']}-{current.isoformat()}"
                now_dt = datetime.combine(current, time(hour=8, minute=rng.choice([0, 15, 30, 45])))
                prev_lat = float(driver["hub_latitude"])
                prev_lon = float(driver["hub_longitude"])
                route_distance = 0.0
                route_travel = 0.0
                day_rows: list[dict] = []
                for sequence, stop in enumerate(ordered, start=1):
                    distance = haversine_km(prev_lat, prev_lon, float(stop["latitude"]), float(stop["longitude"])) * rng.uniform(1.18, 1.46)
                    multiplier = traffic_multiplier(now_dt.hour, current, str(stop["traffic_category"]))
                    travel_minutes = driving_minutes(distance, float(driver["avg_speed_kmph"]), multiplier) * rng.uniform(0.88, 1.15)
                    visit_duration = max(8, int(stop["base_visit_duration_min"] + rng.gauss(0, 4)))
                    now_dt += timedelta(minutes=travel_minutes)
                    visit_time = now_dt.strftime("%H:%M")
                    now_dt += timedelta(minutes=visit_duration)
                    route_distance += distance
                    route_travel += travel_minutes
                    day_rows.append(
                        {
                            "trip_id": trip_id,
                            "driver_id": driver["driver_id"],
                            "date": current.isoformat(),
                            "iso_week": f"{current.isocalendar().year}-W{current.isocalendar().week:02d}",
                            "day_of_week": current.weekday(),
                            "stop_sequence": sequence,
                            "location_id": stop["location_id"],
                            "location_code": stop["location_code"],
                            "location_name": stop["location_name"],
                            "latitude": stop["latitude"],
                            "longitude": stop["longitude"],
                            "visit_time": visit_time,
                            "visit_duration_min": visit_duration,
                            "distance_from_prev_km": round(distance, 3),
                            "travel_duration_from_prev_min": round(travel_minutes, 2),
                            "traffic_category": stop["traffic_category"],
                            "region": stop["region"],
                            "store_density": stop["store_density"],
                            "priority": stop["priority"],
                        }
                    )
                    prev_lat = float(stop["latitude"])
                    prev_lon = float(stop["longitude"])
                straight_line = max(1.0, sum(row["distance_from_prev_km"] for row in day_rows))
                efficiency = min(0.98, max(0.52, (straight_line / max(route_distance, 1.0)) * rng.uniform(0.72, 0.96)))
                for row in day_rows:
                    row["daily_stop_count"] = len(day_rows)
                    row["route_total_distance_km"] = round(route_distance, 3)
                    row["route_total_travel_min"] = round(route_travel, 2)
                    row["route_efficiency_score"] = round(efficiency, 3)
                rows.extend(day_rows)
        current += timedelta(days=1)
    return pd.DataFrame(rows)


def generate_dataset(output_dir: Path = DATA_DIR, force: bool = False) -> SyntheticDatasetSummary:
    """Generate and persist the sample dataset required by the assignment."""

    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and TRIPS_PATH.exists() and LOCATIONS_PATH.exists() and DRIVERS_PATH.exists():
        trips = pd.read_csv(TRIPS_PATH)
        drivers = pd.read_csv(DRIVERS_PATH)
        locations = pd.read_csv(LOCATIONS_PATH)
        return SyntheticDatasetSummary(len(drivers), len(locations), len(trips), output_dir)

    locations = build_locations()
    drivers = build_drivers()
    trips = build_trips(drivers, locations)
    locations.to_csv(LOCATIONS_PATH, index=False)
    drivers.to_csv(DRIVERS_PATH, index=False)
    trips.to_csv(TRIPS_PATH, index=False)
    return SyntheticDatasetSummary(len(drivers), len(locations), len(trips), output_dir)


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load trips, locations, and drivers, generating them if missing."""

    generate_dataset()
    return pd.read_csv(TRIPS_PATH), pd.read_csv(LOCATIONS_PATH), pd.read_csv(DRIVERS_PATH)


def iter_location_refs(locations: pd.DataFrame) -> Iterable[str]:
    for row in locations.itertuples(index=False):
        yield str(row.location_code)
        yield str(row.location_id)
        yield str(row.location_name)


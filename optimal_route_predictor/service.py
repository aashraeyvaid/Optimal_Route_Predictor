"""Application service layer for API and scripts."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DRIVERS_PATH, LOCATIONS_PATH, MODEL_PATH, TRIPS_PATH, ensure_directories
from .data_generator import generate_dataset, load_dataset
from .google_client import GoogleMapsClient
from .modeling import RouteDurationModel, train_route_model
from .optimizer import LocationResolver, RouteOptimizer


DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class RoutePredictionService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        ensure_directories()
        if not (TRIPS_PATH.exists() and LOCATIONS_PATH.exists() and DRIVERS_PATH.exists()):
            generate_dataset()
        trips, locations, drivers = load_dataset()
        if not model_path.exists():
            train_route_model(trips, locations, drivers, model_path)
        self.trips = trips
        self.locations = locations
        self.drivers = drivers
        self.model_path = model_path
        self.model = RouteDurationModel.load(model_path)
        self.resolver = LocationResolver(self._locations_with_profiles())
        self.google_client = GoogleMapsClient(locations=self.locations)
        self.optimizer = RouteOptimizer(self.model, self.google_client)

    def _locations_with_profiles(self) -> pd.DataFrame:
        profiles = pd.DataFrame(self.model.location_profiles.values())
        merged = self.locations.drop(columns=[col for col in ["avg_visit_duration_min", "historical_visits"] if col in self.locations.columns])
        return merged.merge(
            profiles[["location_id", "avg_visit_duration_min", "historical_visits", "avg_arrival_hour"]],
            on="location_id",
            how="left",
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "records": int(len(self.trips)),
            "drivers": int(self.trips["driver_id"].nunique()),
            "locations": int(self.trips["location_id"].nunique()),
            "model": self.model.metrics,
            "google_live_enabled": self.google_client.live_enabled,
        }

    def list_drivers(self) -> list[dict[str, Any]]:
        driver_profiles = self.model.driver_profiles
        rows: list[dict[str, Any]] = []
        for driver in self.drivers.sort_values("driver_id").to_dict("records"):
            driver_id = str(driver["driver_id"])
            profile = driver_profiles.get(driver_id, {})
            rows.append(
                {
                    "driver_id": driver_id,
                    "home_region": str(driver["home_region"]),
                    "hub_latitude": float(driver["hub_latitude"]),
                    "hub_longitude": float(driver["hub_longitude"]),
                    "experience_level": str(driver["experience_level"]),
                    "avg_speed_kmph": round(float(profile.get("avg_speed_kmph", driver["avg_speed_kmph"])), 2),
                    "daily_visits": round(float(profile.get("daily_visits", driver["max_daily_stops"])), 1),
                    "route_efficiency": round(float(profile.get("route_efficiency", 0.72)), 3),
                    "max_daily_stops": int(driver["max_daily_stops"]),
                }
            )
        return rows

    def list_locations(self) -> list[dict[str, Any]]:
        locations = self._locations_with_profiles().sort_values("location_id")
        rows: list[dict[str, Any]] = []
        for location in locations.to_dict("records"):
            rows.append(
                {
                    "location_id": str(location["location_id"]),
                    "location_code": str(location["location_code"]),
                    "location_name": str(location["location_name"]),
                    "place_id": str(location["place_id"]),
                    "latitude": float(location["latitude"]),
                    "longitude": float(location["longitude"]),
                    "region": str(location["region"]),
                    "traffic_category": str(location["traffic_category"]),
                    "store_density": round(float(location["store_density"]), 3),
                    "priority": int(location["priority"]),
                    "avg_visit_duration_min": round(float(location.get("avg_visit_duration_min", 20.0)), 1),
                    "historical_visits": int(float(location.get("historical_visits", 0.0))),
                }
            )
        return rows

    def predict_daily(self, driver_id: str, prediction_date: date, location_refs: list[str]) -> dict[str, Any]:
        locations = self.resolver.resolve_many(location_refs)
        plan = self.optimizer.optimize(driver_id=driver_id, prediction_date=prediction_date, locations=locations)
        recommended_route = [str(row["location_code"]) for row in plan.ordered_locations]
        return {
            "driver_id": driver_id,
            "date": prediction_date.isoformat(),
            "recommended_route": recommended_route,
            "recommended_route_names": [str(row["location_name"]) for row in plan.ordered_locations],
            "predicted_time": f"{round(plan.predicted_minutes / 60.0, 2)} hours",
            "predicted_minutes": plan.predicted_minutes,
            "travel_minutes": plan.travel_minutes,
            "visit_minutes": plan.visit_minutes,
            "total_distance_km": plan.total_distance_km,
            "confidence": plan.confidence,
            "route_score": plan.route_score,
            "legs": plan.legs,
            "google_signal_source": plan.google_source,
        }

    def predict_weekly(self, driver_id: str, iso_week: str) -> dict[str, Any]:
        monday = _parse_iso_week(iso_week)
        used: set[str] = set()
        daily_routes: dict[str, list[str]] = {}
        daily_details: dict[str, Any] = {}
        weekly_distance = 0.0
        weekly_minutes = 0.0
        confidences: list[float] = []
        for offset in range(6):
            current_date = monday + timedelta(days=offset)
            refs = self._locations_for_driver_day(driver_id, current_date.weekday(), used)
            prediction = self.predict_daily(driver_id, current_date, refs)
            day_name = DAY_NAMES[current_date.weekday()]
            daily_routes[day_name] = prediction["recommended_route"]
            daily_details[day_name] = {
                "route_names": prediction["recommended_route_names"],
                "predicted_time": prediction["predicted_time"],
                "distance_km": prediction["total_distance_km"],
                "confidence": prediction["confidence"],
                "route_score": prediction["route_score"],
            }
            weekly_distance += float(prediction["total_distance_km"])
            weekly_minutes += float(prediction["predicted_minutes"])
            confidences.append(float(prediction["confidence"]))
            used.update(prediction["recommended_route"])
        return {
            "driver_id": driver_id,
            "week": iso_week,
            **daily_routes,
            "weekly_distance": f"{round(weekly_distance, 1)}km",
            "weekly_predicted_time": f"{round(weekly_minutes / 60.0, 2)} hours",
            "confidence": round(sum(confidences) / len(confidences), 2),
            "daily_details": daily_details,
        }

    def _locations_for_driver_day(self, driver_id: str, day_of_week: int, used_codes: set[str]) -> list[str]:
        driver_profile = self.model.driver_profiles.get(driver_id)
        target_count = int(round(float(driver_profile.get("daily_visits", 6.0)))) if driver_profile else 6
        if day_of_week == 5:
            target_count = max(3, target_count - 2)
        counts = (
            self.model.route_patterns.get("day_location_counts", {})
            .get(driver_id, {})
            .get(str(day_of_week), {})
        )
        ranked_ids = [location_id for location_id, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
        selected: list[str] = []
        id_to_profile = self.model.location_profiles
        for location_id in ranked_ids:
            profile = id_to_profile[location_id]
            code = str(profile["location_code"])
            if code not in used_codes:
                selected.append(code)
            if len(selected) >= target_count:
                return selected
        fallback = sorted(
            self.model.location_profiles.values(),
            key=lambda row: (str(row.get("region")) != str(driver_profile.get("home_region") if driver_profile else "central"), -float(row.get("historical_visits", 0))),
        )
        for profile in fallback:
            code = str(profile["location_code"])
            if code not in used_codes and code not in selected:
                selected.append(code)
            if len(selected) >= target_count:
                break
        return selected

    def reroute(self, driver_id: str, prediction_date: date, current_location: str, remaining_locations: list[str]) -> dict[str, Any]:
        locations = [current_location, *remaining_locations]
        prediction = self.predict_daily(driver_id, prediction_date, locations)
        prediction["current_location"] = current_location
        prediction["reroute_reason"] = "traffic-aware reorder of remaining stops"
        return prediction

    def retrain(self, regenerate_data: bool = False) -> dict[str, Any]:
        if regenerate_data:
            generate_dataset(force=True)
        trips, locations, drivers = load_dataset()
        report = train_route_model(trips, locations, drivers, self.model_path)
        self.__init__(self.model_path)
        return {
            "status": "retrained",
            "records": report.records,
            "drivers": report.drivers,
            "locations": report.locations,
            "r2": report.r2,
            "mae_minutes": report.mae_minutes,
            "model_path": str(report.model_path),
        }

    def nearby_places(self, latitude: float, longitude: float, radius_m: int = 1500) -> list[dict[str, Any]]:
        return self.google_client.nearby_places(latitude, longitude, radius_m)

    def place_details(self, place_id: str) -> dict[str, Any]:
        return self.google_client.place_details(place_id)

    def geocode(self, address: str) -> dict[str, Any]:
        return self.google_client.geocode(address)

    def monitoring(self) -> dict[str, Any]:
        by_day = self.trips.groupby("day_of_week").size().to_dict()
        return {
            "model_metrics": self.model.metrics,
            "data_volume_by_day": {DAY_NAMES[int(day)]: int(count) for day, count in by_day.items()},
            "latest_trip_date": str(self.trips["date"].max()),
            "google_live_enabled": self.google_client.live_enabled,
            "cache_database": str(self.google_client.cache.path),
        }


def _parse_iso_week(iso_week: str) -> date:
    try:
        year_text, week_text = iso_week.upper().split("-W")
        return date.fromisocalendar(int(year_text), int(week_text), 1)
    except Exception as exc:
        raise ValueError("week must use ISO format like 2026-W20") from exc

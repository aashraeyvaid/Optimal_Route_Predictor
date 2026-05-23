"""Route sequence optimization and confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd

from .google_client import DistanceResult, GoogleMapsClient
from .modeling import RouteDurationModel


@dataclass
class RoutePlan:
    ordered_locations: list[dict[str, Any]]
    total_distance_km: float
    travel_minutes: float
    visit_minutes: float
    predicted_minutes: float
    confidence: float
    google_source: str
    legs: list[dict[str, Any]]
    route_score: float


class LocationResolver:
    def __init__(self, locations: pd.DataFrame) -> None:
        self.locations = locations
        self.index: dict[str, dict[str, Any]] = {}
        for row in locations.to_dict("records"):
            aliases = {
                str(row["location_id"]),
                str(row["location_code"]),
                str(row["location_name"]),
                str(row["place_id"]),
            }
            for alias in aliases:
                self.index[alias.strip().lower()] = row

    def resolve_many(self, refs: list[str]) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        missing: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            row = self.index.get(str(ref).strip().lower())
            if row is None:
                missing.append(str(ref))
                continue
            location_id = str(row["location_id"])
            if location_id not in seen:
                resolved.append(row)
                seen.add(location_id)
        if missing:
            valid_examples = ", ".join(list(self.index)[:8])
            raise ValueError(f"Unknown locations: {', '.join(missing)}. Try codes/names such as {valid_examples}.")
        if not resolved:
            raise ValueError("At least one valid location is required.")
        return resolved


class RouteOptimizer:
    def __init__(self, model: RouteDurationModel, google_client: GoogleMapsClient) -> None:
        self.model = model
        self.google_client = google_client

    def optimize(
        self,
        *,
        driver_id: str,
        prediction_date: date,
        locations: list[dict[str, Any]],
        start_hour: int = 9,
    ) -> RoutePlan:
        hub = self._driver_hub(driver_id)
        if len(locations) == 1:
            departure = datetime.combine(prediction_date, time(hour=start_hour))
            matrix = self.google_client.distance_matrix([hub, locations[0]], [hub, locations[0]], departure)
            return self._single_stop_plan(driver_id, prediction_date, hub, locations[0], matrix, start_hour)
        departure = datetime.combine(prediction_date, time(hour=start_hour))
        matrix_nodes = [hub, *locations]
        matrix = self.google_client.distance_matrix(matrix_nodes, matrix_nodes, departure)
        route = self._greedy_route(hub, locations, matrix)
        route = self._two_opt(hub, route, matrix)
        return self._score_route(driver_id, prediction_date, hub, route, matrix, start_hour)

    def _driver_hub(self, driver_id: str) -> dict[str, Any]:
        driver = self.model.driver_profiles.get(driver_id)
        if driver is None:
            driver = {
                "home_region": "central",
                "hub_latitude": 23.0225,
                "hub_longitude": 72.5714,
            }
        return {
            "location_id": f"HUB-{driver_id}",
            "location_code": f"HUB-{driver_id}",
            "location_name": f"{driver_id} starting hub",
            "latitude": float(driver["hub_latitude"]),
            "longitude": float(driver["hub_longitude"]),
            "region": str(driver["home_region"]),
            "traffic_category": "medium",
            "store_density": 0.0,
            "priority": 3.0,
            "avg_visit_duration_min": 0.0,
            "historical_visits": 0.0,
        }

    def _single_stop_plan(
        self,
        driver_id: str,
        prediction_date: date,
        hub: dict[str, Any],
        location: dict[str, Any],
        matrix: dict[tuple[str, str], DistanceResult],
        start_hour: int,
    ) -> RoutePlan:
        result = matrix[(str(hub["location_id"]), str(location["location_id"]))]
        travel_minutes = self.model.predict_leg_minutes(
            driver_id=driver_id,
            destination=location,
            distance_result=result,
            stop_count=1,
            departure_hour=start_hour,
            prediction_date=prediction_date,
        )
        visit_minutes = float(location.get("avg_visit_duration_min", location.get("base_visit_duration_min", 20.0)))
        confidence = self._confidence(driver_id, [location], matrix)
        predicted_minutes = round(travel_minutes + visit_minutes, 1)
        return RoutePlan(
            ordered_locations=[location],
            total_distance_km=round(result.distance_km, 2),
            travel_minutes=round(travel_minutes, 1),
            visit_minutes=visit_minutes,
            predicted_minutes=predicted_minutes,
            confidence=confidence,
            google_source=result.source,
            legs=[
                {
                    "from": hub["location_code"],
                    "to": location["location_code"],
                    "distance_km": round(result.distance_km, 2),
                    "predicted_minutes": round(travel_minutes, 1),
                    "traffic_minutes": round(result.duration_in_traffic_min, 1),
                    "source": result.source,
                }
            ],
            route_score=self._route_score(predicted_minutes, 1, confidence),
        )

    def _greedy_route(
        self,
        hub: dict[str, Any],
        locations: list[dict[str, Any]],
        matrix: dict[tuple[str, str], DistanceResult],
    ) -> list[dict[str, Any]]:
        route: list[dict[str, Any]] = []
        remaining = locations[:]
        current = hub
        while remaining:
            next_stop = min(remaining, key=lambda row: self._edge_cost(current, row, matrix))
            route.append(next_stop)
            remaining = [row for row in remaining if row["location_id"] != next_stop["location_id"]]
            current = next_stop
        return route

    def _two_opt(
        self,
        hub: dict[str, Any],
        route: list[dict[str, Any]],
        matrix: dict[tuple[str, str], DistanceResult],
    ) -> list[dict[str, Any]]:
        best = route[:]
        best_cost = self._route_cost(best, matrix, hub)
        improved = True
        passes = 0
        while improved and passes < 4:
            improved = False
            passes += 1
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best)):
                    if j - i == 1:
                        continue
                    candidate = best[:]
                    candidate[i:j] = reversed(candidate[i:j])
                    candidate_cost = self._route_cost(candidate, matrix, hub)
                    if candidate_cost + 0.05 < best_cost:
                        best = candidate
                        best_cost = candidate_cost
                        improved = True
        return best

    def _route_cost(
        self,
        route: list[dict[str, Any]],
        matrix: dict[tuple[str, str], DistanceResult],
        hub: dict[str, Any],
    ) -> float:
        cost = 0.0
        full_path = [hub, *route]
        for origin, destination in zip(full_path, full_path[1:]):
            cost += self._edge_cost(origin, destination, matrix)
        return cost

    def _edge_cost(
        self,
        origin: dict[str, Any],
        destination: dict[str, Any],
        matrix: dict[tuple[str, str], DistanceResult],
    ) -> float:
        result = matrix[(str(origin["location_id"]), str(destination["location_id"]))]
        transition_scores = self.model.route_patterns.get("transition_scores", {})
        transition_bonus = float(transition_scores.get(f"{origin['location_id']}->{destination['location_id']}", 0.0))
        density_penalty = float(destination.get("store_density", 0.5)) * 0.35
        priority_bonus = float(destination.get("priority", 3.0)) * 0.08
        return result.duration_in_traffic_min + density_penalty - transition_bonus * 3.0 - priority_bonus

    def _score_route(
        self,
        driver_id: str,
        prediction_date: date,
        hub: dict[str, Any],
        route: list[dict[str, Any]],
        matrix: dict[tuple[str, str], DistanceResult],
        start_hour: int,
    ) -> RoutePlan:
        current_time = datetime.combine(prediction_date, time(hour=start_hour))
        legs: list[dict[str, Any]] = []
        travel_minutes = 0.0
        total_distance = 0.0
        google_sources: set[str] = set()
        full_path = [hub, *route]
        for origin, destination in zip(full_path, full_path[1:]):
            result = matrix[(str(origin["location_id"]), str(destination["location_id"]))]
            predicted_leg = self.model.predict_leg_minutes(
                driver_id=driver_id,
                destination=destination,
                distance_result=result,
                stop_count=len(route),
                departure_hour=current_time.hour,
                prediction_date=prediction_date,
            )
            current_time += timedelta(minutes=predicted_leg)
            travel_minutes += predicted_leg
            total_distance += result.distance_km
            google_sources.add(result.source)
            legs.append(
                {
                    "from": origin["location_code"],
                    "to": destination["location_code"],
                    "distance_km": round(result.distance_km, 2),
                    "predicted_minutes": round(predicted_leg, 1),
                    "traffic_minutes": round(result.duration_in_traffic_min, 1),
                    "source": result.source,
                }
            )
        visit_minutes = sum(float(row.get("avg_visit_duration_min", row.get("base_visit_duration_min", 20.0))) for row in route)
        predicted_minutes = travel_minutes + visit_minutes
        confidence = self._confidence(driver_id, route, matrix)
        route_score = self._route_score(predicted_minutes, len(route), confidence)
        return RoutePlan(
            ordered_locations=route,
            total_distance_km=round(total_distance, 2),
            travel_minutes=round(travel_minutes, 1),
            visit_minutes=round(visit_minutes, 1),
            predicted_minutes=round(predicted_minutes, 1),
            confidence=confidence,
            google_source="+".join(sorted(google_sources)) if google_sources else "not_required",
            legs=legs,
            route_score=route_score,
        )

    def _confidence(
        self,
        driver_id: str,
        route: list[dict[str, Any]],
        matrix: dict[tuple[str, str], DistanceResult],
    ) -> float:
        metrics = self.model.metrics
        model_quality = max(0.45, min(0.96, 0.55 + float(metrics.get("r2", 0.5)) * 0.4))
        driver_known = 1.0 if driver_id in self.model.driver_profiles else 0.72
        location_support = sum(min(float(row.get("historical_visits", 0.0)) / 25.0, 1.0) for row in route) / len(route)
        transition_scores = self.model.route_patterns.get("transition_scores", {})
        if len(route) > 1:
            transition_support = sum(
                float(transition_scores.get(f"{origin['location_id']}->{destination['location_id']}", 0.0))
                for origin, destination in zip(route, route[1:])
            ) / (len(route) - 1)
        else:
            transition_support = 0.8
        source_bonus = 0.92 if any(result.source.startswith("google") for result in matrix.values()) else 0.78
        confidence = (
            model_quality * 0.32
            + driver_known * 0.18
            + location_support * 0.22
            + min(1.0, transition_support * 2.8) * 0.12
            + source_bonus * 0.16
        )
        return round(max(0.45, min(0.97, confidence)), 2)

    @staticmethod
    def _route_score(predicted_minutes: float, stop_count: int, confidence: float) -> float:
        minutes_per_stop = predicted_minutes / max(stop_count, 1)
        efficiency = max(0.35, min(1.0, 55.0 / max(minutes_per_stop, 20.0)))
        return round((efficiency * 0.62 + confidence * 0.38), 2)

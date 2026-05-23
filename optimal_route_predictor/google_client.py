"""Google Maps and Places integration with SQLite caching and offline fallback."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import GOOGLE_CACHE_PATH, GOOGLE_CACHE_TTL_SECONDS, GOOGLE_MAPS_API_KEY
from .geo import driving_minutes, haversine_km, traffic_multiplier


@dataclass(frozen=True)
class DistanceResult:
    origin_id: str
    destination_id: str
    distance_km: float
    duration_min: float
    duration_in_traffic_min: float
    source: str


class SQLiteCache:
    def __init__(self, path: Path = GOOGLE_CACHE_PATH, ttl_seconds: int = GOOGLE_CACHE_TTL_SECONDS) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS google_api_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def get(self, key: str) -> Any | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT payload, created_at FROM google_api_cache WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        payload, created_at = row
        if time.time() - float(created_at) > self.ttl_seconds:
            return None
        return json.loads(payload)

    def set(self, key: str, payload: Any) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "REPLACE INTO google_api_cache(cache_key, payload, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(payload), time.time()),
            )


class GoogleMapsClient:
    """Thin client for required Google APIs.

    When GOOGLE_MAPS_API_KEY is not set, methods return deterministic local
    estimates. This keeps development and tests runnable while preserving the
    production integration points.
    """

    def __init__(
        self,
        api_key: str = GOOGLE_MAPS_API_KEY,
        locations: pd.DataFrame | None = None,
        cache: SQLiteCache | None = None,
    ) -> None:
        self.api_key = api_key
        self.locations = locations
        self.cache = cache or SQLiteCache()

    @property
    def live_enabled(self) -> bool:
        return bool(self.api_key)

    def _cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        stable = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "key": self.api_key}
        cache_key = self._cache_key(endpoint, params)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        url = endpoint + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.cache.set(cache_key, payload)
        return payload

    def distance_matrix(
        self,
        origins: list[dict[str, Any]],
        destinations: list[dict[str, Any]],
        departure_time: datetime | None = None,
    ) -> dict[tuple[str, str], DistanceResult]:
        if self.live_enabled:
            return self._google_distance_matrix(origins, destinations, departure_time)
        return self._offline_distance_matrix(origins, destinations, departure_time)

    def _google_distance_matrix(
        self,
        origins: list[dict[str, Any]],
        destinations: list[dict[str, Any]],
        departure_time: datetime | None,
    ) -> dict[tuple[str, str], DistanceResult]:
        origin_text = "|".join(f"{row['latitude']},{row['longitude']}" for row in origins)
        destination_text = "|".join(f"{row['latitude']},{row['longitude']}" for row in destinations)
        params = {
            "origins": origin_text,
            "destinations": destination_text,
            "mode": "driving",
            "departure_time": "now" if departure_time is None else int(departure_time.timestamp()),
            "traffic_model": "best_guess",
            "units": "metric",
        }
        payload = self._get_json("https://maps.googleapis.com/maps/api/distancematrix/json", params)
        if payload.get("status") != "OK":
            return self._offline_distance_matrix(origins, destinations, departure_time)
        results: dict[tuple[str, str], DistanceResult] = {}
        for origin_index, row in enumerate(payload.get("rows", [])):
            origin_id = str(origins[origin_index]["location_id"])
            for destination_index, element in enumerate(row.get("elements", [])):
                destination_id = str(destinations[destination_index]["location_id"])
                if element.get("status") != "OK":
                    continue
                distance_km = float(element["distance"]["value"]) / 1000.0
                duration_min = float(element["duration"]["value"]) / 60.0
                traffic_min = float(element.get("duration_in_traffic", element["duration"])["value"]) / 60.0
                results[(origin_id, destination_id)] = DistanceResult(
                    origin_id=origin_id,
                    destination_id=destination_id,
                    distance_km=round(distance_km, 3),
                    duration_min=round(duration_min, 2),
                    duration_in_traffic_min=round(traffic_min, 2),
                    source="google_distance_matrix",
                )
        if len(results) < len(origins) * len(destinations):
            offline_results = self._offline_distance_matrix(origins, destinations, departure_time)
            for key, value in offline_results.items():
                results.setdefault(key, value)
        return results

    def _offline_distance_matrix(
        self,
        origins: list[dict[str, Any]],
        destinations: list[dict[str, Any]],
        departure_time: datetime | None,
    ) -> dict[tuple[str, str], DistanceResult]:
        when = departure_time or datetime.now()
        results: dict[tuple[str, str], DistanceResult] = {}
        for origin in origins:
            for destination in destinations:
                origin_id = str(origin["location_id"])
                destination_id = str(destination["location_id"])
                if origin_id == destination_id:
                    distance_km = 0.0
                    duration_min = 0.0
                else:
                    raw_distance = haversine_km(
                        float(origin["latitude"]),
                        float(origin["longitude"]),
                        float(destination["latitude"]),
                        float(destination["longitude"]),
                    )
                    distance_km = raw_distance * 1.32
                    traffic = traffic_multiplier(when.hour, when.date(), str(destination.get("traffic_category", "medium")))
                    duration_min = driving_minutes(distance_km, 30.0, traffic)
                results[(origin_id, destination_id)] = DistanceResult(
                    origin_id=origin_id,
                    destination_id=destination_id,
                    distance_km=round(distance_km, 3),
                    duration_min=round(duration_min, 2),
                    duration_in_traffic_min=round(duration_min, 2),
                    source="offline_haversine_traffic",
                )
        return results

    def nearby_places(self, latitude: float, longitude: float, radius_m: int = 1500, place_type: str = "store") -> list[dict[str, Any]]:
        if self.live_enabled:
            params = {
                "location": f"{latitude},{longitude}",
                "radius": radius_m,
                "type": place_type,
            }
            payload = self._get_json("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params)
            if payload.get("status") == "OK":
                return payload.get("results", [])
        return self._offline_nearby_places(latitude, longitude, radius_m)

    def place_details(self, place_id: str) -> dict[str, Any]:
        if self.live_enabled:
            params = {
                "place_id": place_id,
                "fields": "place_id,name,geometry,formatted_address,rating,user_ratings_total,types",
            }
            payload = self._get_json("https://maps.googleapis.com/maps/api/place/details/json", params)
            if payload.get("status") == "OK":
                return payload.get("result", {})
        if self.locations is not None:
            match = self.locations[self.locations["place_id"].astype(str) == str(place_id)]
            if not match.empty:
                row = match.iloc[0].to_dict()
                return {
                    "place_id": row["place_id"],
                    "name": row["location_name"],
                    "geometry": {"location": {"lat": row["latitude"], "lng": row["longitude"]}},
                    "types": ["store", "synthetic"],
                    "source": "offline_synthetic_places",
                }
        return {"place_id": place_id, "source": "offline_unknown"}

    def geocode(self, address: str) -> dict[str, Any]:
        if self.live_enabled:
            payload = self._get_json("https://maps.googleapis.com/maps/api/geocode/json", {"address": address})
            if payload.get("status") == "OK" and payload.get("results"):
                return payload["results"][0]
        if self.locations is not None:
            normalized = address.strip().lower()
            candidates = self.locations[
                self.locations["location_name"].astype(str).str.lower().eq(normalized)
                | self.locations["location_code"].astype(str).str.lower().eq(normalized)
                | self.locations["location_id"].astype(str).str.lower().eq(normalized)
            ]
            if not candidates.empty:
                row = candidates.iloc[0].to_dict()
                return {
                    "formatted_address": row["location_name"],
                    "geometry": {"location": {"lat": row["latitude"], "lng": row["longitude"]}},
                    "place_id": row["place_id"],
                    "source": "offline_synthetic_geocode",
                }
        return {"formatted_address": address, "source": "offline_unresolved"}

    def _offline_nearby_places(self, latitude: float, longitude: float, radius_m: int) -> list[dict[str, Any]]:
        if self.locations is None:
            return []
        rows: list[dict[str, Any]] = []
        for location in self.locations.to_dict("records"):
            distance_m = haversine_km(latitude, longitude, float(location["latitude"]), float(location["longitude"])) * 1000
            if distance_m <= radius_m:
                rows.append(
                    {
                        "place_id": location["place_id"],
                        "name": location["location_name"],
                        "vicinity": location["region"],
                        "distance_m": round(distance_m, 1),
                        "geometry": {"location": {"lat": location["latitude"], "lng": location["longitude"]}},
                        "source": "offline_synthetic_places",
                    }
                )
        return sorted(rows, key=lambda row: row["distance_m"])[:20]

"""Geospatial helpers and deterministic traffic estimates."""

from __future__ import annotations

import math
from datetime import date

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometers."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def traffic_multiplier(hour: int, day: date, traffic_category: str) -> float:
    """Estimate traffic pressure when live Google traffic is unavailable."""

    category_base = {
        "low": 0.96,
        "medium": 1.08,
        "high": 1.22,
        "severe": 1.42,
    }.get(str(traffic_category).lower(), 1.08)
    rush = 1.0
    if 8 <= hour <= 10:
        rush += 0.22
    if 17 <= hour <= 20:
        rush += 0.28
    if day.weekday() >= 5:
        rush -= 0.08
    return max(0.85, category_base * rush)


def driving_minutes(distance_km: float, avg_speed_kmph: float, multiplier: float = 1.0) -> float:
    """Convert distance to plausible urban driving minutes."""

    speed = max(avg_speed_kmph, 8.0)
    return max(2.0, (distance_km / speed) * 60.0 * multiplier)


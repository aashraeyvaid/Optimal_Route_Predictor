"""API request and response schemas."""

from __future__ import annotations

from datetime import date as date_type
from typing import Any

from pydantic import BaseModel, Field


class DailyRouteRequest(BaseModel):
    driver_id: str = Field(..., examples=["D01"])
    date: date_type = Field(..., examples=["2026-05-20"])
    locations: list[str] = Field(..., min_length=1, examples=[["A", "B", "C", "D"]])


class WeeklyRouteRequest(BaseModel):
    driver_id: str = Field(..., examples=["D01"])
    week: str = Field(..., examples=["2026-W20"])


class RetrainRequest(BaseModel):
    regenerate_data: bool = False


class RerouteRequest(BaseModel):
    driver_id: str = Field(..., examples=["D01"])
    date: date_type = Field(..., examples=["2026-05-20"])
    current_location: str = Field(..., examples=["A"])
    remaining_locations: list[str] = Field(..., min_length=1, examples=[["B", "C", "D"]])


class NearbyPlacesRequest(BaseModel):
    latitude: float
    longitude: float
    radius_m: int = 1500


class GeocodeRequest(BaseModel):
    address: str = Field(..., examples=["Store A"])


class ApiResponse(BaseModel):
    data: dict[str, Any] | list[dict[str, Any]]

"""FastAPI entrypoint for the Optimal Route Predictor."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import DailyRouteRequest, GeocodeRequest, NearbyPlacesRequest, RerouteRequest, RetrainRequest, WeeklyRouteRequest
from optimal_route_predictor import __version__
from optimal_route_predictor.service import RoutePredictionService

ROOT_DIR = Path(__file__).resolve().parents[1]
UI_DIR = ROOT_DIR / "ui"

app = FastAPI(
    title="AI Optimal Route Predictor",
    version=__version__,
    description="AI-based daily and weekly route sequence prediction with Google Maps/Places integration.",
)

if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


@lru_cache(maxsize=1)
def get_service() -> RoutePredictionService:
    return RoutePredictionService()


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return get_service().health()


@app.get("/drivers")
def drivers() -> list[dict]:
    return get_service().list_drivers()


@app.get("/locations")
def locations() -> list[dict]:
    return get_service().list_locations()


@app.post("/predict/daily")
def predict_daily(request: DailyRouteRequest) -> dict:
    try:
        return get_service().predict_daily(request.driver_id, request.date, request.locations)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/weekly")
def predict_weekly(request: WeeklyRouteRequest) -> dict:
    try:
        return get_service().predict_weekly(request.driver_id, request.week)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrain")
def retrain(request: RetrainRequest | None = None) -> dict:
    payload = request or RetrainRequest()
    get_service.cache_clear()
    service = get_service()
    result = service.retrain(regenerate_data=payload.regenerate_data)
    get_service.cache_clear()
    return result


@app.post("/predict/reroute")
def reroute(request: RerouteRequest) -> dict:
    try:
        return get_service().reroute(
            request.driver_id,
            request.date,
            request.current_location,
            request.remaining_locations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/places/nearby")
def nearby_places(request: NearbyPlacesRequest) -> list[dict]:
    return get_service().nearby_places(request.latitude, request.longitude, request.radius_m)


@app.get("/places/details/{place_id}")
def place_details(place_id: str) -> dict:
    return get_service().place_details(place_id)


@app.post("/geocode")
def geocode(request: GeocodeRequest) -> dict:
    return get_service().geocode(request.address)


@app.get("/monitoring/model")
def model_monitoring() -> dict:
    return get_service().monitoring()

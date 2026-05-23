# AI Optimal Route Predictor

An AI-based route prediction system for field sales drivers. It learns from historical trip data, uses Google Maps/Places signals when an API key is available, and exposes daily and weekly route recommendations through REST APIs.

## What Is Included

- Synthetic historical dataset with **4,001 trip records**, **12 drivers**, and **72 locations**.
- Feature engineering for time, weekly patterns, route distance, stop count, driver speed, route efficiency, location density, region, and traffic category.
- Lightweight ML model: NumPy ridge-regression ETA prediction combined with learned route transition patterns.
- Graph route optimizer using nearest-neighbor initialization plus 2-opt improvement.
- Google Maps Platform integration for Distance Matrix, traffic-aware duration, Nearby Places, Place Details, and Geocoding.
- SQLite caching for Google API responses.
- Offline deterministic fallback when `GOOGLE_MAPS_API_KEY` is not configured, so development and tests still run.
- FastAPI endpoints for daily prediction, weekly prediction, retraining, health checks, rerouting, nearby places, and model monitoring.
- Dockerfile and unit tests.

## Project Structure

```text
project/
├── data/
│   ├── drivers.csv
│   ├── locations.csv
│   └── trips.csv
├── notebooks/
├── model/
│   └── route_model.json
├── api/
│   ├── main.py
│   └── schemas.py
├── optimal_route_predictor/
│   ├── data_generator.py
│   ├── features.py
│   ├── google_client.py
│   ├── modeling.py
│   ├── optimizer.py
│   └── service.py
├── scripts/
│   ├── generate_data.py
│   ├── train_model.py
│   └── smoke_test.py
├── tests/
├── README.md
├── requirements.txt
└── Dockerfile
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train_model.py
uvicorn api.main:app --reload
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

If you want live Google traffic and places data, set:

```powershell
$env:GOOGLE_MAPS_API_KEY="your-google-maps-platform-key"
```

Without a key, the app uses the synthetic locations plus a haversine/traffic fallback. This keeps the assignment fully runnable while preserving real Google API integration for production.

## API Endpoints

### Health Check

```http
GET /health
```

### Predict Daily Route

```http
POST /predict/daily
Content-Type: application/json

{
  "driver_id": "D01",
  "date": "2026-05-20",
  "locations": ["A", "B", "C", "D"]
}
```

Example response:

```json
{
  "driver_id": "D01",
  "date": "2026-05-20",
  "recommended_route": ["A", "C", "B", "D"],
  "predicted_time": "3.26 hours",
  "predicted_minutes": 195.7,
  "total_distance_km": 31.09,
  "confidence": 0.82,
  "route_score": 0.93
}
```

### Predict Weekly Route

```http
POST /predict/weekly
Content-Type: application/json

{
  "driver_id": "D01",
  "week": "2026-W20"
}
```

### Retrain Model

```http
POST /retrain
Content-Type: application/json

{
  "regenerate_data": false
}
```

### Bonus Endpoints

- `POST /predict/reroute` for traffic-aware dynamic rerouting.
- `POST /places/nearby` for Google Places nearby search or offline synthetic nearby stores.
- `GET /places/details/{place_id}` for Google Place Details or synthetic place metadata.
- `POST /geocode` for Google Geocoding or synthetic store lookup.
- `GET /monitoring/model` for model metrics and data-health summary.

## Model Selection

The assignment allows Random Forest, XGBoost, K-Means plus heuristics, graph optimization, sequence models, and other approaches. This implementation uses a hybrid that is lightweight, explainable, and production-friendly:

- **ETA model:** ridge regression trained on engineered route, driver, location, and time features.
- **Sequence learning:** historical transition frequencies identify route patterns such as which stores are commonly visited after each other.
- **Optimization:** graph search creates a route order, then 2-opt improves travel cost.
- **External signals:** Google Distance Matrix duration-in-traffic is blended with the ML ETA model when an API key is configured.

The trained sample model currently reports:

```text
R2: 0.9314
MAE: 2.243 minutes
Training records: 4001
```

## Google API Usage

The integration is implemented in `optimal_route_predictor/google_client.py`.

- Distance Matrix API: travel distance, travel duration, and duration in traffic.
- Places Nearby Search: nearby store/customer discovery.
- Place Details: place metadata enrichment.
- Geocoding: address or store lookup when required.
- SQLite cache: avoids duplicate API calls and supports faster repeated predictions.

## Testing

The tests use Python's built-in `unittest`, so they work even before installing `pytest`.

```powershell
python -m unittest discover -s tests
```

Expected result:

```text
Ran 8 tests
OK
```

## Docker

```powershell
docker build -t optimal-route-predictor .
docker run -p 8000:8000 --env GOOGLE_MAPS_API_KEY="your-key" optimal-route-predictor
```

The key is optional for local testing.

## Scalability Notes

- Replace synthetic CSV files with PostgreSQL tables for production trip, driver, and customer data.
- Move Google API cache from SQLite to Redis for multi-instance deployments.
- Add async job processing for retraining and high-volume weekly planning.
- Track model drift using route completion time, missed visits, fuel usage, and driver override rate.
- Introduce OR-Tools or a vehicle-routing solver when constraints expand to time windows, capacity, multi-depot routing, and multi-driver balancing.

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.main import app


class RouteApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_daily_endpoint(self) -> None:
        response = self.client.post(
            "/predict/daily",
            json={"driver_id": "D01", "date": "2026-05-20", "locations": ["A", "B", "C", "D"]},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload["recommended_route"]), {"A", "B", "C", "D"})
        self.assertGreater(payload["confidence"], 0)

    def test_weekly_endpoint(self) -> None:
        response = self.client.post("/predict/weekly", json={"driver_id": "D01", "week": "2026-W20"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["week"], "2026-W20")
        self.assertTrue(payload["monday"])

    def test_places_and_geocoding_endpoints(self) -> None:
        nearby = self.client.post("/places/nearby", json={"latitude": 23.0225, "longitude": 72.5714, "radius_m": 5000})
        self.assertEqual(nearby.status_code, 200)
        self.assertIsInstance(nearby.json(), list)

        geocode = self.client.post("/geocode", json={"address": "Store A"})
        self.assertEqual(geocode.status_code, 200)
        self.assertIn("geometry", geocode.json())


if __name__ == "__main__":
    unittest.main()

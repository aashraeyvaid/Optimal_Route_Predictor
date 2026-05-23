from __future__ import annotations

import unittest
from datetime import date

from optimal_route_predictor.data_generator import load_dataset
from optimal_route_predictor.service import RoutePredictionService


class RoutePredictionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = RoutePredictionService()

    def test_dataset_meets_assignment_minimums(self) -> None:
        trips, locations, drivers = load_dataset()
        self.assertGreaterEqual(len(trips), 1000)
        self.assertGreaterEqual(drivers["driver_id"].nunique(), 10)
        self.assertGreaterEqual(locations["location_id"].nunique(), 50)

    def test_daily_prediction_returns_same_stops_in_optimized_order(self) -> None:
        prediction = self.service.predict_daily("D01", date(2026, 5, 20), ["A", "B", "C", "D"])
        self.assertEqual(set(prediction["recommended_route"]), {"A", "B", "C", "D"})
        self.assertGreater(prediction["predicted_minutes"], 0)
        self.assertGreater(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)
        self.assertGreater(prediction["route_score"], 0)

    def test_weekly_prediction_contains_workweek_routes(self) -> None:
        prediction = self.service.predict_weekly("D01", "2026-W20")
        self.assertEqual(prediction["week"], "2026-W20")
        self.assertTrue(prediction["monday"])
        self.assertTrue(prediction["friday"])
        self.assertTrue(prediction["weekly_distance"].endswith("km"))

    def test_reroute_keeps_remaining_stops(self) -> None:
        prediction = self.service.reroute("D01", date(2026, 5, 20), "A", ["B", "C"])
        self.assertEqual(set(prediction["recommended_route"]), {"A", "B", "C"})
        self.assertTrue(prediction["reroute_reason"])


if __name__ == "__main__":
    unittest.main()


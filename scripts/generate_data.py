"""Generate the assignment sample dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimal_route_predictor.data_generator import generate_dataset


if __name__ == "__main__":
    summary = generate_dataset(force=True)
    print(
        f"Generated {summary.trips} trip records, "
        f"{summary.drivers} drivers, {summary.locations} locations in {summary.output_dir}"
    )

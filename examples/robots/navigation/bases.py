#!/usr/bin/env python3
"""Three Clearpath bases cross a narrow office, the Python version of bases.cpp.

The script loads the office map as a distance grid, plans a Jackal from the lower
aisle into a gap between two desks, measures how much of its travel is sideways,
then plans the same query for every base.

Usage:
  python examples/robots/navigation/bases.py [--json out.json]
"""

import argparse
import json

# [docs-start:map]
from pathlib import Path

import numpy as np

import geodex
import geodex.collision as gc

HERE = Path(__file__).resolve().parent


def office_grid():
    """DistanceGrid of the office map, the signed distance in meters to the nearest
    occupied or unknown cell at the center of every 0.05 m cell."""
    grid = gc.DistanceGrid()
    if not grid.load(str(HERE / "office_dist.txt")):
        raise FileNotFoundError(HERE / "office_dist.txt")
    return grid
# [docs-end:map]


# [docs-start:try-it]
# Footprint length and width in meters and drive, from Clearpath's specifications, and the
# planner's range under the base's metric.
PLATFORMS = {
    "jackal": (0.508, 0.430, "skid_steer", 6.5),
    "dingo_d": (0.551, 0.517, "differential", 12.0),
    "dingo_o": (0.686, 0.517, "holonomic", 6.5),
}

# Weights (w_x, w_y, w_theta) of the base metric for each drive.
DRIVES = {
    "differential": (1.0, 50.0, 1.0),
    "skid_steer": (1.0, 50.0, 2.0),
    "holonomic": (1.0, 1.0, 1.0),
}
# [docs-end:try-it]


# [docs-start:plan]
def plan_base(grid, length, width, weights, planner_range, seed=1):
    """Plan a rectangular base through the office with weights (wx, wy, wtheta). The planner
    adds tree edges of at most planner_range under the metric."""
    wx, wy, wtheta = weights
    x_hi = (grid.width() - 1) * grid.resolution()
    y_hi = (grid.height() - 1) * grid.resolution()
    se2 = geodex.SE2(wx=wx, wy=wy, wtheta=wtheta, x_lo=0.0, x_hi=x_hi, y_lo=0.0,
                     y_hi=y_hi)
    footprint = gc.PolygonFootprint.rectangle(length / 2, width / 2, 6)
    checker = gc.FootprintGridChecker(grid, footprint, 0.05)  # 5 cm safety margin
    # The clearance metric scales the base metric with kappa = 1.5 and beta = 3.
    metric = geodex.ClearanceMetric(geodex.SE2LeftInvariantMetric(wx, wy, wtheta),
                                    checker, 1.5, 3.0)
    space = geodex.ConfigurationSpace(se2, metric)
    settings = geodex.PlanSettings(iterations=1000, seed=seed,
                                   planner=geodex.planners.GreedyRRTstar(range=planner_range),
                                   collision_check_resolution=0.05,
                                   interp="base_geodesic")
    # (x, y, heading) of the start and the goal, in meters and radians.
    start = np.array([3.885, 1.235, np.radians(-2.81)])
    goal = np.array([12.425, 3.675, np.radians(-120.44)])
    return geodex.plan(space, start, goal, checker.is_valid, settings=settings,
                       heuristic=se2.matrix_lower_bound())
# [docs-end:plan]


# [docs-start:sideways]
def sideways_share(path):
    """Share of the base's travel that is sideways in its own frame.

    Each edge moves the base along one constant body twist (v_x, v_y, omega). The
    twist is the SE(2) logarithm between the edge's waypoints."""
    se2 = geodex.SE2()
    sideways = travel = 0.0
    for a, b in zip(path[:-1], path[1:]):
        vx, vy, _ = se2.log(a, b)
        sideways += abs(vy)
        travel += np.sqrt(vx * vx + vy * vy)
    return sideways / travel
# [docs-end:sideways]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    # [docs-start:plan]
    grid = office_grid()
    result = plan_base(grid, 0.508, 0.430, (1.0, 50.0, 2.0), 6.5)  # a Jackal
    print(f"solved={result.solved} cost={result.cost:.3f} waypoints={len(result.path)}")
    # [docs-end:plan]

    # [docs-start:sideways]
    print(f"sideways share={sideways_share(result.path):.3f}")
    # [docs-end:sideways]

    out = {}
    # [docs-start:try-it]
    for name, (length, width, drive, planner_range) in PLATFORMS.items():
        result = plan_base(grid, length, width, DRIVES[drive], planner_range)
        share = sideways_share(result.path)
        print(f"{name:>10}: solved={result.solved} cost={result.cost:.2f} "
              f"sideways share={share:.3f}")
        # [docs-end:try-it]
        out[name] = {"solved": bool(result.solved), "cost": float(result.cost),
                     "sideways_share": float(share), "path": result.path.tolist(),
                     "raw_path": result.raw_path.tolist(), "drive": drive,
                     "footprint": [length, width]}

    if args.json:
        args.json.write_text(json.dumps(out))
    return 0 if all(r["solved"] for r in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Examples of the Planning concept page, the Python version of planning.cpp.

The script makes two plans. The first passes between two caps on the sphere, and the second
sets the fields of PlanSettings.

Usage:
  python examples/concepts/planning.py [--json out.json]
"""

import json
import sys

# [docs-start:first-plan]
import numpy as np

import geodex
# [docs-end:first-plan]


def main():
    out = {}

    # [docs-start:first-plan]
    sphere = geodex.Sphere()

    def point(lon, lat):
        """The point of the unit sphere at a longitude and latitude in degrees."""
        lon, lat = np.radians(lon), np.radians(lat)
        return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    # Two caps are obstacles, each a center and an angular radius.
    caps = [(point(-4, 6), np.radians(18)), (point(14, 36), np.radians(12))]

    def is_valid(q):
        return all(q @ center < np.cos(radius) for center, radius in caps)

    start, goal = point(-40, 8), point(40, 22)

    settings = geodex.PlanSettings(iterations=2000, seed=7)
    result = geodex.plan(sphere, start, goal, is_valid, settings=settings)

    print(f"solved={result.solved} cost={result.cost:.4f} waypoints={len(result.path)}")
    # [docs-end:first-plan]
    out["sphere"] = {
        "solved": bool(result.solved),
        "cost": float(result.cost),
        "raw_path": result.raw_path.tolist(),
        "path": result.path.tolist(),
        "smoothed": bool(result.smoothed),
        "caps": [[center.tolist(), float(radius)] for center, radius in caps],
    }

    # [docs-start:result]
    raw = result.raw_path    # planner waypoints, shape (n, 3)
    path = result.path       # smoothed, checked and evenly spaced waypoints, shape (m, 3)
    print(f"raw waypoints={len(raw)} smoothed={result.smoothed} "
          f"solve={result.time_ms:.1f} ms smooth={result.smooth_ms:.1f} ms")
    # [docs-end:result]
    out["sphere"]["raw_count"] = len(raw)

    # [docs-start:settings]
    settings = geodex.PlanSettings(
        iterations=2000,           # fixed budget, reproducible with a seed
        seed=7,                    # seeds OMPL and the geodex samplers
        time=1.0,                  # wall-clock budget, used when iterations is 0
        planner=geodex.planners.GreedyRRTstar(range=0.0, greedy_ratio=0.9,
                                              rewire_factor=1.1),
        collision_check_resolution=0.01,  # spacing of the validity samples along an edge
        interp="base_geodesic",    # the curve of the planner's edges
        goal_tolerance=0.0,
        smooth=True,               # run the smoother on the planner's path
        smoothing=geodex.PathSmoothingSettings(output_spacing=0.0),
    )
    result = geodex.plan(sphere, start, goal, is_valid, settings=settings)
    # [docs-end:settings]
    out["settings"] = {"solved": bool(result.solved), "cost": float(result.cost),
                       "path": result.path.tolist()}
    return out


if __name__ == "__main__":
    results = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(results, f)

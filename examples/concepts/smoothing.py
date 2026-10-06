#!/usr/bin/env python3
"""Examples of the Path Smoothing concept page, the Python version of smoothing.cpp.

The script runs the smoother on a hand-made path in the plane, then with an edge proof,
then inside plan() for a differential-drive robot on SE(2).

Usage:
  python examples/concepts/smoothing.py [--json out.json]
"""

import json
import sys

# [docs-start:standalone]
import numpy as np

import geodex

# Two discs (center x, center y, radius) in the plane.
DISCS = [(1.0, 0.2, 0.5), (2.5, -0.3, 0.6)]


def is_valid(q):
    """True when q lies outside both discs."""
    return all(np.hypot(q[0] - cx, q[1] - cy) > r for cx, cy, r in DISCS)
# [docs-end:standalone]


# [docs-start:proof]
def segment_clear(a, b):
    """True when the straight segment from a to b stays outside both discs."""
    for cx, cy, r in DISCS:
        d = b - a
        t = ((cx - a[0]) * d[0] + (cy - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])
        t = np.clip(t, 0.0, 1.0)
        if np.hypot(a[0] + t * d[0] - cx, a[1] + t * d[1] - cy) <= r:
            return False
    return True
# [docs-end:proof]


def summary(result):
    return {"collision_free": bool(result.collision_free), "length": float(result.length),
            "path": result.path.tolist(), "fallback": int(result.profile.fallback)}


def main():
    out = {}

    # [docs-start:standalone]
    plane = geodex.Euclidean(2)

    # A feasible but suboptimal path, as a planner might return it.
    path = [np.array(p) for p in
            [(0.0, 0.0), (0.2, 1.0), (1.8, 1.0), (1.8, -1.2), (3.4, -1.1), (4.0, 0.0)]]

    settings = geodex.PathSmoothingSettings(collision_check_resolution=0.01)
    result = geodex.smooth_path(plane, is_valid, path, settings)

    print(f"collision_free={result.collision_free} length={result.length:.4f} "
          f"waypoints={len(result.path)}")
    # [docs-end:standalone]
    out["standalone"] = summary(result)

    # [docs-start:profile]
    p = result.profile
    print(f"{p.input_waypoints} -> {p.output_waypoints} waypoints, "
          f"{p.shortcuts} shortcuts, {p.relax_moves} waypoint moves, "
          f"{p.point_checks} validity checks, fallback stage {p.fallback}")
    # [docs-end:profile]
    out["profile"] = {"input": p.input_waypoints, "output": p.output_waypoints}

    # [docs-start:corners]
    # The same path without corner rounding keeps its corners.
    sharp = geodex.smooth_path(plane, is_valid, path,
                               geodex.PathSmoothingSettings(collision_check_resolution=0.01,
                                                            round_corners=False))
    print(f"rounded={p.rounded_corners} kept={p.kept_corners} cusps={p.cusps} "
          f"points={len(result.path)}, without rounding {len(sharp.path)}")
    # [docs-end:corners]
    out["corners"] = {"rounded": p.rounded_corners, "kept": p.kept_corners, "sharp": summary(sharp)}

    # [docs-start:proof]
    # A sound proof that a whole edge is clear lets the smoother skip its samples.
    settings = geodex.PathSmoothingSettings(collision_check_resolution=0.01,
                                            edge_provably_clear=segment_clear)
    proved = geodex.smooth_path(plane, is_valid, path, settings)
    print(f"same path={np.array_equal(proved.path, result.path)} "
          f"edges settled by the proof={proved.profile.edge_proofs}")
    # [docs-end:proof]
    out["proof"] = summary(proved)

    # [docs-start:spacing]
    # A looser corner tolerance allows a longer step, and output_spacing limits the step.
    spaced = []
    for tolerance, limit in [(1e-3, 0.0), (1e-4, 0.01)]:
        settings = geodex.PathSmoothingSettings(collision_check_resolution=0.01,
                                                corner_tolerance=tolerance,
                                                output_spacing=limit)
        even = geodex.smooth_path(plane, is_valid, path, settings)
        step = max(plane.distance(a, b) for a, b in zip(even.path[:-1], even.path[1:]))
        print(f"corner_tolerance={tolerance:g} output_spacing={limit:g}: "
              f"waypoints={len(even.path)} step={step:.4f}")
        spaced.append(even)
    # [docs-end:spacing]
    out["spacing"] = {"coarse": summary(spaced[0]), "limited": summary(spaced[1])}

    # [docs-start:in-plan]
    # A differential-drive robot on SE(2). The weight 10 on the squared sideways speed
    # makes a meter of sliding cost as much as sqrt(10), about 3.2, meters of driving.
    se2 = geodex.SE2(wx=1.0, wy=10.0, wtheta=1.0,
                     x_lo=-1.0, x_hi=5.0, y_lo=-2.0, y_hi=2.0)

    def pose_is_valid(q):
        return is_valid(q[:2])

    plan_settings = geodex.PlanSettings(
        iterations=3000, seed=5, collision_check_resolution=0.01,
        smoothing=geodex.PathSmoothingSettings(output_spacing=0.2),
    )
    start, goal = np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])
    planned = geodex.plan(se2, start, goal, pose_is_valid, settings=plan_settings)

    def length(waypoints):
        return sum(se2.distance(a, b) for a, b in zip(waypoints[:-1], waypoints[1:]))

    print(f"smoothed={planned.smoothed} raw length={length(planned.raw_path):.3f} "
          f"smoothed length={length(planned.path):.3f}")
    # [docs-end:in-plan]
    out["in_plan"] = {"solved": bool(planned.solved), "smoothed": bool(planned.smoothed),
                      "raw_path": planned.raw_path.tolist(), "path": planned.path.tolist(),
                      "raw_length": float(length(planned.raw_path)),
                      "length": float(length(planned.path))}
    return out


if __name__ == "__main__":
    results = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(results, f)

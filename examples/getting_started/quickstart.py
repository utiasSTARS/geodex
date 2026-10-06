#!/usr/bin/env python3
"""Quickstart example, the Python version of quickstart.cpp.

A Franka Panda swings around a post.

Usage:
  python examples/getting_started/quickstart.py [--json out.json]
"""

import json
import sys

# [docs-start:load]
import numpy as np

import geodex
# [docs-end:load]


def main():
    # [docs-start:load]
    robot = geodex.robots.Panda()  # seven joints, the kinetic-energy metric
    print(robot.name(), "joints:", robot.dim())
    # [docs-end:load]

    # [docs-start:scene]
    scene = geodex.Scene()
    scene.add_box(position=[0.4, 0.0, 0.3], size=[0.1, 0.1, 0.8])  # a post in front
    # [docs-end:scene]

    # [docs-start:plan]
    start = np.array([-1.1, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    goal = np.array([1.1, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    settings = geodex.PlanSettings(iterations=1500, seed=1)
    result = geodex.plan(robot, start, goal, scene, settings=settings)
    print(f"solved={result.solved} cost={result.cost:.4f} waypoints={len(result.path)}")
    # [docs-end:plan]

    # [docs-start:result]
    rows, cols = result.path.shape  # one row per waypoint, one column per joint
    raw_rows, raw_cols = result.raw_path.shape
    print(f"path: {rows} x {cols}, raw path: {raw_rows} x {raw_cols}")
    # [docs-end:result]

    # [docs-start:try-it]
    euclidean = geodex.robots.Panda(metric="euclidean")
    other = geodex.plan(euclidean, start, goal, scene, settings=settings)
    for name, r in (("kinetic energy", result), ("euclidean", other)):
        turned = np.linalg.norm(np.diff(r.path, axis=0), axis=1).sum()
        print(f"{name}: the joints turn {turned:.3f} rad in total")
    # [docs-end:try-it]

    return {"solved": bool(result.solved), "cost": float(result.cost),
            "raw_path": result.raw_path.tolist(), "path": result.path.tolist(),
            "euclidean": {"solved": bool(other.solved), "cost": float(other.cost),
                          "path": other.path.tolist()}}


if __name__ == "__main__":
    out = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f)
    sys.exit(0 if out["solved"] and out["euclidean"]["solved"] else 1)

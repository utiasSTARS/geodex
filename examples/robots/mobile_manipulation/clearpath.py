#!/usr/bin/env python3
"""A UR5e on a Husky and on a Ridgeback, the Python version of clearpath.cpp.

Each robot drives from the shelf of a workcell to a low table and reaches over it with the
tool pointing down. Base and arm move in one motion on SE(2) x R^6. The script plans the
Husky on its skid-steer base, measures how much its base slides, then plans the Ridgeback
on its mecanum base.

Usage:
  python examples/robots/mobile_manipulation/clearpath.py [--json out.json]
"""

import argparse
import json

# [docs-start:load]
from pathlib import Path

import numpy as np

import geodex

# The workcell, a scene file in the scenes directory next to this script.
SCENE = Path(__file__).resolve().parent / "scenes" / "workcell.yaml"
# [docs-end:load]


# [docs-start:sideways]
def sideways_share(path):
    """Share of the base's travel that is sideways in its own frame.

    Each edge moves the base along one constant body twist (v_x, v_y, omega). The
    twist is the SE(2) logarithm between the edge's waypoints."""
    se2 = geodex.SE2()
    sideways = travel = 0.0
    for a, b in zip(path[:-1], path[1:]):
        vx, vy, _ = se2.log(a[:3], b[:3])
        sideways += abs(vy)
        travel += np.sqrt(vx * vx + vy * vy)
    return sideways / travel
# [docs-end:sideways]


def record(robot, result):
    return {"solved": bool(result.solved), "cost": float(result.cost),
            "drive": robot.drive(), "sideways_share": float(sideways_share(result.path)),
            "path": result.path.tolist(), "raw_path": result.raw_path.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    # [docs-start:load]
    scene = geodex.load_scene(str(SCENE))
    workspace = ((-3.0, 3.0), (-2.5, 2.5))
    husky = geodex.robots.HuskyUR5e(workspace=workspace)
    print(husky.name(), "drive:", husky.drive(), "coordinates:", husky.dim())
    # [docs-end:load]

    # [docs-start:plan]
    # (x, y, theta, shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3)
    start = np.array([-1.8, 1.3, np.pi / 2, 0.0, -2.3, 2.3, -1.57, -1.57, 0.0])
    goal = np.array([1.6, 0.0, 0.0, 0.0, -0.6, 0.4, -1.4, -np.pi / 2, 0.0])
    settings = geodex.PlanSettings(iterations=1000, seed=1, collision_check_resolution=0.005)

    husky_result = geodex.plan(husky, start, goal, collision=scene, settings=settings)
    print(f"solved={husky_result.solved} cost={husky_result.cost:.3f}")
    # [docs-end:plan]

    # [docs-start:sideways]
    print(f"sideways share={sideways_share(husky_result.path):.3f}")
    # [docs-end:sideways]

    # [docs-start:try-it]
    ridgeback = geodex.robots.RidgebackUR5e(workspace=workspace)
    ridgeback_result = geodex.plan(ridgeback, start, goal, collision=scene,
                                   settings=settings)
    print(f"solved={ridgeback_result.solved} cost={ridgeback_result.cost:.3f} "
          f"sideways share={sideways_share(ridgeback_result.path):.3f}")
    # [docs-end:try-it]

    out = {"husky_ur5e": record(husky, husky_result),
           "ridgeback_ur5e": record(ridgeback, ridgeback_result)}
    if args.json:
        args.json.write_text(json.dumps(out))
    return 0 if husky_result.solved and ridgeback_result.solved else 1


if __name__ == "__main__":
    raise SystemExit(main())

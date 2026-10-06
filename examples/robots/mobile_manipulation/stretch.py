#!/usr/bin/env python3
"""The Stretch 3 and the Stretch 4 on one kitchen task, the Python version of stretch.cpp.

Each robot starts beside the kitchen table with its wrist tucked and ends on the far side
of the island with its gripper over it. Base and arm move in one motion. The script plans
the Stretch 3 on its differential-drive base, then the Stretch 4 on its holonomic base.

Usage:
  python examples/robots/mobile_manipulation/stretch.py [--json out.json]
"""

import argparse
import json

# [docs-start:load]
from pathlib import Path

import numpy as np

import geodex

# The kitchen, a scene file in the scenes directory next to this script.
SCENE = Path(__file__).resolve().parent / "scenes" / "kitchen.yaml"
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
    stretch3 = geodex.robots.Stretch3(workspace=workspace)
    print(stretch3.name(), "drive:", stretch3.drive(), "coordinates:", stretch3.dim())
    # [docs-end:load]

    # [docs-start:plan]
    # (x, y, theta, lift, arm extension, wrist yaw, wrist pitch, wrist roll)
    start = np.array([1.8, -0.2, np.pi / 2, 0.3, 0.0, 3.0, -0.5, 0.0])  # by the table
    goal = np.array([0.0, 1.9, 0.0, 0.85, 0.4, 0.0, 0.0, 0.0])  # over the island
    settings = geodex.PlanSettings(iterations=1500, seed=1, collision_check_resolution=0.005)

    result3 = geodex.plan(stretch3, start, goal, collision=scene, settings=settings)
    print(f"solved={result3.solved} cost={result3.cost:.3f} "
          f"waypoints={len(result3.path)}")
    # [docs-end:plan]

    # [docs-start:sideways]
    print(f"sideways share={sideways_share(result3.path):.3f}")
    # [docs-end:sideways]

    # [docs-start:try-it]
    stretch4 = geodex.robots.Stretch4(workspace=workspace)
    # The Stretch 4's arm points forward, and its goal faces the island.
    start4 = np.array([1.8, -0.2, np.pi / 2, 0.2, 0.0, 3.0, 0.0, 0.0])
    goal4 = np.array([0.0, 1.85, -np.pi / 2, 0.8, 0.35, 0.0, 0.0, 0.0])

    result4 = geodex.plan(stretch4, start4, goal4, collision=scene, settings=settings)
    print(f"solved={result4.solved} cost={result4.cost:.3f} "
          f"sideways share={sideways_share(result4.path):.3f}")
    # [docs-end:try-it]

    out = {"stretch3": record(stretch3, result3), "stretch4": record(stretch4, result4)}
    if args.json:
        args.json.write_text(json.dumps(out))
    return 0 if result3.solved and result4.solved else 1


if __name__ == "__main__":
    raise SystemExit(main())

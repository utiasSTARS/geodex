#!/usr/bin/env python3
"""A Franka FR3 moves a box between two bays of a shelf, the Python version of arm_ke.cpp.

The script plans the move with the arm's kinetic-energy metric, measures the path, then
plans the same move with the Euclidean metric on the joint angles and measures it too.

Usage:
  python examples/robots/manipulation/arm_ke.py [--json out.json]
"""

import argparse
import json
from pathlib import Path

# [docs-start:load]
import numpy as np

import geodex

# An IKEA BILLY shelf in the arm's base frame, its seven boards (two sides, top, base, two
# shelves, back) as (center, size) in meters.
SHELF = [
    ((0.793492, 0.618718, 0.53), (0.018, 0.28, 1.06)),
    ((0.011492, 0.618718, 0.53), (0.018, 0.28, 1.06)),
    ((0.402492, 0.618718, 1.051), (0.8, 0.28, 0.018)),
    ((0.402492, 0.618718, 0.04), (0.764, 0.28, 0.08)),
    ((0.402492, 0.608718, 0.3977), (0.764, 0.26, 0.018)),
    ((0.402492, 0.608718, 0.7243), (0.764, 0.26, 0.018)),
    ((0.402492, 0.756218, 0.53), (0.78, 0.005, 1.06)),
]
# The held 0.04 x 0.24 x 0.16 m box as 112 spheres (x, y, z, radius) that contain
# it and reach at most 1 cm outside it. The rows give the spheres with x, y, z >= 0
# in the box's frame, and their mirror images give the rest.
OCTANT = [
    [0.012, 0.112, 0.072, 0.018], [0.01, 0.0, 0.07, 0.02],
    [0.01, 0.052, 0.07, 0.02], [0.01, 0.072, 0.07, 0.02],
    [0.01, 0.092, 0.07, 0.02], [0.01, 0.11, 0.032, 0.02],
    [0.01, 0.11, 0.052, 0.02], [0.008, 0.018, 0.068, 0.022],
    [0.008, 0.034, 0.068, 0.022], [0.008, 0.108, 0.0, 0.022],
    [0.008, 0.108, 0.014, 0.022], [0.0, 0.106, 0.066, 0.024],
    [0.0, 0.0, 0.014, 0.03], [0.0, 0.0, 0.046, 0.03], [0.0, 0.026, 0.0, 0.03],
    [0.0, 0.026, 0.036, 0.03], [0.0, 0.05, 0.046, 0.03], [0.0, 0.052, 0.014, 0.03],
    [0.0, 0.08, 0.0, 0.03], [0.0, 0.08, 0.026, 0.03], [0.0, 0.082, 0.046, 0.03],
]
# The gripper holds the box between its finger pads, its 40 mm side along the x axis
# of the TCP frame and its center 0.055 m along the z axis.
BOX = [[sx * x, sy * y, 0.055 + sz * z, r] for x, y, z, r in OCTANT
       for sx in ((1, -1) if x else (1,)) for sy in ((1, -1) if y else (1,))
       for sz in ((1, -1) if z else (1,))]
# [docs-end:load]


# [docs-start:load]
def shelf_with_box():
    """The shelf as a collision scene, with the box attached to the gripper."""
    scene = geodex.Scene()
    for center, size in SHELF:
        scene.add_box(position=center, size=size)
    env = scene.env()
    geodex.vamp.attach_spheres(env, BOX)
    return env
# [docs-end:load]


# [docs-start:measure]
def measure(path, pieces=16):
    """Joint-space and kinetic-energy length of a path with straight edges in joint
    coordinates. The kinetic-energy length uses the midpoint rule."""
    fr3 = geodex.robots.Fr3Gripper()  # fr3.mass_matrix(q) is the arm's inertia M(q)
    joint = energy = 0.0
    for a, b in zip(path[:-1], path[1:]):
        d = b - a
        joint += np.sqrt(d @ d)
        for k in range(pieces):
            q = a + ((k + 0.5) / pieces) * d
            energy += np.sqrt(d @ fr3.mass_matrix(q) @ d) / pieces
    return joint, energy
# [docs-end:measure]


def record(result):
    joint, energy = measure(result.path)
    return {"solved": bool(result.solved), "cost": float(result.cost),
            "smoothed": bool(result.smoothed), "path": result.path.tolist(),
            "raw_path": result.raw_path.tolist(), "joint_length": float(joint),
            "energy_length": float(energy)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    # [docs-start:load]
    env = shelf_with_box()
    robot = geodex.robots.Fr3Gripper()  # the kinetic-energy metric on the joints
    print(robot.name(), "joints:", robot.dim())
    # [docs-end:load]

    # [docs-start:plan]
    start = np.array([0.1758, -0.1952, 0.4451, -2.1536, 1.9130, 2.0966, 1.0981])  # middle bay
    goal = np.array([0.1591, -0.0746, 0.4710, -1.3579, 1.4042, 2.1810, 1.9137])  # top bay
    settings = geodex.PlanSettings(iterations=1500, seed=1)

    result = geodex.plan(robot, start, goal, collision=env, settings=settings)
    print(f"solved={result.solved} cost={result.cost:.3f} waypoints={len(result.path)}")
    # [docs-end:plan]

    # [docs-start:measure]
    joint, energy = measure(result.path)
    print(f"joint length={joint:.3f} rad kinetic-energy length={energy:.3f}")
    # [docs-end:measure]

    # [docs-start:try-it]
    euclidean = geodex.robots.Fr3Gripper(metric="euclidean")
    euclidean_result = geodex.plan(euclidean, start, goal, collision=env, settings=settings)
    joint, energy = measure(euclidean_result.path)
    print(f"joint length={joint:.3f} rad kinetic-energy length={energy:.3f}")
    # [docs-end:try-it]

    out = {"kinetic_energy": record(result), "euclidean": record(euclidean_result)}
    if args.json:
        args.json.write_text(json.dumps(out))
    return 0 if result.solved and euclidean_result.solved else 1


if __name__ == "__main__":
    raise SystemExit(main())

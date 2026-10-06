"""Assets of the gallery of docs/robots/index.rst.

``robots-gallery-<name>``
    One robot viewer scene per built-in robot with its visual meshes. The robot repeats a
    short joint-space motion around a pose that shows its moving parts. The Panda and the
    Husky with a UR5e also have a poster with the collision spheres of their VAMP models.
"""

from __future__ import annotations

import numpy as np

from robot_scene import RobotScene, frame

# Registry name and the pose to show, in planning-group joint order. A mobile robot lists its
# base pose first.
POSES = {
    "panda": [0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.8],
    "ur5": [0.0, -1.2, 1.6, -1.9, -1.57, 0.0],
    "baxter": [0.4, -0.6, -0.3, 1.2, 0.2, 0.9, 0.0, -0.4, -0.6, 0.3, 1.2, -0.2, 0.9, 0.0],
    "pr2": [0.4, 0.2, 0.6, -1.2, 0.0, -0.6, 0.0, -0.4, 0.2, -0.6, -1.2, 0.0, -0.6, 0.0],
    "fr3_arm_gripper": [0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.8],
    "stretch3": [0.0, 0.0, 0.0, 0.7, 0.25, 0.0, -0.3, 0.0],
    "stretch4": [0.0, 0.0, 0.0, 0.7, 0.25, 0.0, 0.0, 0.0],
    "ridgeback_ur5e": [0.0, 0.0, 0.0, 0.0, -1.3, 1.4, -1.7, -1.57, 0.0],
    "husky_ur5e": [0.0, 0.0, 0.0, 0.0, -1.3, 1.4, -1.7, -1.57, 0.0],
}
SIZE = (720, 600)
# Robots whose scene also has a poster with the collision spheres, for the pages that explain
# collision checking.
SPHERES = ("panda", "husky_ur5e")


def motion(q0, count=150, amplitude=0.35):
    """A smooth motion of the arm joints around `q0` that ends where it starts."""
    q0 = np.asarray(q0, dtype=float)
    t = np.linspace(0.0, 2 * np.pi, count)
    arm = slice(3, None) if len(q0) in (8, 9) else slice(0, None)
    frames = np.repeat(q0[None], count, axis=0)
    phase = np.arange(len(q0[arm]))
    frames[:, arm] += amplitude * 0.5 * (np.sin(t[:, None] + phase[None]) - np.sin(phase[None]))
    return frames


def record(name):
    frames = motion(POSES[name])
    scene = RobotScene(f"robots-gallery-{name.replace('_', '-')}", name, frames,
                       camera_position=(0, 0, 0), camera_target=(0, 0, 0), timing="cycle")
    bounds = [scene.chain.bounds(q) for q in frames[::10]]
    lo = np.min([b[0] for b in bounds], axis=0)
    hi = np.max([b[1] for b in bounds], axis=0)
    fov = 35.0
    position, target = frame(lo, hi, fov=fov, aspect=SIZE[0] / SIZE[1], margin=1.0)
    scene.doc["camera"] = {"position": [round(float(v), 4) for v in position],
                           "target": [round(float(v), 4) for v in target], "fov": fov}
    half = float(max(1.2, 1.4 * np.max(hi[:2] - lo[:2])))
    scene.doc["floor"] = {"center": [round(float(v), 3) for v in 0.5 * (lo[:2] + hi[:2])],
                          "half": half, "cell": 0.1, "major": 5, "z": round(float(lo[2]), 4)}
    scene.save(poster_size=SIZE, spheres_poster=name in SPHERES)


def generate(parts=tuple(POSES)):
    for name in parts:
        record(name)

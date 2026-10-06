"""Assets of docs/concepts/planning.rst.

``planning-sphere``
    Viewer scene of the first plan of examples/concepts/planning.py, the sphere with its two
    cap obstacles and the returned path, and a ball that follows the path.
"""

from __future__ import annotations

import numpy as np

from common import densify, run_example
from style import AQUA, BLUE, ORANGE, SURFACE_OBSTACLE

FRAMES = 150


def sphere_scene(result):
    import geodex
    from robot_scene import SURFACE_TONE, PathScene, cap, even, sphere

    s2 = geodex.Sphere()
    path = densify(s2, result["path"], per_edge=4)
    frames = even(path, FRAMES)
    frames /= np.linalg.norm(frames, axis=1, keepdims=True)
    # Look at the middle of the path from outside the sphere.
    middle = frames[len(frames) // 2]
    scene = PathScene("planning-sphere", frames * 1.012, [{"radius": 0.035, "color": BLUE}],
                      camera_position=3.8 * middle + np.array([0.0, 0.0, 0.3]),
                      camera_target=0.3 * middle, fov=30.0)
    scene.add_objects([sphere((0.0, 0.0, 0.0), 1.0, SURFACE_TONE)])
    scene.add_objects([cap(axis, angle, SURFACE_OBSTACLE, radius=1.004)
                       for axis, angle in result["caps"]])
    scene.add_trace(frames * 1.012, BLUE, width=4.5)
    scene.add_marker(frames[0] * 1.012, AQUA, radius=0.05)
    scene.add_marker(frames[-1] * 1.012, ORANGE, radius=0.05)
    scene.save()


def generate(parts=("sphere",)):
    sphere_scene(run_example("concepts/planning")["sphere"])

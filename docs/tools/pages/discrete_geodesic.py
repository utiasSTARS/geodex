"""Assets of docs/concepts/discrete-geodesic-interpolation.rst.

``discrete-geodesic-sphere``
    Viewer scene of the two walks of examples/concepts/discrete_geodesic.py on the unit
    sphere, the great circle of the round metric and the walk under diag(25, 1, 1), with a
    ball that follows each walk.
"""

from __future__ import annotations

import numpy as np

from common import densify, run_example
from style import BLUE, INK_2, ORANGE

FRAMES = 150


def sphere_scene(data):
    import geodex
    from robot_scene import SURFACE_TONE, PathScene, even, sphere

    s2 = geodex.Sphere()
    walks = []
    for key in ("great", "bent"):
        walk = even(densify(s2, data[key], per_edge=8), FRAMES)
        walks.append(walk / np.linalg.norm(walk, axis=1, keepdims=True) * 1.012)
    # Look at the middle of the walks from outside the plane of the great circle.
    middle = walks[0][len(walks[0]) // 2]
    normal = np.cross(walks[0][0], walks[0][-1])
    view = middle / np.linalg.norm(middle) + 0.45 * normal / np.linalg.norm(normal)
    scene = PathScene("discrete-geodesic-sphere", np.hstack(walks),
                      [{"radius": 0.035, "color": BLUE}, {"radius": 0.035, "color": ORANGE}],
                      camera_position=4.2 * view / np.linalg.norm(view),
                      camera_target=0.5 * middle, fov=30.0)
    scene.add_objects([sphere((0.0, 0.0, 0.0), 1.0, SURFACE_TONE)])
    for walk, color in zip(walks, (BLUE, ORANGE)):
        scene.add_trace(walk, color, width=4.0)
    for point in (walks[0][0], walks[0][-1]):
        scene.add_marker(point, INK_2, radius=0.04)
    scene.save(poster_size=(1024, 768))


def generate(parts=("sphere",)):
    sphere_scene(run_example("concepts/discrete_geodesic"))

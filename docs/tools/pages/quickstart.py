"""Assets of docs/getting-started/quickstart.rst.

``quickstart-panda``
    Robot viewer scene of the quickstart plan, the Panda moving along the returned path past
    the post, with the trace of the point between its fingers.
"""

from __future__ import annotations

import numpy as np

from common import run_example
from robot_scene import RobotScene, box, sample
from style import AQUA, BLUE, ORANGE

POST = dict(position=(0.4, 0.0, 0.3), size=(0.1, 0.1, 0.8))
# The point between the open fingers, panda_grasptarget, 0.105 m past the flange panda_link8.
GRASP = np.array([0.0, 0.0, 0.105])


def dense(path, step=0.02):
    """Joint-space points along the path. The robot's geodesics are straight in joint space."""
    path = np.asarray(path)
    points = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        points += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(points)


def generate(parts=("scene",)):
    result = run_example("getting_started/quickstart")
    if not result["solved"]:
        raise RuntimeError("the quickstart plan did not solve")
    frames = sample(dense(result["path"]), 150)
    scene = RobotScene("quickstart-panda", "panda", frames,
                       camera_position=(1.25, -1.45, 1.1), camera_target=(0.2, 0.0, 0.42),
                       fov=36.0,
                       floor={"center": [0.2, 0.0], "half": 1.3, "cell": 0.1, "major": 5})
    scene.add_objects([box(POST["position"], POST["size"])])
    grasp = np.array([scene.chain.end_effector(q, GRASP) for q in frames])
    scene.add_trace(grasp, BLUE)
    scene.add_marker(grasp[0], AQUA, 0.022)
    scene.add_marker(grasp[-1], ORANGE, 0.022)
    scene.ghosts(4)
    scene.save()

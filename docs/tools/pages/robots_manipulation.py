"""Assets of docs/robots/manipulation.rst.

``robots-fr3-euclidean``, ``robots-fr3-kinetic-energy``
    Robot viewer scenes of the documented plans of examples/robots/manipulation/arm_ke.py, the FR3
    moving a box from the middle compartment of a shelf to the top compartment, with faint copies
    along its path and the trace of its gripper's TCP.
``robots-fr3-collision``
    The kinetic-energy plan with the collision spheres of the arm and the sphere cover of the
    box, and its poster with the spheres.
``robots-fr3-joint-travel``
    How far each joint turns along the two documented paths.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from common import ROOT, run_example
from plots import write_plotly
from robot_scene import RobotScene, box, sample, sphere
from robots_draw import densify
from style import AQUA, BLUE, INK_2, ORANGE, plotly_layout

EXAMPLE = ROOT / "examples" / "robots" / "manipulation" / "arm_ke.py"
LABELS = {"euclidean": "Euclidean", "kinetic_energy": "kinetic energy"}
# The held box in the TCP frame, between the finger pads.
BOX_SIZE = (0.04, 0.24, 0.16)
BOX_CENTER = (0.0, 0.0, 0.055)
BOX_TONE = "#8e5ba8"


def _example():
    spec = importlib.util.spec_from_file_location("arm_ke", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(name, path, shelf):
    dense = densify(path, lambda a, b, t: a + t * (b - a),
                    lambda a, b: np.ceil(np.abs(b - a).max() / 0.01))
    scene = RobotScene(name, "fr3_arm_gripper", sample(dense, 150),
                       camera_position=(1.1, -1.16, 1.35), camera_target=(0.35, 0.3, 0.6),
                       fov=38.0, floor={"center": [0.3, 0.2], "half": 1.5, "cell": 0.1, "major": 5})
    scene.add_objects(shelf)
    return scene


def record(documented):
    example = _example()
    shelf = [box(center, size) for center, size in example.SHELF]
    held = box(BOX_CENTER, BOX_SIZE, BOX_TONE)
    for arm, color in (("euclidean", BLUE), ("kinetic_energy", ORANGE)):
        path = np.array(documented[arm]["path"])
        scene = _scene("robots-fr3-" + arm.replace("_", "-"), path, shelf)
        scene.hold([held])
        scene.trace_end_effector(color)
        scene.add_marker(scene.chain.end_effector(path[0]), AQUA)
        scene.add_marker(scene.chain.end_effector(path[-1]), ORANGE)
        scene.ghosts(4)
        scene.save()
    # The kinetic-energy plan with the box's sphere cover, for the page's view of the spheres.
    scene = _scene("robots-fr3-collision", np.array(documented["kinetic_energy"]["path"]), shelf)
    scene.hold([held] + [sphere(c[:3], c[3], BOX_TONE, opacity=0.3) for c in example.BOX])
    scene.save(spheres_poster=True)


def joint_travel(documented):
    import plotly.graph_objects as go

    fig = go.Figure()
    joints = [f"joint {i}" for i in range(1, 8)]
    for arm, color in (("euclidean", BLUE), ("kinetic_energy", ORANGE)):
        path = np.array(documented[arm]["path"])
        travel = np.abs(np.diff(path, axis=0)).sum(axis=0)
        fig.add_trace(go.Bar(x=joints, y=travel, name=f"{LABELS[arm]} metric",
                             marker_color=color,
                             hovertemplate="%{x}: %{y:.2f} rad<extra>" + LABELS[arm] +
                             "</extra>"))
    fig.update_layout(**plotly_layout(
        barmode="group", yaxis_title="total rotation along the path (rad)",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12),
        margin=dict(l=80, r=20, t=60, b=50)))
    fig.update_yaxes(gridcolor="#e6e5e1", linecolor=INK_2)
    write_plotly(fig, "robots-fr3-joint-travel", height=440)


def generate(parts=("scenes", "travel")):
    documented = run_example("robots/manipulation/arm_ke")
    if "scenes" in parts:
        record(documented)
    if "travel" in parts:
        joint_travel(documented)

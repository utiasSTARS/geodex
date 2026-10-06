"""Assets of the Clearpath part of docs/robots/mobile-manipulation.rst.

``robots-husky-ur5e``, ``robots-ridgeback-ur5e``
    Robot viewer scenes of the two documented plans of
    examples/robots/mobile_manipulation/clearpath.py, each robot with its meshes, the base trace on
    the floor and the trace of the tool flange.
``docs/robots/figs/clearpath-manipulators/bases.svg``
    Top view of the workcell with each robot's chassis outline along its path.
"""

from __future__ import annotations

import numpy as np
import yaml

from common import ROOT, figure_path, run_example
from robots_draw import SphereRobot, densify, record_whole_body, whole_body_geodesic, \
    whole_body_pieces
from style import BLUE, OBSTACLE, OBSTACLE_EDGE, ORANGE, matplotlib_style

SCENE = ROOT / "examples" / "robots" / "mobile_manipulation" / "scenes" / "workcell.yaml"
COLORS = {"husky_ur5e": BLUE, "ridgeback_ur5e": ORANGE}
NAMES = {"husky_ur5e": "Husky A200 with a UR5e, skid steer",
         "ridgeback_ur5e": "Ridgeback with a UR5e, mecanum"}
# Chassis length and width from the Clearpath specifications.
CHASSIS = {"husky_ur5e": (0.990, 0.670), "ridgeback_ur5e": (0.960, 0.793)}
FLOOR = ((-3.0, 3.5), (-2.5, 3.0))


def record(documented):
    for name, color in COLORS.items():
        record_whole_body(f"robots-{name.replace('_', '-')}", SphereRobot(name),
                          documented[name]["path"], SCENE, color,
                          camera_position=(0.6, -4.8, 3.6), camera_look_at=(0.3, 0.4, 0.3),
                          floor=FLOOR)


def static_figure(documented):
    plt = matplotlib_style()
    from matplotlib.patches import Polygon, Rectangle

    world = yaml.safe_load(SCENE.read_text())["world"]["collision_objects"]
    fig, ax = plt.subplots(figsize=(7.6, 6.7))
    for obj in world:
        (x, y, _), (dx, dy, _) = obj["primitive_poses"][0]["position"], \
            obj["primitives"][0]["dimensions"]
        ax.add_patch(Rectangle((x - dx / 2, y - dy / 2), dx, dy, facecolor=OBSTACLE,
                               edgecolor=OBSTACLE_EDGE, linewidth=0.6))
    for name in COLORS:
        dense = densify(np.array(documented[name]["path"]), whole_body_geodesic(6),
                        whole_body_pieces)
        steps = np.r_[0.0, np.cumsum(np.hypot(*np.diff(dense[:, :2], axis=0).T))]
        marks = np.searchsorted(steps, np.linspace(0, steps[-1], 10)).clip(0, len(dense) - 1)
        length, width = CHASSIS[name]
        local = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * [length / 2, width / 2]
        for q in dense[marks]:
            c, s = np.cos(q[2]), np.sin(q[2])
            ax.add_patch(Polygon(q[:2] + local @ np.array([[c, s], [-s, c]]), closed=True,
                                 fill=False, edgecolor=COLORS[name], linewidth=1.0, alpha=0.7))
            ax.plot([q[0], q[0] + 0.35 * c], [q[1], q[1] + 0.35 * s], color=COLORS[name],
                    linewidth=1.2, alpha=0.8)
        share = documented[name]["sideways_share"]
        ax.plot(dense[:, 0], dense[:, 1], color=COLORS[name], linewidth=2.4,
                label=f"{NAMES[name]}, sideways share {share:.3f}")
    ax.set_xlim(*FLOOR[0])
    ax.set_ylim(*FLOOR[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    fig.tight_layout()
    fig.savefig(figure_path("robots", "clearpath-manipulators", "bases.svg"),
                bbox_inches="tight")
    plt.close(fig)


def generate(parts=("scenes", "static")):
    documented = run_example("robots/mobile_manipulation/clearpath")
    if "scenes" in parts:
        record(documented)
    if "static" in parts:
        static_figure(documented)

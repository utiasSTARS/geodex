"""Assets of the Stretch part of docs/robots/mobile-manipulation.rst.

``robots-stretch-3``, ``robots-stretch-4``
    Robot viewer scenes of the documented plans of examples/robots/mobile_manipulation/stretch.py,
    each Stretch with its meshes reaching over the kitchen island.
``docs/robots/figs/stretch/bases.svg``
    Top view of the kitchen with each robot's base footprint along its path.
``robots-stretch-sideways``
    The sideways component of the base's velocity along both paths, as the sine of the
    angle between the direction of travel and the heading.
``docs/robots/figs/landing.png``
    The still of the robot guides card on the landing page, shown until the card's video
    exists, cropped from the poster of the Stretch 3 scene.
"""

from __future__ import annotations

import numpy as np
import yaml

from common import ROOT, figure_path, run_example
from plots import write_plotly
from robots_draw import SphereRobot, densify, record_whole_body, whole_body_geodesic, \
    whole_body_pieces
from style import BLUE, OBSTACLE, OBSTACLE_EDGE, ORANGE, matplotlib_style, plotly_layout

SCENE = ROOT / "examples" / "robots" / "mobile_manipulation" / "scenes" / "kitchen.yaml"
COLORS = {"stretch3": BLUE, "stretch4": ORANGE}
NAMES = {"stretch3": "Stretch 3, differential drive", "stretch4": "Stretch 4, holonomic"}


def record(documented):
    for name, color in COLORS.items():
        record_whole_body(f"robots-stretch-{name[-1]}", SphereRobot(name),
                          documented[name]["path"], SCENE, color,
                          camera_position=(3.6, -3.4, 3.6), camera_look_at=(0.5, 0.6, 0.4))


def _outline(name):
    """Base outline in the base frame, a box of 34 cm by 34 cm for the Stretch 3 and a disc
    45 cm across for the Stretch 4."""
    if name == "stretch4":
        a = np.linspace(0.0, 2 * np.pi, 49)
        return np.c_[0.225 * np.cos(a), 0.225 * np.sin(a)]
    return np.array([[0.11, 0.17], [-0.23, 0.17], [-0.23, -0.17], [0.11, -0.17],
                     [0.11, 0.17]])


def static_figure(documented):
    plt = matplotlib_style()
    from matplotlib.patches import Rectangle

    world = yaml.safe_load(SCENE.read_text())["world"]["collision_objects"]
    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    for obj in world:
        (x, y, _), (dx, dy, _) = obj["primitive_poses"][0]["position"], \
            obj["primitives"][0]["dimensions"]
        ax.add_patch(Rectangle((x - dx / 2, y - dy / 2), dx, dy, facecolor=OBSTACLE,
                               edgecolor=OBSTACLE_EDGE, linewidth=0.6))
    for name in COLORS:
        dense = densify(np.array(documented[name]["path"]), whole_body_geodesic(5),
                        whole_body_pieces)
        steps = np.r_[0.0, np.cumsum(np.hypot(*np.diff(dense[:, :2], axis=0).T))]
        marks = np.searchsorted(steps, np.linspace(0, steps[-1], 12)).clip(0, len(dense) - 1)
        outline = _outline(name)
        for q in dense[marks]:
            c, s = np.cos(q[2]), np.sin(q[2])
            pts = q[:2] + outline @ np.array([[c, s], [-s, c]])
            ax.plot(pts[:, 0], pts[:, 1], color=COLORS[name], linewidth=1.0, alpha=0.7)
            ax.plot([q[0], q[0] + 0.2 * c], [q[1], q[1] + 0.2 * s], color=COLORS[name],
                    linewidth=1.2, alpha=0.8)
        ax.plot(dense[:, 0], dense[:, 1], color=COLORS[name], linewidth=2.4, label=NAMES[name])
    ax.set_xlim(-3.1, 3.0)
    ax.set_ylim(-2.5, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_path("robots", "stretch", "bases.svg"), bbox_inches="tight")
    plt.close(fig)


def sideways(documented):
    import geodex
    import plotly.graph_objects as go

    se2 = geodex.SE2()
    fig = go.Figure()
    for name, color in COLORS.items():
        path = np.array(documented[name]["path"])
        dense = densify(path, whole_body_geodesic(5), whole_body_pieces)
        steps, lateral = [], []
        for a, b in zip(dense[:-1], dense[1:]):
            vx, vy, _ = se2.log(a[:3], b[:3])
            steps.append(np.hypot(vx, vy))
            lateral.append(vy)
        steps, lateral = np.array(steps), np.array(lateral)
        s = np.concatenate([[0.0], np.cumsum(steps)])
        mid = 0.5 * (s[:-1] + s[1:]) / s[-1]
        # The ratio says nothing about sliding where the base nearly turns in place, and the
        # curve leaves a gap there.
        moving = steps > 0.25 * np.median(steps)
        value = np.where(moving, lateral / np.where(moving, steps, 1.0), np.nan)
        fig.add_trace(go.Scatter(x=mid, y=value, mode="lines", name=NAMES[name],
                                 line=dict(color=color, width=2.5),
                                 hovertemplate="%{x:.2f} of the base travel<br>"
                                               "sideways %{y:.2f}<extra></extra>"))
    fig.update_layout(**plotly_layout(
        xaxis_title="fraction of the base's travel",
        yaxis_title="sideways component of the base velocity",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
        margin=dict(l=80, r=20, t=50, b=60)))
    fig.update_yaxes(range=[-1.05, 1.05], zeroline=True, zerolinecolor="#c9c8c3")
    fig.update_xaxes(range=[0, 1])
    write_plotly(fig, "robots-stretch-sideways", height=420)


def thumbnail():
    from PIL import Image

    still = Image.open(ROOT / "docs" / "_static" / "robot-scenes" / "robots-stretch-3.jpg")
    w, h = still.size
    box = (int(0.30 * w), int(0.17 * h), int(0.86 * w), int(0.17 * h + 0.28 * w))
    still.crop(box).resize((800, 400), Image.LANCZOS).save(
        figure_path("robots", "", "landing.png"), optimize=True)


def generate(parts=("scenes", "static", "sideways", "thumbnail")):
    documented = run_example("robots/mobile_manipulation/stretch")
    if "scenes" in parts:
        record(documented)
    if "static" in parts:
        static_figure(documented)
    if "sideways" in parts:
        sideways(documented)
    if "thumbnail" in parts:
        thumbnail()

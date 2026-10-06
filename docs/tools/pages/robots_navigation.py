"""Assets of docs/robots/navigation.rst.

``robots-nav-bases``
    The office map with the documented path of each Clearpath base of
    examples/robots/navigation/bases.py, its footprint drawn along the path, one base at a time
    through a dropdown.
``docs/robots/figs/navigation/drives.svg``
    The Dingo-D (differential drive) and the Dingo-O (mecanum) side by side, the two bases
    of one family whose metric differs.
``robots-nav-dingo-d``, ``robots-nav-dingo-o``
    Robot viewer scenes of those two plans, each Dingo with its meshes driving through the
    office with the trace of its base.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import yaml

from common import ROOT, figure_path, run_example
from plots import write_plotly
from robot_scene import SCENE_TONE, WALL_TONE, RobotScene, box
from robots_draw import densify, frames_along
from style import AQUA, BLUE, INK, OBSTACLE, ORANGE, matplotlib_style, plotly_layout

EXAMPLE = ROOT / "examples" / "robots" / "navigation" / "bases.py"
OFFICE = ROOT / "examples" / "robots" / "navigation" / "office.yaml"
# Unknown cells of the office map, lighter than the occupied ones.
UNKNOWN = "#d3d6d9"
DRIVE_COLORS = {"differential": BLUE, "skid_steer": AQUA, "holonomic": ORANGE}
DRIVE_LABELS = {"differential": "differential drive", "skid_steer": "skid steer",
                "holonomic": "holonomic (mecanum)"}
NAMES = {"jackal": "Jackal", "dingo_d": "Dingo-D", "dingo_o": "Dingo-O"}


def _example():
    spec = importlib.util.spec_from_file_location("bases", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _office():
    """Cells of the office map, bottom row first, 0 free, 1 unknown and 2 occupied, and the
    cell size. The thresholds are those of the map server's trinary mode."""
    meta = yaml.safe_load(OFFICE.read_text())
    _, size, _, pixels = (OFFICE.parent / meta["image"]).read_bytes().split(b"\n", 3)
    cols, rows = map(int, size.split())
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(rows, cols)[::-1]
    occupancy = (255.0 - image) / 255.0
    cells = np.where(occupancy > meta["occupied_thresh"], 2,
                     np.where(occupancy < meta["free_thresh"], 0, 1))
    return cells, meta["resolution"]


def _boxes(mask, resolution):
    """The cells of `mask` merged into rectangles, (center x, center y, half length, half
    width) in meters. Cell (r, c) has its center at (c, r) times the resolution."""
    rects, open_runs = [], {}
    for r in range(mask.shape[0] + 1):
        runs = {}
        if r < mask.shape[0]:
            padded = np.r_[False, mask[r], False].astype(np.int8)
            edges = np.flatnonzero(np.diff(padded))
            for c0, c1 in zip(edges[::2], edges[1::2]):
                runs[(c0, c1)] = open_runs.get((c0, c1), r)
        rects += [(run, r0, r) for run, r0 in open_runs.items() if run not in runs]
        open_runs = runs
    return [((c0 + c1 - 1) / 2 * resolution, (r0 + r1 - 1) / 2 * resolution,
             (c1 - c0) / 2 * resolution, (r1 - r0) / 2 * resolution)
            for (c0, c1), r0, r1 in rects]


def _extent(cells, resolution):
    """(x range, y range) of the map's cells in meters."""
    rows, cols = cells.shape
    return ((-resolution / 2, (cols - 0.5) * resolution),
            (-resolution / 2, (rows - 0.5) * resolution))


def _dense(path):
    import geodex

    se2 = geodex.SE2()
    return densify(path, se2.geodesic,
                   lambda a, b: np.ceil(max(np.hypot(*(b[:2] - a[:2])) / 0.02,
                                            abs(se2.log(a, b)[2]) / 0.03, 1.0)))


def _footprint(pose, length, width):
    """Corners of the footprint at `pose`, closed."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    local = np.array([[length, width], [-length, width], [-length, -width], [length, -width],
                      [length, width]]) / 2
    return pose[:2] + local @ np.array([[c, s], [-s, c]])


def _along(dense, spacing):
    """Poses of a densified path every `spacing` meters, the endpoints included."""
    steps = np.r_[0.0, np.cumsum(np.hypot(*np.diff(dense[:, :2], axis=0).T))]
    marks = np.linspace(0.0, steps[-1], max(2, int(steps[-1] / spacing) + 1))
    return dense[np.searchsorted(steps, marks).clip(0, len(dense) - 1)]


def interactive(documented, example):
    import plotly.graph_objects as go

    cells, resolution = _office()
    xrange, yrange = _extent(cells, resolution)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=cells.astype(np.uint8), x=np.arange(cells.shape[1]) * resolution,
        y=np.arange(cells.shape[0]) * resolution, zmin=0, zmax=2, zsmooth=False,
        colorscale=[[0.0, "rgba(0,0,0,0)"], [1 / 3, "rgba(0,0,0,0)"], [1 / 3, UNKNOWN],
                    [2 / 3, UNKNOWN], [2 / 3, OBSTACLE], [1.0, OBSTACLE]],
        showscale=False, hoverinfo="skip"))
    names = list(example.PLATFORMS)
    traces_per = []
    for k, name in enumerate(names):
        length, width, drive, _ = example.PLATFORMS[name]
        color = DRIVE_COLORS[drive]
        dense = _dense(np.array(documented[name]["path"]))
        xs, ys = [], []
        for pose in _along(dense, 0.8):
            corners = _footprint(pose, length, width)
            xs += list(corners[:, 0]) + [None]
            ys += list(corners[:, 1]) + [None]
        visible = k == 0
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1),
                                 opacity=0.55, hoverinfo="skip", showlegend=False,
                                 visible=visible))
        share = documented[name]["sideways_share"]
        fig.add_trace(go.Scatter(
            x=dense[:, 0], y=dense[:, 1], mode="lines", line=dict(color=color, width=3),
            name=f"{NAMES[name]}, {DRIVE_LABELS[drive]}<br>{length:.3f} m x {width:.3f} m, "
                 f"sideways share {share:.3f}", visible=visible,
            customdata=np.degrees(dense[:, 2]),
            hovertemplate="x %{x:.2f} m, y %{y:.2f} m<br>heading %{customdata:.0f} deg"
                          "<extra></extra>"))
        traces_per.append(2)
    start, goal = np.array(documented[names[0]]["path"][0]), np.array(
        documented[names[0]]["path"][-1])
    fig.add_trace(go.Scatter(x=[start[0], goal[0]], y=[start[1], goal[1]], mode="markers+text",
                             marker=dict(size=11, color=[AQUA, ORANGE], line=dict(width=0)),
                             text=["start", "goal"], textposition="bottom center",
                             hoverinfo="skip", showlegend=False))
    buttons = []
    for k, name in enumerate(names):
        visible = [True]
        for j in range(len(names)):
            visible += [j == k] * traces_per[j]
        visible.append(True)
        buttons.append(dict(label=NAMES[name], method="update", args=[{"visible": visible}]))
    fig.update_layout(**plotly_layout(
        showlegend=True, margin=dict(l=60, r=20, t=70, b=50),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14, yanchor="top"),
        updatemenus=[dict(buttons=buttons, direction="down", type="dropdown", x=0.0,
                          xanchor="left", y=1.04, yanchor="bottom", showactive=True,
                          font=dict(size=14))]))
    fig.update_xaxes(range=xrange, title="x (m)", constrain="domain", showgrid=False)
    fig.update_yaxes(range=yrange, title="y (m)", scaleanchor="x", scaleratio=1,
                     showgrid=False)
    write_plotly(fig, "robots-nav-bases", height=560)


def static_figure(documented, example):
    plt = matplotlib_style()
    from matplotlib.patches import PathPatch, Polygon
    from matplotlib.path import Path
    from matplotlib.transforms import Affine2D

    cells, resolution = _office()
    xrange, yrange = _extent(cells, resolution)
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    # Each color is one compound path. A compound path draws adjacent cells without seams.
    for code, color in ((1, UNKNOWN), (2, OBSTACLE)):
        rects = [Path.unit_rectangle().transformed(
            Affine2D().scale(2 * hx, 2 * hy).translate(cx - hx, cy - hy))
            for cx, cy, hx, hy in _boxes(cells == code, resolution)]
        ax.add_patch(PathPatch(Path.make_compound_path(*rects), facecolor=color,
                               edgecolor="none"))
    for name in ("dingo_d", "dingo_o"):
        length, width, drive, _ = example.PLATFORMS[name]
        color = DRIVE_COLORS[drive]
        dense = _dense(np.array(documented[name]["path"]))
        for pose in _along(dense, 0.8):
            ax.add_patch(Polygon(_footprint(pose, length, width), closed=True, fill=False,
                                 edgecolor=color, linewidth=0.9, alpha=0.7))
        share = documented[name]["sideways_share"]
        ax.plot(dense[:, 0], dense[:, 1], color=color, linewidth=2.4,
                label=f"{NAMES[name]}, {DRIVE_LABELS[drive]}, sideways share {share:.3f}")
    start, goal = documented["dingo_d"]["path"][0], documented["dingo_d"]["path"][-1]
    ax.plot(*start[:2], "o", color=AQUA, markersize=9)
    ax.plot(*goal[:2], "o", color=INK, markersize=7)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    fig.tight_layout()
    fig.savefig(figure_path("robots", "navigation", "drives.svg"), bbox_inches="tight")
    plt.close(fig)


def office_objects():
    """The office map as boxes, occupied cells 0.45 m high and unknown cells as flat tiles."""
    cells, resolution = _office()
    objects = []
    for code, height, color in ((2, 0.45, SCENE_TONE), (1, 0.02, WALL_TONE)):
        objects += [box((cx, cy, height / 2), (2 * hx, 2 * hy, height), color)
                    for cx, cy, hx, hy in _boxes(cells == code, resolution)]
    return objects


def record(documented, example):
    objects = office_objects()
    for name in ("dingo_d", "dingo_o"):
        drive = example.PLATFORMS[name][2]
        dense = _dense(np.array(documented[name]["path"]))
        scene = RobotScene(f"robots-nav-{name.replace('_', '-')}", name,
                           frames_along(dense, 240), camera_position=(7.5, -3.8, 8.6),
                           camera_target=(7.5, 4.4, 0.0), fov=40.0,
                           floor={"center": [7.5, 4.6], "half": 8.0, "cell": 0.5, "major": 2})
        scene.add_objects(objects)
        scene.trace_base(DRIVE_COLORS[drive], width=3.0)
        scene.ghosts(5)
        scene.save()


def generate(parts=("interactive", "static", "scenes")):
    documented = run_example("robots/navigation/bases")
    example = _example()
    if "interactive" in parts:
        interactive(documented, example)
    if "static" in parts:
        static_figure(documented, example)
    if "scenes" in parts:
        record(documented, example)

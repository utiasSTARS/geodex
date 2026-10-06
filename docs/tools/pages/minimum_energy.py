"""Assets of docs/tutorials/minimum-energy-planning.rst.

``ke_metric.svg``, ``jacobi_combined.svg``
    Unit balls of the kinetic energy metric and of the Jacobi metric at three energies on
    a grid over [-pi, pi]^2, over the determinant of the metric tensor.
``planning_result.svg``
    The three plans of examples/tutorials/minimum_energy_planning.py, raw and smoothed, over the
    same backgrounds. Raw edges are drawn along the curves the planner checked.
``minimum-energy-arm`` (plotly)
    The planar arm along the three smoothed paths, side by side, with a slider over the
    fraction of each path's metric length.
"""

from __future__ import annotations

import sys

import numpy as np

from common import ROOT, figure_path, run_example
from plots import write_plotly
from style import AQUA, BLUE, INK, INK_2, ORANGE, matplotlib_style

PAGE = "minimum-energy-planning"
TICKS = ([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
         [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
COLORS = {"Euclidean": INK_2, "Kinetic energy": BLUE, "Jacobi": ORANGE}


def _example():
    sys.path.insert(0, str(ROOT / "examples" / "tutorials"))
    import minimum_energy_planning as example
    return example


def _tensors(example, H_factor=None, n=81):
    """Metric tensors on an n x n grid, the kinetic energy one when H_factor is None."""
    mass = example.PlanarArmMassMatrix()
    pmax = 9.81 * (1.0 * 0.5 + 1.0 * (1.0 + 0.5))
    axis = np.linspace(-np.pi, np.pi, n)
    G = np.empty((n, n, 2, 2))
    for i, q2 in enumerate(axis):
        for j, q1 in enumerate(axis):
            q = np.array([q1, q2])
            M = mass(q)
            G[i, j] = M if H_factor is None else 2.0 * (H_factor * pmax - example.potential(q)) * M
    return axis, G


def _panel(ax, axis, G, title, ellipses=True, normalize=False):
    from matplotlib.patches import Ellipse

    det = np.linalg.det(G)
    if normalize:
        det = det / det.max()
    image = ax.imshow(det, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi],
                      cmap="Blues", alpha=0.85, interpolation="bilinear")
    if ellipses:
        step = max(1, (len(axis) - 1) // 8)
        cells = [(i, j) for i in range(0, len(axis), step) for j in range(0, len(axis), step)]
        # Scale each panel to make its largest ellipse fill most of a grid cell.
        largest = max(np.sqrt(np.linalg.eigvalsh(np.linalg.inv(G[i, j]))[-1]) for i, j in cells)
        scale = 0.42 * (axis[step] - axis[0]) / largest
        for i, j in cells:
            vals, vecs = np.linalg.eigh(np.linalg.inv(G[i, j]))
            a, b = np.sqrt(vals[1]) * scale, np.sqrt(vals[0]) * scale
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            ax.add_patch(Ellipse((axis[j], axis[i]), 2 * a, 2 * b, angle=angle, fill=False,
                                 color=INK, lw=1.3))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal")
    ax.set_xticks(*TICKS)
    ax.set_yticks(*TICKS)
    ax.set_xlabel(r"shoulder $q_1$")
    ax.set_title(title, loc="left")
    return image


def _save(fig, name):
    fig.savefig(figure_path("tutorials", PAGE, name), bbox_inches="tight", pad_inches=0.1)


def metric_figures(example):
    # A row of three panels is 11 in wide at 14 pt, and the page shows it at the width of the
    # content column, where the labels match the body text. The single panel has the size of
    # one panel of a row.
    plt = matplotlib_style()
    axis, G = _tensors(example)
    fig, ax = plt.subplots(figsize=(4.9, 4.0), layout="constrained")
    image = _panel(ax, axis, G, "kinetic energy metric")
    ax.set_ylabel(r"elbow $q_2$")
    fig.colorbar(image, ax=ax, shrink=0.85, label=r"$\det M(q)$")
    _save(fig, "ke_metric.svg")
    plt.close(fig)

    # Three energies in one row with one color bar. Each panel divides the determinant by its
    # own largest value.
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), layout="constrained")
    for k, (ax, factor) in enumerate(zip(axes, (1.2, 2.0, 5.0))):
        axis, G = _tensors(example, factor)
        image = _panel(ax, axis, G, rf"$H = {factor:g}\,P_{{\max}}$", normalize=True)
        image.set_clim(0.0, 1.0)
        if k == 0:
            ax.set_ylabel(r"elbow $q_2$")
    fig.colorbar(image, ax=axes, shrink=0.85, label=r"$\det G(q) \,/\, \max\, \det G$")
    _save(fig, "jacobi_combined.svg")
    plt.close(fig)


def _spaces(example):
    import geodex

    mass = example.PlanarArmMassMatrix()
    base = geodex.Euclidean(2)
    base.set_sampling_bounds(np.array([-np.pi] * 2), np.array([np.pi] * 2))
    H = 1.2 * 9.81 * (1.0 * 0.5 + 1.0 * (1.0 + 0.5))
    return {
        "Euclidean": base,
        "Kinetic energy": geodex.ConfigurationSpace(base, geodex.KineticEnergyMetric(mass)),
        "Jacobi": geodex.ConfigurationSpace(
            base, geodex.JacobiMetric(mass, example.potential, H)),
    }


def _curve(space, path, label):
    """Points along a path. Raw edges under a curved metric follow the discrete geodesics
    the planner checked. Everything else follows the manifold's geodesic."""
    import geodex

    path = [np.asarray(p) for p in path]
    points = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        if label != "Euclidean":
            walk = geodex.discrete_geodesic(space, a, b, geodex.InterpolationSettings())
            if str(walk.status).endswith("Converged"):
                points += [np.asarray(p) for p in walk.path[1:]]
                points[-1] = b
                continue
        points += [space.geodesic(a, b, t) for t in np.linspace(0, 1, 20)[1:]]
    return np.array(points)


def planning_figure(example, data):
    plt = matplotlib_style()
    spaces = _spaces(example)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.6), layout="constrained")
    for k, (ax, run) in enumerate(zip(axes, data["runs"])):
        label = run["label"]
        factor = None if label != "Jacobi" else 1.2
        axis, G = _tensors(example, factor, n=61)
        if label == "Euclidean":
            G = np.broadcast_to(np.eye(2), G.shape)
        _panel(ax, axis, G, f"{label}, {run['cost']:.3f}", ellipses=False)
        raw = _curve(spaces[label], run["raw_path"], label)
        smooth = np.array(run["path"])
        ax.plot(raw[:, 0], raw[:, 1], color=INK_2, lw=1.8, ls=(0, (4, 3)), label="planner")
        ax.plot(smooth[:, 0], smooth[:, 1], color=BLUE, lw=3.0, label="returned")
        ax.plot(*data["start"], "o", color=AQUA, ms=11, mec="white", mew=1.5, zorder=5)
        ax.plot(*data["goal"], "o", color=ORANGE, ms=11, mec="white", mew=1.5, zorder=5)
        if k == 0:
            ax.set_ylabel(r"elbow $q_2$")
    from matplotlib.lines import Line2D

    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], ls="", marker="o", color=AQUA, ms=11, mec="white"),
                Line2D([], [], ls="", marker="o", color=ORANGE, ms=11, mec="white")]
    labels += ["start", "goal"]
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    _save(fig, "planning_result.svg")
    plt.close(fig)


def _arm_points(q):
    elbow = np.array([np.cos(q[0]), np.sin(q[0])])
    hand = elbow + np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
    return np.array([[0.0, 0.0], elbow, hand])


def arm_animation(example, data, frames=61):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    spaces = _spaces(example)
    runs = data["runs"]
    samples = []
    for run in runs:
        space = spaces[run["label"]]
        path = [np.asarray(p) for p in run["path"]]
        lengths = np.concatenate([[0.0], np.cumsum(
            [space.distance(a, b) for a, b in zip(path[:-1], path[1:])])])
        qs = []
        for s in np.linspace(0.0, lengths[-1], frames):
            k = min(np.searchsorted(lengths, s, side="right") - 1, len(path) - 2)
            t = (s - lengths[k]) / max(lengths[k + 1] - lengths[k], 1e-12)
            qs.append(space.geodesic(path[k], path[k + 1], t))
        samples.append(np.array(qs))

    fig = make_subplots(rows=1, cols=3, subplot_titles=[r["label"] for r in runs],
                        horizontal_spacing=0.04)

    def traces(f):
        out = []
        for c, (run, qs) in enumerate(zip(runs, samples)):
            color = COLORS[run["label"]] if run["label"] != "Euclidean" else INK
            hands = np.array([_arm_points(q)[2] for q in qs[:f + 1]])
            arm = _arm_points(qs[f])
            out.append(go.Scatter(x=hands[:, 0], y=hands[:, 1], mode="lines",
                                  line=dict(color=color, width=2, dash="dot"),
                                  hoverinfo="skip", showlegend=False, xaxis=f"x{c + 1}",
                                  yaxis=f"y{c + 1}"))
            out.append(go.Scatter(x=arm[:, 0], y=arm[:, 1], mode="lines+markers",
                                  line=dict(color=color, width=9),
                                  marker=dict(size=[16, 12, 10], color="white",
                                              line=dict(color=color, width=3)),
                                  hovertemplate=(f"{run['label']}<br>q = ({qs[f][0]:.2f}, "
                                                 f"{qs[f][1]:.2f})<extra></extra>"),
                                  showlegend=False, xaxis=f"x{c + 1}", yaxis=f"y{c + 1}"))
        return out

    # The page opens on the last frame, with the whole hand trace drawn.
    for trace in traces(frames - 1):
        fig.add_trace(trace)
    fig.frames = [go.Frame(data=traces(f), name=str(f)) for f in range(frames)]
    steps = [dict(method="animate", label=f"{f / (frames - 1):.2f}",
                  args=[[str(f)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0))])
             for f in range(frames)]
    fig.update_layout(
        font=dict(family="Lato, Helvetica, Arial, sans-serif", size=15, color=INK),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", margin=dict(l=20, r=20, t=50, b=20),
        sliders=[dict(active=frames - 1, steps=steps, x=0.12, len=0.86, y=0.02,
                      currentvalue=dict(prefix="fraction of path length ", font=dict(size=14, color=INK)),
                      font=dict(color="rgba(0,0,0,0)"), ticklen=0, pad=dict(t=30))],
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0.02, xanchor="left",
                          yanchor="top", pad=dict(t=30, r=10), buttons=[
                              dict(label="Play", method="animate",
                                   args=[[str(f) for f in range(frames)],
                                         dict(frame=dict(duration=60, redraw=False),
                                              mode="immediate",
                                              transition=dict(duration=0))])])])
    for c in range(1, 4):
        fig.update_xaxes(range=[-2.2, 2.2], visible=False, constrain="domain", row=1, col=c)
        fig.update_yaxes(range=[-2.2, 2.2], visible=False, scaleanchor=f"x{c}",
                         constrain="domain", row=1, col=c)
    fig.update_annotations(font=dict(size=16, color=INK_2))
    write_plotly(fig, "minimum-energy-arm", height=520)


def generate(parts=("metrics", "planning", "arm")):
    example = _example()
    data = run_example("tutorials/minimum_energy_planning")
    if "metrics" in parts:
        metric_figures(example)
    if "planning" in parts:
        planning_figure(example, data)
    if "arm" in parts:
        arm_animation(example, data)

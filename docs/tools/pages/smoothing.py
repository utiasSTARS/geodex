"""Assets of docs/concepts/smoothing.rst.

Every figure draws the output of examples/concepts/smoothing.py, the path around two discs in
the plane and the differential-drive plan on SE(2).

``smoothing-example``
    The input path and the returned path with its waypoints.
``smoothing-shortcut``
    Animation of the first shortcut round on the input path. Each step tests the shortcut that
    saves the most length, and the first valid one replaces the waypoints between its ends.
    The steps replay the smoother's best-first rule with the example's validity function.
``smoothing-shorten``
    The input path and the shortened path, before corner rounding.
``smoothing-corners``
    The rounding curve of a corner and its control points, and a cusp that stays a corner.
``smoothing-shrink``
    A corner next to an obstacle. The full curve and the curve of half its size enter the
    obstacle, the curve of a quarter of the full size passes, and the smoother keeps the
    largest size that passes between a quarter and a half.
``smoothing-check``
    Edges of the shortened path next to a disc and the validity samples along them.
``smoothing-spacing``
    The waypoints over the first disc at the default settings, with a looser corner tolerance
    and with a shorter output spacing.
``smoothing-in-plan``
    The differential-drive plan, raw and smoothed, with arrows for the heading.
"""

from __future__ import annotations

import os

import numpy as np

from common import densify, run_example, video_paths
from plots import write_plotly
from style import (AQUA, BLUE, GRID, INK, INK_2, OBSTACLE, OBSTACLE_EDGE, ORANGE,
                   matplotlib_style, plotly_layout)

DISCS = [(1.0, 0.2, 0.5), (2.5, -0.3, 0.6)]
INPUT = np.array([(0.0, 0.0), (0.2, 1.0), (1.8, 1.0), (1.8, -1.2), (3.4, -1.1), (4.0, 0.0)])
RESOLUTION = 0.01  # collision_check_resolution of the example
FULL = dict(x=[-0.6, 4.6], y=[-1.7, 1.5])


def _discs(xref="x", yref="y"):
    return [dict(type="circle", xref=xref, yref=yref, x0=cx - r, x1=cx + r, y0=cy - r,
                 y1=cy + r, fillcolor=OBSTACLE, opacity=0.55, line=dict(color=OBSTACLE_EDGE))
            for cx, cy, r in DISCS]


def _clearance(q):
    return min(np.hypot(q[0] - cx, q[1] - cy) - r for cx, cy, r in DISCS)


def _path(go, path, name, color, width=3, size=6, dash=None, mode="lines+markers", group=None):
    p = np.asarray(path)
    return go.Scatter(x=p[:, 0], y=p[:, 1], mode=mode, name=name, legendgroup=group or name,
                      line=dict(color=color, width=width, dash=dash),
                      marker=dict(size=size, color=color))


def _ends(go, path):
    p = np.asarray(path)
    return go.Scatter(x=p[[0, -1], 0], y=p[[0, -1], 1], mode="markers",
                      marker=dict(size=13, color=[AQUA, ORANGE], line=dict(color="#ffffff", width=2)),
                      hovertext=["start", "goal"], hoverinfo="text", showlegend=False)


def _axes(fig, view, cols=0):
    """Equal-aspect axes over `view`, on each of `cols` subplots or on the one figure."""
    style = dict(gridcolor=GRID, linecolor=INK_2, ticks="outside", constrain="domain")
    if not cols:
        fig.update_xaxes(range=view["x"], **style)
        fig.update_yaxes(range=view["y"], scaleanchor="x", **style)
        return
    for c in range(1, cols + 1):
        fig.update_xaxes(range=view["x"], row=1, col=c, **style)
        fig.update_yaxes(range=view["y"], scaleanchor="x" if c == 1 else f"x{c}", row=1, col=c,
                         **style)


def _layout(fig, shapes, height, name, top=20):
    fig.update_layout(**plotly_layout(
        shapes=shapes, legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14),
        margin=dict(l=40, r=20, t=top, b=70), hovermode="closest"))
    fig.update_annotations(font=dict(size=15, color=INK_2))
    write_plotly(fig, name, height=height)


def _window(center, half):
    return dict(x=[center[0] - half, center[0] + half], y=[center[1] - half, center[1] + half])


def overview(go, data):
    fig = go.Figure()
    fig.add_trace(_path(go, INPUT, "input path", INK_2, width=2, size=8, dash="dot"))
    fig.add_trace(_path(go, data["standalone"]["path"], "returned path", BLUE, width=2, size=4))
    fig.add_trace(_ends(go, INPUT))
    _axes(fig, FULL)
    _layout(fig, _discs(), 420, "smoothing-example")


def _edge_clear(a, b):
    """The edge check of the example, samples no more than RESOLUTION apart."""
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / RESOLUTION - 1e-9)))
    return all(_clearance(a + (b - a) * k / n) > 0.0 for k in range(n + 1))


def _best_first(path):
    """The steps of one best-first shortcut round, as (path, chord, accepted, saving).

    The candidate shortcuts are sorted by the length they save. The first valid one replaces
    the waypoints between its ends, a shortcut found blocked is not tested again, and the
    round ends when no valid shortcut is left.
    """
    path = [np.asarray(p, dtype=float) for p in path]
    ids = list(range(len(path)))
    blocked, steps = set(), []
    while len(path) > 2:
        prefix = np.concatenate([[0.0], np.cumsum([np.linalg.norm(b - a)
                                                    for a, b in zip(path[:-1], path[1:])])])
        cands = sorted(((prefix[j] - prefix[i] - np.linalg.norm(path[j] - path[i]), i, j)
                        for i in range(len(path)) for j in range(i + 2, len(path))
                        if (ids[i], ids[j]) not in blocked), reverse=True)
        taken = False
        for saving, i, j in cands:
            if saving <= 0.0:
                break
            ok = _edge_clear(path[i], path[j])
            steps.append((list(path), (path[i], path[j]), ok, saving))
            if not ok:
                blocked.add((ids[i], ids[j]))
                continue
            del path[i + 1:j], ids[i + 1:j]
            taken = True
            break
        if not taken:
            break
    steps.append((list(path), None, False, 0.0))
    return steps


def shortcut_animation():
    """The first shortcut round on the input path, one step per tested shortcut."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.patches import Circle

    plt = matplotlib_style()
    plt.rcParams["animation.ffmpeg_path"] = os.environ.get("GEODEX_FFMPEG", "ffmpeg")
    fps, dpi = 30, 120
    steps = _best_first(INPUT)
    hold = (int(0.9 * fps), int(1.3 * fps))  # frames of a blocked and of a taken shortcut
    timeline = []  # (step, phase) per frame, phase 0 shows the chord and 1 the new path
    for k, (_, chord, ok, _) in enumerate(steps):
        if chord is None:
            timeline += [(k, 1)] * (2 * fps)
            continue
        timeline += [(k, 0)] * hold[ok]
        if ok:
            timeline += [(k, 1)] * int(0.5 * fps)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03)
    ax.set_xlim(*FULL["x"])
    ax.set_ylim(*FULL["y"])
    ax.set_aspect("equal")
    ax.axis("off")
    for cx, cy, r in DISCS:
        ax.add_patch(Circle((cx, cy), r, facecolor=OBSTACLE, alpha=0.55, edgecolor=OBSTACLE_EDGE))
    ax.plot(INPUT[:, 0], INPUT[:, 1], ":", color=INK_2, lw=1.6, zorder=2)
    (current,) = ax.plot([], [], "-o", color=BLUE, lw=2.6, ms=7, zorder=4)
    (chord,) = ax.plot([], [], "--", lw=2.8, zorder=5)
    for point, color in ((INPUT[0], AQUA), (INPUT[-1], ORANGE)):
        ax.plot(*point, "o", ms=13, color=color, mec="#ffffff", mew=2, zorder=6)
    title = ax.set_title("", fontsize=15, color=INK)

    def draw(f):
        k, phase = timeline[f]
        path, ends, ok, saving = steps[k]
        if phase == 1 and ends is not None:
            path = steps[k + 1][0]
        p = np.asarray(path)
        current.set_data(p[:, 0], p[:, 1])
        if ends is None:
            chord.set_data([], [])
            title.set_text("no valid shortcut is left")
        elif phase == 0:
            chord.set_data([ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]])
            chord.set_color(AQUA if ok else ORANGE)
            title.set_text(f"a shortcut that saves {saving:.2f} "
                           + ("is valid and taken" if ok else "crosses a disc"))
        else:
            chord.set_data([], [])
            title.set_text("the shortcut replaces the waypoints between its ends")
        return current, chord, title

    webm, mp4, poster = video_paths("smoothing-shortcut")
    draw(len(timeline) - 1)
    fig.savefig(poster, dpi=dpi)
    anim = FuncAnimation(fig, draw, frames=len(timeline), interval=1000 / fps, blit=False)
    anim.save(str(webm), writer=FFMpegWriter(
        fps=fps, codec="libvpx-vp9",
        extra_args=["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-pix_fmt", "yuv420p",
                    "-row-mt", "1"]), dpi=dpi)
    anim.save(str(mp4), writer=FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-crf", "20", "-preset", "slow", "-tune", "animation", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"]), dpi=dpi)
    plt.close(fig)


def shorten(go, data):
    fig = go.Figure()
    fig.add_trace(_path(go, INPUT, "input path", INK_2, width=2, size=8, dash="dot"))
    fig.add_trace(_path(go, data["corners"]["sharp"]["path"], "shortened, rounding off", ORANGE,
                        width=2, size=4))
    fig.add_trace(_ends(go, INPUT))
    _axes(fig, FULL)
    _layout(fig, _discs(), 420, "smoothing-shorten")


def _quintic(p, q, f, g, t):
    """The rounding curve of a corner at the origin with edge vectors p and q."""
    from math import comb

    control = [f * p, 2 / 3 * f * p, 1 / 3 * f * p, 1 / 3 * g * q, 2 / 3 * g * q, g * q]
    t = np.asarray(t)[:, None]
    return sum(comb(5, i) * t**i * (1 - t)**(5 - i) * control[i] for i in range(6)), control


def corners(go, data):
    """A corner of 60 degrees with its rounding curve, and a reversal that stays a corner."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, subplot_titles=(
        "a 60 degree turn, rounded", "a 150 degree turn, a cusp that stays"))
    for col, turn in ((1, np.radians(60.0)), (2, np.radians(150.0))):
        a = np.array([-1.0, 0.0])
        b = np.array([np.cos(turn), np.sin(turn)])
        fig.add_trace(go.Scatter(x=[a[0], 0.0, b[0]], y=[a[1], 0.0, b[1]], mode="lines+markers",
                                 name="edges", legendgroup="edges", showlegend=col == 1,
                                 line=dict(color=INK_2, width=2, dash="dot"),
                                 marker=dict(size=8, color=INK_2)), row=1, col=col)
        if col == 1:
            curve, control = _quintic(a, b, 0.6, 0.6, np.linspace(0.0, 1.0, 120))
            fig.add_trace(go.Scatter(x=curve[:, 0], y=curve[:, 1], mode="lines",
                                     name="rounding curve", line=dict(color=BLUE, width=3)),
                          row=1, col=1)
            ctrl = np.asarray(control)
            fig.add_trace(go.Scatter(x=ctrl[:, 0], y=ctrl[:, 1], mode="markers+text",
                                     name="control points", text=[f"P{i}" for i in range(6)],
                                     textposition="bottom center",
                                     marker=dict(size=9, color=ORANGE)), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[0.0], y=[0.0], mode="markers", name="corner that stays",
                                     marker=dict(size=12, color=ORANGE, symbol="diamond")),
                          row=1, col=2)
    _axes(fig, dict(x=[-1.15, 1.0], y=[-0.45, 1.15]), cols=2)
    _layout(fig, [], 420, "smoothing-corners", top=50)


def _size_search(passes, size, shrinks=5, bisections=2):
    """The curve sizes the smoother tests, as (size, passed), and the size it keeps.

    A failing size shrinks by a factor of two. After a size passes, the smoother tests the
    geometric mean of the passing size and the last failing one, twice, and keeps the largest
    size that passes.
    """
    tried = []
    for attempt in range(shrinks + 1):
        ok = passes(size)
        tried.append((size, ok))
        if not ok:
            size *= 0.5
            continue
        lo, hi = size, 2.0 * size
        for _ in range(bisections if attempt > 0 else 0):
            mid = np.sqrt(lo * hi)
            ok = passes(mid)
            tried.append((mid, ok))
            lo, hi = (mid, hi) if ok else (lo, mid)
        return tried, lo
    return tried, None


def shrink(go):
    """A corner next to an obstacle and the curve sizes the smoother tests."""
    a, b = np.array([-1.0, 0.0]), np.array([np.cos(np.radians(70.0)), np.sin(np.radians(70.0))])
    cx, cy, r = (-0.086, 0.123, 0.065)  # on the bisector inside the corner, clear of both edges
    t = np.linspace(0.0, 1.0, 400)

    def curve(size):
        return _quintic(a, b, size, size, t)[0]

    def passes(size):
        c = curve(size)
        return bool(np.all(np.hypot(c[:, 0] - cx, c[:, 1] - cy) >= r))

    full = 0.7
    tried, kept = _size_search(passes, full)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[a[0], 0.0, b[0]], y=[a[1], 0.0, b[1]], mode="lines+markers",
                             name="edges", line=dict(color=INK_2, width=2, dash="dot"),
                             marker=dict(size=8, color=INK_2)))
    shown = [(full, "full size, enters the obstacle", ORANGE, "dash"),
             (full / 2, "half size, enters the obstacle", ORANGE, "dot"),
             (full / 4, "quarter size, passes", BLUE, "dot"),
             (kept, f"kept, {kept / full:.2f} of the full size", BLUE, None)]
    for size, label, color, dash in shown:
        c = curve(size)
        fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode="lines", name=label,
                                 line=dict(color=color, width=3, dash=dash)))
    shapes = [dict(type="circle", xref="x", yref="y", x0=cx - r, x1=cx + r, y0=cy - r,
                   y1=cy + r, fillcolor=OBSTACLE, opacity=0.55, line=dict(color=OBSTACLE_EDGE))]
    _axes(fig, dict(x=[-0.8, 0.45], y=[-0.12, 0.75]))
    _layout(fig, shapes, 440, "smoothing-shrink")


def check(go, data):
    # With rounding off, the returned path is the checked path.
    path = np.asarray(data["corners"]["sharp"]["path"])
    closest = path[int(np.argmin([_clearance(q) for q in path]))]
    xs, ys = [], []
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / RESOLUTION - 1e-9)))
        for j in range(1, n):
            q = a + (b - a) * j / n
            xs.append(q[0])
            ys.append(q[1])
    fig = go.Figure()
    fig.add_trace(_path(go, path, "shortened path and its waypoints", BLUE, width=2, size=7))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="validity samples",
                             marker=dict(size=4, color=INK_2, symbol="x-thin",
                                         line=dict(width=1, color=INK_2))))
    _axes(fig, _window(closest, 0.08))
    _layout(fig, _discs(), 440, "smoothing-check")


def spacing(go, data):
    from plotly.subplots import make_subplots

    runs = [("default", data["standalone"]["path"]),
            ("corner_tolerance = 0.001", data["spacing"]["coarse"]["path"]),
            ("output_spacing = 0.01", data["spacing"]["limited"]["path"])]
    titles = []
    for name, path in runs:
        p = np.asarray(path)
        step = np.linalg.norm(np.diff(p, axis=0), axis=1).max()
        titles.append(f"{name}<br>{len(p)} waypoints, step {step:.3f}")
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.04, subplot_titles=titles)
    for c, (name, path) in enumerate(runs, start=1):
        fig.add_trace(_path(go, path, name, (BLUE, AQUA, ORANGE)[c - 1], width=1.5, size=6),
                      row=1, col=c)
    window = _window((1.05, 0.68), 0.15)
    _axes(fig, window, cols=3)
    shapes = _discs("x", "y") + _discs("x2", "y2") + _discs("x3", "y3")
    _layout(fig, shapes, 460, "smoothing-spacing", top=70)


def _heading_arrows(path, every, color):
    """Arrows along `path` that point in the heading of every `every`-th pose."""
    return [dict(x=x + 0.25 * np.cos(th), y=y + 0.25 * np.sin(th), ax=x, ay=y, xref="x",
                 yref="y", axref="x", ayref="y", text="", showarrow=True, arrowhead=2,
                 arrowsize=1.2, arrowwidth=2, arrowcolor=color)
            for x, y, th in path[::every]]


def in_plan(go, data):
    import geodex

    se2 = geodex.SE2(wx=1.0, wy=10.0, wtheta=1.0, x_lo=-1.0, x_hi=5.0, y_lo=-2.0, y_hi=2.0)
    plan = data["in_plan"]
    raw = densify(se2, plan["raw_path"], step=0.05)
    smooth = densify(se2, plan["path"], step=0.05)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw[:, 0], y=raw[:, 1], mode="lines", name="planner's path",
                             line=dict(color=INK_2, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=smooth[:, 0], y=smooth[:, 1], mode="lines", name="returned path",
                             legendgroup="smooth", line=dict(color=BLUE, width=3)))
    fig.add_trace(_ends(go, raw[:, :2]))
    fig.update_layout(annotations=_heading_arrows(smooth, max(1, len(smooth) // 12), BLUE))
    _axes(fig, FULL)
    _layout(fig, _discs(), 420, "smoothing-in-plan")


def generate(parts=("example", "shortcut")):
    import plotly.graph_objects as go

    if "example" in parts:
        data = run_example("concepts/smoothing")
        overview(go, data)
        shorten(go, data)
        corners(go, data)
        shrink(go)
        check(go, data)
        spacing(go, data)
        in_plan(go, data)
    if "shortcut" in parts:
        shortcut_animation()

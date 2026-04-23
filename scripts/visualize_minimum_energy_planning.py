#!/usr/bin/env python3
"""Visualize RRT* planning under flat, KE, and Jacobi metrics.

Reads a JSON file produced by minimum_energy_planning and writes an SVG figure
with three panels showing the RRT* tree and solution path for each metric,
overlaid on the respective metric tensor determinant.

Usage:
  python scripts/visualize_minimum_energy_planning.py minimum_energy_planning.json \
      --output-dir docs/tutorials/figs/minimum-energy-planning
"""

import argparse
import json
import math
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PatchCollection

matplotlib.use("Agg")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

pink_cmap = mcolors.ListedColormap(plt.get_cmap("pink")(np.linspace(0.20, 1, 256)))


# ---------------------------------------------------------------------------
# Arm physics (recomputed in Python from embedded JSON parameters)
# ---------------------------------------------------------------------------

def make_mass_matrix_fn(arm):
    """Return a vectorized mass matrix function from arm parameters."""
    l1, lc1, lc2 = arm["l1"], arm["lc1"], arm["lc2"]
    m1, m2 = arm["m1"], arm["m2"]
    I1, I2 = arm["I1"], arm["I2"]

    def mass_matrix(q1, q2):
        """Return (M00, M01, M11) arrays evaluated on a grid."""
        c2 = np.cos(q2)
        h = l1 * lc2 * c2
        M00 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * h)
        M01 = I2 + m2 * (lc2**2 + h)
        M11 = I2 + m2 * lc2**2
        return M00, M01, M11

    return mass_matrix


def compute_potential(q1, q2, arm):
    """Compute gravitational potential P(q) on a grid."""
    g = arm["g"]
    m1, m2 = arm["m1"], arm["m2"]
    l1, lc1, lc2 = arm["l1"], arm["lc1"], arm["lc2"]
    return (m1 * g * lc1 * np.sin(q1)
            + m2 * g * (l1 * np.sin(q1) + lc2 * np.sin(q1 + q2)))


def compute_backgrounds(arm, H, n=80):
    """Compute metric determinant backgrounds on an n x n grid in [-pi, pi].

    Returns (q1, q2, det_flat, det_ke, det_jacobi) where each det is (n, n).
    """
    q1 = np.linspace(-math.pi, math.pi, n)
    q2 = np.linspace(-math.pi, math.pi, n)
    Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")  # shape (n, n)

    mass_fn = make_mass_matrix_fn(arm)
    M00, M01, M11 = mass_fn(Q1, Q2)

    # det(M) = M00*M11 - M01^2
    det_M = M00 * M11 - M01**2

    # Flat: det(I) = 1 everywhere
    det_flat = np.ones_like(det_M)

    # Jacobi: det(J) = [2(H - P)]^2 * det(M)
    P = compute_potential(Q1, Q2, arm)
    conformal = 2.0 * (H - P)
    det_J = conformal**2 * det_M

    return q1, q2, det_flat, det_M, det_J


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pi_ticks():
    """Tick positions and labels for [-pi, pi]."""
    positions = [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]
    labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    return positions, labels


def _finalize_axes(ax, title=None, show_ylabel=True, fontsize=None):
    ticks, labels = _pi_ticks()
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xlabel(r"$q_1$ \textrm{[rad]}", fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel(r"$q_2$ \textrm{[rad]}", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot_planning(data, output_path):
    runs = data["runs"]
    arm = data["arm"]
    H = data["H"]
    start = data["start"]
    goal = data["goal"]

    # Compute metric backgrounds
    q1, q2, det_flat, det_ke, det_jacobi = compute_backgrounds(arm, H)
    backgrounds = [det_flat, det_ke, det_jacobi]
    extent = [-math.pi, math.pi, -math.pi, math.pi]

    titles = [r"\textbf{Euclidean}", r"\textbf{Kinetic Energy}", r"\textbf{Jacobi}"]

    fig, axes = plt.subplots(
        1, 3, figsize=(16.5, 5.5),
        gridspec_kw={"wspace": 0.15},
    )

    for idx, (ax, run, bg, title) in enumerate(zip(axes, runs, backgrounds, titles)):
        # Normalize background independently to [0, 1]
        bg_min, bg_max = bg.min(), bg.max()
        if bg_max > bg_min:
            bg_norm = (bg - bg_min) / (bg_max - bg_min)
        else:
            bg_norm = np.ones_like(bg)

        ax.imshow(
            bg_norm.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=pink_cmap,
            interpolation="bilinear",
            vmin=0.0,
            vmax=1.0,
        )

        # Tree edges
        if "tree" in run:
            segments = [[(e[0][0], e[0][1]), (e[1][0], e[1][1])]
                        for e in run["tree"]]
            lc = LineCollection(
                segments, colors="#2e3236", linewidths=0.15, alpha=0.2, zorder=1
            )
            ax.add_collection(lc)

        # Raw planner path (dashed) vs smoothed path (solid).
        # Falls back to legacy "path" field for backward compatibility.
        raw = run.get("raw_path") or run.get("path") or []
        smoothed = run.get("smoothed_path") or []

        if raw:
            xs = [p[0] for p in raw]
            ys = [p[1] for p in raw]
            ax.plot(
                xs, ys,
                color="#1f77b4", linewidth=1.5, linestyle="--",
                alpha=0.7, zorder=3, label="raw",
            )

        # if smoothed:
        #     xs = [p[0] for p in smoothed]
        #     ys = [p[1] for p in smoothed]
        #     ax.plot(
        #         xs, ys,
        #         color="#d62728", linewidth=2.5, zorder=4, label="smoothed",
        #     )

        # ax.legend(loc="lower right", fontsize=11, framealpha=0.85)

        # Start and goal markers
        ax.plot(start[0], start[1], "o", color="#2ca02c", markersize=9, zorder=5)
        ax.plot(goal[0], goal[1], "*", color="#ff7f0e", markersize=13, zorder=5)

        _finalize_axes(ax, title=title, show_ylabel=(idx == 0), fontsize=16)

    fig.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# Two-link arm animation
# ---------------------------------------------------------------------------

ARM_LINK_COLOR = "#3a5988"
ARM_BASE_COLOR = "#2d4675"
ARM_PATH_COLOR = "#ffb32a"


def forward_kinematics(q1, q2, l1, l2):
    """Return (shoulder, elbow, end-effector) Cartesian positions."""
    sh = np.array([0.0, 0.0])
    el = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    ee = el + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    return sh, el, ee


def _capsule_link(q_abs, length, base_xy, half_width):
    """Return vertices of a horizontal capsule of given length and half-width
    (rounded-rectangle outline), rotated by ``q_abs`` and anchored at ``base_xy``."""
    nb = 30
    t1 = np.linspace(0.0, -np.pi, nb // 2)
    t2 = np.linspace(np.pi, 0.0, nb // 2)
    x = np.concatenate((half_width * np.sin(t1), length + half_width * np.sin(t2)))
    y = np.concatenate((np.cos(t1), np.cos(t2))) * half_width
    xy = np.vstack((x, y))
    R = np.array([[np.cos(q_abs), -np.sin(q_abs)],
                  [np.sin(q_abs), np.cos(q_abs)]])
    return (R @ xy).T + np.asarray(base_xy)


def _base_wedge_verts(base_xy, size):
    """Vertices for a half-circle-on-a-stub wedge that reads as a ground-bolted base."""
    nb = 30
    sz = size * 1.2
    t = np.linspace(0.0, np.pi, nb - 2)
    x = np.concatenate(([1.5], 1.5 * np.cos(t), [-1.5])) * sz
    y = np.concatenate(([-1.2], 1.5 * np.sin(t), [-1.2])) * sz
    return np.c_[x, y] + np.asarray(base_xy)


def draw_arm(ax, q1, q2, l1, l2, *, alpha=1.0, half_width=0.22,
             joint_radius=0.095, link_color=ARM_LINK_COLOR, zorder=5):
    """Draw one arm pose: two capsule-polygon links with a thin white outline,
    plus three joint pins drawn navy-filled with a thicker white outline.
    ``half_width`` sets the link thickness in data units; ``joint_radius`` is smaller than
    ``half_width`` so the joint reads as a small blue disc ringed by white
    against the link body."""
    sh = np.array([0.0, 0.0])
    # Link 1: horizontal capsule rotated by q1, anchored at shoulder.
    verts1 = _capsule_link(q1, l1, sh, half_width)
    ax.add_patch(mpatches.Polygon(
        verts1, closed=True,
        facecolor=link_color, edgecolor="white",
        linewidth=2.0, alpha=alpha, zorder=zorder,
    ))

    # Link 2: horizontal capsule rotated by q1+q2, anchored at elbow.
    el = sh + np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    verts2 = _capsule_link(q1 + q2, l2, el, half_width)
    ax.add_patch(mpatches.Polygon(
        verts2, closed=True,
        facecolor=link_color, edgecolor="white",
        linewidth=2.0, alpha=alpha, zorder=zorder,
    ))

    # Joint pins: solid navy disc with a thick white stroke
    ee = el + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    for pt in (sh, el, ee):
        ax.add_patch(mpatches.Circle(
            pt, joint_radius, facecolor=link_color,
            edgecolor="white", linewidth=3.0,
            alpha=alpha, zorder=zorder + 1))


def draw_base(ax, *, size=0.20, color=ARM_BASE_COLOR, zorder=8):
    """Wedge-shaped base at the origin, a dark navy mount with a white outline"""
    verts = _base_wedge_verts((0.0, 0.0), size)
    ax.add_patch(mpatches.Polygon(
        verts, closed=True,
        facecolor=color, edgecolor="white",
        linewidth=2.0, zorder=zorder,
    ))


def _densify_path(path, step=0.03):
    """Linearly densify a waypoint list so the arm moves smoothly between
    frames. Waypoints coming out of RRT* are typically too sparse for a
    visually continuous animation."""
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        seg_len = np.linalg.norm(b - a)
        n = max(2, int(np.ceil(seg_len / step)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.asarray(out)


def _resample_path(pts, n):
    """Resample a polyline to ``n`` points spaced uniformly by arc length.
    Lets us play paths of different lengths side-by-side in sync."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return np.tile(pts, (n, 1))
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = arc[-1]
    if total == 0:
        return np.tile(pts[0], (n, 1))
    target = np.linspace(0.0, total, n)
    out = np.empty((n, pts.shape[1]))
    for d in range(pts.shape[1]):
        out[:, d] = np.interp(target, arc, pts[:, d])
    return out


def animate_arm(data, output_path, *, fps=15,
                ghost_count=6, ghost_stride_frac=0.11,
                panel_size=(3.2, 3.2), dpi=110, pad=0.08,
                n_frames=150, pause_frames=30,
                bg_color="#f7fbfe",
                titles=(r"\textbf{Euclidean}",
                        r"\textbf{Kinetic Energy}",
                        r"\textbf{Jacobi}")):
    """Write a GIF with three side-by-side panels — one arm per metric —
    animating in sync from start to goal, with a hold at the goal pose
    before the loop restarts.

    The visual style (dark-navy links, white joint pins, amber end-effector
    trace, faded ghosts of previous poses) matches the paper's ``two-link
    planar arm`` figure. Paths are resampled uniformly by arc length so all
    three arms finish at the same frame regardless of planner waypoint count.
    ``bg_color`` defaults to the landing-page card background so the GIF
    blends in when embedded in the docs.
    """
    arm = data["arm"]
    l1, l2 = arm["l1"], arm["l2"]
    runs = data["runs"]
    if len(runs) != len(titles):
        raise ValueError(
            f"Expected {len(titles)} runs (one per metric), got {len(runs)}.")

    paths = []
    for i, run in enumerate(runs):
        raw = run.get("smoothed_path") or run.get("raw_path") or run.get("path")
        if not raw:
            raise ValueError(f"Run {i} ({titles[i]}) has no path to animate.")
        dense = _densify_path(raw, step=0.03)
        paths.append(_resample_path(dense, n_frames))

    ees = [np.array([forward_kinematics(q1, q2, l1, l2)[2] for q1, q2 in p])
           for p in paths]

    # Shared view that encloses every joint (shoulder, elbow, end-effector)
    # of every arm across every frame, then expanded by link half-width +
    # joint radius + `pad` so thick links and the base wedge never clip.
    all_pts = []
    for path in paths:
        for q1, q2 in path:
            sh, el, ee = forward_kinematics(q1, q2, l1, l2)
            all_pts.extend([sh, el, ee])
    all_pts = np.asarray(all_pts)
    slack = 0.22 + 0.095 + pad  # link half-width + joint radius + user pad
    xmin = all_pts[:, 0].min() - slack
    xmax = all_pts[:, 0].max() + slack
    ymin = min(all_pts[:, 1].min(), -0.30) - slack  # include base wedge
    ymax = all_pts[:, 1].max() + slack
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    pw, ph = panel_size
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * pw, ph), dpi=dpi,
        gridspec_kw={"wspace": 0.0},
    )
    # Drop outer figure margins entirely so the three panels fill the frame.
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93)
    fig.patch.set_facecolor(bg_color)
    for ax in axes:
        ax.set_facecolor(bg_color)

    ghost_stride = max(1, int(np.ceil(n_frames * ghost_stride_frac)))

    def render_frame(frame):
        for ax, path, ee, title in zip(axes, paths, ees, titles):
            ax.clear()
            ax.set_facecolor(bg_color)
            ax.set_aspect("equal")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.axis("off")

            for g in range(1, ghost_count + 1):
                idx = frame - g * ghost_stride
                if idx < 0:
                    break
                alpha = max(0.12, 0.65 - 0.09 * g)
                q1g, q2g = path[idx]
                draw_arm(ax, q1g, q2g, l1, l2, alpha=alpha, zorder=5 - g)

            draw_base(ax)
            q1c, q2c = path[frame]
            draw_arm(ax, q1c, q2c, l1, l2, alpha=1.0, zorder=10)

            # Amber trace drawn last so it sits on top of every link and joint.
            ax.plot(ee[: frame + 1, 0], ee[: frame + 1, 1],
                    color=ARM_PATH_COLOR, linewidth=2.6,
                    solid_capstyle="round", zorder=20)

            ax.set_title(title, fontsize=13, pad=4)

    # Render each motion frame once, convert to PIL, then write a GIF with a
    # longer per-frame duration on the final pose. Doing it this way (rather
    # than duplicating the last frame through FuncAnimation) avoids
    # PillowWriter deduplicating identical trailing frames.
    from io import BytesIO
    import PIL.Image

    frame_ms = int(round(1000 / fps))
    pause_ms = frame_ms * max(1, pause_frames)
    durations = [frame_ms] * (n_frames - 1) + [pause_ms]

    images = []
    for f in range(n_frames):
        render_frame(f)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=bg_color)
        buf.seek(0)
        images.append(PIL.Image.open(buf).convert("RGBA").copy())
        buf.close()

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    plt.close(fig)
    print(f"Wrote {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize minimum-energy RRT* planning results."
    )
    parser.add_argument("json_file", help="Path to minimum_energy_planning.json")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for output files (default: .)"
    )
    parser.add_argument(
        "--no-svg", action="store_true",
        help="Skip the three-panel SVG comparison figure.",
    )
    parser.add_argument(
        "--no-gif", action="store_true",
        help="Skip the arm animation GIF.",
    )
    parser.add_argument(
        "--gif-fps", type=int, default=15,
        help="Frames per second for the arm GIF (default: 15).",
    )
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.no_svg:
        svg_path = os.path.join(args.output_dir, "planning_result.svg")
        plot_planning(data, svg_path)

    if not args.no_gif:
        gif_path = os.path.join(args.output_dir, "arm.gif")
        animate_arm(data, gif_path, fps=args.gif_fps)


if __name__ == "__main__":
    main()

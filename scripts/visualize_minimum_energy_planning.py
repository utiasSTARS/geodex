#!/usr/bin/env python3
"""Visualize RRT* planning under flat, KE, and Jacobi metrics.

Reads a JSON file produced by minimum_energy_planning and writes an SVG figure
with three panels showing the RRT* tree and solution path for each metric,
overlaid on the respective metric tensor determinant.

Usage:
  python scripts/visualize_minimum_energy_planning.py minimum_energy_planning.json \
      --output-dir docs/tutorials/figs
"""

import argparse
import json
import math
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

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
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize minimum-energy RRT* planning results."
    )
    parser.add_argument("json_file", help="Path to minimum_energy_planning.json")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for output SVG file (default: .)"
    )
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "minimum_energy_planning.svg")
    plot_planning(data, output_path)


if __name__ == "__main__":
    main()

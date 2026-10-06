#!/usr/bin/env python3
"""Draw the two figures of the sampling concept page, docs/concepts/sampling.rst.

- cube-to-manifold.svg, scrambled Halton points in the unit square and their image on the
  sphere under from_unit_cube.
- low-discrepancy-comparison.svg, 100 points of the pseudo-random, Halton and scrambled
  Halton samplers in the unit square.

pages/sampling.py runs it in the docs style.

Usage:
  pixi run python docs/tools/figures/visualize_sampling.py [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import geodex

# geodex docs style: Lato font, stixsans math, large figures.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "mathtext.fontset": "stixsans",
        "axes.linewidth": 1.2,
    }
)

# Shared palette, matched to the docs blue used by the graphviz/mermaid diagrams.
POINT_BLUE = "#2980b9"
DEEP_BLUE = "#1f5f8b"
SPHERE_FILL = "#e7f0fa"
N_POINTS = 256


def stack_samples(sampler, dim, n):
    """Collect ``n`` draws of a ``dim``-vector from a standalone sampler."""
    return np.array([sampler.sample(dim) for _ in range(n)])


def style_unit_square(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, fontsize=15)
    for spine in ax.spines.values():
        spine.set_color("#7f8c8d")


# ---------------------------------------------------------------------------
# Figure 1: unit cube to manifold
# ---------------------------------------------------------------------------

def make_cube_to_manifold(output):
    cube = stack_samples(geodex.ScrambledHaltonSampler(seed=0), 2, N_POINTS)

    sphere = geodex.Sphere()
    sphere.seed(0)
    pts = np.array([sphere.random_point() for _ in range(N_POINTS)])

    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.32,
                          left=0.06, right=0.97, bottom=0.1, top=0.9)

    # Left: the unit square the sampler actually fills.
    ax_left = fig.add_subplot(gs[0])
    ax_left.scatter(cube[:, 0], cube[:, 1], s=16, color=POINT_BLUE,
                    edgecolors="white", linewidths=0.3, zorder=3)
    style_unit_square(ax_left, r"Unit square $[0,1)^2$")

    # Right: the same coverage carried onto the 2-sphere by from_unit_cube.
    ax_right = fig.add_subplot(gs[1], projection="3d")
    elev, azim = 22, -58
    ax_right.view_init(elev=elev, azim=azim)
    el, az = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    # Light sphere surface.
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax_right.plot_surface(xs, ys, zs, color=SPHERE_FILL, alpha=0.18,
                          linewidth=0, antialiased=True, shade=True, zorder=1)

    # Faint meridians and parallels for a globe cue.
    for phi in np.linspace(0, np.pi, 7)[1:-1]:
        circ = np.linspace(0, 2 * np.pi, 120)
        ax_right.plot(np.cos(circ) * np.sin(phi), np.sin(circ) * np.sin(phi),
                      np.full_like(circ, np.cos(phi)), color="#b7c9d8", lw=0.5, zorder=2)
    for lam in np.linspace(0, 2 * np.pi, 9)[:-1]:
        arc = np.linspace(0, np.pi, 120)
        ax_right.plot(np.cos(lam) * np.sin(arc), np.sin(lam) * np.sin(arc),
                      np.cos(arc), color="#b7c9d8", lw=0.5, zorder=2)

    # Scatter the camera-facing hemisphere, lifted just proud of the surface.
    # A near-opaque surface hides the points because mplot3d does not depth-sort
    # a surface against a scatter, so the surface stays faint and the points sit
    # at radius slightly above 1.
    front = pts @ view > 0.0
    fp = pts[front] * 1.02
    ax_right.scatter(fp[:, 0], fp[:, 1], fp[:, 2], s=24, color=DEEP_BLUE,
                     edgecolors="white", linewidths=0.4, depthshade=False, zorder=5)

    ax_right.set_box_aspect((1, 1, 1))
    ax_right.set_axis_off()
    ax_right.set_title(r"Sphere $\mathbb{S}^2$", fontsize=15, y=0.94)

    # Arrow and label conveying the from_unit_cube map between the panels.
    arrow = mpatches.FancyArrowPatch(
        (0.525, 0.5), (0.615, 0.5), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=22, color=POINT_BLUE, lw=2.2,
    )
    fig.add_artist(arrow)
    fig.text(0.57, 0.55, "from_unit_cube", ha="center", va="bottom",
             fontsize=12, family="monospace", color=DEEP_BLUE)
    fig.text(0.57, 0.44, "equal-area map", ha="center", va="top",
             fontsize=11, color="#5b6b7a")

    fig.savefig(output, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: low-discrepancy comparison
# ---------------------------------------------------------------------------

def make_low_discrepancy_comparison(output, n=100):
    panels = [
        ("Pseudo-random", stack_samples(geodex.PseudoRandomSampler(seed=0), 2, n)),
        ("Halton", stack_samples(geodex.HaltonSampler(), 2, n)),
        ("Scrambled Halton", stack_samples(geodex.ScrambledHaltonSampler(seed=0), 2, n)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, (title, pts) in zip(axes, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=28, color=POINT_BLUE,
                   edgecolors="white", linewidths=0.4, zorder=3)
        style_unit_square(ax, title)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.92, wspace=0.22)
    fig.savefig(output, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "concepts" / "figs",
                        help="Directory to write the SVG figures into.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        ("cube-to-manifold.svg", make_cube_to_manifold),
        ("low-discrepancy-comparison.svg", make_low_discrepancy_comparison),
    ]
    for name, fn in targets:
        svg_path = args.out_dir / name
        fig = fn(svg_path)
        print(f"Saved {svg_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

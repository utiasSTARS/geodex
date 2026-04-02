#!/usr/bin/env python3
"""Visualize KE and Jacobi metric ellipses over T^2 for a 2-link planar arm.

Reads a JSON file produced by minimum_energy_grid and writes two SVG figures:
  ke_metric.svg           -- KE metric ellipses on det(M(q)) background
  jacobi_combined.svg     -- Jacobi ellipses on det(J(q)) background, 3 energy levels

Usage:
  python scripts/visualize_metric_grid.py minimum_energy_grid.json \\
      --output-dir docs/tutorials/figs
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

matplotlib.use("Agg")  # non-interactive backend for SVG output

# Use LaTeX for all text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Create a new colormap by taking a slice
# This skips the first 15% (the black part)
pink_cmap = mcolors.ListedColormap(plt.get_cmap('pink')(np.linspace(0.20, 1, 256)))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pi_ticks():
    """Return tick positions and labels at -pi, -pi/2, 0, pi/2, pi."""
    positions = [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]
    labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    return positions, labels


def _draw_ellipses(ax, ellipse_data, inv_key, scale):
    """Draw metric ellipses as unfilled black outlines on axes *ax*.

    Parameters
    ----------
    ellipse_data : list of dicts with "q" and *inv_key* fields
    inv_key      : key for the 2x2 inverse metric tensor ("M_inv" or "J_inv")
    scale        : multiplicative scale for semi-axes
    """
    for item in ellipse_data:
        q1, q2 = item["q"]
        inv_tensor = np.array(item[inv_key])

        # Eigendecompose the inverse metric tensor.
        # Ellipse is the unit ball {v : v^T G v = 1} = eigenellipse of G^{-1}.
        vals, vecs = np.linalg.eigh(inv_tensor)

        # Semi-axes proportional to sqrt of eigenvalues of G^{-1}.
        # vals are in ascending order from eigh; largest eigenvalue -> major axis.
        a = math.sqrt(max(vals[1], 0.0)) * scale  # semi-major
        b = math.sqrt(max(vals[0], 0.0)) * scale  # semi-minor

        # Angle of the major-axis eigenvector (vecs[:,1]).
        angle_deg = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))

        ellipse = mpatches.Ellipse(
            xy=(q1, q2),
            width=2 * a,
            height=2 * b,
            angle=angle_deg,
            linewidth=0.5,
            edgecolor="black",
            facecolor="none",
            zorder=3,
        )
        ax.add_patch(ellipse)


def _finalize_axes(ax, title=None, show_ylabel=True, fontsize=None):
    """Apply common axis formatting."""
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
# Figure generators
# ---------------------------------------------------------------------------

def plot_ke_metric(data, output_path):
    """KE metric ellipses; background = det(M(q)) heatmap."""
    fine = data["grid_fine"]
    q1_f = np.array(fine["q1"])
    q2_f = np.array(fine["q2"])
    det_M = np.array(fine["det_M"])  # shape (n, n)

    extent = [q1_f[0], q1_f[-1], q2_f[0], q2_f[-1]]

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # Background: det(M(q)) heatmap
    im = ax.imshow(
        det_M.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=pink_cmap,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\det G$")

    # Draw KE metric ellipses (coarse grid)
    ke_data = data["grid_ellipse"]["ke"]
    n_e = data["grid_ellipse"]["n"]
    grid_spacing = 2 * math.pi / n_e
    scale = 0.15 * grid_spacing

    _draw_ellipses(ax, ke_data, "M_inv", scale)

    _finalize_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def plot_jacobi_combined(data, output_path):
    """Side-by-side Jacobi metric ellipses for all H levels; background = det(J(q))."""
    fine = data["grid_fine"]
    q1_f = np.array(fine["q1"])
    q2_f = np.array(fine["q2"])
    det_M = np.array(fine["det_M"])   # shape (n, n)
    pot = np.array(fine["potential"])  # shape (n, n)

    jacobi_entries = data["grid_ellipse"]["jacobi"]
    n_panels = len(jacobi_entries)

    extent = [q1_f[0], q1_f[-1], q2_f[0], q2_f[-1]]

    # Precompute det(J) = [2(H - P)]^2 * det(M) for each panel and find shared range
    det_J_panels = []
    for entry in jacobi_entries:
        H = entry["H"]
        conformal = 2.0 * (H - pot)            # shape (n, n)
        det_J = conformal**2 * det_M            # det of 2x2 conformal scaling
        det_J_panels.append(det_J)

    # Normalize each panel to [0, 1] so the spatial pattern converges to KE
    det_J_panels = [d / d.max() for d in det_J_panels]
    vmin = 0.0
    vmax = 1.0

    # Use gridspec to allocate a thin column for the shared colorbar
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.5 * n_panels, 5.5),
        gridspec_kw={"wspace": 0.08},
    )
    if n_panels == 1:
        axes = [axes]

    n_e = data["grid_ellipse"]["n"]
    grid_spacing = 2 * math.pi / n_e
    scale = 0.5 * grid_spacing

    for idx, (ax, entry, det_J) in enumerate(zip(axes, jacobi_entries, det_J_panels)):
        alpha = entry["H_over_Pmax"]

        im = ax.imshow(
            det_J.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=pink_cmap,
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )

        _draw_ellipses(ax, entry["ellipses"], "J_inv", scale)

        title = rf"$H = {alpha}\,P_{{\max}}$"
        _finalize_axes(ax, title=title, show_ylabel=(idx == 0), fontsize=16)

    # Shared colorbar matching the height of the subplots
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, shrink=1.0)
    cbar.set_label(r"$\det G \;/\; \max \det G$", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate metric-ellipse figures for the minimum-energy tutorial."
    )
    parser.add_argument("json_file", help="Path to minimum_energy_grid.json")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for output SVG files (default: .)"
    )
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating figures in {args.output_dir}/")

    # KE metric figure
    plot_ke_metric(data, os.path.join(args.output_dir, "ke_metric.svg"))

    # Combined Jacobi metric figure
    plot_jacobi_combined(data, os.path.join(args.output_dir, "jacobi_combined.svg"))

    print("Done.")


if __name__ == "__main__":
    main()

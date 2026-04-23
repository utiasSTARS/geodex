#!/usr/bin/env python3
"""Generate concept diagrams for the SE(2) motion planning tutorial.

Produces three SVG figures (pure matplotlib, no C++ data needed):
  1. se2_poses_and_footprints.svg — Three robot types with coordinate axes and footprints
  2. se2_inflation.svg — Inflation of obstacle boundary by robot radius
  3. se2_footprint_checking.svg — Polygon perimeter sampling in body and world frame

Usage:
    python scripts/draw_se2_diagrams.py --output-dir docs/tutorials/figs/
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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


def draw_pose_arrow(ax, x, y, theta, length=0.6, color="black", lw=2.0, head_width=0.08):
    """Draw a heading arrow at (x, y) pointing in direction theta."""
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    ax.arrow(
        x, y, dx, dy, head_width=head_width, head_length=head_width * 0.7,
        fc=color, ec=color, lw=lw, zorder=5,
    )


def draw_axes(ax, x, y, theta, length=0.5, lw=1.5):
    """Draw x (red) and y (green) body-frame axes at a pose."""
    ct, st = np.cos(theta), np.sin(theta)
    # x-axis (body forward)
    ax.arrow(x, y, length * ct, length * st, head_width=0.06, head_length=0.04,
             fc="tab:red", ec="tab:red", lw=lw, zorder=4)
    ax.text(x + (length + 0.12) * ct, y + (length + 0.12) * st, r"$x$",
            color="tab:red", fontsize=12, ha="center", va="center", zorder=6)
    # y-axis (body left)
    ax.arrow(x, y, -length * st, length * ct, head_width=0.06, head_length=0.04,
             fc="tab:green", ec="tab:green", lw=lw, zorder=4)
    ax.text(x - (length + 0.12) * st, y + (length + 0.12) * ct, r"$y$",
            color="tab:green", fontsize=12, ha="center", va="center", zorder=6)


def draw_circle_footprint(ax, x, y, radius, color="royalblue", alpha=0.15):
    """Draw a circular robot footprint."""
    circle = mpatches.Circle((x, y), radius, fc=color, alpha=alpha, ec=color, lw=1.5, zorder=2)
    ax.add_patch(circle)


def draw_rect_footprint(ax, x, y, theta, hl, hw, color="royalblue", alpha=0.15):
    """Draw a rectangular robot footprint at pose (x, y, theta)."""
    ct, st = np.cos(theta), np.sin(theta)
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        lx, ly = sx * hl, sy * hw
        corners.append((x + ct * lx - st * ly, y + st * lx + ct * ly))
    patch = mpatches.Polygon(
        corners, closed=True, fc=color, alpha=alpha, ec=color, lw=1.5, zorder=2,
    )
    ax.add_patch(patch)
    return corners


# ---------------------------------------------------------------------------
# Figure 1: Three robot types with poses and footprints
# ---------------------------------------------------------------------------

def draw_poses_and_footprints(output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    configs = [
        {"title": "Holonomic (circular)", "theta": np.pi / 6,
         "footprint": "circle", "radius": 0.5},
        {"title": "Differential drive", "theta": np.pi / 4,
         "footprint": "rect", "hl": 0.55, "hw": 0.35},
        {"title": "Car-like", "theta": -np.pi / 8,
         "footprint": "rect", "hl": 0.7, "hw": 0.35},
    ]

    for ax, cfg in zip(axes, configs):
        cx, cy = 1.5, 1.5
        theta = cfg["theta"]

        # Draw footprint.
        if cfg["footprint"] == "circle":
            draw_circle_footprint(ax, cx, cy, cfg["radius"])
        else:
            draw_rect_footprint(ax, cx, cy, theta, cfg["hl"], cfg["hw"])

        # Draw body-frame axes.
        draw_axes(ax, cx, cy, theta, length=0.6)

        # Draw heading arrow.
        draw_pose_arrow(ax, cx, cy, theta, length=0.8, color="navy")

        # Draw pose dot.
        ax.plot(cx, cy, "o", color="navy", ms=6, zorder=6)

        # Theta arc.
        arc_r = 0.35
        arc_angles = np.linspace(0, theta, 40) if theta >= 0 else np.linspace(theta, 0, 40)
        ax.plot(cx + arc_r * np.cos(arc_angles), cy + arc_r * np.sin(arc_angles),
                color="gray", lw=1.0, ls="--", zorder=3)
        mid_angle = theta / 2
        ax.text(cx + (arc_r + 0.15) * np.cos(mid_angle),
                cy + (arc_r + 0.15) * np.sin(mid_angle),
                r"$\theta$", fontsize=13, color="gray", ha="center", va="center")

        # Pose label.
        ax.text(cx, cy - 0.85, r"$(x, y, \theta)$", fontsize=13, ha="center", color="navy")

        ax.set_title(cfg["title"], fontsize=14, fontweight="bold")
        ax.set_xlim(0.2, 2.8)
        ax.set_ylim(0.2, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "se2_poses_and_footprints.svg")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Inflation illustration
# ---------------------------------------------------------------------------

def draw_inflation(output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Obstacle (irregular polygon approximated by rectangle for clarity).
    obs_x, obs_y = 4.0, 2.5
    obs_w, obs_h = 3.0, 1.8
    obstacle = mpatches.FancyBboxPatch(
        (obs_x - obs_w / 2, obs_y - obs_h / 2), obs_w, obs_h,
        boxstyle="round,pad=0.1", fc="salmon", ec="darkred", lw=2, alpha=0.7, zorder=2,
    )
    ax.add_patch(obstacle)
    ax.text(obs_x, obs_y, "Obstacle", ha="center", va="center", fontsize=13,
            color="darkred", fontweight="bold", zorder=3)

    # Original boundary (solid).
    # Draw a dashed inflated boundary.
    robot_r = 0.5
    inflated = mpatches.FancyBboxPatch(
        (obs_x - obs_w / 2 - robot_r, obs_y - obs_h / 2 - robot_r),
        obs_w + 2 * robot_r, obs_h + 2 * robot_r,
        boxstyle="round,pad=0.1", fc="none", ec="navy", lw=2, ls="--", zorder=2,
    )
    ax.add_patch(inflated)

    # Robot at the inflated boundary (collision-free).
    rx_safe = obs_x + obs_w / 2 + robot_r + 0.1
    ry_safe = obs_y
    draw_circle_footprint(ax, rx_safe, ry_safe, robot_r, color="forestgreen", alpha=0.25)
    ax.plot(rx_safe, ry_safe, "o", color="forestgreen", ms=5, zorder=6)
    ax.annotate("Safe", xy=(rx_safe, ry_safe + robot_r + 0.15),
                fontsize=12, color="forestgreen", ha="center", fontweight="bold")

    # Robot at the original boundary (collision).
    rx_coll = obs_x + obs_w / 2 - 0.1
    ry_coll = obs_y + obs_h / 2 + 0.3
    draw_circle_footprint(ax, rx_coll, ry_coll, robot_r, color="tomato", alpha=0.25)
    ax.plot(rx_coll, ry_coll, "o", color="tomato", ms=5, zorder=6)
    ax.annotate("Collision", xy=(rx_coll, ry_coll + robot_r + 0.15),
                fontsize=12, color="tomato", ha="center", fontweight="bold")

    # Dimension arrow for inflation radius.
    arr_y = obs_y - obs_h / 2 - 0.7
    arr_x1 = obs_x + obs_w / 2
    arr_x2 = obs_x + obs_w / 2 + robot_r
    ax.annotate("", xy=(arr_x2, arr_y), xytext=(arr_x1, arr_y),
                arrowprops=dict(arrowstyle="<->", color="navy", lw=1.5))
    ax.text((arr_x1 + arr_x2) / 2, arr_y - 0.25, r"$r$", fontsize=14, ha="center", color="navy")

    # Legend labels.
    ax.plot([], [], color="darkred", lw=2, label="Original boundary")
    ax.plot([], [], color="navy", lw=2, ls="--", label="Inflated boundary")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    ax.set_xlim(0.5, 8.0)
    ax.set_ylim(-0.2, 4.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Inflating the obstacle boundary by robot radius", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "se2_inflation.svg")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Footprint collision checking (body frame → world frame)
# ---------------------------------------------------------------------------

def draw_footprint_checking(output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    hl, hw = 0.55, 0.35
    samples_per_edge = 4

    # Generate perimeter samples (matching PolygonFootprint::rectangle).
    body_samples = []
    corners = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    for i in range(4):
        c0 = corners[i]
        c1 = corners[(i + 1) % 4]
        for j in range(samples_per_edge):
            t = j / samples_per_edge
            bx = c0[0] + t * (c1[0] - c0[0])
            by = c0[1] + t * (c1[1] - c0[1])
            body_samples.append((bx, by))
    body_samples = np.array(body_samples)

    # --- Left: body frame ---
    ax = axes[0]
    # Draw rectangle outline.
    rect = mpatches.FancyBboxPatch(
        (-hl, -hw), 2 * hl, 2 * hw,
        boxstyle="square,pad=0", fc="royalblue", alpha=0.1, ec="royalblue", lw=1.5, zorder=2,
    )
    ax.add_patch(rect)

    # Draw samples.
    ax.scatter(body_samples[:, 0], body_samples[:, 1], c="navy", s=30, zorder=5,
               label="Perimeter samples")

    # Draw body axes.
    ax.arrow(0, 0, 0.4, 0, head_width=0.04, head_length=0.03, fc="tab:red", ec="tab:red", lw=1.5)
    ax.text(0.5, 0, r"$x_b$", color="tab:red", fontsize=12, va="center")
    ax.arrow(0, 0, 0, 0.4, head_width=0.04, head_length=0.03, fc="tab:green", ec="tab:green",
             lw=1.5)
    ax.text(0.05, 0.48, r"$y_b$", color="tab:green", fontsize=12)
    ax.plot(0, 0, "o", color="navy", ms=5, zorder=6)

    ax.set_title("Body frame", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")

    # --- Right: world frame (after rotation) ---
    ax = axes[1]
    pose_x, pose_y, pose_theta = 3.0, 2.5, np.pi / 5

    # Draw a simple grid background to suggest the distance grid.
    grid_res = 0.3
    for gx in np.arange(0, 6.01, grid_res):
        ax.axvline(gx, color="lightgray", lw=0.3, zorder=0)
    for gy in np.arange(0, 5.01, grid_res):
        ax.axhline(gy, color="lightgray", lw=0.3, zorder=0)

    # Draw some "obstacles" for context.
    wall1 = mpatches.Rectangle((0, 0), 6, 0.3, fc="salmon", alpha=0.5, ec="darkred", lw=1)
    wall2 = mpatches.Rectangle((0, 4.7), 6, 0.3, fc="salmon", alpha=0.5, ec="darkred", lw=1)
    ax.add_patch(wall1)
    ax.add_patch(wall2)

    # Transform samples to world frame.
    ct, st = np.cos(pose_theta), np.sin(pose_theta)
    world_x = pose_x + ct * body_samples[:, 0] - st * body_samples[:, 1]
    world_y = pose_y + st * body_samples[:, 0] + ct * body_samples[:, 1]

    # Draw robot footprint.
    draw_rect_footprint(ax, pose_x, pose_y, pose_theta, hl, hw)

    # Draw samples in world frame.
    ax.scatter(world_x, world_y, c="navy", s=30, zorder=5, label="Transformed samples")

    # Draw world axes at robot center.
    draw_axes(ax, pose_x, pose_y, pose_theta, length=0.5)
    ax.plot(pose_x, pose_y, "o", color="navy", ms=5, zorder=6)

    # Draw distance query lines from a few samples to nearest wall.
    for i in [0, 4, 8, 12]:
        sx, sy = world_x[i], world_y[i]
        # Distance to bottom wall.
        nearest_y = 0.3 if sy < 2.5 else 4.7
        ax.plot([sx, sx], [sy, nearest_y], color="orange", lw=1.0, ls=":", alpha=0.7, zorder=3)

    ax.set_title("World frame (on distance grid)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 6)
    ax.set_ylim(-0.3, 5.3)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    plt.tight_layout()
    path = os.path.join(output_dir, "se2_footprint_checking.svg")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SE(2) tutorial concept diagrams")
    parser.add_argument(
        "--output-dir",
        default="docs/tutorials/figs",
        help="Output directory for SVG files (default: docs/tutorials/figs)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    draw_poses_and_footprints(args.output_dir)
    draw_inflation(args.output_dir)
    draw_footprint_checking(args.output_dir)


if __name__ == "__main__":
    main()

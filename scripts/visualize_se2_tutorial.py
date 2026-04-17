#!/usr/bin/env python3
"""Visualize SE(2) tutorial planning results.

Reads JSON output from se2_tutorial and produces SVG figures for the tutorial.
Supports multiple rendering modes via --mode:
  - result:     Planning result with tree, paths, and footprints (default)
  - map:        Environment only (occupancy grid + distance heatmap)
  - conformal:  Conformal factor heatmap overlaid on map
  - comparison: Two-panel side-by-side (pass two JSON files)
  - env:        Parking environment layout (no planning data)

Usage:
    python scripts/visualize_se2_tutorial.py result.json -o figure.svg
    python scripts/visualize_se2_tutorial.py result.json -o figure.svg --map corridor.png
    python scripts/visualize_se2_tutorial.py a.json b.json -o comparison.svg --mode comparison
"""

import argparse
import json
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


def draw_pose_arrow(ax, x, y, theta, length=0.4, color="black", lw=1.5):
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    ax.annotate(
        "",
        xy=(x + dx, y + dy),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
    )


def draw_circle_footprint(ax, x, y, radius, color="royalblue", alpha=0.12, ec="navy"):
    circle = mpatches.Circle(
        (x, y), radius, fc=color, alpha=alpha, ec=ec, lw=0.5, zorder=4,
    )
    ax.add_patch(circle)


def draw_rect_footprint(ax, x, y, theta, hl, hw, color="royalblue", alpha=0.12, ec="navy"):
    ct, st = np.cos(theta), np.sin(theta)
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        lx, ly = sx * hl, sy * hw
        corners.append((x + ct * lx - st * ly, y + st * lx + ct * ly))
    patch = mpatches.Polygon(
        corners, closed=True, fc=color, alpha=alpha, ec=ec, lw=0.5, zorder=4,
    )
    ax.add_patch(patch)


def draw_rect_obstacle(ax, obs, fc="salmon", alpha=0.6, ec="darkred", lw=1.5):
    cx, cy = obs["center"]
    theta = obs.get("theta", 0.0)
    hl, hw = obs["half_length"], obs["half_width"]
    ct, st = np.cos(theta), np.sin(theta)
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        lx, ly = sx * hl, sy * hw
        corners.append((cx + ct * lx - st * ly, cy + st * lx + ct * ly))
    patch = mpatches.Polygon(
        corners, closed=True, fc=fc, alpha=alpha, ec=ec, lw=lw, zorder=0,
    )
    ax.add_patch(patch)


def load_map_png(map_path, data):
    """Load map PNG image. Tries explicit path, then sibling of dist map file."""
    from PIL import Image

    candidates = []
    if map_path:
        candidates.append(map_path)
    map_info = data.get("map")
    if map_info and map_info.get("file"):
        dist_file = map_info["file"]
        base = os.path.splitext(os.path.basename(dist_file))[0].replace("_dist", "")
        candidates.append(os.path.join(os.path.dirname(dist_file), base + ".png"))
    for path in candidates:
        if os.path.isfile(path):
            return np.array(Image.open(path).convert("L"))
    return None


def get_extent(data, map_img):
    """Get world extent [x0, x1, y0, y1] from map info or obstacle bounds."""
    map_info = data.get("map")
    if map_info:
        w = map_info["width"] * map_info["resolution"]
        h = map_info["height"] * map_info["resolution"]
        return [0, w, 0, h]
    # For rect obstacle scenarios, compute from obstacles + start/goal.
    rects = data.get("rect_obstacles", [])
    if rects:
        all_x = [o["center"][0] for o in rects] + [data["start"][0], data["goal"][0]]
        all_y = [o["center"][1] for o in rects] + [data["start"][1], data["goal"][1]]
        margin = 3.0
        return [min(all_x) - margin, max(all_x) + margin,
                min(all_y) - margin, max(all_y) + margin]
    return [0, 15, 0, 10]


def draw_background(ax, data, map_img, extent):
    """Draw the environment background (map image or rectangle obstacles)."""
    if map_img is not None:
        ax.imshow(map_img, cmap="gray", extent=extent, origin="lower", alpha=0.7, zorder=0)
    for obs in data.get("rect_obstacles", []):
        draw_rect_obstacle(ax, obs)


def draw_planning_result(ax, data, run, map_img, extent, show_tree=True, arrow_len=None):
    """Draw a full planning result: tree, paths, footprints, start/goal."""
    draw_background(ax, data, map_img, extent)

    start = data["start"]
    goal = data["goal"]
    robot = data.get("robot", {})
    is_map = map_img is not None
    has_rects = len(data.get("rect_obstacles", [])) > 0

    if arrow_len is None:
        arrow_len = 1.0 if is_map else (0.8 if has_rects else 0.5)

    # Tree edges.
    if show_tree:
        for edge in run.get("tree", []):
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]],
                    color="lightgray", lw=0.3, zorder=1)

    # Paths.
    raw = np.array(run.get("raw_path", []))
    smoothed = np.array(run.get("smoothed_path", []))

    if len(smoothed) > 0:
        if len(raw) > 0:
            ax.plot(raw[:, 0], raw[:, 1], color="tomato", lw=1.5, ls="--", zorder=2,
                    label="Raw", alpha=0.7)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="royalblue", lw=2.5, zorder=3,
                label="Smoothed")
        path = smoothed
    elif len(raw) > 0:
        ax.plot(raw[:, 0], raw[:, 1], color="royalblue", lw=2.5, zorder=3, label="Path")
        path = raw
    else:
        path = np.empty((0, 3))

    # Footprints and heading arrows along smoothed path.
    if len(path) > 0:
        n_fp = min(15, len(path))
        indices = np.linspace(0, len(path) - 1, n_fp, dtype=int)
        for i in indices:
            px, py, pth = path[i]
            draw_pose_arrow(ax, px, py, pth, length=arrow_len, color="navy", lw=1.2)
            if robot.get("type") == "circle":
                draw_circle_footprint(ax, px, py, robot["radius"])
            elif robot.get("type") == "rectangle":
                draw_rect_footprint(ax, px, py, pth, robot["half_length"], robot["half_width"])

    # Start / goal.
    ms_s = 12 if is_map else 10
    ms_g = 16 if is_map else 14
    pose_len = 1.5 if is_map else (1.0 if has_rects else 0.6)
    ax.plot(start[0], start[1], "o", color="limegreen", ms=ms_s, zorder=5,
            mec="darkgreen", mew=1.5)
    draw_pose_arrow(ax, start[0], start[1], start[2], length=pose_len, color="darkgreen", lw=2)
    ax.plot(goal[0], goal[1], "*", color="orange", ms=ms_g, zorder=5,
            mec="darkorange", mew=1.5)
    draw_pose_arrow(ax, goal[0], goal[1], goal[2], length=pose_len, color="darkorange", lw=2)

    # Title and labels.
    timing = []
    if "planning_time_ms" in run:
        timing.append(f"plan={run['planning_time_ms']:.0f}ms")
    if "smoothing_time_ms" in run:
        timing.append(f"smooth={run['smoothing_time_ms']:.0f}ms")
    title = run["label"]
    if timing:
        title += f"\n({', '.join(timing)})"
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


# ---------------------------------------------------------------------------
# Mode: result (default)
# ---------------------------------------------------------------------------

def mode_result(data, map_img, output):
    runs = data["runs"]
    extent = get_extent(data, map_img)
    n = len(runs)
    fig_w = 10 if map_img is not None else 9
    fig, axes = plt.subplots(1, n, figsize=(fig_w * n, fig_w), squeeze=False)
    for i, run in enumerate(runs):
        draw_planning_result(axes[0, i], data, run, map_img, extent)
    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode: map (environment with distance heatmap)
# ---------------------------------------------------------------------------

def mode_map(data, map_img, output):
    extent = get_extent(data, map_img)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: occupancy grid.
    if map_img is not None:
        axes[0].imshow(map_img, cmap="gray", extent=extent, origin="lower")
    axes[0].set_title("Occupancy grid", fontsize=14, fontweight="bold")
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(extent[2], extent[3])
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    # Right: distance heatmap.
    map_info = data.get("map", {})
    if map_img is not None:
        # Compute distance transform from the image for visualization.
        from scipy.ndimage import distance_transform_edt

        free = map_img >= 128
        dist = distance_transform_edt(free) * map_info.get("resolution", 0.05)
        im = axes[1].imshow(dist, cmap="inferno", extent=extent, origin="lower")
        plt.colorbar(im, ax=axes[1], label="Distance (m)", shrink=0.8)
    axes[1].set_title("Distance transform", fontsize=14, fontweight="bold")
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[2], extent[3])
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode: conformal factor heatmap
# ---------------------------------------------------------------------------

def mode_conformal(data, map_img, output):
    extent = get_extent(data, map_img)
    cgrid = data.get("conformal_grid")
    if cgrid is None:
        print("Error: no conformal_grid in JSON (run holo_clearance scenario)")
        return

    values = np.array(cgrid["values"]).reshape(cgrid["height"], cgrid["width"])

    fig, ax = plt.subplots(figsize=(10, 8))
    if map_img is not None:
        ax.imshow(map_img, cmap="gray", extent=extent, origin="lower", alpha=0.4, zorder=0)
    im = ax.imshow(values, cmap="YlOrRd", extent=extent, origin="lower", alpha=0.7, zorder=1)
    plt.colorbar(im, ax=ax, label=r"Conformal factor $c(q)$", shrink=0.8)

    ax.set_title(r"Conformal clearance metric: $c(q) = 1 + \kappa \exp(-\beta \cdot \mathrm{sdf}(q))$",
                 fontsize=13)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode: comparison (two-panel side-by-side from two JSON files)
# ---------------------------------------------------------------------------

def mode_comparison(data_list, map_img, output):
    n = len(data_list)
    extent = get_extent(data_list[0], map_img)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 9), squeeze=False)
    for i, data in enumerate(data_list):
        if data["runs"]:
            draw_planning_result(axes[0, i], data, data["runs"][0], map_img, extent)
    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode: env (parking environment layout)
# ---------------------------------------------------------------------------

def mode_env(data, output):
    extent = get_extent(data, None)
    robot = data.get("robot", {})
    start, goal = data["start"], data["goal"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for obs in data.get("rect_obstacles", []):
        draw_rect_obstacle(ax, obs)

    # Draw goal pose (dashed outline).
    if robot.get("type") == "rectangle":
        hl, hw = robot["half_length"], robot["half_width"]
        ct, st = np.cos(goal[2]), np.sin(goal[2])
        corners = []
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            lx, ly = sx * hl, sy * hw
            corners.append((goal[0] + ct * lx - st * ly, goal[1] + st * lx + ct * ly))
        ax.add_patch(mpatches.Polygon(
            corners, closed=True, fc="none", ec="limegreen", lw=2, ls="--", zorder=3))

    # Start and goal markers.
    ax.plot(start[0], start[1], "o", color="limegreen", ms=12, zorder=5,
            mec="darkgreen", mew=1.5)
    draw_pose_arrow(ax, start[0], start[1], start[2], length=1.5, color="darkgreen", lw=2)
    ax.annotate("Start", xy=(start[0], start[1] + 1.5), fontsize=12, ha="center",
                color="darkgreen", fontweight="bold")

    ax.plot(goal[0], goal[1], "*", color="orange", ms=16, zorder=5,
            mec="darkorange", mew=1.5)
    draw_pose_arrow(ax, goal[0], goal[1], goal[2], length=1.0, color="darkorange", lw=2)
    ax.annotate("Goal", xy=(goal[0], goal[1] + 1.5), fontsize=12, ha="center",
                color="darkorange", fontweight="bold")

    ax.set_title("Parallel parking scenario", fontsize=14, fontweight="bold")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize SE(2) tutorial planning results")
    parser.add_argument("json_files", nargs="+", help="JSON from se2_tutorial")
    parser.add_argument("-o", "--output", required=True, help="Output SVG/PNG")
    parser.add_argument("--map", default=None, help="Map PNG for background")
    parser.add_argument("--mode", default="result",
                        choices=["result", "map", "conformal", "comparison", "env"],
                        help="Rendering mode (default: result)")
    args = parser.parse_args()

    data_list = []
    for jf in args.json_files:
        with open(jf) as f:
            data_list.append(json.load(f))
    data = data_list[0]

    map_img = load_map_png(args.map, data)

    if args.mode == "result":
        mode_result(data, map_img, args.output)
    elif args.mode == "map":
        mode_map(data, map_img, args.output)
    elif args.mode == "conformal":
        mode_conformal(data, map_img, args.output)
    elif args.mode == "comparison":
        mode_comparison(data_list, map_img, args.output)
    elif args.mode == "env":
        mode_env(data, args.output)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

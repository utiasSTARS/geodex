#!/usr/bin/env python3
"""Animate robot footprint sweeping along a planned SE(2) path.

Reads JSON output from se2_tutorial and produces a GIF animation showing the
robot footprint translating and rotating along the smoothed path with a
fading trail of previous footprints.

Usage:
    python scripts/animate_se2_tutorial.py result.json -o sweep.gif --fps 15
    python scripts/animate_se2_tutorial.py result.json -o sweep.gif --map corridor.png
"""

import argparse
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# geodex docs style.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "mathtext.fontset": "stixsans",
    }
)


def draw_rect(ax, cx, cy, theta, hl, hw, **kwargs):
    ct, st = np.cos(theta), np.sin(theta)
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        lx, ly = sx * hl, sy * hw
        corners.append((cx + ct * lx - st * ly, cy + st * lx + ct * ly))
    return ax.add_patch(mpatches.Polygon(corners, closed=True, **kwargs))


def load_map_png(map_path, data):
    from PIL import Image

    candidates = []
    if map_path:
        candidates.append(map_path)
    map_info = data.get("map")
    if map_info and map_info.get("file"):
        base = os.path.splitext(os.path.basename(map_info["file"]))[0].replace("_dist", "")
        candidates.append(os.path.join(os.path.dirname(map_info["file"]), base + ".png"))
    for path in candidates:
        if os.path.isfile(path):
            return np.array(Image.open(path).convert("L"))
    return None


def get_extent(data, map_img):
    map_info = data.get("map")
    if map_info:
        w = map_info["width"] * map_info["resolution"]
        h = map_info["height"] * map_info["resolution"]
        return [0, w, 0, h]
    rects = data.get("rect_obstacles", [])
    if rects:
        all_x = [o["center"][0] for o in rects] + [data["start"][0], data["goal"][0]]
        all_y = [o["center"][1] for o in rects] + [data["start"][1], data["goal"][1]]
        margin = 3.0
        return [min(all_x) - margin, max(all_x) + margin,
                min(all_y) - margin, max(all_y) + margin]
    return [0, 15, 0, 10]


def main():
    parser = argparse.ArgumentParser(description="Animate SE(2) footprint sweep")
    parser.add_argument("input", help="JSON from se2_tutorial")
    parser.add_argument("-o", "--output", default="se2_sweep.gif", help="Output GIF")
    parser.add_argument("--map", default=None, help="Map PNG for background")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--trail", type=int, default=8, help="Number of ghost trail footprints")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    map_img = load_map_png(args.map, data)
    extent = get_extent(data, map_img)
    robot = data.get("robot", {})
    start, goal = data["start"], data["goal"]

    # Use smoothed path from first run.
    run = data["runs"][0]
    path = np.array(run.get("smoothed_path", run.get("raw_path", [])))
    if len(path) == 0:
        print("Error: no path data in JSON")
        return

    # Subsample path to ~40 frames for reasonable GIF size.
    n_frames = min(40, len(path))
    indices = np.linspace(0, len(path) - 1, n_frames, dtype=int)
    frames = path[indices]

    # Determine figure size from extent aspect ratio.
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    fig_h = 6
    fig_w = max(6, fig_h * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    def animate(frame_idx):
        ax.clear()

        # Background.
        if map_img is not None:
            ax.imshow(map_img, cmap="gray", extent=extent, origin="lower", alpha=0.7, zorder=0)
        for obs in data.get("rect_obstacles", []):
            cx, cy = obs["center"]
            theta = obs.get("theta", 0.0)
            hl, hw = obs["half_length"], obs["half_width"]
            draw_rect(ax, cx, cy, theta, hl, hw,
                      fc="salmon", alpha=0.6, ec="darkred", lw=1.5, zorder=0)

        # Full path line.
        ax.plot(path[:, 0], path[:, 1], color="royalblue", lw=1.5, alpha=0.4, zorder=1)

        # Ghost trail.
        trail_start = max(0, frame_idx - args.trail)
        for t in range(trail_start, frame_idx):
            alpha = 0.05 + 0.15 * (t - trail_start) / max(1, frame_idx - trail_start)
            px, py, pth = frames[t]
            if robot.get("type") == "circle":
                ax.add_patch(mpatches.Circle(
                    (px, py), robot["radius"], fc="royalblue", alpha=alpha,
                    ec="navy", lw=0.3, zorder=2))
            elif robot.get("type") == "rectangle":
                draw_rect(ax, px, py, pth, robot["half_length"], robot["half_width"],
                          fc="royalblue", alpha=alpha, ec="navy", lw=0.3, zorder=2)

        # Current footprint (solid).
        px, py, pth = frames[frame_idx]
        if robot.get("type") == "circle":
            ax.add_patch(mpatches.Circle(
                (px, py), robot["radius"], fc="royalblue", alpha=0.7,
                ec="navy", lw=2, zorder=5))
        elif robot.get("type") == "rectangle":
            draw_rect(ax, px, py, pth, robot["half_length"], robot["half_width"],
                      fc="royalblue", alpha=0.7, ec="navy", lw=2, zorder=5)

        # Heading arrow on current robot.
        arrow_len = 0.8 if robot.get("type") == "rectangle" else 0.5
        dx = arrow_len * np.cos(pth)
        dy = arrow_len * np.sin(pth)
        ax.annotate(
            "", xy=(px + dx, py + dy), xytext=(px, py),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5), zorder=6,
        )

        # Start / goal.
        ax.plot(start[0], start[1], "o", color="limegreen", ms=10, zorder=7,
                mec="darkgreen", mew=1.5)
        ax.plot(goal[0], goal[1], "*", color="orange", ms=14, zorder=7,
                mec="darkorange", mew=1.5)

        ax.set_title(f"{run['label']}  (frame {frame_idx + 1}/{n_frames})", fontsize=12)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000 // args.fps, repeat=True)
    anim.save(args.output, writer=PillowWriter(fps=args.fps))
    print(f"Saved {args.output} ({n_frames} frames, {args.fps} fps)")
    plt.close()


if __name__ == "__main__":
    main()

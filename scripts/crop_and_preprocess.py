#!/usr/bin/env python3
"""Crop an occupancy grid PNG and produce a distance transform text file.

Reads a grayscale PNG, optionally crops to a pixel rectangle, computes the
Euclidean distance transform, scales by resolution, and writes a text file
usable by the C++ planner.  Also saves the cropped PNG alongside the output
for use as a visualization background.

Usage:
    python scripts/crop_and_preprocess.py input.png -o output_dist.txt \\
        --crop 460,340,760,540 --resolution 0.05
"""

import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def main():
    parser = argparse.ArgumentParser(
        description="Crop occupancy PNG and produce distance transform txt"
    )
    parser.add_argument("input", help="Input PNG (grayscale occupancy grid)")
    parser.add_argument("-o", "--output", required=True, help="Output distance grid txt file")
    parser.add_argument(
        "--crop",
        default=None,
        help="Pixel crop rectangle: x1,y1,x2,y2 (column-start,row-start,column-end,row-end)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Map resolution in m/pixel (default: 0.05)",
    )
    args = parser.parse_args()

    img = Image.open(args.input).convert("L")
    pixels = np.array(img)

    if args.crop:
        parts = [int(x) for x in args.crop.split(",")]
        if len(parts) != 4:
            parser.error("--crop requires exactly 4 values: x1,y1,x2,y2")
        x1, y1, x2, y2 = parts
        pixels = pixels[y1:y2, x1:x2]
        print(f"Cropped to pixel rect ({x1},{y1})–({x2},{y2}): {x2-x1}×{y2-y1} px")

    # Save cropped PNG alongside output for visualization.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    out_base = os.path.splitext(os.path.basename(args.output))[0]
    cropped_png = os.path.join(out_dir, out_base.replace("_dist", "") + ".png")
    Image.fromarray(pixels).save(cropped_png)
    print(f"Saved cropped PNG: {cropped_png}")

    # Binary: occupied where pixel < 128
    occupied = pixels < 128
    free = ~occupied

    # Euclidean distance transform (in pixels), then scale to meters.
    # Interior of obstacles gets negative distance.
    dist_free = distance_transform_edt(free) * args.resolution
    dist_occ = distance_transform_edt(occupied) * args.resolution
    dist_meters = dist_free - dist_occ

    height, width = dist_meters.shape
    world_w = width * args.resolution
    world_h = height * args.resolution
    print(f"Map: {width}×{height} px, resolution={args.resolution} m/px")
    print(f"World size: {world_w:.1f} × {world_h:.1f} m")

    with open(args.output, "w") as f:
        f.write(f"{width} {height} {args.resolution}\n")
        for row in dist_meters:
            f.write(" ".join(f"{v:.4f}" for v in row) + "\n")

    print(f"Wrote distance grid: {args.output}")


if __name__ == "__main__":
    main()

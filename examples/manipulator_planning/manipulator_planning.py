#!/usr/bin/env python3
"""Single-panel viser visualizer for manipulator_planning JSON output.

Reads the JSON written by `manipulator_planning` (built from
`examples/manipulator_planning/manipulator_planning.cpp`) and animates the
selected robot along the planned path among the planning obstacles. The robot
is loaded via yourdfpy from the URDF recorded in the JSON; obstacles come from
the same MBM scene YAML the planner consumed.

The C++ example emits two flavors of the solution path:
  - `raw_path`    — G-RRT*'s direct output (sparse, ~5-10 waypoints)
  - `smooth_path` — L-BFGS energy minimization on the upsampled raw path
This script lets you toggle which one is animated, with per-path stats
(cost / energy / n_waypoints) shown next to the controls.

Usage:
  python examples/manipulator_planning/manipulator_planning.py [path/to/result.json]

Defaults: reads ./manipulator_planning.json, listens on http://localhost:8080.
"""

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

# viser >= 1.0 calls Path.is_relative_to inside its HTTP handler. The polyfill
# (added on Python 3.9+) keeps older interpreters from silently serving a stub.
if not hasattr(Path, "is_relative_to"):

    def _is_relative_to(self, *other):
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    Path.is_relative_to = _is_relative_to  # type: ignore[attr-defined]

import numpy as np
import viser
import yaml
import yourdfpy
from viser.extras import ViserUrdf

# yourdfpy logs a warning whenever a state setter is called and a mimic joint
# does not have its driver in the configuration vector. The PR2 arms-only model
# intentionally omits gripper driver joints, so silence this channel during
# animation.
logging.getLogger("yourdfpy").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "data" / "robots" / "panda" / "urdf" / "panda.urdf"
DEFAULT_SCENE = (
    REPO_ROOT
    / "data"
    / "datasets"
    / "mbm"
    / "scenes"
    / "panda"
    / "table_pick"
    / "scene0002.scene.yaml"
)
DEFAULT_JSON = "manipulator_planning.json"

# Animation densification target. Sparse raw paths and short smoothed paths
# both upsample to this length so the animation feels smooth at any speed.
ANIM_LEN = 200


def load_scene_obstacles(scene_path: Path):
    """Parse the MBM scene YAML and yield (kind, params) per primitive."""
    with scene_path.open() as f:
        scene = yaml.safe_load(f)

    objects = []
    for obj in scene.get("world", {}).get("collision_objects", []):
        primitives = obj.get("primitives", [])
        poses = obj.get("primitive_poses", [])
        for prim, pose in zip(primitives, poses):
            kind = prim.get("type")
            dims = prim.get("dimensions", [])
            position = pose.get("position", [0, 0, 0])
            orientation = pose.get("orientation", [0, 0, 0, 1])  # [qx, qy, qz, qw]
            objects.append({
                "kind": kind,
                "dims": dims,
                "position": tuple(position),
                # viser wants quaternions as (qw, qx, qy, qz).
                "wxyz": (orientation[3], orientation[0], orientation[1], orientation[2]),
                "id": obj.get("id", "anon"),
            })
    return objects


def add_obstacles(server, objects, prefix="/scene", opacity=0.8, color=(190, 0, 0)):
    """Render the parsed obstacle list as static viser primitives."""
    for i, obj in enumerate(objects):
        name = f"{prefix}/obj_{i}_{obj['id']}"
        if obj["kind"] == "box":
            dims = obj["dims"]
            server.scene.add_box(
                name=name,
                dimensions=tuple(dims),
                color=color,
                opacity=opacity,
                position=obj["position"],
                wxyz=obj["wxyz"],
            )
        elif obj["kind"] == "cylinder":
            # MBM cylinder dimensions are [height, radius].
            height, radius = obj["dims"]
            server.scene.add_cylinder(
                name=name,
                radius=float(radius),
                height=float(height),
                color=color,
                opacity=opacity,
                position=obj["position"],
                wxyz=obj["wxyz"],
            )
        elif obj["kind"] == "sphere":
            server.scene.add_icosphere(
                name=name,
                radius=float(obj["dims"][0]),
                color=color,
                opacity=opacity,
                position=obj["position"],
                subdivisions=3,
            )


def make_urdf_cfg(q, joint_names):
    """Return a yOURDFPy configuration using JSON joint names when available."""
    if joint_names:
        return {name: float(value) for name, value in zip(joint_names, q)}
    return np.asarray(q, dtype=np.float64)


def end_effector_polyline(urdf, path, ee_link, joint_names):
    """Forward-kinematics the EE position along the path. Returns (N, 3)."""
    pts = np.empty((len(path), 3), dtype=np.float32)
    for i, q in enumerate(path):
        urdf.update_cfg(make_urdf_cfg(q, joint_names))
        T = urdf.get_transform(ee_link)
        pts[i] = T[:3, 3]
    return pts


def upsample_linear(path, target_len):
    """Linearly upsample a (N, D) joint-space path to `target_len` rows.

    Uses straight-line interpolation between consecutive waypoints, weighted
    by edge L2 distance so the resampled spacing is uniform along the path.
    """
    if len(path) >= target_len:
        return path
    if len(path) < 2:
        return path
    edges = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    if cum[-1] <= 0.0:
        return path
    s = np.linspace(0.0, cum[-1], target_len)
    out = np.empty((target_len, path.shape[1]), dtype=path.dtype)
    for d in range(path.shape[1]):
        out[:, d] = np.interp(s, cum, path[:, d])
    return out


def fmt_stats(label, stats):
    """Format a one-line stats summary for the GUI markdown panel."""
    if not stats:
        return f"**{label}:** —"
    cost = stats.get("cost")
    energy = stats.get("energy")
    n = stats.get("n_waypoints")
    t = stats.get("time_ms")
    parts = [f"**{label}:** "]
    if cost is not None:
        parts.append(f"cost={cost:.4f}")
    if energy is not None:
        parts.append(f"energy={energy:.4f}")
    if n is not None:
        parts.append(f"n={n}")
    if t is not None:
        parts.append(f"t={t:.1f} ms")
    return "  ·  ".join([parts[0] + parts[1]] + parts[2:]) if len(parts) > 1 else parts[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("json", nargs="?", default=DEFAULT_JSON, type=Path,
                        help="Path to the manipulator_planning JSON output.")
    parser.add_argument("--urdf", type=Path, default=None,
                        help="URDF used for visualization (defaults to JSON's "
                        "'urdf' field, falling back to vendored panda).")
    parser.add_argument("--scene", type=Path, default=None,
                        help="Scene YAML for obstacles (defaults to JSON's "
                        "'scene' field, falling back to vendored table-pick).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ee-link", default=None,
                        help="Link to trace for the end-effector trail.")
    parser.add_argument("--loop-seconds", type=float, default=5.0,
                        help="Default loop duration for path animation.")
    args = parser.parse_args()

    if not args.json.exists():
        sys.exit(f"error: {args.json} does not exist; run manipulator_planning first.")

    with args.json.open() as f:
        result = json.load(f)

    if not result.get("solved", False):
        print("warning: planner did not find an exact solution; nothing to animate.",
              file=sys.stderr)

    # Collect the path variants. `path` is the densified smoothed path used
    # as a fallback when `smooth_path` is missing (older JSON outputs).
    raw_arr = result.get("raw_path") or []
    smooth_arr = result.get("smooth_path") or result.get("path") or []

    variants = {}
    if raw_arr:
        variants["raw"] = (np.asarray(raw_arr, dtype=np.float64),
                           result.get("raw_stats", {}))
    if smooth_arr:
        variants["smooth_path"] = (np.asarray(smooth_arr, dtype=np.float64),
                                   result.get("smooth_stats", {}))
    if not variants:
        sys.exit("error: no usable path found in JSON.")

    robot_name = result.get("robot", "panda")
    joint_names = result.get("joint_names") or []
    expected_dim = len(joint_names) if joint_names else None

    # Validate shape against the JSON metadata rather than assuming Panda.
    for name, (arr, _) in variants.items():
        if arr.ndim != 2:
            sys.exit(f"unexpected '{name}' shape: {arr.shape}; expected a 2-D array.")
        if expected_dim is None:
            expected_dim = arr.shape[1]
        if arr.shape[1] != expected_dim:
            sys.exit(f"unexpected '{name}' shape: {arr.shape}; expected (N, {expected_dim}).")

    urdf_path = args.urdf
    if urdf_path is None:
        cand = Path(result.get("urdf", str(DEFAULT_URDF)))
        urdf_path = cand if cand.exists() else DEFAULT_URDF

    default_ee_links = {
        "panda": "panda_link7",
        "pr2": "r_gripper_tool_frame",
    }
    ee_link = args.ee_link or result.get("ee_link") or default_ee_links.get(robot_name)
    if ee_link is None:
        sys.exit("error: --ee-link is required when JSON has no known robot name.")

    scene_path = args.scene
    if scene_path is None:
        cand = Path(result.get("scene", str(DEFAULT_SCENE)))
        scene_path = cand if cand.exists() else DEFAULT_SCENE
    if not scene_path.exists():
        sys.exit(f"error: scene YAML {scene_path} does not exist.")

    print(f"Loading URDF:  {urdf_path}")
    print(f"Loading scene: {scene_path}")
    print(f"Robot:        {robot_name} ({expected_dim} joints)")
    print(f"Variants:      {', '.join(variants.keys())}")
    for name, (arr, stats) in variants.items():
        cost = stats.get("cost", float("nan"))
        print(f"  {name}: {len(arr)} waypoints, cost={cost:.4f}")

    urdf = yourdfpy.URDF.load(str(urdf_path))
    if joint_names:
        missing = sorted(set(joint_names) - set(urdf.actuated_joint_names))
        if missing:
            sys.exit("error: JSON joint_names missing from URDF: " + ", ".join(missing))
    obstacles = load_scene_obstacles(scene_path)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    add_obstacles(server, obstacles)

    robot_handle = ViserUrdf(
        server, urdf, root_node_name="/robot",
        load_meshes=True, load_collision_meshes=False,
    )

    # Pre-compute densified paths and EE trails for each variant. The trails
    # are added with distinct colors so the user can flip between them.
    TRAIL_COLORS = {
        "raw": (200, 200, 50),          # yellow
        "smooth_path": (220, 50, 50),   # red
    }

    densified = {}
    trail_handles = {}
    for name, (arr, _stats) in variants.items():
        dense = upsample_linear(arr, ANIM_LEN)
        densified[name] = dense
        ee = end_effector_polyline(urdf, dense, ee_link=ee_link, joint_names=joint_names)
        trail_handles[name] = server.scene.add_spline_catmull_rom(
            f"/ee_trail/{name}",
            points=ee,
            color=TRAIL_COLORS.get(name, (180, 180, 180)),
            line_width=4.0,
            curve_type="centripetal",
        )

    # Default to smooth_path if available, else raw.
    default_variant = next((v for v in ("smooth_path", "raw") if v in variants),
                           None)
    if default_variant is None:
        sys.exit("error: no variant available to animate.")

    robot_handle.update_cfg(make_urdf_cfg(densified[default_variant][0], joint_names))

    # GUI.
    server.gui.add_markdown(
        "### Path stats\n" +
        fmt_stats("raw", variants.get("raw", (None, {}))[1] if "raw" in variants else None) + "\n\n" +
        fmt_stats("smooth_path", variants.get("smooth_path", (None, {}))[1] if "smooth_path" in variants else None)
    )

    variant_dropdown = server.gui.add_dropdown(
        "Path",
        options=tuple(variants.keys()),
        initial_value=default_variant,
    )
    play_cb = server.gui.add_checkbox("Play", initial_value=True)
    speed_slider = server.gui.add_slider(
        "Loop duration (s)", min=1.0, max=20.0, step=0.5,
        initial_value=args.loop_seconds,
    )
    t_slider = server.gui.add_slider(
        "Time t", min=0.0, max=1.0, step=0.001, initial_value=0.0,
    )
    cb_trail = server.gui.add_checkbox("Show EE trail (selected)", initial_value=True)
    cb_all_trails = server.gui.add_checkbox("Show all trails", initial_value=False)
    cb_obstacles = server.gui.add_checkbox("Show obstacles", initial_value=True)

    obstacle_frame = server.scene.add_frame("/scene", show_axes=False)

    # Lock for the path-pointer swap. The animation thread reads the dense
    # path each frame, the dropdown handler swaps it under the lock.
    state_lock = threading.Lock()
    current = {"name": default_variant, "path": densified[default_variant]}

    def refresh_trails():
        active = current["name"]
        for name, h in trail_handles.items():
            if cb_all_trails.value:
                h.visible = True
            elif name == active:
                h.visible = bool(cb_trail.value)
            else:
                h.visible = False

    def apply_pose(t):
        t = max(0.0, min(1.0, float(t)))
        with state_lock:
            path = current["path"]
        idx = min(int(round(t * (len(path) - 1))), len(path) - 1)
        with server.atomic():
            robot_handle.update_cfg(make_urdf_cfg(path[idx], joint_names))

    @variant_dropdown.on_update
    def _(_):
        name = variant_dropdown.value
        with state_lock:
            current["name"] = name
            current["path"] = densified[name]
        refresh_trails()
        apply_pose(t_slider.value)

    @t_slider.on_update
    def _(_):
        apply_pose(t_slider.value)

    @cb_trail.on_update
    def _(_):
        refresh_trails()

    @cb_all_trails.on_update
    def _(_):
        refresh_trails()

    @cb_obstacles.on_update
    def _(_):
        obstacle_frame.visible = cb_obstacles.value

    refresh_trails()

    stop_flag = threading.Event()

    def animate():
        last = time.time()
        while not stop_flag.is_set():
            time.sleep(0.04)  # ~25 fps
            now = time.time()
            dt = now - last
            last = now
            if play_cb.value:
                duration = max(0.5, float(speed_slider.value))
                new_t = (float(t_slider.value) + dt / duration) % 1.0
                t_slider.value = float(new_t)
                apply_pose(new_t)

    thread = threading.Thread(target=animate, daemon=True)
    thread.start()

    apply_pose(0.0)

    print(f"\nViser running at http://localhost:{args.port}")
    print("Ctrl+C to exit.\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop_flag.set()


if __name__ == "__main__":
    main()

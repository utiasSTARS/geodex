"""Scenes of the docs robot viewer, the ``robot-scene`` directive.

A scene is ``docs/_static/robot-scenes/<name>.json`` with a poster ``<name>.jpg``. It holds
the robot's name, a path of configurations in the robot's joint order sampled at a fixed
rate, the obstacles, the traced paths, the markers, the translucent copies along the path
and the camera. A scene without a robot (``PathScene``) moves small balls along its frames
instead, for a path on a sphere. The robot models come from ``docs/tools/robot_meshes`` and live in
``docs/_static/robots/<robot>/``. The poster is the viewer's own drawing of the final pose,
taken with headless Chrome.

``Chain`` evaluates the forward kinematics of a model's ``chain.json``. The page modules
draw the traces with it, and ``pages/robot_models.py`` checks it against geodex.
"""

from __future__ import annotations

import contextlib
import html
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from browser import Browser
from still import serve

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "_static"
SCENES = STATIC / "robot-scenes"
MODELS = STATIC / "robots"

# Obstacles in a warm grey, walls lighter and see-through.
SCENE_TONE = "#b8b2a8"
WALL_TONE = "#dedbd5"
GHOST_TONE = "#b6c0cc"
MOUNT_TONE = "#8f959c"
# The surface of a manifold drawn in a scene without a robot, such as the unit sphere.
SURFACE_TONE = "#e9eef4"


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def _pose(entry: dict) -> np.ndarray:
    x, y, z, w = entry["quat"]
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = entry["xyz"]
    return T


class Chain:
    """Forward kinematics of ``docs/_static/robots/<robot>/chain.json``."""

    def __init__(self, robot: str):
        self.doc = json.loads((MODELS / robot / "chain.json").read_text())
        self.robot = robot
        self.joint_names = self.doc["joint_names"]
        self.root = self.doc["root"]
        self.joints = [dict(j, T=_pose(j["origin"]), a=np.array(j["axis"], dtype=float))
                       for j in self.doc["joints"]]
        self.ee_parent = self.doc["ee"]["parent"]
        self.ee_offset = _pose(self.doc["ee"])

    def links(self, q) -> dict[str, np.ndarray]:
        """World transform of every moving link at `q`."""
        out = {self.root: np.eye(4)}
        for j in self.joints:
            value = q[j["index"]] * j.get("multiplier", 1.0) + j.get("offset", 0.0)
            M = np.eye(4)
            if j["type"] == "prismatic":
                M[:3, 3] = j["a"] / np.linalg.norm(j["a"]) * value
            else:
                M[:3, :3] = _rotation(j["a"], value)
            out[j["child"]] = out[j["parent"]] @ j["T"] @ M
        return out

    def end_effector(self, q, point=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Position of the end effector of the robot's planning group at `q`, or of `point`
        given in the end-effector frame."""
        T = self.links(q)[self.ee_parent] @ self.ee_offset
        return T[:3, :3] @ np.asarray(point, dtype=float) + T[:3, 3]

    def bounds(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Corners of the box around the robot's visual meshes at `q`."""
        corners = []
        for link, T in self.links(q).items():
            if link not in self.doc["bounds"]:
                continue
            lo, hi = (np.array(b) for b in self.doc["bounds"][link])
            box = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
            corners.append((box @ T.T)[:, :3])
        corners = np.vstack(corners)
        return corners.min(axis=0), corners.max(axis=0)


def frame(lo, hi, direction=(1.0, -0.85, 0.55), fov: float = 40.0, aspect: float = 16 / 9,
          margin: float = 1.15):
    """A camera position and target that show the box from `lo` to `hi` from `direction`."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    center = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    half = np.radians(fov) / 2
    half = min(half, np.arctan(np.tan(half) * aspect))
    distance = margin * radius / np.sin(half)
    d = np.asarray(direction, dtype=float)
    return center + d / np.linalg.norm(d) * distance, center


def scene_objects(path: Path) -> list[dict]:
    """The boxes, cylinders and spheres of a MotionBenchMaker scene file. Objects whose id
    contains "wall" take the lighter wall tone and are see-through."""
    world = yaml.safe_load(Path(path).read_text())["world"]
    out = []
    for obj in world.get("collision_objects", []):
        wall = "wall" in obj["id"]
        for prim, pose in zip(obj["primitives"], obj["primitive_poses"]):
            entry = {"position": _round(pose["position"]),
                     "quat": _round(pose.get("orientation", [0.0, 0.0, 0.0, 1.0])),
                     "color": WALL_TONE if wall else SCENE_TONE}
            if wall:
                entry["opacity"] = 0.35
            dims = prim["dimensions"]
            if prim["type"] == "box":
                entry.update(shape="box", size=_round(dims))
            elif prim["type"] == "cylinder":
                entry.update(shape="cylinder", length=dims[0], radius=dims[1])
            elif prim["type"] == "sphere":
                entry.update(shape="sphere", radius=dims[0])
            out.append(entry)
    return out


def box(position, size, color=SCENE_TONE) -> dict:
    """An axis-aligned box of the given center and edge lengths."""
    return {"shape": "box", "position": _round(position), "size": _round(size), "color": color}


def sphere(position, radius: float, color=SCENE_TONE, opacity: float = 1.0) -> dict:
    """A sphere of the given center and radius, translucent below `opacity` 1."""
    obj = {"shape": "sphere", "position": _round(position), "radius": round(float(radius), 5),
           "color": color}
    if opacity < 1.0:
        obj["opacity"] = round(float(opacity), 3)
    return obj


def cap(axis, angle: float, color=SCENE_TONE, radius: float = 1.0) -> dict:
    """The spherical cap of half-angle `angle` around the direction `axis` on a sphere of
    `radius` around the origin."""
    axis = np.asarray(axis, dtype=float)
    return {"shape": "cap", "axis": _round(axis / np.linalg.norm(axis)),
            "angle": round(float(angle), 6), "radius": round(float(radius), 5), "color": color}


def _round(values, digits: int = 5):
    """`values` as a flat list of floats rounded to `digits` decimals."""
    return [round(float(v), digits) for v in np.asarray(values, dtype=float).ravel()]


def sample(points: np.ndarray, count: int) -> np.ndarray:
    """`count` points of a densified path, evenly spaced in index."""
    index = np.linspace(0, len(points) - 1, count).round().astype(int)
    return points[index]


def even(points, count: int) -> np.ndarray:
    """`count` points of a dense polyline, evenly spaced in length along it."""
    points = np.asarray(points, dtype=float)
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0),
                                                            axis=1))])
    at = np.linspace(0.0, length[-1], count)
    return np.stack([np.interp(at, length, points[:, k]) for k in range(points.shape[1])],
                    axis=1)


class Scene:
    """The frames, obstacles, traced paths, markers and camera of a viewer scene, played at
    `fps` frames per second.

    ``timing="path"`` pauses at both ends of the path and returns along it while the trail
    fades. ``timing="cycle"`` repeats the frames without pauses, for a motion that ends where
    it starts.
    """

    def __init__(self, name: str, frames, *, camera_position, camera_target, fov: float,
                 fps: int, floor, timing: str):
        self.name = name
        self.frames = np.asarray(frames, dtype=float)
        self.doc = {
            "fps": fps, "timing": timing,
            "camera": {"position": _round(camera_position, 4),
                       "target": _round(camera_target, 4), "fov": fov},
            "floor": floor, "objects": [], "traces": [], "markers": [],
        }

    def add_objects(self, objects: list[dict]):
        """Add obstacles, as ``scene_objects``, ``box``, ``sphere`` and ``cap`` return them."""
        self.doc["objects"] += objects

    def add_trace(self, points, color: str, width: float = 3.0, opacity: float = 0.95):
        """A path drawn up to the current frame, one point per frame."""
        points = np.asarray(points, dtype=float)
        if len(points) != len(self.frames):
            raise ValueError("a trace needs one point per frame")
        self.doc["traces"].append({"points": [_round(p, 4) for p in points], "color": color,
                                   "width": width, "opacity": opacity})

    def add_marker(self, position, color: str, radius: float = 0.015):
        """A small ball at `position`, such as the start or the goal of a trace."""
        self.doc["markers"].append({"position": _round(position, 4), "color": color,
                                    "radius": radius})

    def document(self) -> dict:
        """The scene file's contents."""
        self.doc["frames"] = [_round(q, 5) for q in self.frames]
        return self.doc

    def save(self, poster_size=(1280, 720), spheres_poster: bool = False) -> Path:
        """Write the scene and its poster. With `spheres_poster`, also write
        ``<name>-spheres.jpg`` with the collision spheres over the robot."""
        SCENES.mkdir(parents=True, exist_ok=True)
        out = SCENES / f"{self.name}.json"
        out.write_text(json.dumps(self.document(), separators=(",", ":")) + "\n")
        render_poster(self.name, poster_size)
        if spheres_poster:
            render_poster(self.name, poster_size, spheres=True)
        return out


class PathScene(Scene):
    """A viewer scene without a robot. Each mover, a ball of the given radius and color,
    follows three coordinates of every frame. Without `floor`, the scene does not have a
    floor and the camera orbits all the way around."""

    def __init__(self, name: str, frames, movers: list[dict], *, camera_position,
                 camera_target, fov: float = 40.0, fps: int = 30, floor=None,
                 timing: str = "path"):
        super().__init__(name, frames, camera_position=camera_position,
                         camera_target=camera_target, fov=fov, fps=fps, floor=floor,
                         timing=timing)
        if self.frames.shape[1] != 3 * len(movers):
            raise ValueError(f"{name}: frames need three coordinates per mover")
        self.doc["movers"] = movers


class RobotScene(Scene):
    """A viewer scene of `robot` following `frames`, one configuration per frame."""

    def __init__(self, name: str, robot: str, frames, *, camera_position, camera_target,
                 fov: float = 40.0, fps: int = 30, floor=None, timing: str = "path"):
        super().__init__(name, frames, camera_position=camera_position,
                         camera_target=camera_target, fov=fov, fps=fps,
                         floor=floor or {"center": [0.0, 0.0], "half": 1.5, "cell": 0.1,
                                         "major": 5},
                         timing=timing)
        self.robot = robot
        self.chain = Chain(robot)
        if self.frames.shape[1] != len(self.chain.joint_names):
            raise ValueError(f"{name}: frames have {self.frames.shape[1]} coordinates, "
                             f"{robot} has {len(self.chain.joint_names)}")
        if self.chain.joint_names[:3] == ["base_x_joint", "base_y_joint", "base_theta_joint"]:
            self.frames[:, 2] = np.unwrap(self.frames[:, 2])
        self.doc = {"robot": robot, "joint_names": self.chain.joint_names, **self.doc,
                    "ghosts": {"count": 0}}

    def hold(self, objects: list[dict]):
        """Objects the robot holds, as ``box`` returns them, posed in the end-effector frame
        of its planning group. They move with the robot and its translucent copies."""
        self.doc["held"] = objects

    def mount(self, size: float = 0.18):
        """Put the floor under the lowest obstacle and a stand from the floor to the robot's
        base, for a fixed-base robot mounted above the floor of its scene."""
        bottoms = [o["position"][2] - (o["size"][2] / 2 if o["shape"] == "box" else
                                       o.get("length", 2 * o.get("radius", 0.0)) / 2)
                   for o in self.doc["objects"]]
        floor = min(bottoms + [0.0])
        if floor < 0.0:
            self.doc["floor"]["z"] = round(floor, 4)
            self.doc["objects"].append(box((0.0, 0.0, floor / 2), (size, size, -floor),
                                           MOUNT_TONE))

    def trace_end_effector(self, color: str, width: float = 3.0):
        """Trace the end effector of the robot's planning group."""
        self.add_trace([self.chain.end_effector(q) for q in self.frames], color, width)

    def trace_base(self, color: str, width: float = 2.0):
        """Trace the planar base of a mobile robot on the floor."""
        self.add_trace(np.c_[self.frames[:, :2], np.full(len(self.frames), 0.004)], color,
                       width, opacity=0.9)

    def ghosts(self, count: int, color: str = GHOST_TONE, start: float = 0.08,
               end: float = 0.22):
        """`count` translucent copies along the path, from the start, that appear as the
        robot passes them."""
        self.doc["ghosts"] = {"count": count, "color": color, "from": start, "to": end}


def harness(scenes: list[str], mode: str, spheres: bool = False) -> str:
    """A page with one full-window viewer per scene, in `mode` "still" or "capture"."""
    blocks = []
    for name in scenes:
        blocks.append(
            f'<div class="geodex-robot-scene" data-mode="{mode}" '
            f'data-scene="robot-scenes/{html.escape(name)}.json" data-robots="robots/" '
            f'data-spheres="{1 if spheres else 0}" style="position:fixed;inset:0;'
            'border:0;border-radius:0"><div class="geodex-robot-stage"></div></div>')
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<link rel='stylesheet' href='robot-viewer/robot-viewer.css'>"
            "<style>html,body{margin:0;background:#fff}</style></head><body>"
            + "".join(blocks)
            + "<script type='module' src='robot-viewer/robot-viewer.js'></script>"
            "</body></html>")


@contextlib.contextmanager
def site(page: str, scenes: dict | None = None):
    """A served directory with the viewer, three.js, the models, the scenes and `page`.
    `scenes` maps the names of scenes that stay out of the docs to their contents."""
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("three", "robot-viewer", "robots"):
            os.symlink(STATIC / sub, Path(tmp) / sub)
        if scenes:
            (Path(tmp) / "robot-scenes").mkdir()
            for name, doc in scenes.items():
                (Path(tmp) / "robot-scenes" / f"{name}.json").write_text(json.dumps(doc))
        else:
            os.symlink(SCENES, Path(tmp) / "robot-scenes")
        (Path(tmp) / "index.html").write_text(page)
        with serve(Path(tmp)) as url:
            yield f"{url}/index.html"


READY = ("Promise.all(window.geodexRobotViewers.map((v) => v.ready)).then((vs) => "
         "vs.every((v) => v.stage) ? new Promise((r) => requestAnimationFrame(() => "
         "requestAnimationFrame(() => r(true)))) : false)")


def wait_ready(browser: Browser):
    """Wait until every viewer of the open page has drawn its first frame."""
    for _ in range(200):
        if browser.evaluate("(window.geodexRobotViewers || []).length > 0"):
            break
        browser.evaluate("new Promise((r) => setTimeout(r, 50))")
    if not browser.evaluate(READY):
        raise RuntimeError("the robot viewer did not start")


def render_poster(name: str, size=(1280, 720), spheres: bool = False, quality: int = 86):
    """Draw the final pose of a scene with the viewer and write ``<name>.jpg``, or
    ``<name>-spheres.jpg`` with the collision spheres."""
    from PIL import Image

    stem = f"{name}-spheres" if spheres else name
    with site(harness([name], "still", spheres)) as url, Browser(*size) as browser:
        browser.open(url)
        wait_ready(browser)
        png = SCENES / f".{stem}.png"
        browser.screenshot(png)
    Image.open(png).convert("RGB").save(SCENES / f"{stem}.jpg", quality=quality, optimize=True,
                                        progressive=True)
    png.unlink()

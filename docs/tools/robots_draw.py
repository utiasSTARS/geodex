"""Helpers shared by the robot guide assets (docs/robots).

``SphereRobot`` poses a built-in robot's sphere URDF under ``data/robots``, the model its
VAMP kernel checks, with yourdfpy. ``record_whole_body`` writes a robot viewer scene of a
mobile manipulator following a whole-body path. Paths are densified along the planner's
geodesics before they are animated or drawn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOTS = ROOT / "data" / "robots"

# Sphere URDF and joint order of every built-in robot, by registry name.
_DIRECTORY = {"fr3_arm_gripper": "fr3_gripper"}


class SphereRobot:
    """The sphere model of a built-in robot, posed with yourdfpy.

    ``joints`` lists the joints of the robot's first planning group in the planner's
    coordinate order. ``pose(q)`` takes a configuration straight from a plan.
    """

    def __init__(self, name: str):
        import yourdfpy

        logging.getLogger("yourdfpy").setLevel(logging.ERROR)
        directory = ROBOTS / _DIRECTORY.get(name, name)
        meta = yaml.safe_load((directory / "robot.yaml").read_text())
        urdf = meta.get("spherized_urdf", meta["urdf"])
        self.name = name
        self.group = next(iter(meta["planning_groups"].values()))
        self.joints = list(self.group["joints"])
        self.ee_link = self.group["default_ee_link"]
        self.urdf = yourdfpy.URDF.load(str(directory / urdf), load_meshes=False,
                                       build_collision_scene_graph=False,
                                       load_collision_meshes=False)
        self.spheres = {}
        for link in self.urdf.robot.links:
            balls = []
            for c in link.collisions:
                if c.geometry is not None and c.geometry.sphere is not None:
                    origin = np.eye(4) if c.origin is None else np.asarray(c.origin)
                    balls.append((origin[:3, 3], float(c.geometry.sphere.radius)))
            if balls:
                self.spheres[link.name] = balls
        self._defaults = {j: 0.0 for j in self.urdf.actuated_joint_names}

    def pose(self, q) -> dict[str, np.ndarray]:
        """World transform of every link that carries spheres, plus the end effector."""
        cfg = dict(self._defaults)
        cfg.update(zip(self.joints, np.asarray(q, dtype=float)))
        self.urdf.update_cfg(cfg)
        links = list(self.spheres) + [self.ee_link]
        return {link: self.urdf.get_transform(link, self.urdf.base_link) for link in links}

    def end_effector(self, q) -> np.ndarray:
        """Position of the planning group's end effector at `q`."""
        return self.pose(q)[self.ee_link][:3, 3]


def densify(path, geodesic, pieces_of) -> np.ndarray:
    """Points along `geodesic(a, b, t)` between consecutive waypoints, `pieces_of(a, b)`
    pieces per edge."""
    path = [np.asarray(p, dtype=float) for p in path]
    points = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(pieces_of(a, b)))
        points += [np.asarray(geodesic(a, b, k / n)) for k in range(1, n + 1)]
    return np.array(points)


def whole_body_geodesic(n_arm: int):
    """The geodesic of SE(2) x R^n that the whole-body planners interpolate along. It follows
    the SE(2) group geodesic in the base and a straight line in the arm joints."""
    import geodex

    space = geodex.Product([geodex.SE2(), geodex.Euclidean(n_arm)])
    return space.geodesic


def whole_body_pieces(a, b, step_xy=0.02, step_theta=0.04, step_joint=0.03):
    """Pieces per edge. In each piece the base moves at most 2 cm and turns at most 0.04 rad,
    and each arm coordinate moves at most 0.03."""
    d = np.abs(b - a)
    d[2] = abs((b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi)
    return np.ceil(max(np.hypot(d[0], d[1]) / step_xy, d[2] / step_theta,
                       d[3:].max() / step_joint if len(d) > 3 else 0.0, 1.0))


def frames_along(points: np.ndarray, count: int) -> np.ndarray:
    """`count` points of a densified path, evenly spaced in index."""
    index = np.linspace(0, len(points) - 1, count).round().astype(int)
    return points[index]


def record_whole_body(name: str, robot: SphereRobot, path, scene_file: Path, color: str, *,
                      camera_position, camera_look_at, frame_count: int = 240,
                      floor=((-3.0, 3.0), (-2.5, 2.5))):
    """Write a robot viewer scene of a mobile manipulator following a whole-body path, with the
    scene, faint copies along the path, the base trace on the floor and the trace of the end
    effector. Returns the densified path."""
    from robot_scene import RobotScene, scene_objects
    from style import INK_2

    path = np.asarray(path, dtype=float)
    dense = densify(path, whole_body_geodesic(path.shape[1] - 3), whole_body_pieces)
    (x_lo, x_hi), (y_lo, y_hi) = floor
    half = 0.5 * max(x_hi - x_lo, y_hi - y_lo) + 0.5
    scene = RobotScene(name, robot.name, frames_along(dense, frame_count),
                       camera_position=camera_position, camera_target=camera_look_at, fov=45.0,
                       floor={"center": [(x_lo + x_hi) / 2, (y_lo + y_hi) / 2], "half": half,
                              "cell": 0.25, "major": 4})
    scene.add_objects(scene_objects(scene_file))
    scene.ghosts(3)
    scene.trace_base(INK_2)
    scene.trace_end_effector(color)
    scene.save()
    return dense

"""Robot models of the docs robot viewer.

``docs/_static/robots/<robot>/spheres.json``
    The collision spheres of the robot's VAMP model, each in the frame of the moving link
    that carries it, for the viewer's sphere overlay.

The same run checks the viewer's models against geodex. For every robot it compares the
joint order of ``chain.json`` with ``geodex.vamp.robot_joint_names``, places every sphere
of ``geodex.vamp.robot_spheres`` with the viewer's kinematics at random configurations, and
compares the end effector with the sphere URDF in ``data/robots``. The part ``browser``
repeats the forward kinematics in the viewer itself, in headless Chrome, against the same
configurations. The models come from ``docs/tools/robot_meshes/build.py``.
"""

from __future__ import annotations

import json

import numpy as np

from robot_scene import MODELS, Chain, harness, site, wait_ready
from robots_draw import SphereRobot

ROBOTS = ("panda", "ur5", "baxter", "pr2", "fr3_arm_gripper", "stretch3", "stretch4",
          "ridgeback_ur5e", "husky_ur5e")
CLASSES = {"panda": "Panda", "ur5": "UR5", "baxter": "Baxter", "pr2": "PR2",
           "fr3_arm_gripper": "Fr3Gripper", "stretch3": "Stretch3", "stretch4": "Stretch4",
           "ridgeback_ur5e": "RidgebackUR5e", "husky_ur5e": "HuskyUR5e"}
SEED = 7
COUNT = 8
# Largest distance in meters between geodex and the viewer. VAMP computes the sphere centers
# in single precision.
TOLERANCE = 1e-6
SPHERE_TOLERANCE = 1e-5


def configurations(name: str, count: int = COUNT, seed: int = SEED) -> np.ndarray:
    """`count` configurations drawn uniformly inside the robot's joint limits, the base of a
    mobile robot inside a 4 m square."""
    import geodex

    lo, hi = (np.array(v) for v in getattr(geodex.robots, CLASSES[name])().joint_limits())
    if name in ("stretch3", "stretch4", "ridgeback_ur5e", "husky_ur5e"):
        lo[:2], hi[:2] = -2.0, 2.0
    return np.random.default_rng(seed).uniform(lo, hi, size=(count, len(lo)))


def spheres(name: str) -> tuple[dict, float]:
    """The VAMP spheres of `name` by moving link, in link coordinates, and the largest
    distance between a sphere of geodex and the same sphere placed by the viewer's chain."""
    import geodex

    chain = Chain(name)
    qs = configurations(name)
    balls = np.array([geodex.vamp.robot_spheres(name, q) for q in qs])
    frames = [chain.links(q) for q in qs]
    table, worst = {}, 0.0
    for i in range(balls.shape[1]):
        best = None
        for link in frames[0]:
            local = np.array([np.linalg.inv(f[link]) @ np.append(b[i, :3], 1.0)
                              for f, b in zip(frames, balls)])[:, :3]
            spread = float(np.abs(local - local.mean(axis=0)).max())
            if best is None or spread < best[0]:
                best = (spread, link, local.mean(axis=0))
        spread, link, center = best
        worst = max(worst, spread)
        table.setdefault(link, []).append(
            [round(float(v), 5) for v in center] + [round(float(balls[0, i, 3]), 5)])
    return table, worst


def check(name: str) -> dict:
    """Joint order, sphere placement and end-effector position of the viewer's model of
    `name` against geodex."""
    import geodex

    chain = Chain(name)
    order = list(geodex.vamp.robot_joint_names(name)) == chain.joint_names
    table, sphere_error = spheres(name)
    reference = SphereRobot(name)
    qs = configurations(name, seed=SEED + 1)
    ee_error = max(float(np.linalg.norm(chain.end_effector(q) - reference.end_effector(q)))
                   for q in qs)
    return {"order": order, "spheres": sum(len(v) for v in table.values()),
            "sphere_error": sphere_error, "ee_error": ee_error, "table": table}


def write_spheres():
    rows = []
    for name in ROBOTS:
        result = check(name)
        (MODELS / name / "spheres.json").write_text(
            json.dumps(result["table"], separators=(",", ":")) + "\n")
        rows.append((name, result))
    print("robot             joint order  spheres  sphere error (m)  end effector (m)")
    for name, r in rows:
        print(f"{name:17s} {'same' if r['order'] else 'DIFFERS':11s}  {r['spheres']:7d}  "
              f"{r['sphere_error']:16.2e}  {r['ee_error']:16.2e}")
    bad = [n for n, r in rows if not r["order"] or r["sphere_error"] > SPHERE_TOLERANCE
           or r["ee_error"] > TOLERANCE]
    if bad:
        raise RuntimeError(f"the viewer's models of {bad} differ from geodex")


def browser():
    """Forward kinematics of the viewer in headless Chrome against ``Chain``, at the
    configurations of ``check``, through the model and through the posed scene graph."""
    from browser import Browser

    for name in ROBOTS:
        qs = configurations(name, seed=SEED + 1)
        chain = Chain(name)
        scene = f"robots-gallery-{name.replace('_', '-')}"
        script = (
            "(() => { const v = window.geodexRobotViewers[0]; const out = [];"
            f"for (const q of {json.dumps(qs.tolist())}) {{"
            "const ee = v.model.eeMatrix(q).elements;"
            "v.robot.setQ(q); v.robot.updateMatrixWorld(true);"
            "const links = {}; for (const [k, g] of Object.entries(v.robot.links)) "
            "links[k] = g.matrixWorld.elements.slice(12, 15);"
            "out.push({ee: [ee[12], ee[13], ee[14]], links}); } return out; })()")
        with site(harness([scene], "still")) as url, Browser(320, 240) as b:
            b.open(url)
            wait_ready(b)
            result = b.evaluate(script)
        ee = max(float(np.linalg.norm(np.array(r["ee"]) - chain.end_effector(q)))
                 for r, q in zip(result, qs))
        link = max(float(np.linalg.norm(np.array(p) - chain.links(q)[k][:3, 3]))
                   for r, q in zip(result, qs) for k, p in r["links"].items())
        print(f"{name:17s} viewer end effector {ee:.2e} m, link origins {link:.2e} m")
        if ee > TOLERANCE or link > TOLERANCE:
            raise RuntimeError(f"the viewer places {name} differently")


def generate(parts=("spheres",)):
    if "spheres" in parts:
        write_spheres()
    if "browser" in parts:
        browser()

#!/usr/bin/env python3
"""Build the robot models of the docs robot viewer.

For every robot the docs show, the script writes three files under
``docs/_static/robots/<robot>/``.

``chain.json``
    The kinematic tree of the moving links. Every fixed link is folded into its moving
    parent. Each moving joint names the configuration coordinate that moves it, in the order
    of the robot's first planning group in ``data/robots/<robot>/robot.yaml``, the order
    ``geodex.vamp.robot_joint_names`` returns. A mobile robot starts with the planar base
    joints ``base_x_joint``, ``base_y_joint`` and ``base_theta_joint``.
``<robot>.glb``
    The visual meshes with their materials, one node per moving link, simplified and
    compressed with meshoptimizer. A robot that keeps its upstream textures carries them as
    WebP images.
``<robot>_ghost.glb``
    One light mesh per link for the translucent copies along a path.

The kinematics are geodex's own. The vendored robots use the URDF in ``data/robots``, and
the generated robots use the planning model of ``scripts/robotgen`` built from the same
pinned upstream description, with the planar base in front of a mobile robot. The visual
meshes come from the upstream descriptions pinned in ``third_party/robot_descriptions.cmake``
and ``docs/tools/sources.toml``. Parts without a redistributable mesh are drawn as
simple primitives.

Usage::

    pixi run --manifest-path docs/tools/robot_meshes/pixi.toml build [robot...]
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import scipy.ndimage
import trimesh
import yaml
import yourdfpy
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "scripts" / "robotgen"))

import robotgen  # noqa: E402
from fetch import CACHE, fetch  # noqa: E402

logging.getLogger("yourdfpy").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

OUT = ROOT / "docs" / "_static" / "robots"
DATA = ROOT / "data" / "robots"

# Simplification of the full and the ghost meshes, as glTF-Transform's `simplify` ratio and
# error bound (a fraction of the mesh radius).
FULL = {"ratio": 0.3, "error": 0.0015}
GHOST = {"ratio": 0.05, "error": 0.02}

# Largest angle between two faces that share a vertex normal in the full meshes.
CREASE = math.radians(30.0)

# Metalness and roughness of the upstream textured materials, and the WebP quality of their
# textures.
TEXTURED = (0.0, 0.5)
WEBP_QUALITY = 80

# Dark grey of the tool primitives, and the joint value of the Panda's open fingers.
TOOL_GREY = (0.23, 0.24, 0.26, 1.0)
PANDA_OPEN = 0.04

# The Robotiq 2F-85 of ros2_robotiq_gripper, placed at a link of a geodex model. The UR5's
# gripper base sits 7.9 mm below the PickNik base frame.
ROBOTIQ = {
    "ur5": {"link": "robotiq_85_base_link", "xyz": (0.0, 0.0, 0.0079), "rpy": (0.0, 0.0, 0.0)},
}

# The named materials of the Clearpath platforms, which a robot's top-level xacro includes.
CLEARPATH_COLORS = ("CLEARPATH_COMMON", "clearpath_platform_description/urdf/common.urdf.xacro")

# Finishes of real robot parts, as an sRGB base color, metalness and roughness.
FINISHES = {
    "white-plastic": ("#e9ebea", 0.0, 0.45),
    "light-grey-plastic": ("#c3c6ca", 0.0, 0.45),
    "grey-plastic": ("#85888d", 0.0, 0.45),
    "dark-grey-plastic": ("#44474c", 0.0, 0.45),
    "black-plastic": ("#1f2023", 0.0, 0.5),
    "rubber": ("#1a1a1b", 0.0, 0.85),
    "black-anodized": ("#26282c", 0.5, 0.4),
    "graphite-anodized": ("#44474d", 0.6, 0.38),
    "grey-anodized": ("#8c9097", 0.7, 0.35),
    "aluminum": ("#c2c6cc", 0.8, 0.32),
    "marker": ("#d4d6d9", 0.0, 0.7),
    "clearpath-yellow": ("#ddb314", 0.0, 0.6),
}

# Finishes of the robots whose upstream colors are crude or wrong. A rule (links, color,
# finish) applies to the visuals of every source URDF link that matches the fnmatch pattern
# `links` and, when `color` is not None, whose upstream base color has the hex code `color`.
# The first matching rule wins. A rule with the finish None and a visual without a rule keep
# the upstream material. The rules do not apply to the textured visuals of a robot with
# `textures`. The Panda, the FR3, the UR5, Baxter and the UR5e arms keep their upstream
# materials.
MATERIALS = {
    # Black body, dark anodized mast and arm segments, spring steel fingers.
    "stretch3": [("link_*_wheel", None, "rubber"), ("caster_link", None, "rubber"),
                 ("link_mast", None, "black-anodized"),
                 ("link_arm_l*", None, "graphite-anodized"),
                 ("link_gripper_finger_*", None, "aluminum"),
                 ("link_gripper_fingertip_*", None, "rubber"),
                 ("link_aruco_*", None, "marker"),
                 ("camera_link", None, None), ("gripper_camera_link", None, None),
                 ("base_imu", None, None), ("*", None, "black-plastic")],
    # White shells on the base, the lift and the head, grey mast, aluminum arm, black wrist.
    "stretch4": [("wheel_*", None, "rubber"), ("mast_link", None, "grey-anodized"),
                 ("arm_l*", None, "aluminum"), ("base_link", None, "white-plastic"),
                 ("lift_link", None, "white-plastic"), ("head_link", None, "white-plastic"),
                 ("*aruco*", None, "marker"), ("wrist_reflector_link", None, "marker"),
                 ("*", None, "black-plastic")],
    # The Collada meshes keep their upstream textures. The untextured casters and roll links
    # are dark grey, and the bellow is black.
    "pr2": [("base_bellow_link", None, "black-plastic"), ("*", None, "dark-grey-plastic")],
    # Yellow top chassis, black body and tires, black anodized top plate and rails.
    "husky_ur5e": [("*_wheel_link", None, "rubber"),
                   ("top_chassis_link", "cccc00", "clearpath-yellow"),
                   ("top_plate*", None, "black-anodized"), ("base_link", None, "black-plastic")],
    # Yellow side covers, black body and wheels, aluminum riser.
    "ridgeback_ur5e": [("arm_*", None, None), ("*_lights_link", None, None),
                       ("*_side_cover_link", None, "clearpath-yellow"),
                       ("*_wheel_link", None, "rubber"), ("riser_link", None, "aluminum"),
                       ("*", None, "black-plastic")],
    # Yellow covers on a black body, rubber or aluminum mecanum wheels.
    "dingo_d": [("chassis_link", "ffe600", "clearpath-yellow"),
                ("chassis_link", "131315", "black-plastic"), ("*_wheel_link", None, "rubber"),
                ("rear_caster", None, "rubber")],
    "dingo_o": [("chassis_link", "ffe600", "clearpath-yellow"),
                ("chassis_link", "131315", "black-plastic"), ("*_wheel_link", None, "aluminum")],
}

ROBOTS = {
    "panda": {
        "label": "Franka Panda",
        "urdf": DATA / "panda" / "urdf" / "panda.urdf",
        "fixed": {"panda_finger_joint1": PANDA_OPEN, "panda_finger_joint2": PANDA_OPEN},
    },
    "ur5": {
        "label": "Universal Robots UR5 with a Robotiq 2F-85",
        "urdf": DATA / "ur5" / "ur5.urdf",
        "robotiq": True,
        # The FT 300 sensor as a cylinder inside the box of the geodex model.
        "primitives": [{"link": "fts_robotside", "cylinder": (0.0375, 0.0415),
                        "xyz": (0.000165, -0.020745, 0.0), "rpy": (math.pi / 2, 0.0, 0.0)}],
        "drop_visuals": ["fts_robotside", "robotiq_85_"],
    },
    "baxter": {"label": "Rethink Baxter", "urdf": DATA / "baxter" / "baxter.urdf",
               "full": {"ratio": 0.04, "error": 0.003}},
    # The textures shrink to 512 pixels. The finger joints take the origins of the open
    # grippers of the sphere model.
    "pr2": {"label": "Willow Garage PR2", "urdf": DATA / "pr2" / "pr2.urdf",
            "full": {"ratio": 0.2, "error": 0.002}, "textures": 512,
            "origins": (DATA / "pr2" / "pr2_spherized.urdf", "*_gripper_?_finger_joint")},
    "fr3_arm_gripper": {"label": "Franka FR3 with a Robotiq 2F-85", "recipe": "fr3_arm_gripper"},
    "stretch3": {"label": "Hello Robot Stretch 3", "recipe": "stretch3",
                 "full": {"ratio": 0.14, "error": 0.002}},
    # The docking contact mesh covers the whole base shell. The viewer does not draw it.
    "stretch4": {"label": "Hello Robot Stretch 4", "recipe": "stretch4",
                 "full": {"ratio": 0.2, "error": 0.002}, "drop_visuals": ["docking_contact_link"]},
    "ridgeback_ur5e": {"label": "Clearpath Ridgeback with a UR5e", "recipe": "ridgeback_ur5e",
                       "full": {"ratio": 0.2, "error": 0.002}, "materials": [CLEARPATH_COLORS]},
    "husky_ur5e": {"label": "Clearpath Husky with a UR5e", "recipe": "husky_ur5e",
                   "full": {"ratio": 0.2, "error": 0.002}, "materials": [CLEARPATH_COLORS]},
    # The mobile bases of the navigation pages, a planar base (x, y, theta) and no arm.
    "dingo_d": {"label": "Clearpath Dingo-D", "platform": "dingo_d.urdf.xacro",
                "full": {"ratio": 0.8, "error": 0.001}},
    "dingo_o": {"label": "Clearpath Dingo-O", "platform": "dingo_o.urdf.xacro",
                "full": {"ratio": 0.1, "error": 0.002}},
}
PLANAR = ["base_x_joint", "base_y_joint", "base_theta_joint"]


# Sources of the visual meshes of each model and their licenses, for the NOTICE next to the
# models. The license texts come from THIRD_PARTY_LICENSES.txt.
MESH_SOURCES = [
    ("panda", "Franka Panda, franka_ros edba362 (data/robots/panda/meshes)", "Apache-2.0"),
    ("ur5", "UR5, ros-industrial/universal_robot f287d22 (data/robots/ur5/meshes)",
     "BSD-3-Clause"),
    ("ur5", "Robotiq 2F-85, PickNikRobotics/ros2_robotiq_gripper a74d007", "BSD-3-Clause"),
    ("fr3_arm_gripper",
     "Robotiq 2F-85 and coupling, PickNikRobotics/ros2_robotiq_gripper a74d007",
     "BSD-3-Clause"),
    ("fr3_arm_gripper", "Franka FR3, frankarobotics/franka_description 2.9.0 (7aeeddc)",
     "Apache-2.0"),
    ("baxter", "Rethink Baxter, RethinkRobotics/baxter_common 6c4b0f3 (data/robots/baxter)",
     "BSD-3-Clause"),
    ("pr2", "Willow Garage PR2, PR2/pr2_common 9a8e4fb (data/robots/pr2)", "BSD-3-Clause"),
    ("stretch3", "Hello Robot Stretch 3, hello-robot-stretch-urdf 0.1.2 (PyPI)", "Clear BSD"),
    ("stretch4", "Hello Robot Stretch 4, hello-robot-stretch4-urdf 2026.8.21 (PyPI)",
     "Clear BSD"),
    ("ridgeback_ur5e, husky_ur5e, dingo_d, dingo_o",
     "Clearpath Ridgeback, Husky and Dingo, clearpathrobotics/clearpath_common 2.9.17 "
     "(726d7fb)", "BSD-3-Clause"),
    ("ridgeback_ur5e, husky_ur5e",
     "UR5e, UniversalRobots/Universal_Robots_ROS2_Description 4.3.1 (ae33328)",
     "BSD-3-Clause"),
]
LICENSE_TEXTS = ["franka-description", "ur5-robotiq", "baxter-common", "pr2-description",
                 "stretch-urdf", "clearpath-common", "universal-robots-description",
                 "ros2-robotiq-gripper"]


def write_notice() -> None:
    """Write docs/_static/robots/NOTICE with the mesh sources and their license texts."""
    lines = ["Robot models of the geodex docs robot viewer, built by",
             "docs/tools/robot_meshes/build.py. Each model holds meshes simplified",
             "from the sources below. The PR2 model also holds downscaled copies of",
             "the textures of its meshes. The FT 300 sensor of the UR5 is a cylinder.", ""]
    width = max(len(m) for m, _, _ in MESH_SOURCES)
    for models, source, spdx in MESH_SOURCES:
        lines.append(f"  {models:{width}s}  {source}  [{spdx}]")
    lines.append("")
    texts = license_sections((ROOT / "THIRD_PARTY_LICENSES.txt").read_text())
    for name in LICENSE_TEXTS:
        lines += [f"----- {name} (THIRD_PARTY_LICENSES.txt) -----", texts[name], ""]
    (OUT / "NOTICE").write_text("\n".join(lines))


def license_sections(text: str) -> dict[str, str]:
    """The sections of THIRD_PARTY_LICENSES.txt by name. A section starts with its name
    between two lines of equals signs."""
    bar = "=" * 80
    parts = text.split(bar + "\n")
    sections = {}
    for k in range(1, len(parts) - 1, 2):
        sections[parts[k].strip()] = parts[k + 1].strip()
    return sections


# ------------------------------------------------------------------------------ transforms


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """Rotation matrix of URDF roll, pitch and yaw angles."""
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Homogeneous transform of a translation and URDF roll, pitch and yaw angles."""
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def matrix_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch and yaw of a rotation matrix, the URDF convention."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])
    return math.atan2(-R[1, 2], R[1, 1]), pitch, 0.0


def quaternion(R: np.ndarray) -> list[float]:
    """Unit quaternion (x, y, z, w) of a rotation matrix, three.js order, with w >= 0."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        q = [(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s,
             0.25 / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
             (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
             (R[1, 0] - R[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [round(float(v), 12) for v in q]


def pose_json(T: np.ndarray) -> dict:
    """A transform as the translation and quaternion of chain.json."""
    return {"xyz": [round(float(v), 9) for v in T[:3, 3]], "quat": quaternion(T[:3, :3])}


def origin_of(el: ET.Element | None) -> np.ndarray:
    """Transform of a URDF origin element, the identity when it is missing."""
    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    return transform(xyz, rpy)


def set_origin(parent: ET.Element, T: np.ndarray) -> None:
    """Replace the origin element of `parent` with the transform `T`."""
    for el in parent.findall("origin"):
        parent.remove(el)
    ET.SubElement(parent, "origin", {"xyz": " ".join(repr(float(v)) for v in T[:3, 3]),
                                     "rpy": " ".join(repr(v) for v in matrix_rpy(T[:3, :3]))})


def joint_motion(kind: str, axis: np.ndarray, value: float) -> np.ndarray:
    """Transform of a revolute or prismatic joint at `value` about or along `axis`."""
    T = np.eye(4)
    if kind == "prismatic":
        T[:3, 3] = axis * value
    elif kind in ("revolute", "continuous"):
        T[:3, :3] = trimesh.transformations.rotation_matrix(value, axis)[:3, :3]
    return T


# ------------------------------------------------------------------------------ models


def planning_group(name: str) -> tuple[list[str], str]:
    """Joints of the robot's first planning group, in configuration order, and its end
    effector link. A mobile base of the navigation pages has the planar joints and ends at
    its base link."""
    if "platform" in ROBOTS[name]:
        return PLANAR, "base_link"
    directory = DATA / ("fr3_gripper" if name == "fr3_arm_gripper" else name)
    meta = yaml.safe_load((directory / "robot.yaml").read_text())
    group = next(iter(meta["planning_groups"].values()))
    return list(group["joints"]), group["default_ee_link"]


def resolve_meshes(root: ET.Element, base: Path) -> None:
    """Make every visual mesh path of a vendored URDF absolute."""
    for mesh in (m for visual in root.iter("visual") for m in visual.iter("mesh")):
        name = mesh.get("filename")
        if name.startswith("package://"):
            name = name[len("package://"):]
            name = name.split("/", 1)[1] if not (base / name).exists() else name
        path = (base / name).resolve()
        if not path.exists():
            raise SystemExit(f"mesh {mesh.get('filename')} not found under {base}")
        mesh.set("filename", str(path))


def vendored_model(name: str, cfg: dict) -> ET.Element:
    """The URDF of a fixed-base robot in data/robots, with absolute visual mesh paths."""
    urdf = cfg["urdf"]
    root = ET.parse(urdf).getroot()
    resolve_meshes(root, urdf.parent)
    return root


def generated_model(name: str, cfg: dict) -> ET.Element:
    """The robotgen planning model of a generated robot with its visuals, and the planar base
    of a mobile robot."""
    cache = CACHE / "robotgen"
    robotgen.fetch(cache, ROOT)
    recipe = json.loads((ROOT / "scripts" / "robotgen" / "robots" /
                         f"{cfg['recipe']}.json").read_text())
    root = robotgen.planning_model(recipe, cache)
    if "drive" in recipe:
        root = robotgen.with_planar_base(root, recipe["base_link"],
                                         recipe.get("base_height", 0.0))
    # The planning model keeps links and joints only. The visuals refer to the named
    # materials of the source description.
    source, _ = robotgen.load_source_urdf(recipe, cache)
    materials = source.findall("material")
    for package, rel in cfg.get("materials", []):
        materials += ET.parse(robotgen.source_dir(cache, package) / rel).getroot().findall(
            "material")
    for material in materials:
        root.insert(0, copy.deepcopy(material))
    return root


def platform_model(name: str, cfg: dict) -> ET.Element:
    """A Clearpath base from its pinned macros, every wheel fixed, with the planar base in
    front and the lowest point of its meshes on the floor."""
    cache = CACHE / "robotgen"
    robotgen.fetch(cache, ROOT)
    recipe = {"source": {"xacro_file": str(HERE / "platforms" / cfg["platform"]),
                         "packages": {"clearpath_platform_description":
                                      ["CLEARPATH_COMMON", "clearpath_platform_description"],
                                      "clearpath_control": ["CLEARPATH_COMMON",
                                                            "clearpath_control"]}}}
    root = ET.fromstring(robotgen.expand_xacro(recipe, cache))
    packages = robotgen.package_dirs(recipe, cache)
    for el in list(root):
        if el.tag not in ("link", "joint", "material"):
            root.remove(el)
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            robotgen.fix_joint(joint)
    for mesh in (m for visual in root.iter("visual") for m in visual.iter("mesh")):
        mesh.set("filename", robotgen.resolve_mesh(mesh.get("filename"), cache, packages))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "base.urdf"
        ET.ElementTree(root).write(path)
        urdf = yourdfpy.URDF.load(str(path), build_scene_graph=True, load_meshes=True,
                                  build_collision_scene_graph=False, load_collision_meshes=False)
        lowest = float(urdf.scene.bounds[0][2] - urdf.get_transform("base_link")[2, 3])
    return robotgen.with_planar_base(root, "base_link", -lowest)


def add_visual(link: ET.Element, T: np.ndarray, geometry: ET.Element, rgba=None) -> None:
    """Add a visual element with `geometry` at `T` to `link`, colored `rgba` when given."""
    visual = ET.SubElement(link, "visual")
    set_origin(visual, T)
    visual.append(geometry)
    if rgba is not None:
        material = ET.SubElement(visual, "material", {"name": ""})
        ET.SubElement(material, "color", {"rgba": " ".join(repr(float(c)) for c in rgba)})


def robotiq_parts() -> list[tuple[Path, np.ndarray]]:
    """The visual meshes of the ros2_robotiq_gripper 2F-85 with the open fingers, each
    with its pose in the gripper's base frame."""
    source = next(fetch("ros2_robotiq_gripper").iterdir()) / "robotiq_description"
    macro = (source / "urdf" / "robotiq_2f_85_macro.urdf.xacro").read_text()
    macro = macro.replace("${prefix}", "")
    xml = ET.fromstring(macro)
    body = next(el for el in xml if el.tag.endswith("}macro"))
    joints = {j.find("child").get("link"): j for j in body.findall("joint")}
    base = "robotiq_85_base_link"

    def pose(link: str) -> np.ndarray:
        T = np.eye(4)
        while link != base:
            joint = joints[link]
            T = origin_of(joint.find("origin")) @ T
            link = joint.find("parent").get("link")
        return T

    parts = []
    for link in body.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh").get("filename")
            rel = mesh.split("robotiq_description/", 1)[1]
            parts.append((source / rel, pose(link.get("name")) @ origin_of(visual.find("origin"))))
    return parts


def add_robotiq(root: ET.Element, name: str) -> None:
    mount = ROBOTIQ[name]
    link = next(l for l in root.findall("link") if l.get("name") == mount["link"])
    T_mount = transform(mount["xyz"], mount["rpy"])
    for path, T in robotiq_parts():
        geometry = ET.Element("geometry")
        ET.SubElement(geometry, "mesh", {"filename": str(path)})
        add_visual(link, T_mount @ T, geometry)


def add_primitives(root: ET.Element, primitives: list[dict]) -> None:
    """Add the cylinders of a robot's configuration as grey visuals."""
    links = {l.get("name"): l for l in root.findall("link")}
    for p in primitives:
        geometry = ET.Element("geometry")
        radius, length = p["cylinder"]
        ET.SubElement(geometry, "cylinder", {"radius": repr(radius), "length": repr(length)})
        add_visual(links[p["link"]], transform(p["xyz"], p["rpy"]), geometry, TOOL_GREY)


def copy_origins(root: ET.Element, urdf: Path, pattern: str) -> None:
    """Give every joint whose name matches the fnmatch `pattern` the origin of the joint of
    the same name in the URDF `urdf`."""
    source = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    for joint in root.findall("joint"):
        if fnmatch.fnmatchcase(joint.get("name"), pattern):
            set_origin(joint, origin_of(source[joint.get("name")].find("origin")))


def drop_visuals(root: ET.Element, prefixes: list[str]) -> None:
    """Remove the visuals of every link whose name starts with one of `prefixes`."""
    for link in root.findall("link"):
        if any(link.get("name").startswith(p) for p in prefixes):
            for visual in link.findall("visual"):
                link.remove(visual)


def visual_model(name: str, cfg: dict) -> ET.Element:
    """The robot's geodex kinematics with the visual meshes of its configuration and no
    collision or inertial elements."""
    if "recipe" in cfg:
        root = generated_model(name, cfg)
    elif "platform" in cfg:
        root = platform_model(name, cfg)
    else:
        root = vendored_model(name, cfg)
    root = copy.deepcopy(root)
    if cfg.get("origins"):
        copy_origins(root, *cfg["origins"])
    if cfg.get("drop_visuals"):
        drop_visuals(root, cfg["drop_visuals"])
    if cfg.get("robotiq"):
        add_robotiq(root, name)
    if cfg.get("primitives"):
        add_primitives(root, cfg["primitives"])
    for link in root.findall("link"):
        for tag in ("collision", "inertial"):
            for el in link.findall(tag):
                link.remove(el)
    return root


# ------------------------------------------------------------------------------ chain


class Tree:
    """The joints of a URDF with the configuration coordinate that moves each moving one."""

    def __init__(self, root: ET.Element, joints: list[str], fixed: dict[str, float]):
        self.joints = {}
        for j in root.findall("joint"):
            axis = j.find("axis")
            self.joints[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"),
                "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                "origin": origin_of(j.find("origin")),
                "axis": np.array([float(v) for v in axis.get("xyz").split()])
                if axis is not None else np.array([1.0, 0.0, 0.0]),
                "mimic": j.find("mimic"),
            }
        children = set(self.joints)
        self.root = next(l.get("name") for l in root.findall("link")
                         if l.get("name") not in children)
        by_name = {j["name"]: j for j in self.joints.values()}
        for i, name in enumerate(joints):
            by_name[name]["index"], by_name[name]["multiplier"] = i, 1.0
        for j in self.joints.values():
            mimic = j["mimic"]
            if mimic is not None and mimic.get("joint") in joints:
                j["index"] = joints.index(mimic.get("joint"))
                j["multiplier"] = float(mimic.get("multiplier", "1"))
                j["offset"] = float(mimic.get("offset", "0"))
            j["value"] = fixed.get(j["name"], 0.0)
        missing = [n for n in joints if "index" not in by_name[n]]
        if missing:
            raise SystemExit(f"planning joints {missing} are not in the model")

    def moves(self, link: str) -> bool:
        """True when the joint above `link` follows a configuration coordinate."""
        joint = self.joints.get(link)
        return joint is not None and "index" in joint and joint["type"] != "fixed"

    def local(self, joint: dict) -> np.ndarray:
        """Transform of a non-moving joint at its fixed value."""
        return joint["origin"] @ joint_motion(joint["type"], joint["axis"], joint["value"])

    def moving_ancestor(self, link: str) -> tuple[str, np.ndarray]:
        """The first moving link at or above `link` and the transform from it to `link`."""
        T = np.eye(4)
        while link != self.root and not self.moves(link):
            joint = self.joints[link]
            T = self.local(joint) @ T
            link = joint["parent"]
        return link, T

    def chain(self) -> list[dict]:
        out = []
        for link, joint in self.joints.items():
            if not self.moves(link):
                continue
            parent, T_fixed = self.moving_ancestor(joint["parent"])
            entry = {"name": joint["name"], "type": "prismatic" if joint["type"] == "prismatic"
                     else "revolute", "parent": parent, "child": link,
                     "origin": pose_json(T_fixed @ joint["origin"]),
                     "axis": [round(float(v), 12) for v in joint["axis"]],
                     "index": joint["index"]}
            if joint["multiplier"] != 1.0 or joint.get("offset"):
                entry["multiplier"] = joint["multiplier"]
                entry["offset"] = joint.get("offset", 0.0)
            out.append(entry)
        # Parents before children.
        order, placed = [], {self.root}
        while out:
            ready = [e for e in out if e["parent"] in placed]
            if not ready:
                raise SystemExit("the moving links do not form a tree")
            for e in ready:
                order.append(e)
                placed.add(e["child"])
                out.remove(e)
        return order


# ------------------------------------------------------------------------------ meshes


def average_color(geometry: trimesh.Trimesh) -> tuple:
    """Base color (0 to 255), emissive color and roughness of a mesh's material. A textured
    mesh takes the mean color of its texture."""
    visual = geometry.visual
    material = getattr(visual, "material", None)
    rough = None
    emissive = np.zeros(3)
    if material is not None:
        image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
        if image is not None and getattr(visual, "uv", None) is not None:
            base = visual.to_color().vertex_colors.astype(float).mean(axis=0)
        else:
            base = getattr(material, "baseColorFactor", None)
            if base is None:
                base = getattr(material, "diffuse", None)
            base = np.array(base if base is not None else [180, 180, 180, 255], dtype=float)
            if base.max() <= 1.0:
                base = base * 255.0
        if getattr(material, "emissiveFactor", None) is not None:
            emissive = np.asarray(material.emissiveFactor, dtype=float)
        rough = getattr(material, "roughnessFactor", None)
    else:
        colors = getattr(visual, "face_colors", None)
        base = (np.asarray(colors, dtype=float).mean(axis=0) if colors is not None
                and len(colors) else np.array([180.0, 180.0, 180.0, 255.0]))
    base = np.asarray(base, dtype=float)
    if len(base) == 3:
        base = np.append(base, 255.0)
    return (tuple(int(round(v)) for v in base[:4]), tuple(round(float(v), 3) for v in emissive),
            1.0 if rough is None else round(float(rough), 3))


def srgb_to_linear(hex_color: str) -> list[float]:
    """Linear RGBA of an sRGB hex color, the glTF base color factor."""
    srgb = [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return [round(c, 5) for c in linear] + [1.0]


def material_key(rules: list, link: str, upstream: tuple) -> tuple:
    """Material of a visual of the source link `link` with the upstream material `upstream`.
    The key holds the base color (0 to 255), emissive color, roughness and finish name. The
    finish name is empty for the upstream material."""
    rgba = upstream[0]
    for pattern, color, finish in rules:
        if fnmatch.fnmatchcase(link, pattern) and (color is None or
                                                   color == "%02x%02x%02x" % rgba[:3]):
            if finish is None:
                break
            base = srgb_to_linear(FINISHES[finish][0])
            return (tuple(int(round(255 * c)) for c in base), (0.0, 0.0, 0.0),
                    FINISHES[finish][2], finish)
    return (*upstream, "")


def texture_of(geometry: trimesh.Trimesh):
    """The base color texture of a mesh with texture coordinates, otherwise None."""
    material = getattr(geometry.visual, "material", None)
    if material is None or getattr(geometry.visual, "uv", None) is None:
        return None
    return getattr(material, "baseColorTexture", None) or getattr(material, "image", None)


def link_meshes(model: ET.Element, tree: Tree, workdir: Path, rules: list,
                textures: bool = False) -> dict[str, list]:
    """Visual geometry of every link in the frame of its moving ancestor, with the material
    that `rules` of MATERIALS give it. Each material key ends with a texture name. With
    `textures`, a textured mesh keeps its texture and takes the name of its mesh file as the
    texture name. The texture name is empty otherwise."""
    path = workdir / "visual.urdf"
    ET.ElementTree(model).write(path)
    urdf = yourdfpy.URDF.load(str(path), build_scene_graph=True, load_meshes=True,
                              build_collision_scene_graph=False, load_collision_meshes=False)
    cfg = {name: 0.0 for name in urdf.actuated_joint_names}
    for joint in tree.joints.values():
        if joint["name"] in cfg and "index" not in joint:
            cfg[joint["name"]] = joint["value"]
    urdf.update_cfg(cfg)
    scene = urdf.scene
    parents = scene.graph.transforms.parents
    groups = {}
    for node in scene.graph.nodes_geometry:
        T_node, geometry_name = scene.graph[node]
        link = node
        while link in parents and link not in tree.joints and link != tree.root:
            link = parents[link]
        mover, _ = tree.moving_ancestor(link)
        T_mover = urdf.get_transform(mover, tree.root)
        T_world = urdf.get_transform(tree.root, scene.graph.base_frame)
        local = np.linalg.inv(T_mover) @ np.linalg.inv(T_world) @ T_node
        geometry = scene.geometry[geometry_name].copy()
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        geometry.apply_transform(local)
        if textures and texture_of(geometry) is not None:
            key = ((255, 255, 255, 255), (0.0, 0.0, 0.0), TEXTURED[1], "",
                   geometry_name.split(".")[0])
        else:
            key = (*material_key(rules, link, average_color(geometry)), "")
        groups.setdefault(mover, []).append((key, geometry))
    return groups


def plain_mesh(meshes: list, smooth: bool) -> trimesh.Trimesh:
    """Concatenated geometry without materials. The full meshes share a vertex across
    faces that meet at less than CREASE and keep hard edges above it. The ghost meshes are
    welded and smooth."""
    parts = []
    for g in meshes:
        part = trimesh.Trimesh(vertices=g.vertices, faces=g.faces, process=False)
        part.merge_vertices()
        if not smooth:
            part = trimesh.graph.smooth_shade(part, angle=CREASE)
        parts.append(part)
    mesh = trimesh.util.concatenate(parts)
    if smooth:
        mesh.merge_vertices()
        mesh.fix_normals()
    mesh.metadata = {}
    return mesh


def textured_mesh(meshes: list) -> trimesh.Trimesh:
    """Concatenated geometry with the texture coordinates of `meshes`, one vertex per face
    corner. Corners at one position share a vertex normal across faces that meet at less than
    CREASE."""
    vertices, uv, normals = [], [], []
    for g in meshes:
        part = trimesh.Trimesh(vertices=g.vertices, faces=g.faces, process=False)
        part.merge_vertices()
        smooth = trimesh.graph.smooth_shade(part, angle=CREASE)
        order = np.hstack(smooth.metadata.get("original_components", [np.arange(len(g.faces))]))
        corner = np.empty((len(g.faces), 3, 3))
        corner[order] = smooth.vertex_normals[smooth.faces]
        vertices.append(g.vertices[g.faces].reshape(-1, 3))
        uv.append(g.visual.uv[g.faces].reshape(-1, 2))
        normals.append(corner.reshape(-1, 3))
    vertices = np.vstack(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3),
                           vertex_normals=np.vstack(normals), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.vstack(uv))
    return mesh


def texture_image(meshes: list, size: int) -> Image.Image:
    """The base color texture of `meshes` at `size` pixels. Texels outside the faces of the
    meshes take the color of the nearest texel inside."""
    image = texture_of(meshes[0])
    w, h = image.size
    mask = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for g in meshes:
        corners = g.visual.uv[g.faces] * (w, -h) + (0, h)
        for triangle in corners:
            draw.polygon([tuple(p) for p in triangle], fill=1)
    inside = scipy.ndimage.binary_dilation(np.asarray(mask), iterations=2)
    _, (rows, cols) = scipy.ndimage.distance_transform_edt(~inside, return_indices=True)
    filled = Image.fromarray(np.asarray(image)[rows, cols])
    return filled.resize((size, size), Image.Resampling.LANCZOS)


def export_glb(groups: dict, path: Path, ghost: bool, texture_size: int = 0) -> None:
    scene = trimesh.Scene()
    for link, items in sorted(groups.items()):
        scene.graph.update(frame_from=scene.graph.base_frame, frame_to=link, matrix=np.eye(4))
        if ghost:
            mesh = plain_mesh([g for _, g in items], smooth=True)
            mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    name="ghost", baseColorFactor=[200, 200, 200, 255], metallicFactor=0.0,
                    roughnessFactor=1.0))
            scene.add_geometry(mesh, node_name=f"{link}__0", geom_name=f"{link}__0",
                               parent_node_name=link)
            continue
        by_key = {}
        for key, g in items:
            by_key.setdefault(key, []).append(g)
        for i, (key, gs) in enumerate(sorted(by_key.items())):
            rgba, emissive, rough, finish, texture = key
            if texture:
                mesh = textured_mesh(gs)
                mesh.visual.material = trimesh.visual.material.PBRMaterial(
                    name=texture, baseColorTexture=texture_image(gs, texture_size),
                    metallicFactor=TEXTURED[0], roughnessFactor=rough)
            else:
                mesh = plain_mesh(gs, smooth=False)
                mesh.visual = trimesh.visual.TextureVisuals(
                    material=trimesh.visual.material.PBRMaterial(
                        name=finish or "c%02x%02x%02x" % rgba[:3], baseColorFactor=list(rgba),
                        metallicFactor=FINISHES[finish][1] if finish else 0.0,
                        roughnessFactor=rough, emissiveFactor=list(emissive)))
            scene.add_geometry(mesh, node_name=f"{link}__{i}", geom_name=f"{link}__{i}",
                               parent_node_name=link)

    def exact_finishes(tree: dict) -> None:
        # trimesh stores base colors as 8-bit values. The finishes get their exact linear color.
        for material in tree.get("materials", []):
            if material.get("name") in FINISHES:
                material["pbrMetallicRoughness"]["baseColorFactor"] = srgb_to_linear(
                    FINISHES[material["name"]][0])

    path.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True,
                                                      tree_postprocessor=exact_finishes))


def gltf_transform() -> Path:
    """The glTF-Transform CLI that package-lock.json pins, installed once into the cache."""
    prefix = CACHE / "robot-meshes-node"
    lock = HERE / "package-lock.json"
    stamp = prefix / "package-lock.json"
    tool = prefix / "node_modules" / ".bin" / "gltf-transform"
    if not tool.exists() or not stamp.exists() or stamp.read_bytes() != lock.read_bytes():
        prefix.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "package.json", prefix / "package.json")
        shutil.copy2(lock, stamp)
        subprocess.run(["npm", "ci", "--no-audit", "--no-fund", "--ignore-scripts"],
                       cwd=prefix, check=True, stdout=subprocess.DEVNULL)
    return tool


def compress(tool: Path, src: Path, dst: Path, ratio: float, error: float,
             textures: bool = False) -> int:
    """Weld, simplify, deduplicate, quantize and meshopt-compress a GLB. With `textures`, the
    textures become WebP images. Returns the triangle count after simplification."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = Path(tmp) / "a.glb", Path(tmp) / "b.glb", Path(tmp) / "c.glb"
        steps = [["weld", src, a], ["simplify", a, b, "--ratio", ratio, "--error", error],
                 ["dedup", b, c],
                 ["quantize", c, a, "--quantize-position", 14, "--quantize-normal", 8]]
        if textures:
            steps.append(["webp", a, b, "--quality", WEBP_QUALITY])
        steps.append(["meshopt", b if textures else a, dst, "--level", "high"])
        for step in steps:
            subprocess.run([str(tool), *map(str, step)], check=True, stdout=subprocess.DEVNULL)
        return triangles(c)


def triangles(path: Path) -> int:
    """Number of triangles in a GLB file."""
    scene = trimesh.load(str(path), force="scene")
    return int(sum(len(g.faces) for g in scene.geometry.values()))


# ------------------------------------------------------------------------------ main


def build(name: str, tool: Path) -> None:
    """Write chain.json and both GLB files of the robot `name`."""
    cfg = ROBOTS[name]
    joints, ee_link = planning_group(name)
    model = visual_model(name, cfg)
    tree = Tree(model, joints, cfg.get("fixed", {}))
    chain = tree.chain()
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        texture_size = cfg.get("textures", 0)
        groups = link_meshes(model, tree, Path(tmp), MATERIALS.get(name, []), texture_size > 0)
        raw_full, raw_ghost = Path(tmp) / "full.glb", Path(tmp) / "ghost.glb"
        export_glb(groups, raw_full, ghost=False, texture_size=texture_size)
        export_glb(groups, raw_ghost, ghost=True)
        raw = triangles(raw_full)
        full = compress(tool, raw_full, out / f"{name}.glb", **cfg.get("full", FULL),
                        textures=texture_size > 0)
        ghost = compress(tool, raw_ghost, out / f"{name}_ghost.glb", **cfg.get("ghost", GHOST))
    ee_parent, T_ee = tree.moving_ancestor(ee_link)
    bounds = {}
    for link, items in sorted(groups.items()):
        points = np.vstack([g.vertices for _, g in items])
        bounds[link] = [[round(float(v), 4) for v in points.min(axis=0)],
                        [round(float(v), 4) for v in points.max(axis=0)]]
    doc = {
        "robot": name,
        "label": cfg["label"],
        "root": tree.root,
        "joint_names": joints,
        "links": [tree.root] + [j["child"] for j in chain],
        "joints": chain,
        "ee": {"link": ee_link, "parent": ee_parent, **pose_json(T_ee)},
        "meshes": {"full": f"{name}.glb", "ghost": f"{name}_ghost.glb"},
        "bounds": bounds,
    }
    (out / "chain.json").write_text(json.dumps(doc, indent=1) + "\n")
    sizes = ", ".join(f"{p.name} {p.stat().st_size / 1024:.0f} KB" for p in sorted(out.iterdir()))
    print(f"{name:16s} raw {raw:7d} tris, full {full:6d}, ghost {ghost:5d} | {sizes}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("robots", nargs="*", default=list(ROBOTS), help="robots to build")
    args = parser.parse_args()
    unknown = sorted(set(args.robots) - set(ROBOTS))
    if unknown:
        parser.error(f"unknown robots {unknown}; choose from {list(ROBOTS)}")
    tool = gltf_transform()
    for name in args.robots:
        build(name, tool)
    write_notice()


if __name__ == "__main__":
    main()

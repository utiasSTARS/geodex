#!/usr/bin/env python3
"""Regenerate the robot data of geodex from pinned sources.

For a robot whose recipe in scripts/robotgen/robots/<name>.json has a base link, `build`

1. builds a planning model from the upstream description. Every joint outside the recipe's
   planning joints is fixed, and an arm of nested segments becomes one prismatic coordinate (a
   massless driver joint with the segments as URDF mimic joints).
2. writes the arm dynamics URDF (fixed base, inertials only) for the CRBA.
3. writes the whole-body URDF. A planar base adds prismatic x, prismatic y and revolute yaw
   joints ahead of the base link, and the configuration is (x, y, theta, arm joints...).
4. spherizes the collision meshes with foam and finds the self-collision pairs to skip by
   sampling the mesh model, following the MoveIt Setup Assistant rules.
5. generates the VAMP kernel with cricket and the CRBA source with pinocchio_codegen.

A recipe without a `drive` has a fixed base. A recipe can hold joints at fixed positions
(`joint_positions`), drop leaf links (`drop_links`) and add frames such as a tool center
point (`frames`).

For the other robots, `crba` regenerates the CRBA from a vendored dynamics URDF, and
`kernel` regenerates the VAMP kernel from a vendored sphere model. `sweep` writes the sphere
travel bounds and joint limits of every robot's kernel, and `fit` measures how closely a
sphere model fits its meshes. generate.sh runs these steps and the Loewner bound tool. Run
this script inside the scripts/robotgen pixi environment.

    robotgen.py fetch --cache DIR [--repo REPO]
    robotgen.py build NAME --cache DIR --work DIR --repo REPO --tools DIR [--samples N]
        [--stages LIST]
    robotgen.py kernel NAME --repo REPO --tools DIR --work DIR
    robotgen.py pin NAME [--repo REPO]
    robotgen.py crba NAME --cache DIR --work DIR --repo REPO --tools DIR
    robotgen.py sweep NAME --repo REPO --vamp VAMP_SOURCE
    robotgen.py fit MESH_URDF SPHERE_URDF [--pitch M] [--output JSON]
    robotgen.py get NAME KEY
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLANAR_JOINTS = ("base_x_joint", "base_y_joint", "base_theta_joint")
# Half-width of the planar base box in the VAMP kernel. The kernel rejects a base
# position outside it. The value is far larger than any scene geodex plans in.
BASE_XY_LIMIT = 1000.0


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pins(repo: Path) -> dict[str, str]:
    """Return the `GEODEX_<NAME>` values from the `set()` calls in third_party/*.cmake."""
    pins: dict[str, str] = {}
    for f in sorted((repo / "third_party").glob("*.cmake")):
        for name, value in re.findall(r'set\(GEODEX_(\w+)\s+"([^"]*)"\s*\)', f.read_text()):
            pins[name] = value
    for name, value in list(pins.items()):
        pins[name] = re.sub(r"\$\{GEODEX_(\w+)\}", lambda m: pins[m.group(1)], value)
    return pins


def description_sources(repo: Path) -> list[str]:
    """Names of the archives in third_party/robot_descriptions.cmake, in file order."""
    text = (repo / "third_party" / "robot_descriptions.cmake").read_text()
    return re.findall(r"set\(GEODEX_(\w+)_URL\b", text)


def fetch(cache: Path, repo: Path) -> None:
    """Download every archive of third_party/robot_descriptions.cmake into @p cache and
    unpack it once."""
    pins = read_pins(repo)
    cache.mkdir(parents=True, exist_ok=True)
    for name in description_sources(repo):
        url, sha = pins[f"{name}_URL"], pins[f"{name}_SHA256"]
        archive = cache / Path(url).name
        if not archive.exists():
            print(f"[fetch] {name}: {url}")
            urllib.request.urlretrieve(url, archive)
        digest = sha256_of(archive)
        if digest != sha:
            raise SystemExit(f"{name}: sha256 {digest} does not match the pinned {sha}")
        dest = cache / name
        if dest.exists():
            continue
        dest.mkdir()
        if archive.suffix == ".whl":
            zipfile.ZipFile(archive).extractall(dest)
        else:
            with tarfile.open(archive) as tar:
                tar.extractall(dest)
        print(f"[fetch] {name}: sha256 ok, unpacked into {dest}")


def source_dir(cache: Path, package: str) -> Path:
    root = cache / package
    entries = list(root.iterdir())
    # GitHub archives unpack into one top-level directory.
    if len(entries) == 1 and entries[0].is_dir() and not package.endswith("_URDF"):
        return entries[0]
    return root


def expand_xacro(recipe: dict, cache: Path) -> str:
    """Expand the recipe's top-level xacro with `$(find pkg)` mapped to the pinned sources.

    xacro resolves packages through ament_index_python. A two-line stand-in module on the
    path maps each package name to its directory in @p cache.
    """
    import tempfile

    import xacro

    src = recipe["source"]
    packages = {name: str(path) for name, path in package_dirs(recipe, cache).items()}
    with tempfile.TemporaryDirectory(dir=cache) as shim:
        pkg = Path(shim) / "ament_index_python"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "packages.py").write_text(
            f"PACKAGES = {packages!r}\n"
            "def get_package_share_directory(name):\n"
            "    return PACKAGES[name]\n")
        sys.path.insert(0, shim)
        for mod in [m for m in sys.modules if m.startswith("ament_index_python")]:
            del sys.modules[mod]
        try:
            doc = xacro.process_file(str(HERE / "robots" / src["xacro_file"]))
        finally:
            sys.path.pop(0)
    return doc.toxml()


def load_source_urdf(recipe: dict, cache: Path) -> tuple[ET.Element, Path]:
    """Return the upstream URDF root and the directory its mesh paths resolve against."""
    src = recipe["source"]
    root_dir = source_dir(cache, src["package"])
    if "xacro_file" in src:
        root = ET.fromstring(expand_xacro(recipe, cache))
        return root, cache
    if "xacro_call" in src:
        sys.path.insert(0, str(root_dir))
        module = __import__(src["xacro_call"]["module"], fromlist=["get_urdf"])
        xml = getattr(module, src["xacro_call"]["function"])(**src["xacro_call"]["kwargs"])
        sys.path.pop(0)
        root = ET.fromstring(xml)
    else:
        urdf = root_dir / src["urdf"]
        return ET.parse(urdf).getroot(), urdf.parent
    return root, root_dir


def package_dirs(recipe: dict, cache: Path) -> dict[str, Path]:
    return {name: source_dir(cache, archive) / sub
            for name, (archive, sub) in recipe["source"].get("packages", {}).items()}


def resolve_mesh(filename: str, base_dir: Path, packages: dict[str, Path]) -> str:
    """Return the absolute path of a mesh reference.

    `package://` resolves through the recipe's package map, and a relative path resolves
    against the directory of the URDF that names it. A name found anywhere else must be
    unique. A description package can hold several robot versions with equally named
    meshes.
    """
    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    if filename.startswith("package://"):
        package, _, rel = filename[len("package://"):].partition("/")
        if package in packages and (packages[package] / rel).exists():
            return str((packages[package] / rel).resolve())
        return unique_match(base_dir, rel, filename)
    path = Path(filename)
    if path.is_absolute():
        if path.exists():
            return str(path)
        raise SystemExit(f"mesh {filename} does not exist")
    if (base_dir / path).exists():
        return str((base_dir / path).resolve())
    return unique_match(base_dir, str(path), filename)


def unique_match(root: Path, rel: str, filename: str) -> str:
    candidates = sorted(root.rglob(rel))
    if len(candidates) != 1:
        raise SystemExit(f"mesh {filename}: {len(candidates)} candidates under {root}")
    return str(candidates[0].resolve())


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------


def joint_map(root: ET.Element) -> dict[str, ET.Element]:
    return {j.get("name"): j for j in root.findall("joint")}


def link_map(root: ET.Element) -> dict[str, ET.Element]:
    return {l.get("name"): l for l in root.findall("link")}


def fix_joint(joint: ET.Element) -> None:
    joint.set("type", "fixed")
    for tag in ("axis", "limit", "mimic", "dynamics", "safety_controller", "calibration"):
        for el in joint.findall(tag):
            joint.remove(el)


def origin_matrix(el: ET.Element | None):
    """Return the homogeneous transform of a URDF origin element, the identity when absent."""
    import numpy as np
    from trimesh.transformations import euler_matrix

    if el is None:
        return np.eye(4)
    T = euler_matrix(*[float(v) for v in el.get("rpy", "0 0 0").split()], axes="sxyz")
    T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def set_origin(joint: ET.Element, T) -> None:
    """Replace the origin of a joint with the transform @p T."""
    from trimesh.transformations import euler_from_matrix

    for old in joint.findall("origin"):
        joint.remove(old)
    origin = ET.Element("origin", {"xyz": " ".join(repr(float(v)) for v in T[:3, 3]),
                                   "rpy": " ".join(repr(float(v))
                                                   for v in euler_from_matrix(T, axes="sxyz"))})
    joint.insert(0, origin)


def hold_joints(root: ET.Element, positions: dict[str, float]) -> None:
    """Fix each named joint at its position and every joint that mimics it at the mimicked value.

    The joint motion at that position moves into the joint origin. The fixed model places every
    link where the upstream model places it at those positions.
    """
    import numpy as np
    from trimesh.transformations import rotation_matrix

    joints = joint_map(root)
    values = {}
    for name, value in positions.items():
        lim = joints[name].find("limit")
        if not float(lim.get("lower")) <= value <= float(lim.get("upper")):
            raise SystemExit(f"{name}: position {value} is outside its limits")
        values[name] = float(value)
    for name, joint in joints.items():
        m = joint.find("mimic")
        if m is not None and m.get("joint") in positions:
            values[name] = (float(m.get("multiplier", "1")) * values[m.get("joint")] +
                            float(m.get("offset", "0")))
    for name, value in values.items():
        joint = joints[name]
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        if joint.get("type") == "prismatic":
            motion = np.eye(4)
            motion[:3, 3] = axis * value
        elif joint.get("type") in ("revolute", "continuous"):
            motion = rotation_matrix(value, axis)
        else:
            raise SystemExit(f"{name}: cannot hold a {joint.get('type')} joint")
        set_origin(joint, origin_matrix(joint.find("origin")) @ motion)
        fix_joint(joint)


def drop_links(root: ET.Element, patterns: list[str]) -> None:
    """Remove every leaf link whose name matches an fnmatch pattern, with the joint above it."""
    import fnmatch

    parents = {j.find("parent").get("link") for j in root.findall("joint")}
    for link in list(root.findall("link")):
        name = link.get("name")
        if not any(fnmatch.fnmatchcase(name, p) for p in patterns):
            continue
        if name in parents:
            raise SystemExit(f"{name} is not a leaf link")
        root.remove(link)
        for joint in root.findall("joint"):
            if joint.find("child").get("link") == name:
                root.remove(joint)


def add_frames(root: ET.Element, frames: list[dict]) -> None:
    """Add each recipe frame as a massless link on a fixed joint to its parent link."""
    links = link_map(root)
    for frame in frames:
        if frame["parent"] not in links:
            raise SystemExit(f"frame {frame['name']}: no parent link {frame['parent']}")
        ET.SubElement(root, "link", {"name": frame["name"]})
        joint = ET.SubElement(root, "joint", {"name": f"{frame['name']}_joint", "type": "fixed"})
        ET.SubElement(joint, "origin", {"xyz": " ".join(repr(float(v)) for v in frame["xyz"]),
                                        "rpy": " ".join(repr(float(v)) for v in frame["rpy"])})
        ET.SubElement(joint, "parent", {"link": frame["parent"]})
        ET.SubElement(joint, "child", {"link": frame["name"]})


def planning_model(recipe: dict, cache: Path) -> ET.Element:
    """Reduce the upstream URDF to the recipe's planning joints and resolve its meshes."""
    root, root_dir = load_source_urdf(recipe, cache)
    packages = package_dirs(recipe, cache)
    root = copy.deepcopy(root)
    root.attrib = {"name": recipe["name"]}
    for el in list(root):
        if el.tag not in ("link", "joint"):
            root.remove(el)

    hold_joints(root, recipe.get("joint_positions", {}))
    drop_links(root, recipe.get("drop_links", []))
    add_frames(root, recipe.get("frames", []))
    joints = joint_map(root)
    coupled_segments = {s for c in recipe.get("couplings", []) for s in c["segments"]}
    keep = set(recipe["planning_joints"]) | coupled_segments
    for name, joint in joints.items():
        if joint.get("type") != "fixed" and name not in keep:
            fix_joint(joint)

    for name, (lo, hi) in recipe.get("limit_overrides", {}).items():
        lim = joints[name].find("limit")
        lim.set("lower", repr(float(lo)))
        lim.set("upper", repr(float(hi)))

    for coupling in recipe.get("couplings", []):
        add_coupling(root, coupling)

    drop = set(recipe.get("drop_collision_links", []))
    for link in root.findall("link"):
        if link.get("name") in drop:
            for el in link.findall("collision"):
                link.remove(el)
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                for mesh in el.iter("mesh"):
                    mesh.set("filename", resolve_mesh(mesh.get("filename"), root_dir, packages))
                org = el.find("origin")
                if org is None:
                    org = ET.SubElement(el, "origin")
                org.set("xyz", org.get("xyz", "0 0 0"))
                org.set("rpy", org.get("rpy", "0 0 0"))
    for joint in root.findall("joint"):
        if joint.get("type") in ("revolute", "prismatic"):
            lim = joint.find("limit")
            lim.set("effort", lim.get("effort") or "100")
            lim.set("velocity", lim.get("velocity") or "1")
    return root


def add_coupling(root: ET.Element, coupling: dict) -> None:
    """Move a chain of nested segments with one prismatic coordinate, the total extension.

    A massless driver joint is added next to the first segment, and each segment joint
    mimics it with multiplier upper_i / sum(upper). Segment i extends its share of the
    total. Pinocchio and cricket read the mimic tags directly.
    """
    joints = joint_map(root)
    segments = [joints[s] for s in coupling["segments"]]
    uppers = [float(s.find("limit").get("upper")) for s in segments]
    lowers = [float(s.find("limit").get("lower")) for s in segments]
    if any(lo != 0.0 for lo in lowers):
        raise SystemExit(f"{coupling['joint']}: coupling assumes segments start at 0")
    total = sum(uppers)
    first = segments[0]
    driver_link = coupling["joint"] + "_driver_link"
    ET.SubElement(root, "link", {"name": driver_link})
    driver = ET.SubElement(root, "joint", {"name": coupling["joint"], "type": "prismatic"})
    ET.SubElement(driver, "parent", {"link": first.find("parent").get("link")})
    ET.SubElement(driver, "child", {"link": driver_link})
    ET.SubElement(driver, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    first_origin = first.find("origin")
    driver.find("origin").set("rpy", first_origin.get("rpy", "0 0 0"))
    ET.SubElement(driver, "axis", {"xyz": first.find("axis").get("xyz")})
    ET.SubElement(driver, "limit", {"lower": "0", "upper": repr(total), "effort": "100",
                                    "velocity": "1"})
    for seg, upper in zip(segments, uppers):
        for el in seg.findall("mimic"):
            seg.remove(el)
        ET.SubElement(seg, "mimic", {"joint": coupling["joint"],
                                     "multiplier": repr(upper / total), "offset": "0"})


def strip_geometry(root: ET.Element, keep_collision: bool = False) -> ET.Element:
    out = copy.deepcopy(root)
    for link in out.findall("link"):
        for tag in ("visual",) if keep_collision else ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    return out


def with_planar_base(root: ET.Element, base_link: str, base_height: float = 0.0) -> ET.Element:
    """Prepend prismatic x, prismatic y and revolute yaw joints ahead of @p base_link.

    @p base_height lifts the base link above the chain and puts the floor at world z = 0.
    """
    out = ET.Element("robot", {"name": root.get("name")})
    for name in ("world", "base_x_link", "base_y_link"):
        ET.SubElement(out, "link", {"name": name})
    chain = [("base_x_joint", "prismatic", "world", "base_x_link", "1 0 0", BASE_XY_LIMIT),
             ("base_y_joint", "prismatic", "base_x_link", "base_y_link", "0 1 0", BASE_XY_LIMIT),
             ("base_theta_joint", "revolute", "base_y_link", base_link, "0 0 1", math.pi)]
    for name, kind, parent, child, axis, lim in chain:
        j = ET.SubElement(out, "joint", {"name": name, "type": kind})
        ET.SubElement(j, "parent", {"link": parent})
        ET.SubElement(j, "child", {"link": child})
        z = base_height if name == "base_theta_joint" else 0.0
        ET.SubElement(j, "origin", {"xyz": f"0 0 {z!r}", "rpy": "0 0 0"})
        ET.SubElement(j, "axis", {"xyz": axis})
        ET.SubElement(j, "limit", {"lower": repr(-lim), "upper": repr(lim), "effort": "100",
                                   "velocity": "1"})
    for el in root:
        out.append(copy.deepcopy(el))
    return out


def repair_collision_meshes(root: ET.Element, work: Path) -> list[str]:
    """Write re-oriented copies of inside-out collision meshes into @p work for spherization."""
    import trimesh

    fixed = []
    for link in root.findall("link"):
        for el in link.findall("collision"):
            mesh = el.find("geometry/mesh")
            if mesh is None:
                continue
            tm = trimesh.load(mesh.get("filename"), force="mesh")
            if tm.is_watertight and tm.volume > 0:
                continue
            trimesh.repair.fix_normals(tm)
            if tm.volume < 0:
                tm.invert()
            out = work / "meshes" / f"{link.get('name')}_{Path(mesh.get('filename')).stem}.stl"
            out.parent.mkdir(parents=True, exist_ok=True)
            tm.export(out)
            mesh.set("filename", str(out))
            fixed.append(link.get("name"))
    return fixed


def data_dir(recipe: dict, repo: Path) -> Path:
    """Return the robot's data directory, data/robots/<name> unless the recipe names another."""
    return repo / recipe.get("data_dir", f"data/robots/{recipe['name']}")


def write_urdf(root: ET.Element, path: Path, header: str) -> None:
    tree = ET.ElementTree(copy.deepcopy(root))
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(f"<!-- {header} -->\n")
        f.write(ET.tostring(tree.getroot(), encoding="unicode"))
        f.write("\n")


# ---------------------------------------------------------------------------
# Spherization, self-collision pairs, VAMP kernel, CRBA source
# ---------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, log: Path | None = None):
    print("[run]", " ".join(str(c) for c in cmd))
    with open(log, "w") if log else open(os.devnull, "w") as out:
        res = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=out,
                             stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise SystemExit(f"command failed ({res.returncode}): {cmd}; see {log}")


def link_mesh_in(root: ET.Element, link: str, frame: str):
    """Return the collision geometry of @p link as one trimesh in the rigidly attached @p frame."""
    import numpy as np
    import trimesh
    from trimesh.transformations import euler_matrix

    def origin(el):
        o = el.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        T = euler_matrix(*rpy, axes="sxyz")
        T[:3, 3] = xyz
        return T

    parent = {j.find("child").get("link"): j for j in root.findall("joint")}
    T = np.eye(4)
    cur = link
    while cur != frame:
        joint = parent[cur]
        if joint.get("type") != "fixed":
            raise SystemExit(f"{link} is not rigidly attached to {frame}")
        T = origin(joint) @ T
        cur = joint.find("parent").get("link")
    parts = []
    for c in link_map(root)[link].findall("collision"):
        g = c.find("geometry")
        if g.find("mesh") is not None:
            tm = trimesh.load(g.find("mesh").get("filename"), force="mesh")
        elif g.find("box") is not None:
            tm = trimesh.creation.box([float(v) for v in g.find("box").get("size").split()])
        elif g.find("cylinder") is not None:
            cy = g.find("cylinder")
            tm = trimesh.creation.cylinder(float(cy.get("radius")), float(cy.get("length")))
        else:
            continue
        tm.apply_transform(T @ origin(c))
        parts.append(tm)
    return trimesh.util.concatenate(parts)


def collision_parts(link: ET.Element) -> list:
    """Return the mesh, box and cylinder collision geometry of @p link as trimeshes in its frame."""
    import trimesh

    parts = []
    for c in link.findall("collision"):
        g = c.find("geometry")
        if g.find("mesh") is not None:
            tm = trimesh.load(g.find("mesh").get("filename"), force="mesh")
            if g.find("mesh").get("scale"):
                tm.apply_scale([float(v) for v in g.find("mesh").get("scale").split()])
        elif g.find("box") is not None:
            tm = trimesh.creation.box([float(v) for v in g.find("box").get("size").split()])
        elif g.find("cylinder") is not None:
            cy = g.find("cylinder")
            tm = trimesh.creation.cylinder(float(cy.get("radius")), float(cy.get("length")))
        else:
            continue
        tm.apply_transform(origin_matrix(c.find("origin")))
        parts.append(tm)
    return parts


def rigid_frames(root: ET.Element) -> dict:
    """Return, for every link, the link it moves rigidly with and the transform between them.

    That link is the first link at or above it whose parent joint moves, or the root.
    """
    import numpy as np

    parent = {j.find("child").get("link"): j for j in root.findall("joint")}
    out = {}
    for name in link_map(root):
        T, cur = np.eye(4), name
        while cur in parent and parent[cur].get("type") == "fixed":
            T = origin_matrix(parent[cur].find("origin")) @ T
            cur = parent[cur].find("parent").get("link")
        out[name] = (cur, T)
    return out


# Surface samples per square meter at which the sphere fit is measured.
FIT_DENSITY = 1.0e6


def surface_points(meshes: list, density: float, seed: int):
    """Return every vertex of @p meshes and uniform surface samples at @p density per m^2."""
    import numpy as np

    rng = np.random.default_rng(seed)
    points = []
    for tm in meshes:
        n = max(1, int(tm.area * density))
        face = rng.choice(len(tm.faces), size=n, p=tm.area_faces / tm.area)
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1.0
        u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
        tri = tm.triangles[face]
        points += [tm.vertices, tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) +
                   v[:, None] * (tri[:, 2] - tri[:, 0])]
    return np.vstack(points)


def inside(tm, points):
    """Return True for each point whose generalized winding number about @p tm exceeds 1/2.

    The winding number does not need a closed or consistently oriented mesh.
    """
    import numpy as np

    w = np.zeros(len(points))
    step = 1 << 12
    for s in range(0, len(points), step):
        a, b, c = (tm.triangles[None, :, k, :] - points[s:s + step, None, :] for k in range(3))
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("pfi,pfi->pf", a, b) * lc +
               np.einsum("pfi,pfi->pf", b, c) * la + np.einsum("pfi,pfi->pf", c, a) * lb)
        w[s:s + step] = np.arctan2(det, den).sum(axis=1) / (2.0 * np.pi)
    return np.abs(w) > 0.5


def outside(points, balls):
    """Return how far each point lies outside the union of @p balls, negative inside."""
    import numpy as np

    out = np.empty(len(points))
    step = 1 << 15
    for s in range(0, len(points), step):
        p = points[s:s + step]
        d = np.linalg.norm(p[:, None, :] - balls[None, :, :3], axis=2) - balls[None, :, 3]
        out[s:s + step] = d.min(axis=1)
    return out


def body_fit(args) -> dict:
    """Return the fit of one rigid body's spheres to its meshes. See `sphere_fit`."""
    import numpy as np

    meshes, b, pitch = args
    entry = {"spheres": len(b)}
    if not meshes or not len(b):
        return entry
    entry["max_outside_m"] = max(0.0, float(outside(surface_points(meshes, FIT_DENSITY, 2),
                                                    b).max()))
    lo = np.minimum((b[:, :3] - b[:, 3:]).min(axis=0),
                    np.min([tm.bounds[0] for tm in meshes], axis=0))
    hi = np.maximum((b[:, :3] + b[:, 3:]).max(axis=0),
                    np.max([tm.bounds[1] for tm in meshes], axis=0))
    axes = [np.arange(a + pitch / 2, c, pitch) for a, c in zip(lo, hi)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    in_union = outside(grid, b) <= 0.0
    in_mesh = np.zeros(len(grid), dtype=bool)
    for tm in meshes:
        near = np.all((grid >= tm.bounds[0]) & (grid <= tm.bounds[1]), axis=1) & ~in_mesh
        in_mesh[near] = inside(tm, grid[near])
    cell = pitch ** 3
    entry.update({"mesh_m3": float(in_mesh.sum() * cell),
                  "union_m3": float(in_union.sum() * cell),
                  "excess_m3": float((in_union & ~in_mesh).sum() * cell),
                  "uncovered_m3": float((in_mesh & ~in_union).sum() * cell)})
    return entry


def sphere_fit(mesh_urdf: Path, sphere_urdf: Path, pitch: float = 0.002) -> dict:
    """Measure how closely the spheres of @p sphere_urdf fit the meshes of @p mesh_urdf.

    The two URDFs share their kinematics. For each rigid body, the result holds the sphere
    count, the largest distance by which a mesh point (every vertex and 10^6 surface samples
    per m^2, seed 2) lies outside every sphere, and the volumes of the meshes, of the union
    of spheres, of the union outside the meshes (excess) and of the meshes outside the union
    (uncovered), counted on a grid of spacing @p pitch.
    """
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np

    mesh_root, sphere_root = ET.parse(mesh_urdf).getroot(), ET.parse(sphere_urdf).getroot()
    for mesh in mesh_root.iter("mesh"):
        mesh.set("filename", resolve_mesh(mesh.get("filename"), mesh_urdf.parent, {}))
    parts, balls, counts = {}, {}, {}
    for link, (body, T) in rigid_frames(mesh_root).items():
        for tm in collision_parts(link_map(mesh_root)[link]):
            tm.apply_transform(T)
            parts.setdefault(body, []).append(tm)
    for link, (body, T) in rigid_frames(sphere_root).items():
        for c in link_map(sphere_root)[link].findall("collision"):
            s = c.find("geometry/sphere")
            if s is not None:
                centre = (T @ origin_matrix(c.find("origin")))[:3, 3]
                balls.setdefault(body, []).append([*centre, float(s.get("radius"))])
                counts[link] = counts.get(link, 0) + 1
    names = sorted(set(parts) | set(balls))
    with ProcessPoolExecutor() as pool:
        bodies = dict(zip(names, pool.map(body_fit, [
            (parts.get(n, []), np.array(balls.get(n, []), dtype=float).reshape(-1, 4), pitch)
            for n in names])))
    total = {"spheres": sum(e["spheres"] for e in bodies.values()),
             "max_outside_m": max(e.get("max_outside_m", 0.0) for e in bodies.values())}
    for key in ("mesh_m3", "union_m3", "excess_m3", "uncovered_m3"):
        total[key] = sum(e.get(key, 0.0) for e in bodies.values())
    total["excess_ratio"] = total["excess_m3"] / total["mesh_m3"] if total["mesh_m3"] else 0.0
    return {"mesh_urdf": str(mesh_urdf), "sphere_urdf": str(sphere_urdf), "pitch_m": pitch,
            "surface_samples_per_m2": FIT_DENSITY, "links": counts, "bodies": bodies,
            "total": total}


def box_cover(root: ET.Element, cover: dict) -> list[tuple[list[float], float]]:
    """Return spheres that cover the bounding box of the listed links in the cover link's frame.

    The box is split into `cells` equal cells, and each sphere circumscribes one cell.
    The union contains the box and every listed mesh.
    """
    import numpy as np

    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for link in cover["links"]:
        b = link_mesh_in(root, link, cover["link"]).bounds
        lo, hi = np.minimum(lo, b[0]), np.maximum(hi, b[1])
    n = np.array(cover["cells"])
    size = (hi - lo) / n
    r = 0.5 * float(np.linalg.norm(size))
    spheres = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                c = lo + size * (np.array([i, j, k]) + 0.5)
                spheres.append(([float(v) for v in c], r))
    return spheres


def spherize(recipe: dict, mesh_urdf: Path, out: Path, tools: Path, work: Path) -> None:
    """Spherize the collision geometry, then keep kinematics, inertials and spheres.

    Links in a recipe box cover get a regular sphere grid, and foam spherizes the rest.
    """
    foam = tools / "foam"
    params = recipe.get("spherize", {})
    raw = work / "foam_spherized.urdf"
    mesh_root = ET.parse(mesh_urdf).getroot()
    covered = {l for c in recipe.get("box_covers", []) for l in c["links"]}
    foam_input = work / "foam_input.urdf"
    foam_root = copy.deepcopy(mesh_root)
    for link in foam_root.findall("link"):
        if link.get("name") in covered:
            for el in link.findall("collision"):
                link.remove(el)
    write_urdf(foam_root, foam_input, "foam input")
    cmd = [sys.executable, foam / "scripts" / "generate_sphere_urdf.py", foam_input,
           "--output", raw, "--database", work / "sphere_database.json",
           "--depth", "1", "--branch", str(params.get("branch", 8)), "--method",
           params.get("method", "medial"), "--threads", str(params.get("threads", 16))]
    if "shrinkage" in params:
        cmd += ["--shrinkage", str(params["shrinkage"])]
    for link, scale in params.get("link_scale", {}).items():
        cmd += [f"--{link}", str(scale)]
    tmp = work / "tmp"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(foam), TMPDIR=str(tmp))
    run(cmd, cwd=foam, env=env, log=work / "foam.log")
    root = ET.parse(raw).getroot()
    root.attrib = {"name": recipe["name"]}
    for link in root.findall("link"):
        for el in link.findall("visual"):
            link.remove(el)
        for el in link.findall("collision"):
            if el.find("geometry/sphere") is None:
                link.remove(el)
    links = link_map(root)
    for cover in recipe.get("box_covers", []):
        for centre, radius in box_cover(mesh_root, cover):
            col = ET.SubElement(links[cover["link"]], "collision")
            ET.SubElement(col, "origin", {"xyz": " ".join(repr(v) for v in centre),
                                          "rpy": "0 0 0"})
            geo = ET.SubElement(col, "geometry")
            ET.SubElement(geo, "sphere", {"radius": repr(radius)})
    write_urdf(root, out, f"{recipe['description']}. Whole-body planning model with foam "
                          "collision spheres, generated by scripts/robotgen/robotgen.py.")


def srdf_pairs(recipe: dict, mesh_urdf: Path, out: Path, samples: int) -> dict:
    """Return the self-collision pairs to skip under the MoveIt Setup Assistant rules.

    A link pair is skipped when both links belong to the same rigid body, when their bodies
    are parent and child in the kinematic tree, when the recipe groups them as nested
    parts (nested arm tubes and the carriage they slide through) or lists them as
    collision meshes that overlap where the hardware moves freely, or when the pair is
    in collision in at least 95 percent or in none of @p samples uniform configurations
    of the planning joints.
    """
    import numpy as np
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(mesh_urdf), mimic=True)
    geom = pin.buildGeomFromUrdf(model, str(mesh_urdf), pin.GeometryType.COLLISION)
    link_of = [model.frames[g.parentFrame].name for g in geom.geometryObjects]
    body_of = {name: g.parentJoint for g, name in zip(geom.geometryObjects, link_of)}
    links = sorted(set(link_of))
    geom.addAllCollisionPairs()
    data = model.createData()
    gdata = pin.GeometryData(geom)

    def pairs_where(pred):
        return {(a, b) for i, a in enumerate(links) for b in links[i + 1:] if pred(a, b)}

    rigid = pairs_where(lambda a, b: body_of[a] == body_of[b])
    adjacent = pairs_where(lambda a, b: model.parents[body_of[a]] == body_of[b] or
                           model.parents[body_of[b]] == body_of[a])
    nested = set()
    for group in recipe.get("nested_groups", []):
        members = sorted(l for l in group if l in body_of)
        nested |= {(a, b) for i, a in enumerate(members) for b in members[i + 1:]}
    for entry in recipe.get("overlapping_pairs", []):
        nested.add(tuple(sorted(entry["links"])))

    rng = np.random.default_rng(0)
    lo, hi = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    # Hold the base pose at the origin. It does not change self-collision.
    for name in PLANAR_JOINTS:
        if model.existJointName(name):
            i = model.idx_qs[model.getJointId(name)]
            lo[i] = hi[i] = 0.0
    counts: dict[tuple[str, str], int] = {}
    for _ in range(samples):
        q = lo + (hi - lo) * rng.random(model.nq)
        pin.computeCollisions(model, data, geom, gdata, q, False)
        hit = set()
        for k, cp in enumerate(geom.collisionPairs):
            if gdata.collisionResults[k].isCollision():
                a, b = link_of[cp.first], link_of[cp.second]
                if a != b:
                    hit.add(tuple(sorted((a, b))))
        for pair in hit:
            counts[pair] = counts.get(pair, 0) + 1
    all_pairs = pairs_where(lambda a, b: True)
    always = {p for p, c in counts.items() if c >= 0.95 * samples}
    never = {p for p in all_pairs if counts.get(p, 0) == 0}
    reasons = {}
    for label, group in (("Adjacent", adjacent), ("Default", rigid), ("Nested", nested),
                         ("Always", always), ("Never", never)):
        for p in group:
            reasons.setdefault(p, label)
    lines = ['<?xml version="1.0"?>',
             f"<!-- Self-collision pairs the VAMP kernel for {recipe['name']} skips: rigid and "
             "adjacent bodies, nested parts, and pairs always or never in collision over "
             f"{samples} configurations of the mesh model. -->",
             f'<robot name="{recipe["name"]}">']
    for (a, b) in sorted(reasons):
        lines.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="{reasons[(a, b)]}"/>')
    lines.append("</robot>")
    out.write_text("\n".join(lines) + "\n")
    return {"links": len(links), "pairs": len(all_pairs), "skipped": len(reasons),
            "checked": len(all_pairs) - len(reasons), "rigid": len(rigid),
            "adjacent": len(adjacent), "nested": len(nested), "always": len(always),
            "never": len(never), "samples": samples}


def cricket(recipe: dict, spherized: Path, srdf: Path, out: Path, tools: Path, work: Path) -> None:
    templates = tools / "cricket" / "resources" / "templates"
    config = {
        "name": recipe["vamp_struct"],
        "urdf": str(spherized),
        "srdf": str(srdf),
        "end_effector": recipe["end_effector"],
        "resolution": 32,
        "template": str(templates / "fk_template.hh"),
        "subtemplates": [{"name": "ccfk", "template": str(templates / "ccfk_template.hh")}],
        "output": str(out),
    }
    cfg = work / f"{recipe['name']}_cricket.json"
    cfg.write_text(json.dumps(config, indent=2) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    run([tools / "cricket-build" / "fkcc_gen", cfg], cwd=work, log=work / "cricket.log")
    if not out.exists():
        raise SystemExit(f"cricket wrote no kernel to {out}; see {work / 'cricket.log'}")


def crba(recipe: dict, dynamics: Path, generated: Path, tools: Path, repo: Path, work: Path,
         name: str | None = None):
    name = name or recipe["name"]
    run([tools / "geodex-build" / "pinocchio_codegen", dynamics, name, generated],
        log=work / "codegen.log")
    cpp = generated / f"{name}_crba.cpp"
    run([sys.executable, HERE / "post_process_sincos_simd.py", cpp, cpp, name],
        log=work / "post_process.log")


def crba_vendored(name: str, cache: Path, work: Path, repo: Path, tools: Path) -> None:
    """Regenerate the CRBA of a robot whose dynamics URDF is a vendored input.

    A recipe with an `expansion` first rebuilds that URDF from the expansion by replacing
    the listed meshes with their bounding boxes (`mesh_boxes.py`), taking the meshes from
    the pinned source archive. The recipe's `header` becomes the URDF's leading comment.
    """
    recipe = json.loads((HERE / "robots" / f"{name}.json").read_text())
    spec = recipe["crba"]
    work = work / name
    work.mkdir(parents=True, exist_ok=True)
    urdf = repo / spec["urdf"]
    if "expansion" in spec:
        source = source_dir(cache, spec["mesh_source"])
        header = ["--header", spec["header"]] if "header" in spec else []
        run([sys.executable, HERE / "mesh_boxes.py", repo / spec["expansion"], urdf, "--meshes",
             *[source / d for d in spec["mesh_dirs"]], *header], log=work / "mesh_boxes.log")
    crba(recipe, urdf, repo / "src" / "robots" / "generated", tools, repo, work, spec["name"])


# ---------------------------------------------------------------------------
# Sphere travel bounds and joint limits of a VAMP kernel
# ---------------------------------------------------------------------------


def kernel_layout(header: Path) -> tuple[list[str], str]:
    """Return the joint names in configuration order and the end-effector link of a kernel."""
    text = header.read_text()
    names = re.search(r"joint_names\s*=\s*\{([^}]*)\}", text)
    ee = re.search(r"end_effector\s*=\s*\"([^\"]+)\"", text)
    if names is None or ee is None:
        raise SystemExit(f"{header}: no joint_names or end_effector")
    return re.findall(r'"([^"]+)"', names.group(1)), ee.group(1)


def xyz_of(el: ET.Element | None) -> list[float]:
    if el is None:
        return [0.0, 0.0, 0.0]
    return [float(v) for v in el.get("xyz", "0 0 0").split()]


def sweep_model(urdf: Path, kernel_joints: list[str], ee: str) -> dict:
    """Bound how far a sphere center and the end-effector frame move per unit of each
    kernel coordinate, from the triangle inequality over the kinematic chain.

    A point at distance d from a revolute joint's origin moves at most d per radian of
    that joint, and any point moves at most 1 per unit of a prismatic joint. d is bounded
    by the point's offset in its link plus, for every joint between, the length of the
    joint's origin offset and the largest travel of a prismatic joint. A coordinate that
    moves mimic joints adds their contributions with their multipliers. The bound does
    not use sampling.
    """
    root = ET.parse(urdf).getroot()
    joints = joint_map(root)
    parent_joint = {j.find("child").get("link"): j for j in joints.values()}
    mimic = {}
    for name, j in joints.items():
        m = j.find("mimic")
        if m is not None:
            mimic[name] = (m.get("joint"), float(m.get("multiplier", "1")),
                           float(m.get("offset", "0")))

    def limits(name: str) -> tuple[float, float]:
        lim = joints[name].find("limit")
        return float(lim.get("lower", "0")), float(lim.get("upper", "0"))

    def travel(j: ET.Element) -> float:
        if j.get("type") != "prismatic":
            return 0.0
        if j.get("name") in mimic:
            primary, mult, offset = mimic[j.get("name")]
            lo, hi = limits(primary)
            return abs(mult) * max(abs(lo), abs(hi)) + abs(offset)
        lo, hi = limits(j.get("name"))
        return max(abs(lo), abs(hi))

    def distances(link: str, offset: float) -> dict[str, float]:
        """Bound the point's distance from each joint origin up the chain."""
        out, acc = {}, offset
        while link in parent_joint:
            j = parent_joint[link]
            if j.get("type") not in ("revolute", "continuous", "prismatic", "fixed"):
                raise SystemExit(f"{urdf.name}: joint {j.get('name')} of type {j.get('type')}")
            out[j.get("name")] = acc
            acc += math.dist(xyz_of(j.find("origin")), [0.0, 0.0, 0.0]) + travel(j)
            link = j.find("parent").get("link")
        return out

    def driven(coordinate: str) -> list[tuple[str, float]]:
        return [(coordinate, 1.0)] + [(n, m) for n, (p, m, _) in mimic.items() if p == coordinate]

    def speed(dist: dict[str, float], coordinate: str) -> tuple[float, float]:
        """Return the point speed bound per unit of the coordinate and its rotational weight."""
        total = rotation = 0.0
        for name, mult in driven(coordinate):
            if name not in dist:
                continue
            if joints[name].get("type") == "prismatic":
                total += abs(mult)
            else:
                total += abs(mult) * dist[name]
                rotation += abs(mult)
        return total, rotation

    for name in kernel_joints:
        if name not in joints:
            raise SystemExit(f"{urdf.name}: kernel joint {name} is not in the URDF")
    points = []
    for link in root.findall("link"):
        for col in link.findall("collision"):
            if col.find("geometry/sphere") is not None:
                points.append(distances(link.get("name"), math.dist(xyz_of(col.find("origin")),
                                                                    [0.0, 0.0, 0.0])))
    if not points:
        raise SystemExit(f"{urdf.name}: no collision spheres")
    ee_dist = distances(ee, 0.0)
    planar = kernel_joints[:3] == list(PLANAR_JOINTS)
    reach = [max(speed(d, c)[0] for d in points) for c in kernel_joints]
    ee_reach = [speed(ee_dist, c)[0] for c in kernel_joints]
    ee_rotation = [speed(ee_dist, c)[1] for c in kernel_joints]
    theta = PLANAR_JOINTS[2]
    return {
        "planar_base": planar,
        "base_reach": max(d.get(theta, 0.0) for d in points) if planar else 0.0,
        "ee_base_reach": ee_dist.get(theta, 0.0) if planar else 0.0,
        "reach": reach,
        "ee_reach": ee_reach,
        "ee_rotation": ee_rotation,
    }


def joint_limits(urdfs: list[Path], names: list[str]) -> tuple[list[float], list[float]]:
    """Return the limits of each named joint from the first URDF that has it."""
    lo, hi = [], []
    for name in names:
        for urdf in urdfs:
            j = joint_map(ET.parse(urdf).getroot()).get(name)
            if j is not None and j.find("limit") is not None:
                lo.append(float(j.find("limit").get("lower")))
                hi.append(float(j.find("limit").get("upper")))
                break
        else:
            raise SystemExit(f"no limits for joint {name} in {[u.name for u in urdfs]}")
    return lo, hi


def resolve(path: str, repo: Path, vamp: Path) -> Path:
    return Path(path.format(vamp=vamp)) if "{vamp}" in path else repo / path


def sweep(name: str, repo: Path, vamp: Path) -> Path:
    """Write include/geodex/integration/vamp/robots/generated/<name>_sweep.hh."""
    recipe = json.loads((HERE / "robots" / f"{name}.json").read_text())
    generated = repo / "include" / "geodex" / "integration" / "vamp" / "robots" / "generated"
    kernel = resolve(recipe.get("vamp_kernel", str(generated.relative_to(repo) / f"{name}.hh")),
                     repo, vamp)
    default = data_dir(recipe, Path("")) / f"{name}_spherized.urdf"
    spherized = resolve(recipe.get("spherized_urdf", str(default)), repo, vamp)
    limit_urdfs = [resolve(u, repo, vamp) for u in recipe.get("limits_urdfs", [])] + [spherized]
    names, ee = kernel_layout(kernel)
    model = sweep_model(spherized, names, ee)
    lo, hi = joint_limits(limit_urdfs, names)

    def arr(values: list[float]) -> str:
        return "{" + ", ".join(repr(float(v)) for v in values) + "}"

    n = len(names)
    out = generated / f"{name}_sweep.hh"
    out.write_text(
        f"// AUTO-GENERATED by scripts/robotgen/robotgen.py sweep {name}. DO NOT EDIT.\n"
        f"// Sphere model {spherized.name} (sha256 {sha256_of(spherized)}),\n"
        f"// joint layout and end effector from {kernel.name}; joint limits by name from\n"
        f"// {', '.join(u.name for u in limit_urdfs)}.\n"
        "// Bounds on how far a sphere centre or the end-effector frame moves per unit of each\n"
        "// coordinate, from the triangle inequality over the kinematic chain.\n\n"
        "#pragma once\n\n"
        '#include "geodex/integration/vamp/detail/sweep_model.hpp"\n\n'
        "namespace geodex::integration::vamp::detail::generated {\n\n"
        f"inline constexpr SweepModel<{n}> {name}_sweep{{\n"
        f"    .planar_base = {'true' if model['planar_base'] else 'false'},\n"
        f"    .base_reach = {model['base_reach']!r},\n"
        f"    .ee_base_reach = {model['ee_base_reach']!r},\n"
        f"    .reach = {arr(model['reach'])},\n"
        f"    .ee_reach = {arr(model['ee_reach'])},\n"
        f"    .ee_rotation = {arr(model['ee_rotation'])},\n"
        f"    .lower = {arr(lo)},\n"
        f"    .upper = {arr(hi)},\n"
        "};\n\n"
        "}  // namespace geodex::integration::vamp::detail::generated\n")
    print(f"[sweep] {name}: wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build(name: str, cache: Path, work: Path, repo: Path, tools: Path, samples: int,
          stages: set[str]) -> None:
    recipe = json.loads((HERE / "robots" / f"{name}.json").read_text())
    work = work / name
    work.mkdir(parents=True, exist_ok=True)
    data = data_dir(recipe, repo)
    data.mkdir(parents=True, exist_ok=True)
    model = planning_model(recipe, cache)

    dynamics = data / f"{name}_dynamics.urdf"
    mesh_urdf = work / f"{name}_mesh.urdf"
    spherized = data / f"{name}_spherized.urdf"
    srdf = data / f"{name}.srdf"
    kernel = repo / "include" / "geodex" / "integration" / "vamp" / "robots" / "generated" / f"{name}.hh"

    if "urdf" in stages:
        write_urdf(strip_geometry(model), dynamics,
                   f"{recipe['description']}. Fixed-base arm, kinematics and inertials only, the "
                   "CRBA source. Generated by scripts/robotgen/robotgen.py.")
        whole = strip_geometry(model, keep_collision=True)
        if "drive" in recipe:
            whole = with_planar_base(whole, recipe["base_link"], recipe.get("base_height", 0.0))
        repaired = repair_collision_meshes(whole, work)
        if repaired:
            print("[urdf] re-oriented inside-out collision meshes:", ", ".join(repaired))
        write_urdf(whole, mesh_urdf, "whole-body mesh model")
    if "spheres" in stages:
        spherize(recipe, mesh_urdf, spherized, tools, work)
    if "srdf" in stages:
        stats = srdf_pairs(recipe, mesh_urdf, srdf, samples)
        (work / "srdf_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print("[srdf]", stats)
    if "vamp" in stages:
        cricket(recipe, spherized, srdf, kernel, tools, work)
    if "crba" in stages:
        crba(recipe, dynamics, repo / "src" / "robots" / "generated", tools, repo, work,
             recipe.get("crba", {}).get("name"))


def kernel(name: str, repo: Path, tools: Path, work: Path) -> None:
    """Regenerate a VAMP kernel whose sphere model and SRDF are vendored inputs."""
    recipe = json.loads((HERE / "robots" / f"{name}.json").read_text())
    work = work / name
    work.mkdir(parents=True, exist_ok=True)
    out = repo / "include" / "geodex" / "integration" / "vamp" / "robots" / "generated" / f"{name}.hh"
    cricket(recipe, repo / recipe["spherized_urdf"], repo / recipe["srdf"], out, tools, work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--cache", type=Path, required=True)
    f.add_argument("--repo", type=Path, default=HERE.parents[1])
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--cache", type=Path, required=True)
    b.add_argument("--work", type=Path, required=True)
    b.add_argument("--repo", type=Path, required=True)
    b.add_argument("--tools", type=Path, required=True)
    b.add_argument("--samples", type=int, default=100000)
    b.add_argument("--stages", default="urdf,spheres,srdf,vamp,crba")
    q = sub.add_parser("pin")
    q.add_argument("name", help="pin name without the GEODEX_ prefix, e.g. CRICKET_REF")
    q.add_argument("--repo", type=Path, default=HERE.parents[1])
    k = sub.add_parser("kernel")
    k.add_argument("name")
    k.add_argument("--repo", type=Path, required=True)
    k.add_argument("--tools", type=Path, required=True)
    k.add_argument("--work", type=Path, required=True)
    c = sub.add_parser("crba")
    c.add_argument("name")
    c.add_argument("--cache", type=Path, required=True)
    c.add_argument("--work", type=Path, required=True)
    c.add_argument("--repo", type=Path, required=True)
    c.add_argument("--tools", type=Path, required=True)
    g = sub.add_parser("get")
    g.add_argument("name")
    g.add_argument("key", help="dotted recipe key, e.g. crba.name")
    w = sub.add_parser("sweep")
    w.add_argument("name")
    w.add_argument("--repo", type=Path, required=True)
    w.add_argument("--vamp", type=Path, required=True)
    t = sub.add_parser("fit")
    t.add_argument("mesh_urdf", type=Path)
    t.add_argument("sphere_urdf", type=Path)
    t.add_argument("--pitch", type=float, default=0.002)
    t.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.cache, args.repo.resolve())
    elif args.cmd == "build":
        build(args.name, args.cache.resolve(), args.work.resolve(), args.repo.resolve(),
              args.tools.resolve(), args.samples, set(args.stages.split(",")))
    elif args.cmd == "pin":
        print(read_pins(args.repo.resolve()).get(args.name, ""))
    elif args.cmd == "kernel":
        kernel(args.name, args.repo.resolve(), args.tools.resolve(), args.work.resolve())
    elif args.cmd == "crba":
        crba_vendored(args.name, args.cache.resolve(), args.work.resolve(), args.repo.resolve(),
                      args.tools.resolve())
    elif args.cmd == "get":
        value = json.loads((HERE / "robots" / f"{args.name}.json").read_text())
        for part in args.key.split("."):
            value = value.get(part, "") if isinstance(value, dict) else ""
        print(value)
    elif args.cmd == "sweep":
        sweep(args.name, args.repo.resolve(), args.vamp.resolve())
    elif args.cmd == "fit":
        report = sphere_fit(args.mesh_urdf.resolve(), args.sphere_urdf.resolve(), args.pitch)
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(text)
        print(text, end="")


if __name__ == "__main__":
    main()

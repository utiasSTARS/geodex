#!/usr/bin/env python3
"""Write the PR2 assets (URDF, sphere URDF, SRDF, meshes, materials) into a directory.

It fixes every joint except the 14 arm joints and turns the continuous arm joints into
revolute joints over [-pi, pi]. data/robots/pr2/ holds its output, and generate.sh does not
run it.
"""

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ARM_JOINT_NAMES = (
    "l_shoulder_pan_joint",
    "l_shoulder_lift_joint",
    "l_upper_arm_roll_joint",
    "l_elbow_flex_joint",
    "l_forearm_roll_joint",
    "l_wrist_flex_joint",
    "l_wrist_roll_joint",
    "r_shoulder_pan_joint",
    "r_shoulder_lift_joint",
    "r_upper_arm_roll_joint",
    "r_elbow_flex_joint",
    "r_forearm_roll_joint",
    "r_wrist_flex_joint",
    "r_wrist_roll_joint",
)
ARM_JOINTS = set(ARM_JOINT_NAMES)


def indent(elem: ET.Element, level: int = 0) -> None:
    space = "\n" + level * "  "
    child_space = "\n" + (level + 1) * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_space
        for child in elem:
            indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = space
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = space


def normalize_joint(joint: ET.Element) -> None:
    name = joint.attrib.get("name", "")
    joint_type = joint.attrib.get("type", "fixed")
    if name not in ARM_JOINTS:
        if joint_type != "fixed":
            joint.set("type", "fixed")
        return

    if joint_type == "continuous":
        joint.set("type", "revolute")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("lower", f"{-math.pi:.17g}")
        limit.set("upper", f"{math.pi:.17g}")
        if "effort" not in limit.attrib:
            limit.set("effort", "30.0")
        if "velocity" not in limit.attrib:
            limit.set("velocity", "3.6")


def normalize_urdf(src: Path, dst: Path) -> None:
    tree = ET.parse(src)
    root = tree.getroot()
    root.set("name", "pr2")
    for joint in root.findall("joint"):
        normalize_joint(joint)
    indent(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", required=True, type=Path)
    parser.add_argument("--source-spherized-urdf", required=True, type=Path)
    parser.add_argument("--source-srdf", required=True, type=Path)
    parser.add_argument("--source-meshes", required=True, type=Path)
    parser.add_argument("--source-materials", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalize_urdf(args.source_urdf, args.out_dir / "pr2.urdf")
    normalize_urdf(args.source_spherized_urdf, args.out_dir / "pr2_spherized.urdf")
    shutil.copy2(args.source_srdf, args.out_dir / "pr2.srdf")
    copy_tree(args.source_meshes, args.out_dir / "meshes")
    copy_tree(args.source_materials, args.out_dir / "materials")


if __name__ == "__main__":
    main()

"""Check the data files of every robot viewer scene on the docs pages.

Each ``robot-scene`` directive loads a scene, its poster, a robot model and the viewer. The
test finds every directive in the pages, checks that each file exists, that the scene lists
the joints of its robot model in the order of ``data/robots/<robot>/robot.yaml``, and that
every configuration of the path and every trace has that many coordinates. A scene without a
robot has three coordinates per mover in every frame. It also checks the
size budget of each model and the vendored three.js license.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

DOCS = Path(__file__).resolve().parents[1]
STATIC = DOCS / "_static"
DATA = DOCS.parent / "data" / "robots"
sys.path.insert(0, str(DOCS / "_ext"))

from geodex_docs import robot_scene_files  # noqa: E402

DIRECTIVE = re.compile(r"^(\s*)\.\. robot-scene:: (\S+)\s*$")
# Largest size of one robot model's two meshes, in bytes.
MODEL_BUDGET = 800 * 1024


def scenes() -> list[tuple[str, str, bool | str]]:
    """(page, scene name, spheres) of every robot-scene directive in the docs pages, where
    spheres is False, "shown" or "hidden"."""
    found = []
    for page in sorted(DOCS.rglob("*.rst")):
        if "_build" in page.relative_to(DOCS).parts:
            continue
        lines = page.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            match = DIRECTIVE.match(line)
            if not match:
                continue
            indent = len(match.group(1))
            spheres = False
            for option in lines[i + 1:]:
                if option.strip() and len(option) - len(option.lstrip()) <= indent:
                    break
                if option.strip().startswith(":spheres:"):
                    spheres = option.strip()[len(":spheres:"):].strip() or "shown"
            found.append((str(page.relative_to(DOCS)), match.group(2), spheres))
    return found


def planning_joints(robot: str) -> list[str]:
    """Joints of the robot's first planning group. A mobile base of the navigation pages,
    which has no geodex robot model, has the planar base joints."""
    directory = DATA / ("fr3_gripper" if robot == "fr3_arm_gripper" else robot)
    if not directory.exists():
        return ["base_x_joint", "base_y_joint", "base_theta_joint"]
    meta = yaml.safe_load((directory / "robot.yaml").read_text())
    return list(next(iter(meta["planning_groups"].values()))["joints"])


def test_pages_use_the_viewer():
    assert scenes(), "no page holds a robot-scene"


@pytest.mark.parametrize("page,name,spheres", scenes())
def test_scene_files(page, name, spheres):
    files = robot_scene_files(STATIC, name, spheres)
    missing = [str(p.relative_to(DOCS)) for p in files if not p.exists()]
    assert not missing, f"{page}: {name} misses {missing}"

    scene = json.loads(files[0].read_text())
    if "robot" in scene:
        chain = json.loads(files[2].read_text())
        joints = planning_joints(scene["robot"])
        assert scene["joint_names"] == joints
        assert chain["joint_names"] == joints
        n = len(joints)
    else:
        assert scene["movers"]
        n = 3 * len(scene["movers"])
    assert len(scene["frames"]) >= 2
    assert all(len(q) == n for q in scene["frames"])
    for trace in scene["traces"]:
        assert len(trace["points"]) == len(scene["frames"])
    for key in ("position", "target"):
        assert len(scene["camera"][key]) == 3


def test_model_sizes():
    for model in sorted(p for p in (STATIC / "robots").iterdir() if p.is_dir()):
        chain = json.loads((model / "chain.json").read_text())
        size = sum((model / f).stat().st_size for f in chain["meshes"].values())
        assert size <= MODEL_BUDGET, f"{model.name}: {size / 1024:.0f} KB of meshes"


def test_three_license():
    text = (STATIC / "three" / "LICENSE").read_text()
    assert "three.js" in text and "meshoptimizer" in text and "MIT License" in text

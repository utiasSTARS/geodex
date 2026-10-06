"""Tests for the geodex.vamp submodule.

Skipped wholesale when geodex was built without `-DGEODEX_VAMP=ON`.
"""

from pathlib import Path

import numpy as np
import pytest

import geodex

if not hasattr(geodex._geodex_core, "vamp"):
    pytest.skip("geodex was built without VAMP bindings", allow_module_level=True)
vamp = geodex.vamp


REPO = Path(__file__).resolve().parent.parent.parent
PANDA_EMPTY = REPO / "tests" / "fixtures" / "vamp" / "panda" / "empty.yaml"
PANDA_ENCLOSURE = REPO / "tests" / "fixtures" / "vamp" / "panda" / "enclosure.yaml"

if not PANDA_EMPTY.is_file():
    pytest.skip(
        f"Panda VAMP scene fixture missing at {PANDA_EMPTY}", allow_module_level=True
    )


PANDA_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


class TestRegisteredRobots:
    def test_returns_sorted_list(self):
        robots = vamp.registered_robots()
        assert isinstance(robots, list)
        assert robots == sorted(robots)
        assert "panda" in robots

    def test_joint_names_and_end_effector(self):
        for name in vamp.registered_robots():
            joints = vamp.robot_joint_names(name)
            assert len(joints) == vamp.robot_dimension(name), name
            assert len(set(joints)) == len(joints), name
            assert vamp.robot_end_effector(name), name
        assert vamp.robot_joint_names("panda") == [f"panda_joint{i}" for i in range(1, 8)]
        assert vamp.robot_end_effector("panda") == "panda_grasptarget"
        assert vamp.robot_joint_names("ridgeback_ur5e")[:3] == [
            "base_x_joint", "base_y_joint", "base_theta_joint"]
        with pytest.raises(RuntimeError):
            vamp.robot_joint_names("not_a_robot")
        with pytest.raises(RuntimeError):
            vamp.robot_end_effector("not_a_robot")

    def test_robot_spheres(self):
        spheres = vamp.robot_spheres("panda", PANDA_READY)
        assert spheres.ndim == 2 and spheres.shape[1] == 4 and spheres.shape[0] > 10
        assert np.all(spheres[:, 3] > 0.0)

    def test_fr3_arm_gripper_model(self):
        assert vamp.robot_joint_names("fr3_arm_gripper") == [f"fr3_joint{i}" for i in range(1, 8)]
        assert vamp.robot_end_effector("fr3_arm_gripper") == "2f85_tcp"
        spheres = vamp.robot_spheres("fr3_arm_gripper", PANDA_READY)
        assert spheres.shape == (60, 4)
        assert np.all((spheres[:, 3] > 0.0) & (spheres[:, 3] < 0.093))
        with pytest.raises(ValueError):
            vamp.robot_spheres("panda", np.zeros(3))

    def test_joint_names_match_robot_yaml(self):
        yaml = pytest.importorskip("yaml")
        seen = set()
        for path in sorted((REPO / "data" / "robots").glob("*/robot.yaml")):
            spec = yaml.safe_load(path.read_text())
            name = spec.get("vamp_name")
            if name not in vamp.registered_robots():
                continue
            groups = [g["joints"] for g in spec["planning_groups"].values()]
            assert vamp.robot_joint_names(name) in groups, (name, groups)
            seen.add(name)
        assert seen == set(vamp.registered_robots())


class TestLoadScene:
    def test_load_empty_scene(self):
        env = vamp.load_scene(str(PANDA_EMPTY))
        assert env is not None

    def test_load_nonexistent_raises(self):
        with pytest.raises(Exception):
            vamp.load_scene("/tmp/this-file-does-not-exist.yaml")


class TestMakeVampChecker:
    def test_panda_unknown_robot_raises(self):
        env = vamp.load_scene(str(PANDA_EMPTY))
        with pytest.raises(Exception):
            vamp.make_vamp_checker("not_a_robot", env)

    def test_panda_ready_pose_valid_in_empty_scene(self):
        env = vamp.load_scene(str(PANDA_EMPTY))
        checker = vamp.make_vamp_checker("panda", env)
        assert checker.is_valid(PANDA_READY) is True

    def test_panda_ready_pose_invalid_in_enclosure(self):
        if not PANDA_ENCLOSURE.is_file():
            pytest.skip("Panda enclosure fixture missing")
        env = vamp.load_scene(str(PANDA_ENCLOSURE))
        checker = vamp.make_vamp_checker("panda", env)
        # The enclosure scene's box collides with the ready pose, as in the C++
        # EnclosureRejectsReadyPose test.
        assert checker.is_valid(PANDA_READY) is False


class TestBatchAndAttachments:
    def test_all_valid_matches_the_conjunction(self):
        env = vamp.load_scene(str(PANDA_EMPTY))
        checker = vamp.make_vamp_checker("panda", env)
        assert checker.batch_width() >= 1
        rows = np.tile(PANDA_READY, (11, 1))
        assert checker.all_valid(rows) is True
        blocked = vamp.make_vamp_checker("panda", vamp.load_scene(str(PANDA_ENCLOSURE)))
        assert blocked.all_valid(rows) is False
        with pytest.raises(ValueError):
            checker.all_valid(rows[:, :6])

    def test_robot_dimension(self):
        assert vamp.robot_dimension("panda") == 7
        assert vamp.robot_dimension("fr3_arm_gripper") == 7
        with pytest.raises(Exception):
            vamp.robot_dimension("not_a_robot")

    def test_attached_sphere_changes_validity(self):
        ready = np.array([0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0,
                          1.5707963267948966, 0.7853981633974483])
        env = vamp.load_scene(str(PANDA_EMPTY))
        vamp.attach_spheres(env, [[0.0, 0.0, 0.0, 0.4]])
        checker = vamp.make_vamp_checker("fr3_arm_gripper", env)
        assert checker.is_valid(ready) is False
        # A checker keeps the attachment it was built with. A new one sees the change.
        vamp.attach_spheres(env, [])
        assert checker.is_valid(ready) is False
        assert vamp.make_vamp_checker("fr3_arm_gripper", env).is_valid(ready) is True


class TestSceneConvention:
    def test_box_size_is_full_extents_like_yaml(self):
        box_yaml = REPO / "tests" / "fixtures" / "vamp" / "panda" / "box.yaml"
        scene = geodex.Scene()
        scene.add_box([0.45, 0.0, 0.4], [0.2, 0.3, 0.4])
        a = vamp.make_vamp_checker("panda", vamp.load_scene(str(box_yaml)))
        b = vamp.make_vamp_checker("panda", scene.env())
        rng = np.random.default_rng(3)
        lo = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        hi = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        qs = lo + (hi - lo) * rng.random((300, 7))
        verdicts_a = [a.is_valid(q) for q in qs]
        verdicts_b = [b.is_valid(q) for q in qs]
        assert verdicts_a == verdicts_b
        assert not all(verdicts_a)

    def test_scene_env_takes_an_attachment(self):
        scene = geodex.Scene()
        scene.add_sphere([3.0, 0.0, 0.5], 0.1)
        env = scene.env()
        vamp.attach_spheres(env, [[0.0, 0.0, 0.0, 0.4]])
        checker = vamp.make_vamp_checker("panda", env)
        assert checker.is_valid(PANDA_READY) is False

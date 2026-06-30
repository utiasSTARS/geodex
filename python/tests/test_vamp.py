"""Tests for the geodex.vamp submodule.

Skipped wholesale when geodex was built without `-DGEODEX_VAMP=ON`.
"""

from pathlib import Path

import numpy as np
import pytest

import geodex

if not hasattr(geodex, "vamp"):
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
        # The enclosure scene's box collides with the ready pose; mirrors
        # the EnclosureRejectsReadyPose test on the C++ side.
        assert checker.is_valid(PANDA_READY) is False

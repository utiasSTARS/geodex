"""Tests for in-memory scene collision planning (``geodex.Scene`` and ``geodex.plan``).

These tests plan robot motions against an in-memory VAMP collision scene built in Python,
with no dataset files. ``geodex.Scene``, ``geodex.robots`` and the VAMP collision path of
``geodex.plan`` exist only in a full build (``-DGEODEX_VAMP=ON -DGEODEX_ROBOTS=ON`` with
OMPL), and the module is skipped otherwise. The module imports without the planning build.
Every Scene, robots or plan reference lives inside a test body or a helper.
"""

import numpy as np
import pytest

import geodex

pytestmark = pytest.mark.skipif(
    not (hasattr(geodex._geodex_core, "Scene") and hasattr(geodex._geodex_core, "robots") and hasattr(geodex._geodex_core, "plan")),
    reason="scene collision planning requires the full VAMP build",
)


# A 0.3 x 0.6 x 0.8 m box (full extents) in front of the Panda. Both endpoints of
# _interior_endpoints are free, and the straight joint-space segment between them passes
# through the box.
_BOX_POSITION = [0.55, 0.0, 0.4]
_BOX_SIZE = [0.3, 0.6, 0.8]
_BOX_ORIENTATION = [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Helpers. They touch geodex.Scene, geodex.robots and geodex.plan only when called.
# ---------------------------------------------------------------------------


def _panda():
    """Construct the 7-DoF Panda RobotModel."""
    return geodex.robots.Panda()


def _interior_endpoints(model):
    """Two distinct joint vectors at 30% and 60% of each joint's range, well inside the
    limits."""
    lo, hi = model.joint_limits()
    start = np.asarray(lo + 0.3 * (hi - lo), dtype=np.float64)
    goal = np.asarray(lo + 0.6 * (hi - lo), dtype=np.float64)
    return start, goal


def _settings(*, time=5.0, seed=42):
    """PlanSettings with a fixed seed."""
    return geodex.PlanSettings(time=time, seed=seed)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def test_scene_constructs_and_accepts_primitives():
    """Scene() builds and takes one box, one sphere, and one cylinder."""
    scene = geodex.Scene()
    assert scene is not None

    # size is the full box extent; orientation is [qx, qy, qz, qw].
    scene.add_box(
        position=[0.5, 0.0, 0.4],
        size=[0.2, 0.2, 0.2],
        orientation=[0.0, 0.0, 0.0, 1.0],
    )
    # center + radius; a sphere carries no orientation.
    scene.add_sphere(center=[0.4, 0.3, 0.5], radius=0.08)
    # orientation omitted to exercise its documented [0, 0, 0, 1] default.
    scene.add_cylinder(position=[0.3, -0.3, 0.5], radius=0.05, height=0.3)


# ---------------------------------------------------------------------------
# Free space vs. empty scene
# ---------------------------------------------------------------------------


def test_free_space_and_empty_scene_agree():
    """A Panda query that solves in free space also solves against an empty Scene."""
    geodex.seed(0)
    panda = _panda()
    start, goal = _interior_endpoints(panda)

    # collision=None -> free-space planning; must solve for two interior configs.
    free = geodex.plan(panda, start, goal, collision=None, settings=_settings(time=3.0))
    assert free.solved
    assert free.path.ndim == 2
    assert free.path.shape[1] == 7

    # An empty in-memory Scene adds no obstacles, so the same query still solves.
    empty = geodex.Scene()
    scened = geodex.plan(panda, start, goal, collision=empty, settings=_settings(time=3.0))
    assert scened.solved
    assert scened.path.shape[1] == 7


# ---------------------------------------------------------------------------
# Collision avoidance
# ---------------------------------------------------------------------------


def test_panda_plans_around_box_obstacle():
    """Panda plans a collision-free path with a box obstacle in the scene."""
    geodex.seed(0)
    panda = _panda()
    start, goal = _interior_endpoints(panda)

    scene = geodex.Scene()
    scene.add_box(position=_BOX_POSITION, size=_BOX_SIZE, orientation=_BOX_ORIENTATION)
    checker = geodex.vamp.make_vamp_checker("panda", scene.env())

    # Both endpoints are free, and the straight segment between them is blocked.
    assert checker.is_valid(start)
    assert checker.is_valid(goal)
    segment = start + np.linspace(0.0, 1.0, 201)[:, None] * (goal - start)
    assert not checker.all_valid(segment)

    result = geodex.plan(
        panda, start, goal, collision=scene, settings=_settings(time=5.0, seed=42)
    )

    assert result.solved
    assert result.path.ndim == 2
    assert result.path.shape[1] == 7
    assert result.path.shape[0] >= 2
    np.testing.assert_allclose(result.path[0], start, atol=1e-6)
    np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)
    for q in result.path:
        assert checker.is_valid(q)

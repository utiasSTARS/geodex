"""Whole-body planning for the mobile manipulators (``geodex.robots``).

A mobile robot is SE(2) x R^n, the base pose (x, y, theta) followed by the arm joints.
Only the metric changes between a holonomic and a differential-drive base.
"""

from pathlib import Path

import numpy as np
import pytest

import geodex

pytestmark = pytest.mark.skipif(
    not (hasattr(geodex._geodex_core, "robots") and hasattr(geodex._geodex_core, "plan")),
    reason="whole-body models require the robots and planning build",
)

# geodex.vamp is a placeholder that raises on use when the loaded module does not have
# VAMP. The tests that check against a scene ask the compiled module.
needs_vamp = pytest.mark.skipif(not hasattr(geodex._geodex_core, "vamp"),
                                reason="needs the module with VAMP")

WALL = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "vamp" / "mobile" / "wall.yaml"

# (class, start, goal); the wall stands between the two base positions.
_CASES = [
    ("Stretch3", [-1.5, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0],
     [1.5, 0.5, 1.57, 0.9, 0.3, 1.0, -0.3, 0.0]),
    ("Stretch4", [-1.5, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0],
     [1.5, 0.5, -1.57, 0.9, 0.3, 1.0, 0.3, 0.0]),
    ("RidgebackUR5e", [-1.5, 0.0, 0.0, 0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
     [1.5, 0.5, 3.0, 1.0, -0.8, 1.2, -1.9, -1.57, 0.3]),
    ("HuskyUR5e", [-1.5, 0.0, 0.0, 0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
     [1.5, 0.5, 3.0, 1.0, -0.8, 1.2, -1.9, -1.57, 0.3]),
]


def _geodesic_space(dof):
    return geodex.Product([geodex.SE2(), geodex.Euclidean(dof - 3)])


# The box of wall.yaml, its center and half extents.
WALL_CENTER = np.array([0.0, 0.0, 0.6])
WALL_HALF = np.array([0.15, 0.8, 0.6])


def _depth_in_wall(robot, q):
    """Deepest reach of a collision sphere of the robot at q into the wall, 0 outside it."""
    spheres = geodex.vamp.robot_spheres(robot.name(), np.asarray(q))
    d = np.abs(spheres[:, :3] - WALL_CENTER) - WALL_HALF
    signed = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
    return float(max(0.0, np.max(spheres[:, 3] - signed)))


def _check_path(robot, path, env):
    """Assert that the waypoints pass the scene, that no state along the path reaches into
    the wall by more than half the smoother's 5 mm check travel, and that the base stays in
    bounds.

    The check resamples each edge along its SE(2) x R^n geodesic every 0.25 mm of sphere
    travel at most. Between two of the smoother's checks, a sphere moves at most 5 mm, and it
    comes within 2.5 mm of a checked position.
    """
    in_scene = geodex.vamp.make_vamp_checker(robot.name(), env)
    speed = geodex.vamp.sphere_speed(robot.name(), env)
    space = _geodesic_space(robot.dof())
    lo, hi = (np.asarray(x) for x in robot.joint_limits())
    assert all(in_scene.is_valid(q) for q in path)
    depth = 0.0
    outside = 0
    for a, b in zip(path[:-1], path[1:]):
        m = max(1, int(np.ceil(speed * np.linalg.norm(space.log(a, b)) / 2.5e-4)))
        for k in range(m + 1):
            q = np.asarray(space.geodesic(a, b, k / m))
            depth = max(depth, _depth_in_wall(robot, q))
            outside += bool(np.any(q[:2] < lo[:2] - 1e-9) or np.any(q[:2] > hi[:2] + 1e-9))
    assert depth < 0.0025
    assert outside == 0


def test_drives_and_periods():
    s3, s4, rb = geodex.robots.Stretch3(), geodex.robots.Stretch4(), geodex.robots.RidgebackUR5e()
    assert s3.drive() == "differential_drive"
    assert s4.drive() == "holonomic"
    assert rb.drive() == "holonomic"
    assert geodex.robots.HuskyUR5e().drive() == "differential_drive"
    assert geodex.robots.Panda().drive() == "none"
    np.testing.assert_allclose(s3.periods(), [0, 0, 2 * np.pi, 0, 0, 0, 0, 0])
    assert geodex.robots.Panda().periods().size == 0


def test_options_change_only_the_metric():
    q = np.array([0.0, 0.0, 0.7, 0.6, 0.1, 0.0, 0.0, 0.0])
    diff = geodex.robots.Stretch3(base="differential_drive").mass_matrix(q)
    holo = geodex.robots.Stretch3(base="holonomic").mass_matrix(q)
    custom = geodex.robots.Stretch3(base_weights=(1.0, 1.0, 1.0), metric="euclidean").mass_matrix(q)
    np.testing.assert_allclose(diff[3:, 3:], holo[3:, 3:])
    assert not np.allclose(diff[:3, :3], holo[:3, :3])
    np.testing.assert_allclose(custom, np.eye(8), atol=1e-12)
    with pytest.raises(ValueError):
        geodex.robots.Stretch3(base="tank")
    with pytest.raises(ValueError):
        geodex.robots.Stretch3(workspace=((1, -1), (0, 1)))


def test_workspace_bounds_the_base():
    robot = geodex.robots.Stretch4(workspace=((-1.0, 2.0), (0.5, 1.5)))
    lo, hi = robot.joint_limits()
    np.testing.assert_allclose(lo[:3], [-1.0, 0.5, -np.pi])
    np.testing.assert_allclose(hi[:3], [2.0, 1.5, np.pi])
    geodex.seed(0)
    for _ in range(20):
        q = robot.random_point()
        assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)


def test_heuristic_wraps_the_heading():
    robot = geodex.robots.RidgebackUR5e(base="holonomic")
    h = robot.heuristic()
    a = np.zeros(9)
    b = np.zeros(9)
    a[2], b[2] = np.pi - 0.05, -np.pi + 0.05
    assert h(a, b) == pytest.approx(0.1)


def test_heuristic_is_the_product_of_the_base_and_arm_bounds():
    robot = geodex.robots.Stretch3(metric="euclidean", base_weights=(1.0, 20.0, 2.0))
    metric = geodex.SE2LeftInvariantMetric(1.0, 20.0, 2.0)
    h = geodex.heuristics.product_lower_bound(
        [(metric.coordinate_lower_bound(), geodex.SE2().periods()), np.eye(5)])
    np.testing.assert_array_equal(h.periods, robot.periods())
    geodex.seed(3)
    for _ in range(50):
        a, b = robot.random_point(), robot.random_point()
        assert h(a, b) == robot.heuristic()(a, b)


@needs_vamp
def test_recheck_flags_a_path_through_the_wall():
    cls, start, goal = _CASES[2]
    robot = getattr(geodex.robots, cls)(workspace=((-3.0, 3.0), (-3.0, 3.0)))
    env = geodex.load_scene(str(WALL))
    with pytest.raises(AssertionError):
        _check_path(robot, np.array([start, goal]), env)


@needs_vamp
@pytest.mark.parametrize("cls, start, goal", _CASES)
def test_whole_body_plan_is_collision_free(cls, start, goal):
    geodex.seed(7)
    robot = getattr(geodex.robots, cls)(workspace=((-3.0, 3.0), (-3.0, 3.0)))
    env = geodex.load_scene(str(WALL))
    start, goal = np.array(start), np.array(goal)
    checker = geodex.vamp.make_vamp_checker(robot.name(), env)
    assert checker.is_valid(start) and checker.is_valid(goal)
    result = geodex.plan(robot, start, goal, collision=env,
                         settings=geodex.PlanSettings(time=10.0, seed=7))
    assert result.solved
    np.testing.assert_allclose(result.path[0], start, atol=1e-9)
    np.testing.assert_allclose(result.path[-1], goal, atol=1e-9)
    _check_path(robot, result.path, env)


@needs_vamp
def test_differential_and_holonomic_bases_both_plan():
    start = np.array([-1.5, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0])
    goal = np.array([1.5, 0.5, 1.57, 0.9, 0.3, 1.0, -0.3, 0.0])
    env = geodex.load_scene(str(WALL))
    for base in ("differential_drive", "holonomic"):
        geodex.seed(7)
        robot = geodex.robots.Stretch3(base=base, workspace=((-3.0, 3.0), (-3.0, 3.0)))
        result = geodex.plan(robot, start, goal, collision=env,
                             settings=geodex.PlanSettings(time=10.0, seed=7))
        assert result.solved, base
        _check_path(robot, result.path, env)


@needs_vamp
@pytest.mark.parametrize("cls, start, goal", _CASES)
def test_known_free_and_colliding_configurations(cls, start, goal):
    robot = getattr(geodex.robots, cls)()
    checker = geodex.vamp.make_vamp_checker(robot.name(), geodex.load_scene(str(WALL)))
    free = np.array(start)
    on_wall = free.copy()
    on_wall[:3] = [0.0, 0.0, 0.3]
    out_of_box = free.copy()
    out_of_box[0] = 2000.0
    assert checker.is_valid(free)
    assert not checker.is_valid(on_wall)
    assert not checker.is_valid(out_of_box)
    assert checker.all_valid(np.stack([free, np.array(goal)]))
    assert not checker.all_valid(np.stack([free, on_wall]))


SCENES = (Path(__file__).resolve().parents[2] / "examples" / "robots" / "mobile_manipulation"
          / "scenes")


def _max_turn(path, cols):
    """Largest angle in degrees between consecutive steps of the given coordinates, the base
    heading wrapped, where both steps move."""
    d = np.diff(np.asarray(path), axis=0)
    d[:, 2] = (d[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    d = d[:, cols]
    n = np.linalg.norm(d, axis=1)
    ok = (n[:-1] > 1e-4) & (n[1:] > 1e-4)
    c = (d[:-1] * d[1:]).sum(axis=1)[ok] / (n[:-1] * n[1:])[ok]
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))).max())


@needs_vamp
def test_holonomic_plan_turns_smoothly():
    # The documented holonomic plan of the mobile-manipulation guide. Every coordinate turns
    # along curves, the base included.
    robot = geodex.robots.Stretch4(workspace=((-3.0, 3.0), (-2.5, 2.5)))
    r = geodex.plan(robot, np.array([-1.6, -1.2, 0.0, 0.2, 0.0, 3.0, 0.0, 0.0]),
                    np.array([1.0, -1.2, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0]),
                    collision=geodex.load_scene(str(SCENES / "kitchen.yaml")),
                    settings=geodex.PlanSettings(iterations=3000, seed=1,
                                                 collision_check_resolution=0.005))
    assert r.solved and r.smoothed
    assert _max_turn(r.path, list(range(8))) < 20.0


@needs_vamp
def test_differential_drive_arm_turns_smoothly():
    # The documented Stretch 3 plan of the mobile-manipulation guide. The base drives forward,
    # stops and backs into the goal, and the arm's joints turn along a curve there.
    robot = geodex.robots.Stretch3(workspace=((-3.0, 3.0), (-2.5, 2.5)))
    r = geodex.plan(robot, np.array([1.8, -0.2, np.pi / 2, 0.3, 0.0, 3.0, -0.5, 0.0]),
                    np.array([0.0, 1.9, 0.0, 0.85, 0.4, 0.0, 0.0, 0.0]),
                    collision=geodex.load_scene(str(SCENES / "kitchen.yaml")),
                    settings=geodex.PlanSettings(iterations=1500, seed=1,
                                                 collision_check_resolution=0.005))
    assert r.solved and r.smoothed
    assert _max_turn(r.path, list(range(3, 8))) < 20.0

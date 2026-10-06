"""Tests for the built-in robot models and robot planning (``geodex.robots``).

The robots submodule and the ``geodex.plan`` API exist only in a full build
(``-DGEODEX_ROBOTS=ON`` with OMPL bindings), and the module is skipped when either is
absent. The module imports without the planning build. Every robots or plan reference
lives inside a test body or a helper.
"""

import numpy as np
import pytest

import geodex

pytestmark = pytest.mark.skipif(
    not (hasattr(geodex._geodex_core, "robots") and hasattr(geodex._geodex_core, "plan")),
    reason="robots planning requires the full build",
)


# (constructor attribute on geodex.robots, lowercase registry name, expected dof)
_ROBOTS = [
    ("Panda", "panda", 7),
    ("UR5", "ur5", 6),
    ("Baxter", "baxter", 14),
    ("PR2", "pr2", 14),
    ("Fr3Gripper", "fr3_arm_gripper", 7),
    ("Stretch3", "stretch3", 8),
    ("Stretch4", "stretch4", 8),
    ("RidgebackUR5e", "ridgeback_ur5e", 9),
    ("HuskyUR5e", "husky_ur5e", 9),
]

# Documented order of geodex.robots.available(), alphabetical.
_EXPECTED_NAMES = [
    "baxter", "fr3_arm_gripper", "husky_ur5e", "panda", "pr2", "ridgeback_ur5e", "stretch3",
    "stretch4", "ur5",
]

# Free-space planning is exercised only on the small, fast arms.
_PLANNING_ROBOTS = [("Panda", 7), ("UR5", 6)]


# ---------------------------------------------------------------------------
# Lazily evaluated helpers. They touch geodex.robots and geodex.plan only when called, and
# the module imports in a lean build without those symbols.
# ---------------------------------------------------------------------------


def _make(ctor):
    """Construct a RobotModel by its constructor name on geodex.robots."""
    return getattr(geodex.robots, ctor)()


def _interior_endpoints(model):
    """Two distinct joint vectors at 30% and 60% of each joint's range, well inside the
    limits."""
    lo, hi = model.joint_limits()
    start = np.asarray(lo + 0.3 * (hi - lo), dtype=np.float64)
    goal = np.asarray(lo + 0.6 * (hi - lo), dtype=np.float64)
    return start, goal


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_available_returns_the_registry_names():
    names = geodex.robots.available()
    assert isinstance(names, list)
    assert len(names) == len(_EXPECTED_NAMES)
    # Order-independent correctness of the registered set.
    assert set(names) == set(_EXPECTED_NAMES)
    # The registry is alphabetical.
    assert names == _EXPECTED_NAMES


# ---------------------------------------------------------------------------
# RobotModel contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctor, name, dof", _ROBOTS)
def test_robot_model_basic_properties(ctor, name, dof):
    model = _make(ctor)

    # Degrees of freedom and configuration-space dimension.
    assert model.dof() == dof
    assert model.dim() == model.dof()

    # The name matches the lowercase registry name.
    assert model.name().lower() == name

    # __repr__ returns a non-empty string.
    text = repr(model)
    assert isinstance(text, str)
    assert text

    # joint_limits() returns a (lo, hi) pair of float64 arrays of length dof with lo <= hi.
    lo, hi = model.joint_limits()
    assert isinstance(lo, np.ndarray)
    assert isinstance(hi, np.ndarray)
    assert lo.dtype == np.float64
    assert hi.dtype == np.float64
    assert lo.shape == (dof,)
    assert hi.shape == (dof,)
    assert np.all(lo <= hi)

    # random_point() returns a length-dof array inside the joint limits.
    geodex.seed(0)
    for _ in range(8):
        q = model.random_point()
        assert isinstance(q, np.ndarray)
        assert q.shape == (dof,)
        assert np.all(q >= lo - 1e-9)
        assert np.all(q <= hi + 1e-9)


# ---------------------------------------------------------------------------
# Free-space robot planning (small/fast arms only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctor, dof", _PLANNING_ROBOTS)
def test_free_space_planning_solves(ctor, dof):
    geodex.seed(0)
    model = _make(ctor)
    start, goal = _interior_endpoints(model)
    assert start.shape == (dof,)
    assert goal.shape == (dof,)

    # No collision predicate -> free-space planning. Robot planning uses the
    # kinetic-energy metric and the precomputed Loewner-bound heuristic automatically.
    settings = geodex.PlanSettings(time=3.0, seed=42)
    result = geodex.plan(model, start, goal, settings=settings)

    assert result.solved
    assert result.path.ndim == 2
    assert result.path.shape[1] == model.dof()
    assert result.path.shape[0] >= 2

    # Endpoints of the returned path coincide with the requested start/goal.
    np.testing.assert_allclose(result.path[0], start, atol=1e-6)
    np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)


def test_plan_cost_is_finite_and_positive():
    geodex.seed(0)
    model = _make("Panda")
    start, goal = _interior_endpoints(model)

    result = geodex.plan(
        model, start, goal, settings=geodex.PlanSettings(time=3.0, seed=42)
    )

    assert result.solved
    assert isinstance(result.cost, float)
    assert np.isfinite(result.cost)
    # A distinct start/goal has strictly positive geodesic length.
    assert result.cost > 0.0


@pytest.mark.parametrize("ctor, name, dof", _ROBOTS)
def test_mass_matrix_is_symmetric_positive_definite(ctor, name, dof):
    model = _make(ctor)
    assert model.has_mass_matrix()
    geodex.seed(0)
    q = model.random_point()
    M = model.mass_matrix(q)
    assert M.shape == (dof, dof)
    np.testing.assert_allclose(M, M.T, atol=1e-12)
    assert np.linalg.eigvalsh(M).min() > 0.0
    with pytest.raises(ValueError):
        model.mass_matrix(np.zeros(dof + 1))


def test_fr3_gripper_limits_and_mass_matrix():
    # The FR3 of franka_description 2.9.0 with the Robotiq 2F-85 on its coupling, fingers
    # 40 mm apart. The reference entries are Pinocchio's CRBA at the Franka ready pose.
    model = geodex.robots.Fr3Gripper()
    lo, hi = model.joint_limits()
    np.testing.assert_array_equal(lo, [-2.9007, -1.8361, -2.9007, -3.077, -2.8763, 0.4398,
                                       -3.0508])
    np.testing.assert_array_equal(hi, [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
    ready = np.array([0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0,
                      1.5707963267948966, 0.7853981633974483])
    M = model.mass_matrix(ready)
    assert M[0, 0] == pytest.approx(0.587072849706287, abs=1e-12)
    assert M[1, 3] == pytest.approx(-0.7636729416010635, abs=1e-12)
    assert M[6, 6] == pytest.approx(0.000991798201424664, abs=1e-12)


# ---------------------------------------------------------------------------
# Sampler control
# ---------------------------------------------------------------------------


def _panda_plan(model, seed):
    start, goal = _interior_endpoints(model)
    return geodex.plan(model, start, goal,
                       settings=geodex.PlanSettings(iterations=300, seed=seed, smooth=False))


def test_robot_seed_repeats_its_samples_and_plans():
    a, b = geodex.robots.Panda(), geodex.robots.Panda()
    a.seed(3)
    b.seed(3)
    np.testing.assert_array_equal(a.random_point(), b.random_point())
    a.seed(4)
    b.seed(4)
    first_a, first_b = _panda_plan(a, 0), _panda_plan(b, 0)
    assert np.array_equal(first_a.raw_path, first_b.raw_path)
    # An unseeded plan advances the robot's sampler. The next plan differs.
    assert not np.array_equal(first_a.raw_path, _panda_plan(a, 0).raw_path)


def test_robot_set_sampler_switches_the_kind():
    panda = geodex.robots.Panda()
    panda.set_sampler("halton")
    lo, hi = panda.joint_limits()
    # The first Halton point takes 1/p on the axis of the p-th prime.
    first = np.array([1 / 2, 1 / 3, 1 / 5, 1 / 7, 1 / 11, 1 / 13, 1 / 17])
    np.testing.assert_allclose(panda.random_point(), lo + first * (hi - lo), rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        panda.set_sampler("unknown")

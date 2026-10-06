"""Bound C++ collision objects that ClearanceMetric and geodex.plan call in C++.

A ClearanceMetric whose sdf is a geodex.collision SDF, and a plan whose validity is the
is_valid method of a FootprintGridChecker or a geodex.vamp CollisionChecker, run without
calling into Python. A ConfigurationSpace over SE2 with such a metric plans on a typed copy
of the space. Each test runs the same computation through a Python callable that wraps the
object and asserts that both give the same bits.
"""

import gc
import weakref

import numpy as np
import pytest

import geodex
import geodex.collision as gcol

has_plan = hasattr(geodex._geodex_core, "plan")
needs_plan = pytest.mark.skipif(not has_plan, reason="planning requires the full OMPL build")

RES = 0.05
WIDTH, HEIGHT = 161, 101  # cells of an 8 m x 5 m room


def _room():
    """Distance grid of an 8 m x 5 m room with a wall from the floor to y = 3.2 m at x = 4 m."""
    x = np.arange(WIDTH) * RES
    y = np.arange(HEIGHT) * RES
    X, Y = np.meshgrid(x, y)
    wall = np.hypot(np.maximum(np.maximum(3.8 - X, X - 4.2), 0.0), np.maximum(Y - 3.2, 0.0))
    walls = np.minimum(np.minimum(X, x[-1] - X), np.minimum(Y, y[-1] - Y))
    return gcol.DistanceGrid(WIDTH, HEIGHT, RES, np.minimum(wall, walls).ravel().tolist())


GRID = _room()
FOOTPRINT = gcol.PolygonFootprint.rectangle(0.25, 0.2, 6)
CHECKER = gcol.FootprintGridChecker(GRID, FOOTPRINT, 0.05)
START = np.array([1.0, 1.0, 0.0])
GOAL = np.array([7.0, 1.0, 0.0])
WEIGHTS = (1.0, 10.0, 1.0)


def _se2(**kwargs):
    wx, wy, wtheta = WEIGHTS
    return geodex.SE2(wx=wx, wy=wy, wtheta=wtheta, x_lo=0.0, x_hi=8.0, y_lo=0.0, y_hi=5.0,
                      **kwargs)


def _metric(sdf):
    return geodex.ClearanceMetric(geodex.SE2LeftInvariantMetric(*WEIGHTS), sdf, 1.5, 3.0)


def _wrap(obj):
    """A Python callable that calls obj in Python."""
    return lambda q: obj(q)


def _same_plan(a, b):
    assert a.solved and b.solved
    np.testing.assert_array_equal(a.raw_path, b.raw_path)
    np.testing.assert_array_equal(a.path, b.path)
    assert a.cost == b.cost
    assert a.smoothed == b.smoothed
    assert (a.informed_samples, a.focused_samples, a.uniform_samples) == (
        b.informed_samples, b.focused_samples, b.uniform_samples)


def _settings(**kwargs):
    kwargs.setdefault("iterations", 1500)
    kwargs.setdefault("seed", 3)
    kwargs.setdefault("collision_check_resolution", RES)
    return geodex.PlanSettings(**kwargs)


SDFS = {
    "footprint": lambda: CHECKER,
    "grid": lambda: gcol.GridSDF(GRID),
    "circle": lambda: gcol.CircleSDF(4.0, 2.0, 0.5),
    "circles": lambda: gcol.CircleSmoothSDF([gcol.CircleSDF(4.0, 2.0, 0.5),
                                             gcol.CircleSDF(2.0, 3.0, 0.3)], 20.0),
    "rects": lambda: gcol.RectSmoothSDF([gcol.RectObstacle(4.0, 1.6, 0.2, 0.2, 1.6)], 20.0, 0.1),
    "inflated": lambda: gcol.InflatedSDF(gcol.GridSDF(GRID), 0.3),
    "memoized": lambda: gcol.MemoizedSDF(CHECKER),
}


class TestClearanceMetric:
    @pytest.mark.parametrize("make", SDFS.values(), ids=SDFS.keys())
    def test_native_sdf_matches_the_python_callable(self, make):
        sdf = make()
        native, python = _metric(sdf), _metric(_wrap(sdf))
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = np.array([rng.uniform(0.0, 8.0), rng.uniform(0.0, 5.0), rng.uniform(-3.0, 3.0)])
            u, v = rng.normal(size=3), rng.normal(size=3)
            assert native.inner(p, u, v) == python.inner(p, u, v)
            assert native.norm(p, v) == python.norm(p, v)

    def test_wrappers_call_a_native_sdf(self):
        grid_sdf = gcol.GridSDF(GRID)
        q = np.array([3.0, 3.5, 0.4])
        assert gcol.InflatedSDF(grid_sdf, 0.3)(q) == gcol.InflatedSDF(_wrap(grid_sdf), 0.3)(q)
        assert gcol.MemoizedSDF(CHECKER)(q) == CHECKER(q)

    def test_a_short_point_raises_the_error_of_the_sdf(self):
        base = geodex.ConstantSPDMetric(np.eye(2))
        p, u = np.zeros(2), np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="FootprintGridChecker: q has 2 entries"):
            geodex.ClearanceMetric(base, CHECKER).inner(p, u, u)
        with pytest.raises(ValueError, match="FootprintGridChecker: q has 2 entries"):
            geodex.ClearanceMetric(base, _wrap(CHECKER)).inner(p, u, u)

    def test_a_subclass_override_runs_in_python(self):
        class Far(gcol.FootprintGridChecker):
            def __call__(self, q):
                return 100.0

        far = Far(GRID, FOOTPRINT, 0.05)
        p, v = np.array([4.0, 3.4, 0.0]), np.array([1.0, 0.0, 0.0])
        base = geodex.SE2LeftInvariantMetric(*WEIGHTS)
        assert _metric(far).norm(p, v) == pytest.approx(base.norm(p, v))
        assert _metric(far).norm(p, v) < _metric(CHECKER).norm(p, v)

    @pytest.mark.parametrize("wrap", [lambda f: gcol.InflatedSDF(f, 0.1), gcol.MemoizedSDF],
                             ids=["inflated", "memoized"])
    def test_a_wrapper_cycle_is_collected(self, wrap):
        class Holder:
            pass

        holder = Holder()

        def sdf(q):
            return 1.0

        sdf.holder = holder
        holder.obj = wrap(sdf)
        probe = weakref.ref(holder)
        del holder, sdf
        gc.collect()
        assert probe() is None

    def test_the_metric_keeps_its_sdf_alive(self):
        metric = _metric(gcol.FootprintGridChecker(gcol.DistanceGrid(WIDTH, HEIGHT, RES,
                                                                     [1.0] * (WIDTH * HEIGHT)),
                                                   FOOTPRINT))
        gc.collect()
        p, v = np.array([4.0, 2.5, 0.0]), np.array([1.0, 0.0, 0.0])
        assert metric.norm(p, v) > 1.0


@needs_plan
class TestPlan:
    def test_footprint_validity_matches_the_python_callable(self):
        native = geodex.plan(_se2(), START, GOAL, CHECKER.is_valid, settings=_settings())
        python = geodex.plan(_se2(), START, GOAL, _wrap(CHECKER.is_valid), settings=_settings())
        _same_plan(native, python)

    @pytest.mark.parametrize("interp", ["base_geodesic", "auto"])
    def test_clearance_space_matches_the_python_callables(self, interp):
        settings = _settings(interp=interp, iterations=1000 if interp == "auto" else 1500)
        native = geodex.plan(geodex.ConfigurationSpace(_se2(), _metric(CHECKER)), START, GOAL,
                             CHECKER.is_valid, settings=settings)
        python = geodex.plan(geodex.ConfigurationSpace(_se2(), _metric(_wrap(CHECKER))), START,
                             GOAL, _wrap(CHECKER.is_valid), settings=settings)
        _same_plan(native, python)

    def test_clearance_space_with_a_python_validity(self):
        def is_valid(q):
            return CHECKER(q) > 0.0

        native = geodex.plan(geodex.ConfigurationSpace(_se2(), _metric(CHECKER)), START, GOAL,
                             is_valid, settings=_settings())
        python = geodex.plan(geodex.ConfigurationSpace(_se2(), _metric(_wrap(CHECKER))), START,
                             GOAL, is_valid, settings=_settings())
        _same_plan(native, python)

    @pytest.mark.parametrize("heuristic", ["zero", "euclidean", "eigenvalue", "matrix"])
    def test_every_heuristic_matches(self, heuristic):
        def make():
            return {
                "zero": geodex.heuristics.Zero(),
                "euclidean": geodex.heuristics.Euclidean(),
                "eigenvalue": geodex.heuristics.EigenvalueLowerBound(1.0),
                "matrix": _se2().matrix_lower_bound(),
            }[heuristic]

        native = geodex.plan(geodex.ConfigurationSpace(_se2(), _metric(CHECKER)), START, GOAL,
                             CHECKER.is_valid, settings=_settings(iterations=800),
                             heuristic=make())
        python = geodex.plan(geodex.ConfigurationSpace(_se2(), _metric(_wrap(CHECKER))), START,
                             GOAL, _wrap(CHECKER.is_valid), settings=_settings(iterations=800),
                             heuristic=make())
        _same_plan(native, python)

    @pytest.mark.parametrize("kwargs", [{"frame": "world"}, {"retraction": "euler"}])
    def test_every_se2_retraction_matches(self, kwargs):
        native = geodex.plan(geodex.ConfigurationSpace(_se2(**kwargs), _metric(CHECKER)), START,
                             GOAL, CHECKER.is_valid, settings=_settings(iterations=800))
        python = geodex.plan(geodex.ConfigurationSpace(_se2(**kwargs), _metric(_wrap(CHECKER))),
                             START, GOAL, _wrap(CHECKER.is_valid),
                             settings=_settings(iterations=800))
        _same_plan(native, python)

    def test_directional_validator_matches(self):
        validator = geodex.DirectionalMotionValidator(max_reverse_length=0.2)
        native = geodex.plan(geodex.ConfigurationSpace(_se2(), _metric(CHECKER)), START, GOAL,
                             CHECKER.is_valid, settings=_settings(), motion_validator=validator)
        python = geodex.plan(geodex.ConfigurationSpace(_se2(), _metric(_wrap(CHECKER))), START,
                             GOAL, _wrap(CHECKER.is_valid), settings=_settings(),
                             motion_validator=validator)
        _same_plan(native, python)

    def test_unseeded_plans_advance_the_base_sampler_alike(self):
        def run(sdf, valid):
            geodex.seed(9)
            se2 = _se2()
            space = geodex.ConfigurationSpace(se2, _metric(sdf))
            plans = [geodex.plan(space, START, GOAL, valid, settings=_settings(seed=0,
                                                                              iterations=500))
                     for _ in range(2)]
            return plans, se2.random_point()

        (n1, n2), n_next = run(CHECKER, CHECKER.is_valid)
        (p1, p2), p_next = run(_wrap(CHECKER), _wrap(CHECKER.is_valid))
        _same_plan(n1, p1)
        _same_plan(n2, p2)
        assert not np.array_equal(n1.raw_path, n2.raw_path)
        np.testing.assert_array_equal(n_next, p_next)

    def test_a_plain_python_validity_still_plans(self):
        # The plan keeps half the check spacing and the corner tolerance from the walls.
        margin = 0.5 * RES + 1e-4

        def is_valid(q):
            return GRID.distance_at(q[0], q[1]) > 0.4 + margin

        result = geodex.plan(_se2(), START, GOAL, is_valid, settings=_settings())
        assert result.solved
        assert all(GRID.distance_at(q[0], q[1]) > 0.4 for q in result.path)

    def test_a_wrong_point_size_raises_like_python(self):
        plane = geodex.Euclidean(2)
        start, goal = np.array([-0.5, 0.0]), np.array([0.5, 0.0])
        for valid in (CHECKER.is_valid, _wrap(CHECKER.is_valid)):
            with pytest.raises(TypeError):
                geodex.plan(plane, start, goal, valid, settings=_settings(iterations=10))


@needs_plan
@pytest.mark.skipif(not (hasattr(geodex._geodex_core, "Scene")
                         and hasattr(geodex._geodex_core, "robots")),
                    reason="scene planning requires the full VAMP build")
class TestScene:
    def test_vamp_checker_matches_the_python_callable(self):
        panda = geodex.robots.Panda()
        lo, hi = panda.joint_limits()
        start, goal = lo + 0.3 * (hi - lo), lo + 0.6 * (hi - lo)
        scene = geodex.Scene()
        scene.add_box(position=[0.55, 0.0, 0.4], size=[0.3, 0.6, 0.8])
        checker = geodex.vamp.make_vamp_checker("panda", scene.env())
        settings = geodex.PlanSettings(iterations=600, seed=7)
        native = geodex.plan(panda, start, goal, checker.is_valid, settings=settings)
        python = geodex.plan(panda, start, goal, _wrap(checker.is_valid), settings=settings)
        _same_plan(native, python)

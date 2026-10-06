"""Tests for the planning API (``geodex.plan``).

The planning symbols exist only in the full OMPL build, and the module is skipped when
``geodex.plan`` is absent. The module imports without the planning build. Every
planning-API reference lives inside a test body or ``setup_method``.
"""

import os

import numpy as np
import pytest

import geodex

pytestmark = pytest.mark.skipif(
    not hasattr(geodex._geodex_core, "plan"), reason="planning requires the full OMPL build"
)


def _settings(planner, *, time=2.0, seed=42, **kwargs):
    """PlanSettings with a fixed seed."""
    return geodex.PlanSettings(time=time, planner=planner, seed=seed, **kwargs)


def _greedy(**kwargs):
    return geodex.planners.GreedyRRTstar(**kwargs)


# ---------------------------------------------------------------------------
# Euclidean(2) free space
# ---------------------------------------------------------------------------


class TestEuclideanFreeSpace:
    def setup_method(self):
        # The default Euclidean sampling range is [-1, 1]^n. Keep the endpoints inside.
        self.euc = geodex.Euclidean(2)
        self.start = np.array([-0.5, -0.5], dtype=np.float64)
        self.goal = np.array([0.5, 0.5], dtype=np.float64)

    def test_plan_solves_with_matching_endpoints(self):
        result = geodex.plan(
            self.euc, self.start, self.goal, settings=_settings(_greedy())
        )
        assert result.solved is True
        assert result.path.ndim == 2
        assert result.path.shape[1] == 2
        assert result.path.shape[0] >= 2
        np.testing.assert_allclose(result.path[0], self.start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], self.goal, atol=1e-6)

    def test_greedy_rrtstar_cost_near_straight_line(self):
        result = geodex.plan(
            self.euc, self.start, self.goal, settings=_settings(_greedy(), time=2.0)
        )
        assert result.solved
        dist = float(np.linalg.norm(self.goal - self.start))
        # A path is never shorter than the straight line. On this 2-D problem, GreedyRRTstar
        # with smoothing comes within 25% of it.
        assert result.cost >= dist - 1e-9
        assert result.cost <= 1.25 * dist


# ---------------------------------------------------------------------------
# SE2() free space
# ---------------------------------------------------------------------------


class TestSE2FreeSpace:
    def setup_method(self):
        # The default SE2 workspace is [0, 10] x [0, 10], with theta in [-pi, pi).
        self.se2 = geodex.SE2()
        self.start = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        self.goal = np.array([3.0, 3.0, 0.5], dtype=np.float64)

    def test_plan_solves_with_matching_endpoints(self):
        result = geodex.plan(
            self.se2, self.start, self.goal, settings=_settings(_greedy())
        )
        assert result.solved
        assert result.path.shape[1] == 3
        np.testing.assert_allclose(result.path[0], self.start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], self.goal, atol=1e-6)


# ---------------------------------------------------------------------------
# Sphere() obstacle avoidance
# ---------------------------------------------------------------------------


class TestSphereObstacleAvoidance:
    def setup_method(self):
        self.sphere = geodex.Sphere()
        self.start = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.goal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # Collision-free everywhere except the corner where x and z are both large.
        # The direct geodesic runs through that corner, and the path must detour. The plan
        # checks the corner grown by half the check spacing and the corner tolerance.
        self.free = lambda q: not (q[0] > 0.3 and q[2] > 0.3)
        grown = 0.3 - (0.5 * 0.02 + 1e-4)
        self.padded = lambda q: not (q[0] > grown and q[2] > grown)

    def test_plan_avoids_obstacle_and_stays_on_sphere(self):
        result = geodex.plan(
            self.sphere,
            self.start,
            self.goal,
            self.padded,
            settings=_settings(_greedy(), collision_check_resolution=0.02),
        )
        assert result.solved
        # Every returned point is a unit vector on S^2.
        norms = np.linalg.norm(result.path, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        # Every returned point is collision-free.
        for row in result.path:
            assert self.free(row)
        np.testing.assert_allclose(result.path[0], self.start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], self.goal, atol=1e-6)


# ---------------------------------------------------------------------------
# Torus(2) free space
# ---------------------------------------------------------------------------


class TestTorusFreeSpace:
    def setup_method(self):
        # Torus coordinates lie in [0, 2*pi)^n. Keep the endpoints inside.
        self.torus = geodex.Torus(2)
        self.start = np.array([0.5, 0.5], dtype=np.float64)
        self.goal = np.array([2.0, 2.0], dtype=np.float64)

    def test_plan_solves(self):
        result = geodex.plan(
            self.torus, self.start, self.goal, settings=_settings(_greedy())
        )
        assert result.solved
        assert result.path.shape[1] == 2


# ---------------------------------------------------------------------------
# Product([Euclidean(2), SE2()]) free space
# ---------------------------------------------------------------------------


class TestProductFreeSpace:
    def setup_method(self):
        self.space = geodex.Product([geodex.Euclidean(2), geodex.SE2()])
        # Sample the endpoints with the product's own sampler. Each block stays in its range,
        # Euclidean in [-1, 1]^2 and SE2 in its workspace.
        self.space.seed(1)
        self.start = self.space.random_point()
        self.goal = self.space.random_point()

    def test_plan_solves(self):
        # Combined ambient size = Euclidean(2) + SE2 = 2 + 3 = 5.
        d = self.space.random_point().shape[0]
        assert d == 5
        assert self.start.shape[0] == d
        assert self.goal.shape[0] == d

        result = geodex.plan(
            self.space, self.start, self.goal, settings=_settings(_greedy(), time=3.0)
        )
        assert result.solved
        assert result.path.ndim == 2
        assert result.path.shape[1] == d


# ---------------------------------------------------------------------------
# SphereN(3) obstacle avoidance  (the 3-sphere S^3, ambient point size 4)
# ---------------------------------------------------------------------------


class TestSphereNObstacleAvoidance:
    def setup_method(self):
        # SphereN(3) is S^3 embedded in R^4, and points are unit vectors of size 4.
        self.sphere = geodex.SphereN(3)
        self.start = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.goal = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        # Block the corner where the first and last coordinates are both large,
        # which the direct geodesic passes through. The plan checks the corner grown by half
        # the check spacing and the corner tolerance.
        self.free = lambda q: not (q[0] > 0.3 and q[3] > 0.3)
        grown = 0.3 - (0.5 * 0.02 + 1e-4)
        self.padded = lambda q: not (q[0] > grown and q[3] > grown)

    def test_plan_solves_on_unit_3_sphere(self):
        result = geodex.plan(
            self.sphere,
            self.start,
            self.goal,
            is_valid=self.padded,
            settings=_settings(_greedy(), time=3.0, collision_check_resolution=0.02),
        )
        assert result.solved
        assert result.path.shape[1] == 4
        # Every returned point lies on the unit 3-sphere.
        norms = np.linalg.norm(result.path, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        for row in result.path:
            assert self.free(row)


# ---------------------------------------------------------------------------
# PlanResult attribute / type contract
# ---------------------------------------------------------------------------


class TestPlanResultContract:
    def setup_method(self):
        self.euc = geodex.Euclidean(2)
        self.start = np.array([-0.5, -0.5], dtype=np.float64)
        self.goal = np.array([0.5, 0.5], dtype=np.float64)
        self.result = geodex.plan(
            self.euc, self.start, self.goal, settings=_settings(_greedy())
        )

    def test_path_is_2d_float64_ndarray(self):
        assert isinstance(self.result.path, np.ndarray)
        assert self.result.path.ndim == 2
        assert self.result.path.dtype == np.float64

    def test_raw_path_is_2d_ndarray(self):
        assert isinstance(self.result.raw_path, np.ndarray)
        assert self.result.raw_path.ndim == 2
        assert self.result.raw_path.shape[1] == self.result.path.shape[1]

    def test_waypoints_is_list_of_arrays(self):
        assert isinstance(self.result.waypoints, list)
        assert all(isinstance(w, np.ndarray) for w in self.result.waypoints)
        assert len(self.result.waypoints) == self.result.path.shape[0]

    def test_scalar_field_types(self):
        assert isinstance(self.result.cost, float)
        assert isinstance(self.result.time_ms, float)
        assert isinstance(self.result.solved, bool)

    def test_first_solution_within_the_search(self):
        assert 0.0 <= self.result.first_solution_ms <= self.result.time_ms
        assert self.result.first_solution_iterations >= 1

    def test_bool_matches_solved(self):
        assert bool(self.result) == self.result.solved

    def test_len_matches_path_rows(self):
        assert len(self.result) == self.result.path.shape[0]


# ---------------------------------------------------------------------------
# PlanSettings construction and field round-trip
# ---------------------------------------------------------------------------


class TestPlanSettings:
    def test_explicit_settings_plan_solves(self):
        euc = geodex.Euclidean(2)
        start = np.array([-0.5, -0.5], dtype=np.float64)
        goal = np.array([0.5, 0.5], dtype=np.float64)
        settings = geodex.PlanSettings(
            time=2.0, planner=geodex.planners.GreedyRRTstar(range=1.0)
        )
        result = geodex.plan(euc, start, goal, settings=settings)
        assert result.solved

    def test_field_round_trip(self):
        settings = geodex.PlanSettings(
            time=2.0, planner=geodex.planners.GreedyRRTstar(range=1.0)
        )
        # interp defaults to the string "base_geodesic" and reads back unchanged.
        assert settings.interp == "base_geodesic"
        assert settings.time == pytest.approx(2.0)
        # Untouched fields keep their documented defaults.
        assert settings.smooth is True
        assert settings.smoothing.output_spacing == pytest.approx(0.0)
        assert settings.seed == 0


# ---------------------------------------------------------------------------
# ConfigurationSpace planning with a custom (anisotropic) metric
# ---------------------------------------------------------------------------


def _planar_arm_mass(q):
    """Two-link planar arm mass matrix, anisotropic in the elbow angle q[1]."""
    h = 0.5 * np.cos(q[1])
    m00 = 1 / 12 + 1 / 12 + 0.25 + (1.0 + 0.25 + 2 * h)
    m01 = 1 / 12 + (0.25 + h)
    return np.array([[m00, m01], [m01, 1 / 12 + 0.25]])


def _kinetic_energy_cspace(lo=-np.pi, hi=np.pi):
    base = geodex.Euclidean(2)
    base.set_sampling_bounds(np.array([lo, lo]), np.array([hi, hi]))
    return geodex.ConfigurationSpace(base, geodex.KineticEnergyMetric(_planar_arm_mass))


class TestConfigurationSpacePlanning:
    def setup_method(self):
        self.cspace = _kinetic_energy_cspace()
        self.start = np.array([-np.pi / 4, -np.pi / 4], dtype=np.float64)
        self.goal = np.array([3 * np.pi / 4, 3 * np.pi / 4], dtype=np.float64)

    def test_greedy_rrtstar_with_zero_heuristic_solves(self):
        # The informed planner needs the base manifold's sampling interface. A
        # ConfigurationSpace forwards it, and plan() derives a search domain from it. The
        # zero heuristic is admissible for any metric.
        result = geodex.plan(
            self.cspace, self.start, self.goal,
            settings=_settings(_greedy(greedy_ratio=0.0)),
            heuristic=geodex.heuristics.Zero(),
        )
        assert result.solved
        np.testing.assert_allclose(result.path[0], self.start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], self.goal, atol=1e-6)

    def test_kinetic_energy_path_leaves_the_straight_line(self):
        # Under the anisotropic mass metric, the minimum-energy path curves toward
        # the low-inertia folded configurations rather than tracking the chord.
        result = geodex.plan(
            self.cspace, self.start, self.goal,
            settings=_settings(_greedy(greedy_ratio=0.0)),
            heuristic=geodex.heuristics.Zero(),
        )
        chord = self.goal - self.start
        t = (result.path - self.start) @ chord / (chord @ chord)
        proj = self.start + np.outer(t, chord)
        deviation = np.linalg.norm(result.path - proj, axis=1)
        assert deviation.max() > 1.0


class TestHeuristicSelection:
    def setup_method(self):
        self.cspace = _kinetic_energy_cspace()
        self.start = np.array([-np.pi / 4, -np.pi / 4], dtype=np.float64)
        self.goal = np.array([3 * np.pi / 4, 3 * np.pi / 4], dtype=np.float64)

    @pytest.mark.parametrize(
        "heuristic",
        [
            None,
            geodex.heuristics.Zero(),
            geodex.heuristics.Euclidean(),
            geodex.heuristics.EigenvalueLowerBound(0.3),
            geodex.heuristics.MatrixLowerBound(0.3 * np.eye(2)),
        ],
    )
    def test_supported_heuristics_solve(self, heuristic):
        result = geodex.plan(
            self.cspace, self.start, self.goal,
            settings=_settings(_greedy(greedy_ratio=0.0), time=1.0),
            heuristic=heuristic,
        )
        assert result.solved

    def test_invalid_heuristic_raises(self):
        with pytest.raises((TypeError, ValueError)):
            geodex.plan(
                self.cspace, self.start, self.goal,
                settings=_settings(_greedy()), heuristic="zero",
            )


class TestEuclideanSamplingBounds:
    def test_bounds_round_trip(self):
        euc = geodex.Euclidean(2)
        euc.set_sampling_bounds(np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi]))
        np.testing.assert_allclose(euc.lo(), [-np.pi, -np.pi])
        np.testing.assert_allclose(euc.hi(), [np.pi, np.pi])

    def test_bounds_widen_the_search_domain(self):
        # The wall forces the crossing to |q2| >= 1.5, outside the default
        # [-1, 1]^2 sampling range. The planner finds the detour only when the widened
        # bounds enclose it.
        euc = geodex.Euclidean(2)
        euc.set_sampling_bounds(np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi]))
        start = np.array([-2.5, 0.0], dtype=np.float64)
        goal = np.array([2.5, 0.0], dtype=np.float64)

        def free(q):
            return not (abs(q[0]) < 0.5 and abs(q[1]) < 1.5)

        # Sample uniformly (greedy_ratio 0, zero heuristic). The planner explores the
        # detour outside the straight-line corridor and does not prune it.
        result = geodex.plan(
            euc, start, goal, free,
            settings=_settings(_greedy(greedy_ratio=0.0), time=3.0),
            heuristic=geodex.heuristics.Zero(),
        )
        assert result.solved
        # The path detoured through the widened region to clear the wall.
        assert np.abs(result.path[:, 1]).max() > 1.4


class TestSE2InformedDefault:
    def test_plan_defaults_to_the_certified_wrapped_bound(self):
        # A start near +pi and a goal near -pi are a short turn apart through the
        # cut. An unwrapped chord measures nearly a full turn there, and the informed set
        # collapses. This problem exercises the deck-group union.
        se2 = geodex.SE2(x_lo=0.0, x_hi=10.0, y_lo=0.0, y_hi=10.0)
        start = np.array([5.0, 5.0, np.pi - 0.15])
        goal = np.array([6.0, 5.0, -np.pi + 0.15])

        result = geodex.plan(
            se2, start, goal,
            settings=_settings(_greedy(greedy_ratio=0.9), iterations=1500, seed=1),
        )
        assert result.solved
        np.testing.assert_allclose(result.path[0], start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)
        # The cost stays near the short turn through the cut and does not wind the long
        # way round.
        assert result.cost < 3.0

    def test_plan_is_reproducible_under_a_fixed_budget(self):
        se2 = geodex.SE2(x_lo=0.0, x_hi=10.0, y_lo=0.0, y_hi=10.0)
        start = np.array([1.0, 1.0, 0.0])
        goal = np.array([8.0, 8.0, np.pi / 2.0])
        costs = []
        for _ in range(2):
            geodex.seed(7)
            r = geodex.plan(
                se2, start, goal,
                settings=_settings(_greedy(), iterations=800, seed=7),
            )
            assert r.solved
            costs.append(r.cost)
        assert costs[0] == pytest.approx(costs[1])

    def test_explicit_heuristic_still_overrides(self):
        se2 = geodex.SE2(x_lo=0.0, x_hi=10.0, y_lo=0.0, y_hi=10.0)
        result = geodex.plan(
            se2, np.array([1.0, 1.0, 0.0]), np.array([6.0, 6.0, 0.0]),
            settings=_settings(_greedy(), iterations=800, seed=3),
            heuristic=geodex.heuristics.Zero(),
        )
        assert result.solved


# ---------------------------------------------------------------------------
# The space's sampler kind, instance and seed reach plan()
# ---------------------------------------------------------------------------


def _wall_free(q):
    return not (abs(q[0]) < 0.1 and q[1] > -0.5)


def _plan_around_wall(space, seed):
    """Raw GreedyRRTstar path around a wall under a fixed iteration budget."""
    return geodex.plan(
        space, np.array([-0.8, -0.8]), np.array([0.8, 0.8]), _wall_free,
        settings=geodex.PlanSettings(iterations=500, seed=seed, smooth=False),
    )


def _seeded_euclidean(seed, sampler="scrambled"):
    space = geodex.Euclidean(2, sampler=sampler)
    space.seed(seed)
    return space


def _same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


class TestSamplerReachesThePlanner:
    def test_sampler_kind_changes_and_reproduces_the_plan(self):
        paths = {}
        for kind in ("scrambled", "halton", "random"):
            a = _plan_around_wall(geodex.Euclidean(2, sampler=kind), 7)
            b = _plan_around_wall(geodex.Euclidean(2, sampler=kind), 7)
            assert a.solved
            assert _same(a.raw_path, b.raw_path)
            paths[kind] = a.raw_path
        assert not _same(paths["scrambled"], paths["halton"])
        assert not _same(paths["scrambled"], paths["random"])
        assert not _same(paths["halton"], paths["random"])

    def test_plan_seed_changes_the_plan(self):
        a = _plan_around_wall(geodex.Euclidean(2), 7)
        b = _plan_around_wall(geodex.Euclidean(2), 8)
        assert not _same(a.raw_path, b.raw_path)

    def test_set_sampler_reaches_the_planner(self):
        space = geodex.Euclidean(2)
        space.set_sampler("random")
        assert _same(
            _plan_around_wall(space, 7).raw_path,
            _plan_around_wall(geodex.Euclidean(2, sampler="random"), 7).raw_path,
        )

    def test_unseeded_plans_on_one_space_are_independent(self):
        space = geodex.Euclidean(2)
        a = _plan_around_wall(space, 0)
        b = _plan_around_wall(space, 0)
        assert a.solved and b.solved
        assert not _same(a.raw_path, b.raw_path)

    def test_unseeded_plan_advances_the_space_by_one_sample(self):
        space = _seeded_euclidean(11)
        reference = _seeded_euclidean(11)
        _plan_around_wall(space, 0)
        reference.random_point()
        np.testing.assert_array_equal(space.random_point(), reference.random_point())

    def test_seeded_space_repeats_its_sequence_of_plans(self):
        a, b = _seeded_euclidean(11), _seeded_euclidean(11)
        a1, a2 = _plan_around_wall(a, 0), _plan_around_wall(a, 0)
        b1, b2 = _plan_around_wall(b, 0), _plan_around_wall(b, 0)
        assert _same(a1.raw_path, b1.raw_path)
        assert _same(a2.raw_path, b2.raw_path)
        assert not _same(a1.raw_path, a2.raw_path)
        assert not _same(a1.raw_path, _plan_around_wall(_seeded_euclidean(12), 0).raw_path)

    def test_plan_seed_ignores_the_space_state(self):
        advanced = _seeded_euclidean(11)
        for _ in range(3):
            advanced.random_point()
        first = _plan_around_wall(_seeded_euclidean(11), 5).raw_path
        assert _same(first, _plan_around_wall(_seeded_euclidean(12), 5).raw_path)
        assert _same(first, _plan_around_wall(advanced, 5).raw_path)

    def test_seeded_plan_leaves_the_space_untouched(self):
        space = _seeded_euclidean(11)
        _plan_around_wall(space, 5)
        np.testing.assert_array_equal(space.random_point(), _seeded_euclidean(11).random_point())

    def test_configuration_space_and_product_forward_the_base_sampler(self):
        def cspace(seed):
            return geodex.ConfigurationSpace(
                _seeded_euclidean(seed), geodex.ConstantSPDMetric(np.eye(2))
            )

        def product(seed):
            space = geodex.Product([geodex.Euclidean(1), geodex.Euclidean(1)])
            space.seed(seed)
            return space

        for make in (cspace, product):
            space = make(11)
            a1, a2 = _plan_around_wall(space, 0), _plan_around_wall(space, 0)
            b = _plan_around_wall(make(11), 0)
            c = _plan_around_wall(make(12), 0)
            assert a1.solved
            assert _same(a1.raw_path, b.raw_path)
            assert not _same(a1.raw_path, a2.raw_path)
            assert not _same(a1.raw_path, c.raw_path)


class TestInformedSamplingFocuses:
    def setup_method(self):
        self.space = geodex.Euclidean(2)
        self.start = np.array([-0.8, -0.8])
        self.goal = np.array([0.8, 0.8])

    def _plan(self, planner):
        return geodex.plan(
            self.space, self.start, self.goal,
            settings=geodex.PlanSettings(iterations=2000, seed=1, smooth=False, planner=planner),
        )

    def test_greedy_plan_samples_the_informed_and_greedy_sets(self):
        r = self._plan(_greedy())
        assert r.solved
        assert r.informed_samples > 10 * r.uniform_samples
        assert r.focused_samples > r.informed_samples // 2

    def test_zero_greedy_ratio_gives_no_greedy_samples(self):
        r = self._plan(_greedy(greedy_ratio=0.0))
        assert r.informed_samples > 10 * r.uniform_samples
        assert r.focused_samples == 0


# ---------------------------------------------------------------------------
# Lie groups and periodic manifolds
# ---------------------------------------------------------------------------


def _rot_z(angle):
    return np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)])


class TestLieGroupPlanning:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_so2_plans_through_the_cut(self, seed):
        # A raw chord across the cut overestimates the wrapped distance. plan() certifies
        # a periodic bound, and informed sampling keeps working.
        r = geodex.plan(
            geodex.SO2(), np.array([-2.5]), np.array([2.5]),
            settings=geodex.PlanSettings(iterations=1000, seed=seed, smooth=False),
        )
        assert r.solved
        assert r.informed_samples > 0
        assert r.cost < 2 * np.pi - 5.0 + 0.01

    def test_torus_plans_through_the_cut(self):
        r = geodex.plan(
            geodex.Torus(2), np.array([0.5, 3.0]), np.array([5.8, 3.0]),
            settings=geodex.PlanSettings(iterations=2000, seed=1, smooth=False),
        )
        assert r.solved
        assert r.informed_samples > 0
        assert r.cost < 1.1 * (2 * np.pi - 5.3)

    def test_so3_stays_on_the_manifold_around_an_obstacle(self):
        so3 = geodex.SO3()
        start, goal, middle = _rot_z(0.0), _rot_z(2.5), _rot_z(1.25)

        def free(q):
            return so3.distance(q, middle) > 0.4

        def padded(q):
            return so3.distance(q, middle) > 0.4 + 0.5 * 0.02 + 1e-4

        r = geodex.plan(
            so3, start, goal, padded,
            settings=geodex.PlanSettings(iterations=1500, seed=2,
                                         collision_check_resolution=0.02),
        )
        assert r.solved
        for path in (r.raw_path, r.path):
            np.testing.assert_allclose(np.linalg.norm(path, axis=1), 1.0, atol=1e-9)
            assert all(free(q) for q in path)
        assert so3.distance(r.path[0], start) < 1e-9
        assert so3.distance(r.path[-1], goal) < 1e-9
        assert r.cost > so3.distance(start, goal)

    def test_se3_stays_on_the_manifold_around_an_obstacle(self):
        se3 = geodex.SE3(x_lo=0.0, x_hi=4.0, y_lo=0.0, y_hi=4.0, z_lo=0.0, z_hi=4.0)
        start = np.concatenate([[0.5, 0.5, 0.5], _rot_z(0.0)])
        goal = np.concatenate([[3.5, 3.5, 3.5], _rot_z(2.0)])

        def free(q):
            return np.linalg.norm(q[:3] - 2.0) > 0.8

        def padded(q):
            return np.linalg.norm(q[:3] - 2.0) > 0.8 + 0.02 + 1e-4

        r = geodex.plan(se3, start, goal, padded,
                        settings=geodex.PlanSettings(iterations=2000, seed=3,
                                                     collision_check_resolution=0.02))
        assert r.solved
        for path in (r.raw_path, r.path):
            np.testing.assert_allclose(np.linalg.norm(path[:, 3:], axis=1), 1.0, atol=1e-9)
            assert all(free(q) for q in path)
        assert se3.distance(r.path[0], start) < 1e-9
        assert se3.distance(r.path[-1], goal) < 1e-9


# ---------------------------------------------------------------------------
# DirectionalMotionValidator
# ---------------------------------------------------------------------------


class TestDirectionalMotionValidator:
    def test_every_edge_drives_forward(self):
        se2 = geodex.SE2(wx=1.0, wy=20.0, wtheta=0.5)
        # Start facing away from the goal. A direct reverse would be shorter.
        start = np.array([5.0, 5.0, 0.0])
        goal = np.array([2.0, 5.0, 0.0])
        r = geodex.plan(
            se2, start, goal,
            settings=geodex.PlanSettings(iterations=3000, seed=4),
            motion_validator=geodex.DirectionalMotionValidator(max_reverse_length=0.0),
        )
        assert r.solved
        for path in (r.raw_path, r.path):
            for a, b in zip(path[:-1], path[1:]):
                assert se2.log(a, b)[0] >= -1e-9

    @pytest.mark.parametrize(
        "kwargs", [{"frame": "world"}, {"retraction": "euler"}]
    )
    def test_refuses_an_se2_without_a_body_twist_log(self, kwargs):
        with pytest.raises(ValueError, match="body twist"):
            geodex.plan(
                geodex.SE2(**kwargs), np.array([5.0, 5.0, 0.0]), np.array([2.0, 5.0, 0.0]),
                settings=geodex.PlanSettings(iterations=10, seed=1),
                motion_validator=geodex.DirectionalMotionValidator(),
            )


# ---------------------------------------------------------------------------
# Parity with C++ for automatic interpolation, the shared seed source, limits and log level
# ---------------------------------------------------------------------------


class TestAutoInterpolation:
    @pytest.mark.parametrize(
        "space, start, goal",
        [
            (lambda: geodex.Sphere(), [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            (lambda: geodex.SE2(retraction="euler"), [1.0, 1.0, 0.0], [8.0, 8.0, 1.0]),
            (lambda: geodex.Euclidean(2), [-0.8, -0.8], [0.8, 0.8]),
        ],
    )
    def test_auto_uses_the_manifold_geodesic_when_log_is_riemannian(self, space, start, goal):
        # These manifolds report a Riemannian log, as in C++.
        def run(interp):
            return geodex.plan(
                space(), np.array(start), np.array(goal),
                settings=geodex.PlanSettings(iterations=300, seed=5, interp=interp),
            )

        auto, base = run("auto"), run("base_geodesic")
        assert auto.solved
        np.testing.assert_array_equal(auto.raw_path, base.raw_path)
        np.testing.assert_array_equal(auto.path, base.path)

    def test_auto_keeps_the_discrete_geodesic_for_a_custom_metric(self):
        cspace = _kinetic_energy_cspace()
        start = np.array([-np.pi / 4, -np.pi / 4])
        goal = np.array([3 * np.pi / 4, 3 * np.pi / 4])

        def run(interp):
            return geodex.plan(
                cspace, start, goal, heuristic=geodex.heuristics.Zero(),
                settings=geodex.PlanSettings(iterations=300, seed=5, interp=interp),
            )

        assert not np.array_equal(run("auto").raw_path, run("base_geodesic").raw_path)

    def test_auto_keeps_the_discrete_geodesic_on_the_se2_group(self):
        # The screw motion of the group exponential is not a geodesic of the metric.
        start, goal = np.array([1.0, 1.0, 0.0]), np.array([8.0, 8.0, 1.0])

        def run(interp):
            return geodex.plan(
                geodex.SE2(), start, goal,
                settings=geodex.PlanSettings(iterations=300, seed=5, interp=interp,
                                             smooth=False),
            )

        auto, riemannian = run("auto"), run("riemannian_geodesic")
        np.testing.assert_array_equal(auto.raw_path, riemannian.raw_path)
        assert not np.array_equal(auto.path, run("base_geodesic").path)


def test_discrete_geodesic_on_unit_se2_has_the_flat_length():
    # Unit weights give the flat metric of R^2 x S^1, whose geodesic drives straight
    # while turning at a constant rate, shorter than the screw motion of the twist.
    a, b = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, np.pi / 2])
    settings = geodex.InterpolationSettings()
    settings.step_size = 0.01
    settings.max_steps = 2000
    flat = np.hypot(1.0, np.pi / 2)
    for space in (geodex.SE2(), geodex.SE2(retraction="euler")):
        path = np.asarray(geodex.discrete_geodesic(space, a, b, settings).path)
        d = np.diff(path, axis=0)
        d[:, 2] = (d[:, 2] + np.pi) % (2 * np.pi) - np.pi
        assert np.sum(np.linalg.norm(d, axis=1)) == pytest.approx(flat, rel=2e-3)
    assert geodex.SE2().distance(a, b) > flat + 1e-2


class TestSharedSeedSource:
    @pytest.mark.parametrize(
        "make",
        [geodex.Sphere, lambda: geodex.SphereN(3), geodex.SE2, geodex.SE3, geodex.SO3,
         lambda: geodex.Euclidean(2), lambda: geodex.Torus(2), geodex.SO2],
    )
    def test_each_space_takes_one_seed_on_construction(self, make):
        # As in C++, each manifold takes one seed from the shared source when built. The
        # second space built after geodex.seed(s) is the same whatever the first one was.
        geodex.seed(7)
        geodex.Euclidean(2)
        second = make().random_point()
        geodex.seed(7)
        make()
        np.testing.assert_array_equal(second, make().random_point())


class TestPlanLimits:
    def test_path_stays_inside_the_limits(self):
        def free(q):
            return not (abs(q[0]) < 0.1 and q[1] < 0.3)

        settings = geodex.PlanSettings(iterations=1500, seed=3, collision_check_resolution=0.01,
                                       limits=(np.array([-1.0, -1.0]), np.array([1.0, 0.5])))
        r = geodex.plan(geodex.Euclidean(2), np.array([-0.8, -0.8]), np.array([0.8, -0.8]),
                        free, settings=settings)
        assert r.solved
        assert r.raw_path[:, 1].max() <= 0.5 + 1e-12
        assert r.path[:, 1].max() <= 0.5 + 1e-12
        lo, hi = settings.limits
        np.testing.assert_array_equal(hi, [1.0, 0.5])

    def test_limits_of_the_wrong_size_raise(self):
        settings = geodex.PlanSettings(limits=(np.zeros(3), np.ones(3)))
        with pytest.raises(ValueError):
            geodex.plan(geodex.Euclidean(2), np.array([-0.5, 0.0]), np.array([0.5, 0.0]),
                        settings=settings)


class TestLogLevel:
    def _plan(self):
        geodex.plan(geodex.Euclidean(2), np.array([-0.8, -0.8]), np.array([0.8, 0.8]),
                    settings=geodex.PlanSettings(iterations=200, seed=1))

    def test_plans_are_quiet_by_default_and_can_be_verbose(self, capfd):
        # GEODEX_LOG_LEVEL sets the starting level, Warn without it.
        if "GEODEX_LOG_LEVEL" not in os.environ:
            assert geodex.log_level() == geodex.LogLevel.Warn
        geodex.set_log_level(geodex.LogLevel.Warn)
        self._plan()
        assert "Info:" not in capfd.readouterr().out
        geodex.set_log_level(geodex.LogLevel.Info)
        try:
            self._plan()
            assert "Info:" in capfd.readouterr().out
        finally:
            geodex.set_log_level(geodex.LogLevel.Warn)

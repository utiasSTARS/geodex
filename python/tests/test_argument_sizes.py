"""Arrays of the wrong size raise ValueError instead of reading past a buffer."""

import numpy as np
import pytest

import geodex


def _raises(fn, *args):
    with pytest.raises(ValueError):
        fn(*args)


class TestManifolds:
    @pytest.mark.parametrize(
        "space, n",
        [
            (geodex.Euclidean(3), 3),
            (geodex.Torus(3), 3),
            (geodex.SO2(), 1),
            (geodex.SphereN(3), 4),
        ],
    )
    def test_direct_methods_check_sizes(self, space, n):
        good = space.random_point()
        bad = np.zeros(n + 1)
        short = np.zeros(max(n - 1, 0))
        _raises(space.exp, good, bad)
        _raises(space.log, good, short)
        _raises(space.inner, bad, np.zeros(n), np.zeros(n))
        _raises(space.norm, good, bad)
        _raises(space.distance, short, good)
        _raises(space.geodesic, good, bad, 0.5)

    def test_euclidean_sampling_bounds(self):
        _raises(geodex.Euclidean(3).set_sampling_bounds, np.zeros(2), np.ones(2))

    def test_product_log_checks_sizes(self):
        space = geodex.Product([geodex.SE2(), geodex.Euclidean(1)])
        _raises(space.log, np.zeros(3), np.zeros(3))
        _raises(space.exp, np.zeros(4), np.zeros(5))
        assert space.log(np.zeros(4), np.zeros(4)).shape == (4,)

    def test_configuration_space_checks_sizes(self):
        space = geodex.ConfigurationSpace(
            geodex.Euclidean(2), geodex.KineticEnergyMetric(lambda q: np.eye(2))
        )
        _raises(space.inner, np.zeros(2), np.ones(3), np.ones(3))
        _raises(space.log, np.zeros(1), np.zeros(1))

    def test_algorithms_check_sizes(self):
        _raises(geodex.distance_midpoint, geodex.Torus(2), np.zeros(3), np.zeros(3))
        _raises(geodex.discrete_geodesic, geodex.Euclidean(2), np.zeros(2), np.zeros(5))


class TestMetrics:
    def test_kinetic_energy_checks_the_mass_matrix(self):
        metric = geodex.KineticEnergyMetric(lambda q: np.eye(3))
        _raises(metric.inner, np.zeros(2), np.ones(2), np.ones(2))
        _raises(metric.norm, np.zeros(3), np.ones(2))
        assert metric.inner(np.zeros(3), np.ones(3), np.ones(3)) == pytest.approx(3.0)

    def test_kinetic_energy_in_a_space_checks_the_mass_matrix(self):
        space = geodex.ConfigurationSpace(
            geodex.Euclidean(2), geodex.KineticEnergyMetric(lambda q: np.eye(3))
        )
        _raises(space.norm, np.zeros(2), np.ones(2))

    def test_jacobi_checks_the_mass_matrix(self):
        metric = geodex.JacobiMetric(lambda q: np.eye(3), lambda q: 0.0, 1.0)
        _raises(metric.inner, np.zeros(2), np.ones(2), np.ones(2))
        assert metric.inner(np.zeros(3), np.ones(3), np.ones(3)) == pytest.approx(6.0)

    def test_pullback_checks_its_matrices(self):
        metric = geodex.PullbackMetric(lambda q: np.ones((2, 3)), lambda q: np.eye(2))
        _raises(metric.inner, np.zeros(3), np.ones(2), np.ones(2))
        bad_task = geodex.PullbackMetric(lambda q: np.ones((2, 3)), lambda q: np.eye(3))
        _raises(bad_task.inner, np.zeros(3), np.ones(3), np.ones(3))

    def test_constant_spd_checks_sizes(self):
        _raises(geodex.ConstantSPDMetric, np.ones((2, 3)))
        metric = geodex.ConstantSPDMetric(np.eye(2))
        _raises(metric.inner, np.zeros(2), np.ones(3), np.ones(3))

    def test_se2_left_invariant_checks_sizes(self):
        metric = geodex.SE2LeftInvariantMetric(1.0, 2.0, 3.0)
        _raises(metric.inner, np.zeros(2), np.ones(3), np.ones(3))
        _raises(metric.norm, np.zeros(3), np.ones(4))


class TestHeuristics:
    def test_matrix_lower_bound_checks_sizes(self):
        h = geodex.heuristics.MatrixLowerBound(np.eye(3))
        _raises(h, np.zeros(2), np.zeros(2))
        _raises(h, np.zeros(3), np.zeros(4))
        _raises(h.update, np.eye(2))
        _raises(geodex.heuristics.MatrixLowerBound, np.ones((2, 3)))
        assert h(np.zeros(3), np.ones(3)) == pytest.approx(np.sqrt(3.0))

    def test_chord_heuristics_check_sizes(self):
        _raises(geodex.heuristics.Euclidean(), np.zeros(2), np.zeros(3))
        _raises(geodex.heuristics.EigenvalueLowerBound(1.0), np.zeros(2), np.zeros(3))

    def test_precompute_checks_the_metric(self):
        _raises(
            geodex.precompute_matrix_lower_bound,
            lambda q: np.eye(3),
            np.zeros(2),
            np.ones(2),
        )

    @pytest.mark.skipif(
        not hasattr(geodex, "robots") or not hasattr(geodex.robots, "RidgebackUR5e"),
        reason="robots not built",
    )
    def test_robot_heuristic_checks_sizes(self):
        h = geodex.robots.RidgebackUR5e().heuristic()
        for _ in range(1000):
            _raises(h, np.zeros(1), np.full(1, 3.0))


class TestCollision:
    def test_sdfs_check_the_query(self):
        _raises(geodex.collision.CircleSDF(0.0, 0.0, 1.0), np.zeros(1))
        grid = geodex.collision.DistanceGrid(4, 4, 0.1, [1.0] * 16)
        _raises(geodex.collision.GridSDF(grid), np.zeros(1))
        fp = geodex.collision.PolygonFootprint.rectangle(0.1, 0.1, 2)
        checker = geodex.collision.FootprintGridChecker(grid, fp)
        _raises(checker, np.zeros(2))
        _raises(geodex.collision.MemoizedSDF(checker), np.zeros(2))

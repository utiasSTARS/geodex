"""Tests for algorithm bindings: distance_midpoint."""

import numpy as np
import pytest

import geodex


# ---------------------------------------------------------------------------
# distance_midpoint
# ---------------------------------------------------------------------------


class TestDistanceMidpointSphere:
    def setup_method(self):
        self.sphere = geodex.Sphere()

    def test_zero_distance(self):
        p = np.array([0.0, 0.0, 1.0])
        d = geodex.distance_midpoint(self.sphere, p, p)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_agrees_with_sphere_distance(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            p = self.sphere.random_point()
            q = self.sphere.random_point()
            d_mid = geodex.distance_midpoint(self.sphere, p, q)
            d_exact = self.sphere.distance(p, q)
            assert d_mid == pytest.approx(d_exact, abs=1e-8)

    def test_symmetry(self):
        p = self.sphere.random_point()
        q = self.sphere.random_point()
        d1 = geodex.distance_midpoint(self.sphere, p, q)
        d2 = geodex.distance_midpoint(self.sphere, q, p)
        assert d1 == pytest.approx(d2, abs=1e-10)


class TestDistanceMidpointEuclidean:
    def setup_method(self):
        self.euc = geodex.Euclidean(4)

    def test_zero_distance(self):
        p = np.ones(4)
        assert geodex.distance_midpoint(self.euc, p, p) == pytest.approx(0.0, abs=1e-10)

    def test_agrees_with_euclidean_distance(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            p = rng.standard_normal(4)
            q = rng.standard_normal(4)
            d_mid = geodex.distance_midpoint(self.euc, p, q)
            d_exact = np.linalg.norm(p - q)
            assert d_mid == pytest.approx(d_exact, abs=1e-8)


class TestDistanceMidpointTorus:
    def setup_method(self):
        self.torus = geodex.Torus(3)

    def test_zero_distance(self):
        p = self.torus.random_point()
        assert geodex.distance_midpoint(self.torus, p, p) == pytest.approx(0.0, abs=1e-10)

    def test_agrees_with_torus_distance(self):
        rng = np.random.default_rng(2)
        for _ in range(5):
            p = self.torus.random_point()
            q = self.torus.random_point()
            d_mid = geodex.distance_midpoint(self.torus, p, q)
            d_exact = self.torus.distance(p, q)
            assert d_mid == pytest.approx(d_exact, abs=1e-8)


class TestDistanceMidpointSE2:
    def setup_method(self):
        self.se2 = geodex.SE2()

    def test_zero_distance(self):
        p = self.se2.random_point()
        assert geodex.distance_midpoint(self.se2, p, p) == pytest.approx(0.0, abs=1e-10)

    def test_agrees_with_se2_distance(self):
        for _ in range(5):
            p = self.se2.random_point()
            q = self.se2.random_point()
            d_mid = geodex.distance_midpoint(self.se2, p, q)
            d_exact = self.se2.distance(p, q)
            assert d_mid == pytest.approx(d_exact, abs=1e-8)


class TestDistanceMidpointConfigSpace:
    def test_with_config_space(self):
        torus = geodex.Torus(2)
        metric = geodex.KineticEnergyMetric(lambda q: np.eye(2))
        cs = geodex.ConfigurationSpace(torus, metric)
        p = np.array([1.0, 2.0])
        q = np.array([1.5, 2.5])
        d = geodex.distance_midpoint(cs, p, q)
        assert isinstance(d, float)
        assert d > 0.0

    def test_invalid_manifold_raises(self):
        with pytest.raises(Exception):
            geodex.distance_midpoint("not_a_manifold", np.zeros(3), np.ones(3))

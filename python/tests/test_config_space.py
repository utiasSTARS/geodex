import numpy as np
import pytest

import geodex


class TestConfigurationSpaceTorusIdentity:
    """ConfigurationSpace with Torus base and identity metric should behave like Torus."""

    def setup_method(self):
        self.torus = geodex.Torus(3)
        self.metric = geodex.KineticEnergyMetric(lambda q: np.eye(3))
        self.cs = geodex.ConfigurationSpace(self.torus, self.metric)

    def test_dim(self):
        assert self.cs.dim() == 3

    def test_repr(self):
        r = repr(self.cs)
        assert "ConfigurationSpace" in r
        assert "Torus" in r

    def test_random_point_in_range(self):
        for _ in range(10):
            p = self.cs.random_point()
            assert p.shape == (3,)
            assert np.all(p >= 0.0)
            assert np.all(p < 2 * np.pi)

    def test_exp_from_base(self):
        p = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, 0.5, 0.5])
        q_cs = self.cs.exp(p, v)
        q_torus = self.torus.exp(p, v)
        np.testing.assert_allclose(q_cs, q_torus, atol=1e-12)

    def test_log_from_base(self):
        p = np.array([1.0, 2.0, 3.0])
        q = np.array([2.0, 3.0, 4.0])
        v_cs = self.cs.log(p, q)
        v_torus = self.torus.log(p, q)
        np.testing.assert_allclose(v_cs, v_torus, atol=1e-12)

    def test_inner_from_metric(self):
        p = np.array([1.0, 2.0, 3.0])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert self.cs.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)
        assert self.cs.inner(p, u, u) == pytest.approx(1.0, abs=1e-12)

    def test_norm_from_metric(self):
        p = np.array([1.0, 2.0, 3.0])
        v = np.array([3.0, 4.0, 0.0])
        assert self.cs.norm(p, v) == pytest.approx(5.0, abs=1e-12)

    def test_distance_symmetry(self):
        p = self.cs.random_point()
        q = self.cs.random_point()
        d1 = self.cs.distance(p, q)
        d2 = self.cs.distance(q, p)
        assert d1 == pytest.approx(d2, abs=1e-10)

    def test_distance_self_is_zero(self):
        p = self.cs.random_point()
        assert self.cs.distance(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_geodesic_endpoints(self):
        p = np.array([1.0, 2.0, 3.0])
        q = np.array([2.0, 3.0, 4.0])
        start = self.cs.geodesic(p, q, 0.0)
        end = self.cs.geodesic(p, q, 1.0)
        np.testing.assert_allclose(start, p, atol=1e-12)
        # end should be close to q modulo 2pi
        diff = np.abs(end - q)
        diff = np.minimum(diff, 2 * np.pi - diff)
        np.testing.assert_allclose(diff, 0.0, atol=1e-10)


class TestConfigurationSpaceWeightedMetric:
    """ConfigurationSpace with non-trivial constant SPD metric."""

    def setup_method(self):
        self.torus = geodex.Torus(2)
        A = np.diag([4.0, 1.0])
        self.metric = geodex.ConstantSPDMetric(A)
        self.cs = geodex.ConfigurationSpace(self.torus, self.metric)

    def test_inner_uses_custom_metric(self):
        p = np.array([1.0, 2.0])
        u = np.array([1.0, 0.0])
        # inner = u^T A u = 4
        assert self.cs.inner(p, u, u) == pytest.approx(4.0, abs=1e-12)
        v = np.array([0.0, 1.0])
        # inner = v^T A v = 1
        assert self.cs.inner(p, v, v) == pytest.approx(1.0, abs=1e-12)

    def test_norm_uses_custom_metric(self):
        p = np.array([0.0, 0.0])
        v = np.array([1.0, 1.0])
        # norm = sqrt(4 + 1) = sqrt(5)
        assert self.cs.norm(p, v) == pytest.approx(np.sqrt(5.0), abs=1e-12)

    def test_distance_anisotropic(self):
        # Points differing only in first coordinate (heavy direction).
        p = np.array([0.0, 0.0])
        q1 = np.array([0.1, 0.0])
        q2 = np.array([0.0, 0.1])
        d1 = self.cs.distance(p, q1)
        d2 = self.cs.distance(p, q2)
        # First direction is 4x heavier, so d1 should be ~2x d2
        assert d1 > d2


class TestConfigurationSpaceEuclidean:
    """ConfigurationSpace with Euclidean base and kinetic energy metric."""

    def setup_method(self):
        self.euclidean = geodex.Euclidean(3)
        self.metric = geodex.KineticEnergyMetric(lambda q: np.eye(3))
        self.cs = geodex.ConfigurationSpace(self.euclidean, self.metric)

    def test_dim(self):
        assert self.cs.dim() == 3

    def test_distance_matches_euclidean(self):
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0])
        d_cs = self.cs.distance(p, q)
        d_euc = self.euclidean.distance(p, q)
        assert d_cs == pytest.approx(d_euc, abs=1e-6)


class TestConfigurationSpaceSphere:
    """ConfigurationSpace with Sphere base."""

    def setup_method(self):
        self.sphere = geodex.Sphere()
        self.metric = geodex.ConstantSPDMetric(np.eye(3))
        self.cs = geodex.ConfigurationSpace(self.sphere, self.metric)

    def test_dim(self):
        assert self.cs.dim() == 2

    def test_exp_stays_on_sphere(self):
        p = np.array([0.0, 0.0, 1.0])
        v = np.array([0.3, 0.4, 0.0])
        q = self.cs.exp(p, v)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10


class TestConfigurationSpaceInvalidArgs:
    def test_invalid_base_raises(self):
        metric = geodex.KineticEnergyMetric(lambda q: np.eye(2))
        with pytest.raises(Exception):
            geodex.ConfigurationSpace("not_a_manifold", metric)

    def test_invalid_metric_raises(self):
        torus = geodex.Torus(2)
        with pytest.raises(Exception):
            geodex.ConfigurationSpace(torus, "not_a_metric")

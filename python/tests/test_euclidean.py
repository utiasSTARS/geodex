import numpy as np
import pytest

import geodex


class TestEuclidean:
    def setup_method(self):
        self.e3 = geodex.Euclidean(3)
        self.e5 = geodex.Euclidean(5)

    def test_dim(self):
        assert self.e3.dim() == 3
        assert self.e5.dim() == 5

    def test_repr(self):
        assert "3" in repr(self.e3)
        assert "5" in repr(self.e5)

    def test_random_point_shape(self):
        p = self.e3.random_point()
        assert p.shape == (3,)
        p = self.e5.random_point()
        assert p.shape == (5,)

    def test_distance_known_value(self):
        origin = np.zeros(3)
        p = np.array([3.0, 4.0, 0.0])
        assert self.e3.distance(origin, p) == pytest.approx(5.0, abs=1e-10)

    def test_distance_symmetry(self):
        p = self.e5.random_point()
        q = self.e5.random_point()
        assert self.e5.distance(p, q) == pytest.approx(
            self.e5.distance(q, p), abs=1e-10
        )

    def test_distance_self_is_zero(self):
        p = self.e5.random_point()
        assert self.e5.distance(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_exp_is_addition(self):
        p = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(self.e3.exp(p, v), p + v, atol=1e-12)

    def test_log_is_subtraction(self):
        p = np.array([1.0, 2.0, 3.0])
        q = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(self.e3.log(p, q), q - p, atol=1e-12)

    def test_exp_log_roundtrip(self):
        p = self.e5.random_point()
        q = self.e5.random_point()
        v = self.e5.log(p, q)
        r = self.e5.exp(p, v)
        np.testing.assert_allclose(r, q, atol=1e-10)

    def test_geodesic_endpoints(self):
        p = np.zeros(3)
        q = np.ones(3)
        start = self.e3.geodesic(p, q, 0.0)
        end = self.e3.geodesic(p, q, 1.0)
        np.testing.assert_allclose(start, p, atol=1e-12)
        np.testing.assert_allclose(end, q, atol=1e-12)

    def test_geodesic_midpoint(self):
        p = np.zeros(3)
        q = np.array([2.0, 4.0, 6.0])
        mid = self.e3.geodesic(p, q, 0.5)
        np.testing.assert_allclose(mid, np.array([1.0, 2.0, 3.0]), atol=1e-12)

    def test_inner_product(self):
        p = np.zeros(3)
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert self.e3.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)
        assert self.e3.inner(p, u, u) == pytest.approx(1.0, abs=1e-12)

    def test_norm(self):
        p = np.zeros(3)
        v = np.array([3.0, 4.0, 0.0])
        assert self.e3.norm(p, v) == pytest.approx(5.0, abs=1e-12)

    def test_high_dimension(self):
        e100 = geodex.Euclidean(100)
        assert e100.dim() == 100
        p = e100.random_point()
        assert p.shape == (100,)

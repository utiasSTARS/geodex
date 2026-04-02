import numpy as np
import pytest

import geodex


class TestTorus:
    def setup_method(self):
        self.t2 = geodex.Torus(2)
        self.t3 = geodex.Torus(3)

    def test_dim(self):
        assert self.t2.dim() == 2
        assert self.t3.dim() == 3

    def test_repr(self):
        assert "2" in repr(self.t2)

    def test_random_point_in_range(self):
        for _ in range(20):
            p = self.t3.random_point()
            assert p.shape == (3,)
            assert np.all(p >= 0.0)
            assert np.all(p < 2 * np.pi)

    def test_exp_wraps_to_2pi(self):
        p = np.array([5.0, 5.0])
        v = np.array([2.0, 2.0])
        q = self.t2.exp(p, v)
        # (5+2) mod 2pi should be < 2pi
        assert np.all(q >= 0.0)
        assert np.all(q < 2 * np.pi + 1e-10)

    def test_log_wraps_to_pi(self):
        p = np.array([0.1, 0.1])
        q = np.array([2 * np.pi - 0.1, 2 * np.pi - 0.1])
        v = self.t2.log(p, q)
        # shortest path should be negative (wrapping backward)
        assert np.all(np.abs(v) <= np.pi + 1e-10)

    def test_exp_log_roundtrip(self):
        p = self.t3.random_point()
        q = self.t3.random_point()
        v = self.t3.log(p, q)
        r = self.t3.exp(p, v)
        # r should be close to q (modulo 2pi)
        diff = np.abs(r - q)
        diff = np.minimum(diff, 2 * np.pi - diff)
        np.testing.assert_allclose(diff, 0.0, atol=1e-10)

    def test_distance_symmetry(self):
        p = self.t3.random_point()
        q = self.t3.random_point()
        assert self.t3.distance(p, q) == pytest.approx(
            self.t3.distance(q, p), abs=1e-10
        )

    def test_distance_self_is_zero(self):
        p = self.t3.random_point()
        assert self.t3.distance(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_distance_across_boundary(self):
        # Two points near the 0/2pi boundary should be close
        p = np.array([0.1])
        q = np.array([2 * np.pi - 0.1])
        t1 = geodex.Torus(1)
        d = t1.distance(p, q)
        assert d == pytest.approx(0.2, abs=1e-6)

    def test_geodesic_endpoints(self):
        p = self.t2.random_point()
        q = self.t2.random_point()
        start = self.t2.geodesic(p, q, 0.0)
        end = self.t2.geodesic(p, q, 1.0)
        # start should be p (mod 2pi)
        diff_start = np.abs(start - p)
        diff_start = np.minimum(diff_start, 2 * np.pi - diff_start)
        np.testing.assert_allclose(diff_start, 0.0, atol=1e-10)
        diff_end = np.abs(end - q)
        diff_end = np.minimum(diff_end, 2 * np.pi - diff_end)
        np.testing.assert_allclose(diff_end, 0.0, atol=1e-10)

    def test_inner_product(self):
        p = np.array([1.0, 2.0])
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert self.t2.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)

    def test_norm(self):
        p = np.array([1.0, 2.0])
        v = np.array([3.0, 4.0])
        assert self.t2.norm(p, v) == pytest.approx(5.0, abs=1e-12)

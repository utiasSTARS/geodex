import numpy as np
import pytest

import geodex


class TestSO2:
    def setup_method(self):
        self.so2 = geodex.SO2()

    def test_dim(self):
        assert self.so2.dim() == 1

    def test_random_point_shape(self):
        p = self.so2.random_point()
        assert p.shape == (1,)
        assert -np.pi <= p[0] <= np.pi

    def test_exp_log_roundtrip(self):
        p = np.array([0.5])
        q = np.array([2.0])
        v = self.so2.log(p, q)
        r = self.so2.exp(p, v)
        d = abs(r[0] - q[0])
        d = min(d, 2 * np.pi - d)
        assert d < 1e-9

    def test_shortest_arc(self):
        # 3.0 -> -3.0 the short way wraps through +/- pi (arc length ~0.283), not through 0.
        p = np.array([3.0])
        q = np.array([-3.0])
        assert self.so2.distance(p, q) == pytest.approx(2 * np.pi - 6.0, abs=1e-9)
        mid = self.so2.geodesic(p, q, 0.5)
        assert abs(abs(mid[0]) - np.pi) < 0.2

    def test_distance_symmetry(self):
        p = self.so2.random_point()
        q = self.so2.random_point()
        assert self.so2.distance(p, q) == pytest.approx(self.so2.distance(q, p), abs=1e-10)

    def test_weight_scales_distance(self):
        p = np.array([0.0])
        q = np.array([1.0])
        d1 = geodex.SO2(weight=1.0).distance(p, q)
        d4 = geodex.SO2(weight=4.0).distance(p, q)
        assert d4 == pytest.approx(2.0 * d1, rel=1e-9)

    def test_discrete_geodesic(self):
        p = np.array([0.2])
        q = np.array([2.5])
        r = geodex.discrete_geodesic(self.so2, p, q)
        assert r.final_distance < 1e-2
        assert isinstance(r.path, np.ndarray)

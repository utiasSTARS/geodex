import numpy as np
import pytest

import geodex


class TestSO3:
    def setup_method(self):
        self.so3 = geodex.SO3()

    def test_dim(self):
        assert self.so3.dim() == 3

    def test_random_point_unit(self):
        for _ in range(50):
            q = self.so3.random_point()
            assert q.shape == (4,)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-9

    def test_exp_log_roundtrip(self):
        q0 = self.so3.random_point()
        q1 = self.so3.random_point()
        v = self.so3.log(q0, q1)
        assert v.shape == (3,)
        assert self.so3.distance(self.so3.exp(q0, v), q1) < 1e-8

    def test_geodesic_endpoints(self):
        q0 = self.so3.random_point()
        q1 = self.so3.random_point()
        assert self.so3.distance(self.so3.geodesic(q0, q1, 0.0), q0) < 1e-9
        assert self.so3.distance(self.so3.geodesic(q0, q1, 1.0), q1) < 1e-8

    def test_geodesic_constant_speed(self):
        # SLERP is constant speed: the midpoint is equidistant from both ends.
        q0 = self.so3.random_point()
        q1 = self.so3.random_point()
        d = self.so3.distance(q0, q1)
        mid = self.so3.geodesic(q0, q1, 0.5)
        assert self.so3.distance(q0, mid) == pytest.approx(d / 2, abs=1e-7)
        assert self.so3.distance(mid, q1) == pytest.approx(d / 2, abs=1e-7)

    def test_distance_bounded_by_pi(self):
        for _ in range(20):
            q0 = self.so3.random_point()
            q1 = self.so3.random_point()
            assert self.so3.distance(q0, q1) <= np.pi + 1e-9

    def test_double_cover(self):
        # q and -q represent the same rotation, so their distance is 0.
        q = self.so3.random_point()
        assert self.so3.distance(q, -q) < 1e-8

    def test_frames_roundtrip(self):
        for frame in ("body", "world"):
            so3 = geodex.SO3(frame=frame)
            q0 = so3.random_point()
            q1 = so3.random_point()
            assert so3.distance(so3.exp(q0, so3.log(q0, q1)), q1) < 1e-8

    def test_invalid_frame_raises(self):
        with pytest.raises(Exception):
            geodex.SO3(frame="invalid")

    def test_discrete_geodesic(self):
        q0 = self.so3.random_point()
        q1 = self.so3.random_point()
        r = geodex.discrete_geodesic(self.so3, q0, q1)
        assert r.final_distance < 1e-2
        assert isinstance(r.path, np.ndarray)
        assert r.path.shape[1] == 4

import numpy as np
import pytest

import geodex


class TestSE3:
    def setup_method(self):
        self.se3 = geodex.SE3()

    def test_dim(self):
        assert self.se3.dim() == 6

    def test_random_point(self):
        g = self.se3.random_point()
        assert g.shape == (7,)
        assert abs(np.linalg.norm(g[3:7]) - 1.0) < 1e-9  # quaternion part is a unit quaternion

    def test_exp_log_roundtrip(self):
        g0 = self.se3.random_point()
        g1 = self.se3.random_point()
        xi = self.se3.log(g0, g1)
        assert xi.shape == (6,)
        assert self.se3.distance(self.se3.exp(g0, xi), g1) < 1e-7

    def test_geodesic_endpoints(self):
        g0 = self.se3.random_point()
        g1 = self.se3.random_point()
        assert self.se3.distance(self.se3.geodesic(g0, g1, 0.0), g0) < 1e-9
        assert self.se3.distance(self.se3.geodesic(g0, g1, 1.0), g1) < 1e-7

    def test_pure_translation_keeps_orientation(self):
        g0 = self.se3.random_point()
        xi = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        g1 = self.se3.exp(g0, xi)
        np.testing.assert_allclose(g1[3:7], g0[3:7], atol=1e-10)

    def test_body_vs_world_differ(self):
        # Body- and world-frame distances differ for a general pose pair.
        g0 = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        g1 = np.array([4.0, 1.0, 2.0, 0.0, 0.0, 0.70710678, 0.70710678])
        db = geodex.SE3(frame="body").distance(g0, g1)
        dw = geodex.SE3(frame="world").distance(g0, g1)
        assert abs(db - dw) > 1e-3

    def test_translation_weight_scales_distance(self):
        g0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        g1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # pure translation
        d1 = geodex.SE3(w_trans=1.0).distance(g0, g1)
        d4 = geodex.SE3(w_trans=4.0).distance(g0, g1)
        assert d4 == pytest.approx(2.0 * d1, rel=1e-9)

    def test_invalid_frame_raises(self):
        with pytest.raises(Exception):
            geodex.SE3(frame="invalid")

    def test_discrete_geodesic(self):
        g0 = self.se3.random_point()
        g1 = self.se3.random_point()
        r = geodex.discrete_geodesic(self.se3, g0, g1)
        assert r.final_distance < 1e-2
        assert isinstance(r.path, np.ndarray)
        assert r.path.shape[1] == 7

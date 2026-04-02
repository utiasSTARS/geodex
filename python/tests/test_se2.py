import numpy as np
import pytest

import geodex


class TestSE2Exponential:
    """Test SE2 with default exponential map."""

    def setup_method(self):
        self.se2 = geodex.SE2()

    def test_dim(self):
        assert self.se2.dim() == 3

    def test_repr(self):
        assert "exponential" in repr(self.se2)

    def test_random_point_shape(self):
        p = self.se2.random_point()
        assert p.shape == (3,)

    def test_random_point_bounds(self):
        for _ in range(20):
            p = self.se2.random_point()
            assert 0.0 <= p[0] <= 10.0
            assert 0.0 <= p[1] <= 10.0
            assert -np.pi <= p[2] <= np.pi

    def test_distance_symmetry(self):
        p = self.se2.random_point()
        q = self.se2.random_point()
        assert self.se2.distance(p, q) == pytest.approx(
            self.se2.distance(q, p), abs=1e-10
        )

    def test_distance_self_is_zero(self):
        p = self.se2.random_point()
        assert self.se2.distance(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_exp_log_roundtrip(self):
        p = np.array([1.0, 2.0, 0.5])
        q = np.array([3.0, 4.0, 1.0])
        v = self.se2.log(p, q)
        r = self.se2.exp(p, v)
        np.testing.assert_allclose(r[:2], q[:2], atol=1e-8)
        # angle comparison (mod 2pi)
        angle_diff = abs(r[2] - q[2])
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        assert angle_diff < 1e-8

    def test_geodesic_endpoints(self):
        p = np.array([1.0, 2.0, 0.5])
        q = np.array([3.0, 4.0, 1.0])
        start = self.se2.geodesic(p, q, 0.0)
        end = self.se2.geodesic(p, q, 1.0)
        np.testing.assert_allclose(start, p, atol=1e-10)
        np.testing.assert_allclose(end[:2], q[:2], atol=1e-8)

    def test_identity_exp(self):
        p = np.array([1.0, 2.0, 0.0])
        v = np.array([0.0, 0.0, 0.0])
        r = self.se2.exp(p, v)
        np.testing.assert_allclose(r, p, atol=1e-10)


class TestSE2Weights:
    """Test that metric weights affect distances."""

    def test_weights_affect_distance(self):
        se2_iso = geodex.SE2(wx=1.0, wy=1.0, wtheta=1.0)
        se2_aniso = geodex.SE2(wx=1.0, wy=1.0, wtheta=100.0)

        p = np.array([0.0, 0.0, 0.0])
        # Pure rotation
        q = np.array([0.0, 0.0, 1.0])

        d_iso = se2_iso.distance(p, q)
        d_aniso = se2_aniso.distance(p, q)
        # Higher weight on theta should increase distance for rotation
        assert d_aniso > d_iso


class TestSE2Retractions:
    """Test retraction policies."""

    def test_euler_basic(self):
        se2 = geodex.SE2(retraction="euler")
        assert "euler" in repr(se2)
        p = np.array([1.0, 2.0, 0.0])
        v = np.array([0.5, 0.3, 0.1])
        q = se2.exp(p, v)
        assert q.shape == (3,)

    def test_invalid_retraction_raises(self):
        with pytest.raises(Exception):
            geodex.SE2(retraction="invalid")

    def test_euler_agrees_at_identity_orientation(self):
        # Euler retraction ignores group structure (no rotation of v),
        # so it only agrees with exp when theta=0
        p = np.array([1.0, 2.0, 0.0])
        v = np.array([0.01, 0.01, 0.01])

        exp_result = geodex.SE2(retraction="exponential").exp(p, v)
        eul_result = geodex.SE2(retraction="euler").exp(p, v)

        np.testing.assert_allclose(exp_result, eul_result, atol=1e-3)


class TestSE2CustomBounds:
    def test_custom_workspace(self):
        se2 = geodex.SE2(x_lo=-5.0, x_hi=5.0, y_lo=-5.0, y_hi=5.0)
        for _ in range(20):
            p = se2.random_point()
            assert -5.0 <= p[0] <= 5.0
            assert -5.0 <= p[1] <= 5.0

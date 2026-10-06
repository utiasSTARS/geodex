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


class TestSE2Frames:
    """Test body- and world-frame (left- and right-invariant) metrics."""

    def test_world_frame_repr(self):
        assert "world" in repr(geodex.SE2(frame="world"))

    def test_body_vs_world_differ(self):
        # SE(2) has no bi-invariant metric, so the two frames disagree for a
        # general pose pair.
        p = np.array([1.0, 2.0, 0.0])
        q = np.array([3.0, 4.0, 1.57])
        db = geodex.SE2(frame="body").distance(p, q)
        dw = geodex.SE2(frame="world").distance(p, q)
        assert abs(db - dw) > 1e-3

    def test_body_is_default(self):
        p = np.array([1.0, 2.0, 0.3])
        q = np.array([3.0, 4.0, 1.2])
        default = geodex.SE2().distance(p, q)
        body = geodex.SE2(frame="body").distance(p, q)
        assert default == pytest.approx(body, abs=1e-12)

    def test_invalid_frame_raises(self):
        with pytest.raises(Exception):
            geodex.SE2(frame="invalid")

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


class TestSE2CoordinateMetricAndPeriods:
    TWO_PI = 2.0 * np.pi

    def test_periods_are_theta_only(self):
        se2 = geodex.SE2()
        np.testing.assert_allclose(se2.periods(), [0.0, 0.0, self.TWO_PI])

    def test_coordinate_metric_matches_the_body_metric_at_zero(self):
        se2 = geodex.SE2(wx=1.0, wy=20.0, wtheta=2.25)
        G = se2.coordinate_metric(np.array([3.0, 4.0, 0.0]))
        np.testing.assert_allclose(G, np.diag([1.0, 20.0, 2.25]), atol=1e-12)

    def test_coordinate_metric_rotates_with_theta(self):
        # At theta = pi/2 the coordinate frame has swapped, so the weights swap.
        se2 = geodex.SE2(wx=1.0, wy=20.0, wtheta=2.25)
        G = se2.coordinate_metric(np.array([0.0, 0.0, np.pi / 2.0]))
        np.testing.assert_allclose(G, np.diag([20.0, 1.0, 2.25]), atol=1e-10)

    def test_coordinate_metric_is_the_frame_pullback(self):
        se2 = geodex.SE2(wx=1.0, wy=20.0, wtheta=2.25)
        M = np.diag([1.0, 20.0, 2.25])
        for theta in np.linspace(-np.pi, np.pi, 17):
            c, s = np.cos(theta), np.sin(theta)
            J = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            expected = J.T @ M @ J
            G = se2.coordinate_metric(np.array([1.0, 2.0, theta]))
            np.testing.assert_allclose(G, expected, atol=1e-12)

    def test_matrix_lower_bound_meets_to_the_isotropic_block(self):
        se2 = geodex.SE2.car_like(turning_radius=1.5, lateral_penalty=20.0)
        h = se2.matrix_lower_bound()
        np.testing.assert_allclose(h.matrix(), np.diag([1.0, 1.0, 2.25]), atol=1e-6)
        np.testing.assert_allclose(h.periods, [0.0, 0.0, self.TWO_PI])

    def test_derived_bound_never_exceeds_the_planner_distance(self):
        se2 = geodex.SE2.car_like(turning_radius=1.5, lateral_penalty=20.0,
                                  x_hi=30.0, y_hi=12.0)
        h = se2.matrix_lower_bound()
        rng = np.random.default_rng(31337)
        for i in range(600):
            at_cut = i % 4 == 0
            ta = rng.uniform(np.pi - 0.4, np.pi) if at_cut else rng.uniform(-np.pi, np.pi)
            tb = -rng.uniform(np.pi - 0.4, np.pi) if at_cut else rng.uniform(-np.pi, np.pi)
            a = np.array([rng.uniform(0, 30), rng.uniform(0, 12), ta])
            b = np.array([rng.uniform(0, 30), rng.uniform(0, 12), tb])
            assert h(a, b) <= se2.distance(a, b) * (1.0 + 1e-9)

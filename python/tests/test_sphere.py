import numpy as np
import pytest

import geodex


class TestSphereExponential:
    """Test Sphere with the default round metric and exponential map."""

    def setup_method(self):
        self.sphere = geodex.Sphere()
        self.north = np.array([0.0, 0.0, 1.0])
        self.east = np.array([1.0, 0.0, 0.0])

    def test_dim(self):
        assert self.sphere.dim() == 2

    def test_repr(self):
        assert "exponential" in repr(self.sphere)

    def test_random_point_on_sphere(self):
        for _ in range(10):
            p = self.sphere.random_point()
            assert p.shape == (3,)
            assert abs(np.linalg.norm(p) - 1.0) < 1e-12

    def test_distance_known_value(self):
        d = self.sphere.distance(self.north, self.east)
        assert d == pytest.approx(np.pi / 2, abs=1e-6)

    def test_distance_symmetry(self):
        p = self.sphere.random_point()
        q = self.sphere.random_point()
        assert self.sphere.distance(p, q) == pytest.approx(
            self.sphere.distance(q, p), abs=1e-10
        )

    def test_distance_self_is_zero(self):
        p = self.sphere.random_point()
        assert self.sphere.distance(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_exp_log_roundtrip(self):
        p = self.north
        q = self.east
        v = self.sphere.log(p, q)
        r = self.sphere.exp(p, v)
        np.testing.assert_allclose(r, q, atol=1e-10)

    def test_log_exp_roundtrip(self):
        p = self.north
        v = np.array([0.3, 0.4, 0.0])  # tangent at north pole
        q = self.sphere.exp(p, v)
        v_back = self.sphere.log(p, q)
        np.testing.assert_allclose(v_back, v, atol=1e-10)

    def test_geodesic_endpoints(self):
        p = self.north
        q = self.east
        start = self.sphere.geodesic(p, q, 0.0)
        end = self.sphere.geodesic(p, q, 1.0)
        np.testing.assert_allclose(start, p, atol=1e-10)
        np.testing.assert_allclose(end, q, atol=1e-10)

    def test_geodesic_midpoint_on_sphere(self):
        p = self.north
        q = self.east
        mid = self.sphere.geodesic(p, q, 0.5)
        assert abs(np.linalg.norm(mid) - 1.0) < 1e-10

    def test_inner_product(self):
        p = self.north
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert self.sphere.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)
        assert self.sphere.inner(p, u, u) == pytest.approx(1.0, abs=1e-12)

    def test_norm(self):
        p = self.north
        v = np.array([3.0, 4.0, 0.0])
        assert self.sphere.norm(p, v) == pytest.approx(5.0, abs=1e-12)

    def test_project(self):
        p = self.north
        v = np.array([1.0, 2.0, 3.0])
        projected = self.sphere.project(p, v)
        # projected should be orthogonal to p
        assert abs(np.dot(projected, p)) < 1e-12
        # tangential components preserved
        np.testing.assert_allclose(projected[:2], v[:2], atol=1e-12)


class TestSphereProjection:
    """Test Sphere with projection retraction."""

    def setup_method(self):
        self.sphere = geodex.Sphere(retraction="projection")
        self.north = np.array([0.0, 0.0, 1.0])

    def test_repr(self):
        assert "projection" in repr(self.sphere)

    def test_exp_stays_on_sphere(self):
        p = self.north
        v = np.array([0.5, 0.3, 0.0])
        q = self.sphere.exp(p, v)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-12

    def test_distance_positive(self):
        p = self.sphere.random_point()
        q = self.sphere.random_point()
        d = self.sphere.distance(p, q)
        assert d >= 0.0

    def test_distance_self_is_zero(self):
        p = self.sphere.random_point()
        assert self.sphere.distance(p, p) == pytest.approx(0.0, abs=1e-10)


class TestSphereInvalidRetraction:
    def test_invalid_retraction_raises(self):
        with pytest.raises(Exception):
            geodex.Sphere(retraction="invalid")

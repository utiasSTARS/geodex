"""Tests for M5 algorithm bindings: InterpolationSettings, distance_midpoint,
discrete_geodesic, EuclideanHeuristic."""

import numpy as np
import pytest

import geodex


# ---------------------------------------------------------------------------
# InterpolationSettings
# ---------------------------------------------------------------------------


class TestInterpolationSettings:
    def test_default_values(self):
        s = geodex.InterpolationSettings()
        assert s.step_size == pytest.approx(0.5)
        assert s.convergence_tol == pytest.approx(1e-4)
        assert s.convergence_rel == pytest.approx(1e-3)
        assert s.max_steps == 100
        assert s.fd_epsilon == pytest.approx(0.0)
        assert s.distortion_ratio == pytest.approx(1.5)
        assert s.growth_factor == pytest.approx(1.5)

    def test_keyword_construction(self):
        s = geodex.InterpolationSettings(step_size=0.1, max_steps=50)
        assert s.step_size == pytest.approx(0.1)
        assert s.max_steps == 50
        # Other fields keep defaults
        assert s.convergence_tol == pytest.approx(1e-4)

    def test_field_mutation(self):
        s = geodex.InterpolationSettings()
        s.step_size = 0.25
        s.max_steps = 200
        assert s.step_size == pytest.approx(0.25)
        assert s.max_steps == 200

    def test_repr(self):
        s = geodex.InterpolationSettings()
        r = repr(s)
        assert "InterpolationSettings" in r
        assert "step_size" in r


# ---------------------------------------------------------------------------
# EuclideanHeuristic
# ---------------------------------------------------------------------------


class TestEuclideanHeuristic:
    def test_zero_distance(self):
        h = geodex.EuclideanHeuristic()
        a = np.array([1.0, 2.0, 3.0])
        assert h(a, a) == pytest.approx(0.0)

    def test_known_distance(self):
        h = geodex.EuclideanHeuristic()
        a = np.zeros(3)
        b = np.array([3.0, 4.0, 0.0])
        assert h(a, b) == pytest.approx(5.0)

    def test_symmetry(self):
        h = geodex.EuclideanHeuristic()
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert h(a, b) == pytest.approx(h(b, a))

    def test_agrees_with_numpy(self):
        h = geodex.EuclideanHeuristic()
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.standard_normal(5)
            b = rng.standard_normal(5)
            assert h(a, b) == pytest.approx(np.linalg.norm(a - b))


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


# ---------------------------------------------------------------------------
# discrete_geodesic
# ---------------------------------------------------------------------------


class TestDiscreteGeodesicSphere:
    def setup_method(self):
        self.sphere = geodex.Sphere()
        self.settings = geodex.InterpolationSettings(step_size=0.3, max_steps=200)

    def test_returns_result_with_list_of_arrays(self):
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([0.0, 1.0, 0.0])
        r = geodex.discrete_geodesic(self.sphere, p, q, self.settings)
        assert isinstance(r, geodex.InterpolationResult)
        assert isinstance(r.path, list)
        assert len(r.path) >= 2
        assert all(isinstance(pt, np.ndarray) for pt in r.path)

    def test_first_point_is_start(self):
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([0.0, 1.0, 0.0])
        r = geodex.discrete_geodesic(self.sphere, p, q, self.settings)
        np.testing.assert_allclose(r.path[0], p, atol=1e-12)

    def test_last_point_near_goal(self):
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([0.0, 1.0, 0.0])
        r = geodex.discrete_geodesic(self.sphere, p, q, self.settings)
        # Final point should be close to goal
        d_end = self.sphere.distance(r.path[-1], q)
        assert d_end < 0.1
        assert r.status == geodex.InterpolationStatus.Converged

    def test_all_points_on_sphere(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        r = geodex.discrete_geodesic(self.sphere, p, q, self.settings)
        for pt in r.path:
            assert abs(np.linalg.norm(pt) - 1.0) < 1e-8

    def test_start_equals_goal_returns_single_point(self):
        p = np.array([0.0, 0.0, 1.0])
        r = geodex.discrete_geodesic(self.sphere, p, p, self.settings)
        assert len(r.path) == 1
        assert r.status == geodex.InterpolationStatus.DegenerateInput
        np.testing.assert_allclose(r.path[0], p, atol=1e-12)

    def test_default_settings(self):
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([0.0, 1.0, 0.0])
        # Should work without explicit settings
        r = geodex.discrete_geodesic(self.sphere, p, q)
        assert len(r.path) >= 2

    def test_projection_retraction(self):
        sphere_proj = geodex.Sphere(retraction="projection")
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        r = geodex.discrete_geodesic(sphere_proj, p, q, self.settings)
        assert len(r.path) >= 2
        np.testing.assert_allclose(r.path[0], p, atol=1e-12)


class TestDiscreteGeodesicEuclidean:
    def setup_method(self):
        self.euc = geodex.Euclidean(3)
        self.settings = geodex.InterpolationSettings(step_size=0.3, max_steps=200)

    def test_reaches_goal(self):
        p = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0])
        r = geodex.discrete_geodesic(self.euc, p, q, self.settings)
        assert len(r.path) >= 2
        np.testing.assert_allclose(r.path[0], p, atol=1e-12)
        np.testing.assert_allclose(r.path[-1], q, atol=0.05)

    def test_path_is_approximately_straight(self):
        """Points on a Euclidean path should lie close to the straight line."""
        p = np.zeros(3)
        q = np.array([2.0, 0.0, 0.0])
        r = geodex.discrete_geodesic(self.euc, p, q, self.settings)
        for pt in r.path:
            # y and z components should stay near 0
            assert abs(pt[1]) < 0.05
            assert abs(pt[2]) < 0.05


class TestDiscreteGeodesicTorus:
    def setup_method(self):
        self.torus = geodex.Torus(2)
        self.settings = geodex.InterpolationSettings(step_size=0.3, max_steps=300)

    def test_first_point_is_start(self):
        p = np.array([0.5, 0.5])
        q = np.array([2.0, 2.0])
        r = geodex.discrete_geodesic(self.torus, p, q, self.settings)
        assert len(r.path) >= 1
        np.testing.assert_allclose(r.path[0], p, atol=1e-12)

    def test_path_length_positive(self):
        p = np.array([0.5, 0.5])
        q = np.array([2.0, 2.0])
        r = geodex.discrete_geodesic(self.torus, p, q, self.settings)
        assert len(r.path) >= 2


class TestDiscreteGeodesicSE2:
    def setup_method(self):
        self.se2 = geodex.SE2(wx=1.0, wy=1.0, wtheta=0.5, x_lo=0.0, x_hi=5.0,
                               y_lo=0.0, y_hi=5.0)
        self.settings = geodex.InterpolationSettings(step_size=0.3, max_steps=300)

    def test_first_point_is_start(self):
        p = np.array([1.0, 1.0, 0.0])
        q = np.array([3.0, 3.0, 0.5])
        r = geodex.discrete_geodesic(self.se2, p, q, self.settings)
        assert len(r.path) >= 1
        np.testing.assert_allclose(r.path[0], p, atol=1e-12)

    def test_returns_list(self):
        p = np.array([1.0, 1.0, 0.0])
        q = np.array([2.0, 2.0, 0.0])
        r = geodex.discrete_geodesic(self.se2, p, q, self.settings)
        assert isinstance(r.path, list)
        assert len(r.path) >= 2


class TestDiscreteGeodesicConfigSpace:
    def setup_method(self):
        self.sphere = geodex.Sphere()
        metric = geodex.ConstantSPDMetric(np.eye(3))
        self.cs = geodex.ConfigurationSpace(self.sphere, metric)
        self.settings = geodex.InterpolationSettings(step_size=0.3, max_steps=200)

    def test_first_point_is_start(self):
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([0.0, 1.0, 0.0])
        r = geodex.discrete_geodesic(self.cs, p, q, self.settings)
        assert len(r.path) >= 1
        np.testing.assert_allclose(r.path[0], p, atol=1e-12)

    def test_all_points_on_sphere(self):
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        r = geodex.discrete_geodesic(self.cs, p, q, self.settings)
        for pt in r.path:
            assert abs(np.linalg.norm(pt) - 1.0) < 1e-8

    def test_torus_with_kinetic_metric(self):
        torus = geodex.Torus(2)
        metric = geodex.KineticEnergyMetric(lambda q: np.eye(2))
        cs = geodex.ConfigurationSpace(torus, metric)
        p = np.array([0.5, 0.5])
        q = np.array([2.0, 2.0])
        settings = geodex.InterpolationSettings(step_size=0.3, max_steps=300)
        r = geodex.discrete_geodesic(cs, p, q, settings)
        assert len(r.path) >= 1
        np.testing.assert_allclose(r.path[0], p, atol=1e-12)


class TestDiscreteGeodesicInvalidInput:
    def test_invalid_manifold_raises(self):
        with pytest.raises(Exception):
            geodex.discrete_geodesic("not_a_manifold", np.zeros(3), np.ones(3))


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


class TestInterpolationStatus:
    def setup_method(self):
        self.sphere = geodex.Sphere()

    def test_converged_status(self):
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([np.sin(1.0), 0.0, np.cos(1.0)])
        r = geodex.discrete_geodesic(self.sphere, p, q)
        assert r.status == geodex.InterpolationStatus.Converged
        assert r.iterations > 0
        assert r.initial_distance > 0.5
        assert r.final_distance < 1e-3
        assert r.distortion_halvings == 0

    def test_max_steps_status(self):
        # Target far away with a tight iteration budget.
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([np.sin(2.5), 0.0, np.cos(2.5)])
        settings = geodex.InterpolationSettings(step_size=0.1, max_steps=2)
        r = geodex.discrete_geodesic(self.sphere, p, q, settings)
        assert r.status == geodex.InterpolationStatus.MaxStepsReached
        assert r.iterations == 2

    def test_cut_locus_status(self):
        # Antipodal points on the sphere — log collapses to zero.
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([0.0, 0.0, -1.0])
        r = geodex.discrete_geodesic(self.sphere, p, q)
        assert r.status == geodex.InterpolationStatus.CutLocus
        assert len(r.path) == 1

    def test_degenerate_input_status(self):
        p = np.array([0.0, 0.0, 1.0])
        r = geodex.discrete_geodesic(self.sphere, p, p)
        assert r.status == geodex.InterpolationStatus.DegenerateInput
        assert r.iterations == 0
        assert r.initial_distance == 0.0

    def test_repr_includes_status(self):
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([np.sin(1.0), 0.0, np.cos(1.0)])
        r = geodex.discrete_geodesic(self.sphere, p, q)
        text = repr(r)
        assert "InterpolationResult" in text
        assert "Converged" in text

    def test_final_distance_reported(self):
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([np.sin(1.0), 0.0, np.cos(1.0)])
        r = geodex.discrete_geodesic(self.sphere, p, q)
        # Final distance should match distance from last path point to target.
        expected = self.sphere.distance(r.path[-1], q)
        assert r.final_distance == pytest.approx(expected, abs=1e-6)

"""Tests for algorithm bindings: InterpolationSettings, distance_midpoint,
discrete_geodesic, simplify_path, precompute_matrix_lower_bound."""

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
        assert s.force_log_direction is False
        assert s.fd_midpoint_guard_tau == pytest.approx(0.25)

    def test_keyword_construction(self):
        s = geodex.InterpolationSettings(step_size=0.1, max_steps=50)
        assert s.step_size == pytest.approx(0.1)
        assert s.max_steps == 50
        # Other fields keep defaults
        assert s.convergence_tol == pytest.approx(1e-4)

    def test_new_fields_keyword_construction(self):
        s = geodex.InterpolationSettings(force_log_direction=True, fd_midpoint_guard_tau=0.1)
        assert s.force_log_direction is True
        assert s.fd_midpoint_guard_tau == pytest.approx(0.1)
        # Unrelated fields keep defaults
        assert s.step_size == pytest.approx(0.5)

    def test_field_mutation(self):
        s = geodex.InterpolationSettings()
        s.step_size = 0.25
        s.max_steps = 200
        s.force_log_direction = True
        s.fd_midpoint_guard_tau = 0.05
        assert s.step_size == pytest.approx(0.25)
        assert s.max_steps == 200
        assert s.force_log_direction is True
        assert s.fd_midpoint_guard_tau == pytest.approx(0.05)

    def test_repr(self):
        s = geodex.InterpolationSettings()
        r = repr(s)
        assert "InterpolationSettings" in r
        assert "step_size" in r


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
        assert isinstance(r.path, np.ndarray)
        assert r.path.ndim == 2 and r.path.shape[0] >= 2
        assert isinstance(r.waypoints, list)
        assert all(isinstance(pt, np.ndarray) for pt in r.waypoints)

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

    def test_returns_ndarray_and_waypoints(self):
        p = np.array([1.0, 1.0, 0.0])
        q = np.array([2.0, 2.0, 0.0])
        r = geodex.discrete_geodesic(self.se2, p, q, self.settings)
        assert isinstance(r.path, np.ndarray)
        assert r.path.ndim == 2 and r.path.shape[0] >= 2
        assert isinstance(r.waypoints, list)
        assert len(r.waypoints) == r.path.shape[0]


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


# ---------------------------------------------------------------------------
# force_log_direction: skip the FD fallback even when the metric is non-Riemannian
# ---------------------------------------------------------------------------


class TestForceLogDirection:
    """``force_log_direction = True`` makes the walk always use -log as its
    descent direction. Under a non-identity SPD metric attached to the sphere
    (where ``is_riemannian_log`` is false), this changes the walk from the FD
    path to the fast path."""

    def setup_method(self):
        self.sphere = geodex.Sphere()
        self.metric = geodex.ConstantSPDMetric(np.diag([4.0, 1.0, 1.0]))
        self.cs = geodex.ConfigurationSpace(self.sphere, self.metric)
        self.p = np.array([0.0, 0.0, 1.0])
        self.q = np.array([np.sin(1.0), 0.0, np.cos(1.0)])

    def test_default_converges_via_fd(self):
        settings = geodex.InterpolationSettings(step_size=0.1, max_steps=500)
        r = geodex.discrete_geodesic(self.cs, self.p, self.q, settings)
        assert r.status == geodex.InterpolationStatus.Converged
        assert r.fd_midpoint_fallbacks >= 0  # field readable, value meaningful

    def test_force_log_skips_fd(self):
        """With force_log_direction=True the FD path never runs, so the
        midpoint fallback counter stays at zero."""
        settings = geodex.InterpolationSettings(
            step_size=0.1, max_steps=500, force_log_direction=True
        )
        r = geodex.discrete_geodesic(self.cs, self.p, self.q, settings)
        assert r.status == geodex.InterpolationStatus.Converged
        assert r.fd_midpoint_fallbacks == 0


# ---------------------------------------------------------------------------
# fd_midpoint_guard_tau + InterpolationResult.fd_midpoint_fallbacks
# ---------------------------------------------------------------------------


class TestFdMidpointGuard:
    def test_fallbacks_zero_on_clean_sphere_geodesic(self):
        """Clean round sphere uses the fast path; FD never runs."""
        sphere = geodex.Sphere()
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([np.sin(1.0), 0.0, np.cos(1.0)])
        r = geodex.discrete_geodesic(sphere, p, q)
        assert isinstance(r.fd_midpoint_fallbacks, int)
        assert r.fd_midpoint_fallbacks == 0

    def test_guard_tau_zero_forces_via_log_samples(self):
        """``fd_midpoint_guard_tau = 0`` rejects every midpoint sample, so
        every FD basis direction falls back to |log|_R and the counter
        increments throughout the walk."""
        sphere = geodex.Sphere()
        metric = geodex.ConstantSPDMetric(np.diag([4.0, 1.0, 1.0]))
        cs = geodex.ConfigurationSpace(sphere, metric)
        p = np.array([0.0, 0.0, 1.0])
        q = np.array([np.sin(1.0), 0.0, np.cos(1.0)])
        settings = geodex.InterpolationSettings(
            step_size=0.1, max_steps=500, fd_midpoint_guard_tau=0.0
        )
        r = geodex.discrete_geodesic(cs, p, q, settings)
        assert r.status == geodex.InterpolationStatus.Converged
        assert r.fd_midpoint_fallbacks > 0


# ---------------------------------------------------------------------------
# simplify_path
# ---------------------------------------------------------------------------


class TestSimplifyPathSettings:
    def test_defaults(self):
        s = geodex.SimplifyPathSettings()
        assert s.max_shortcut_attempts == 200
        assert s.smooth_target_segments == 128
        assert s.max_displacement == pytest.approx(0.0)


class TestSimplifyPathEuclidean:
    def setup_method(self):
        self.euc = geodex.Euclidean(2)
        self.cs = geodex.ConfigurationSpace(self.euc, geodex.ConstantSPDMetric(np.eye(2)))

    def test_collinear_path_collapses(self):
        # Three points along the x-axis, middle redundant.
        path = [np.array([0.0, 0.0]), np.array([0.5, 0.0]), np.array([1.0, 0.0])]
        settings = geodex.SimplifyPathSettings()
        settings.max_shortcut_attempts = 50
        settings.smooth_target_segments = 4
        settings.max_iter_per_level = 50
        r = geodex.simplify_path(self.cs, lambda q: True, path, settings)
        assert isinstance(r, geodex.SimplifyPathResult)
        assert r.collision_free is True
        assert r.distance == pytest.approx(1.0, abs=2e-2)

    def test_validity_fn_respected(self):
        # Block any q with x in [0.45, 0.55]: shortcut from (0,0) to (1,0)
        # passes through that band, so the middle vertex must survive.
        def validity(q):
            return not (0.45 <= q[0] <= 0.55 and abs(q[1]) < 0.1)

        path = [
            np.array([0.0, 0.0]),
            np.array([0.5, -0.5]),  # detour avoiding the band
            np.array([1.0, 0.0]),
        ]
        settings = geodex.SimplifyPathSettings()
        settings.max_shortcut_attempts = 100
        settings.smooth_target_segments = 4
        settings.max_iter_per_level = 50
        r = geodex.simplify_path(self.cs, validity, path, settings)
        # Result must remain collision-free under the same predicate.
        for q in r.path:
            assert validity(q)

    def test_endpoints_preserved(self):
        path = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([1.0, 1.0]),
        ]
        settings = geodex.SimplifyPathSettings()
        settings.smooth_target_segments = 4
        r = geodex.simplify_path(self.cs, lambda q: True, path, settings)
        np.testing.assert_allclose(r.path[0], path[0], atol=1e-12)
        np.testing.assert_allclose(r.path[-1], path[-1], atol=1e-12)


# ---------------------------------------------------------------------------
# precompute_matrix_lower_bound
# ---------------------------------------------------------------------------


class TestPrecomputeMatrixLowerBoundSettings:
    def test_defaults(self):
        s = geodex.PrecomputeMatrixLowerBoundSettings()
        assert s.max_outer == 50
        assert s.tol == pytest.approx(1e-6)
        assert s.n_starts_per_iter == 0  # auto
        assert s.seed == 42


class TestPrecomputeMatrixLowerBound:
    def test_constant_identity_metric_certifies_immediately(self):
        result = geodex.precompute_matrix_lower_bound(
            metric_fn=lambda q: np.eye(3),
            lo=np.array([-1.0, -1.0, -1.0]),
            hi=np.array([1.0, 1.0, 1.0]),
        )
        np.testing.assert_allclose(result.M_lower, np.eye(3), atol=1e-12)
        assert result.lambda_min_certificate == pytest.approx(1.0, abs=1e-6)
        assert result.converged is True

    def test_constant_anisotropic_metric_recovers_matrix(self):
        M = np.diag([4.0, 1.0])
        result = geodex.precompute_matrix_lower_bound(
            metric_fn=lambda q: M,
            lo=np.array([-1.0, -1.0]),
            hi=np.array([1.0, 1.0]),
        )
        np.testing.assert_allclose(result.M_lower, M, atol=1e-8)
        assert result.converged is True

    def test_deterministic_with_seed(self):
        def metric(q):
            # q-dependent SPD: 1 + q² * I
            scale = 1.0 + q[0] ** 2 + q[1] ** 2
            return scale * np.eye(2)

        s = geodex.PrecomputeMatrixLowerBoundSettings()
        s.seed = 13
        s.max_outer = 5
        a = geodex.precompute_matrix_lower_bound(
            metric, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), s
        )
        b = geodex.precompute_matrix_lower_bound(
            metric, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), s
        )
        np.testing.assert_allclose(a.M_lower, b.M_lower, atol=1e-12)
        assert a.lambda_min_certificate == pytest.approx(b.lambda_min_certificate, abs=1e-12)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(Exception):
            geodex.precompute_matrix_lower_bound(
                metric_fn=lambda q: np.eye(2),
                lo=np.array([-1.0, -1.0]),
                hi=np.array([1.0]),  # wrong size
            )

    def test_lo_greater_than_hi_raises(self):
        with pytest.raises(Exception):
            geodex.precompute_matrix_lower_bound(
                metric_fn=lambda q: np.eye(2),
                lo=np.array([1.0, 1.0]),
                hi=np.array([-1.0, -1.0]),
            )

    def test_bound_is_loewner_dominated_by_metric(self):
        rng = np.random.default_rng(0)

        # Position-dependent metric: M(q) = (1 + ||q||²) * I_2
        def metric(q):
            return (1.0 + q.dot(q)) * np.eye(2)

        result = geodex.precompute_matrix_lower_bound(
            metric_fn=metric,
            lo=np.array([-1.0, -1.0]),
            hi=np.array([1.0, 1.0]),
        )
        for _ in range(20):
            q = rng.uniform(-1.0, 1.0, size=2)
            diff = metric(q) - result.M_lower
            evals = np.linalg.eigvalsh(diff)
            assert evals.min() > -1e-6  # M(q) - M_lower ⪰ 0 (Loewner)

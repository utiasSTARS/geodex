"""Tests for the geodex.heuristics admissible-heuristics submodule."""

import numpy as np
import pytest

from geodex import heuristics


# ---------------------------------------------------------------------------
# Submodule wiring
# ---------------------------------------------------------------------------


class TestSubmoduleSurface:
    def test_submodule_exposes_four_classes(self):
        assert hasattr(heuristics, "Zero")
        assert hasattr(heuristics, "Euclidean")
        assert hasattr(heuristics, "EigenvalueLowerBound")
        assert hasattr(heuristics, "MatrixLowerBound")

    def test_submodule_exposes_the_product_combiner(self):
        assert callable(heuristics.product_lower_bound)


# ---------------------------------------------------------------------------
# Zero
# ---------------------------------------------------------------------------


class TestZero:
    def test_returns_zero_for_distinct_points(self):
        h = heuristics.Zero()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert h(a, b) == 0.0

    def test_returns_zero_for_same_point(self):
        h = heuristics.Zero()
        p = np.array([1.0, 2.0, 3.0])
        assert h(p, p) == 0.0

    def test_non_negative(self):
        h = heuristics.Zero()
        rng = np.random.default_rng(123)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            assert h(a, b) >= 0.0


# ---------------------------------------------------------------------------
# Euclidean
# ---------------------------------------------------------------------------


class TestEuclidean:
    def test_zero_for_same_point(self):
        h = heuristics.Euclidean()
        a = np.array([1.0, 2.0, 3.0])
        assert h(a, a) == pytest.approx(0.0)

    def test_known_distance(self):
        h = heuristics.Euclidean()
        a = np.zeros(3)
        b = np.array([3.0, 4.0, 0.0])
        assert h(a, b) == pytest.approx(5.0)

    def test_symmetry(self):
        h = heuristics.Euclidean()
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert h(a, b) == pytest.approx(h(b, a))

    def test_agrees_with_numpy(self):
        h = heuristics.Euclidean()
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.standard_normal(5)
            b = rng.standard_normal(5)
            assert h(a, b) == pytest.approx(np.linalg.norm(a - b))

    def test_non_negative(self):
        h = heuristics.Euclidean()
        rng = np.random.default_rng(1)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            assert h(a, b) >= 0.0

    def test_triangle_inequality(self):
        h = heuristics.Euclidean()
        rng = np.random.default_rng(7)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            c = rng.standard_normal(3)
            assert h(a, c) <= h(a, b) + h(b, c) + 1e-12


# ---------------------------------------------------------------------------
# EigenvalueLowerBound
# ---------------------------------------------------------------------------


class TestEigenvalueLowerBound:
    def test_lambda_one_matches_euclidean(self):
        h_e = heuristics.Euclidean()
        h_lb = heuristics.EigenvalueLowerBound(1.0)
        rng = np.random.default_rng(0)
        for _ in range(5):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            assert h_lb(a, b) == pytest.approx(h_e(a, b))

    def test_scalar_factor(self):
        h = heuristics.EigenvalueLowerBound(4.0)
        a = np.zeros(2)
        b = np.array([3.0, 4.0])
        assert h(a, b) == pytest.approx(np.sqrt(4.0) * 5.0)

    def test_sqrt_lambda_min_getter(self):
        h = heuristics.EigenvalueLowerBound(9.0)
        assert h.sqrt_lambda_min == pytest.approx(3.0)

    def test_zero_for_same_point(self):
        h = heuristics.EigenvalueLowerBound(2.0)
        p = np.array([1.0, 2.0])
        assert h(p, p) == 0.0

    def test_dominates_zero(self):
        h_z = heuristics.Zero()
        h_lb = heuristics.EigenvalueLowerBound(0.5)
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert h_lb(a, b) >= h_z(a, b)

    def test_non_negative(self):
        h = heuristics.EigenvalueLowerBound(2.5)
        rng = np.random.default_rng(11)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            assert h(a, b) >= 0.0

    def test_triangle_inequality(self):
        h = heuristics.EigenvalueLowerBound(2.0)
        rng = np.random.default_rng(8)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            c = rng.standard_normal(3)
            assert h(a, c) <= h(a, b) + h(b, c) + 1e-12


# ---------------------------------------------------------------------------
# MatrixLowerBound
# ---------------------------------------------------------------------------


class TestMatrixLowerBound:
    def test_identity_matches_euclidean(self):
        h = heuristics.MatrixLowerBound(np.eye(3))
        h_e = heuristics.Euclidean()
        rng = np.random.default_rng(1)
        for _ in range(5):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            assert h(a, b) == pytest.approx(h_e(a, b))

    def test_anisotropic_diagonal_matches_closed_form(self):
        # M = diag(4, 1) → ‖Lᵀ Δ‖ = sqrt(4 dx² + dy²)
        h = heuristics.MatrixLowerBound(np.diag([4.0, 1.0]))
        a = np.zeros(2)
        b = np.array([1.0, 1.0])
        assert h(a, b) == pytest.approx(np.sqrt(5.0))

    def test_matrix_recovers_input(self):
        M = np.diag([4.0, 1.0, 9.0])
        h = heuristics.MatrixLowerBound(M)
        np.testing.assert_allclose(h.matrix(), M, atol=1e-12)

    def test_det(self):
        h = heuristics.MatrixLowerBound(np.diag([4.0, 1.0, 9.0]))
        assert h.det() == pytest.approx(36.0)

    def test_eigenvalues_ascending(self):
        h = heuristics.MatrixLowerBound(np.diag([4.0, 1.0, 9.0]))
        evals = h.eigenvalues()
        np.testing.assert_allclose(evals, [1.0, 4.0, 9.0], atol=1e-10)

    def test_construction_keeps_the_input_matrix(self):
        h = heuristics.MatrixLowerBound(np.eye(3))
        np.testing.assert_allclose(h.matrix(), np.eye(3), atol=1e-12)

    def test_update_loosens_when_smaller(self):
        h = heuristics.MatrixLowerBound(2.0 * np.eye(2))  # M_lower = 2I
        loosened = h.update(np.eye(2))  # M_new = I  ⪯  2I
        assert loosened is True
        np.testing.assert_allclose(h.matrix(), np.eye(2), atol=1e-12)

    def test_update_skipped_when_dominated(self):
        h = heuristics.MatrixLowerBound(np.eye(2))  # M_lower = I
        loosened = h.update(2.0 * np.eye(2))  # M_new = 2I  ⪰  I
        assert loosened is False
        np.testing.assert_allclose(h.matrix(), np.eye(2), atol=1e-12)

    def test_update_lowers_det_monotonically(self):
        h = heuristics.MatrixLowerBound(4.0 * np.eye(2))
        rng = np.random.default_rng(7)
        prev_det = h.det()
        for _ in range(5):
            # Random SPD via QQᵀ + epsI, then scale to keep things small.
            Q = rng.standard_normal((2, 2))
            M_new = Q @ Q.T + 0.1 * np.eye(2)
            h.update(M_new)
            cur_det = h.det()
            assert cur_det <= prev_det + 1e-12
            prev_det = cur_det

    def test_eigenvalue_floor_dominates_when_matrix_thin(self):
        # M_lower has tiny eigenvalue along x; floor pulls h up to sqrt(0.25)*||Δ||.
        M = np.diag([1e-10, 1.0])
        lambda_floor = 0.25
        h = heuristics.MatrixLowerBound(M, lambda_floor)
        a = np.zeros(2)
        b = np.array([1.0, 0.0])
        # h_mlb ≈ sqrt(1e-10) ≈ 1e-5; h_elb = sqrt(0.25)*1 = 0.5; max = 0.5
        assert h(a, b) == pytest.approx(0.5, abs=1e-3)
        assert h.has_eigenvalue_floor is True

    def test_no_floor_by_default(self):
        h = heuristics.MatrixLowerBound(np.eye(2))
        assert h.has_eigenvalue_floor is False

    def test_non_negative(self):
        M = np.array(
            [
                [4.0, 0.5, 0.0],
                [0.5, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        h = heuristics.MatrixLowerBound(M)
        rng = np.random.default_rng(55)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            assert h(a, b) >= 0.0

    def test_triangle_inequality(self):
        # h(a, b) = ‖Lᵀ Δ‖ is a norm on (a - b), so the triangle inequality holds.
        M = np.array(
            [
                [3.0, 0.4, 0.0],
                [0.4, 1.5, 0.0],
                [0.0, 0.0, 0.8],
            ]
        )
        h = heuristics.MatrixLowerBound(M)
        rng = np.random.default_rng(9)
        for _ in range(10):
            a = rng.standard_normal(3)
            b = rng.standard_normal(3)
            c = rng.standard_normal(3)
            assert h(a, c) <= h(a, b) + h(b, c) + 1e-9

    def test_symmetry(self):
        M = np.diag([4.0, 1.0, 0.7, 1.0])
        h = heuristics.MatrixLowerBound(M)
        rng = np.random.default_rng(77)
        for _ in range(6):
            a = rng.standard_normal(4)
            b = rng.standard_normal(4)
            assert h(a, b) == pytest.approx(h(b, a))

    def test_loewner_order_preserved_after_updates(self):
        # After any sequence of updates with SPD observations M_k, the current
        # M_lower must satisfy M_lower ≼ M_k in the Loewner order for every k
        # (in particular for each M_k we just supplied).
        h = heuristics.MatrixLowerBound(np.eye(2))
        observations = [
            np.array([[0.5, 0.0], [0.0, 3.0]]),
            np.array([[2.0, 0.5], [0.5, 0.4]]),
            np.array([[0.7, 0.0], [0.0, 0.7]]),
        ]
        for X in observations:
            h.update(X)
        M_lower = h.matrix()
        for X in observations:
            evals = np.linalg.eigvalsh(X - M_lower)
            assert evals.min() >= -1e-9


# ---------------------------------------------------------------------------
# Periodic coordinates
# ---------------------------------------------------------------------------


class TestMatrixLowerBoundPeriods:
    TWO_PI = 2.0 * np.pi

    def test_empty_periods_reproduce_the_raw_difference(self):
        M = np.diag([1.0, 1.0, 2.25])
        plain = heuristics.MatrixLowerBound(M)
        empty = heuristics.MatrixLowerBound(M, np.zeros(0))
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, 0.5, -3.0])
        assert empty.periods.size == 0
        assert empty(a, b) == plain(a, b)

    def test_wraps_across_the_branch_cut(self):
        M = np.diag([1.0, 1.0, 4.0])
        periods = np.array([0.0, 0.0, self.TWO_PI])
        wrapped = heuristics.MatrixLowerBound(M, periods)
        unwrapped = heuristics.MatrixLowerBound(M)
        eps = 0.05
        a = np.array([0.0, 0.0, np.pi - eps])
        b = np.array([0.0, 0.0, -np.pi + eps])
        assert wrapped(a, b) == pytest.approx(2.0 * 2.0 * eps)
        assert unwrapped(a, b) == pytest.approx(2.0 * (self.TWO_PI - 2.0 * eps))

    def test_aperiodic_axes_are_left_alone(self):
        periods = np.array([0.0, 0.0, self.TWO_PI])
        h = heuristics.MatrixLowerBound(np.eye(3), periods)
        assert h(np.zeros(3), np.array([100.0, 0.0, 0.0])) == pytest.approx(100.0)

    def test_periods_are_exposed(self):
        periods = np.array([0.0, 0.0, self.TWO_PI])
        h = heuristics.MatrixLowerBound(np.eye(3), periods)
        np.testing.assert_allclose(h.periods, periods)

    def test_matches_the_deck_group_minimum(self):
        M = np.array([[1.5, 0.4, 0.0], [0.4, 2.0, 0.0], [0.0, 0.0, 3.0]])
        h = heuristics.MatrixLowerBound(M, np.array([0.0, 0.0, self.TWO_PI]))
        Lt = np.linalg.cholesky(M).T
        rng = np.random.default_rng(4242)
        for _ in range(200):
            a = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)])
            b = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)])
            best = min(
                np.linalg.norm(Lt @ (a - b - np.array([0.0, 0.0, k * self.TWO_PI])))
                for k in range(-3, 4)
            )
            assert h(a, b) == pytest.approx(best)

    def test_rejects_a_coupled_periodic_axis(self):
        M = np.array([[1.0, 0.0, 0.3], [0.0, 1.0, 0.0], [0.3, 0.0, 1.0]])
        with pytest.raises(ValueError):
            heuristics.MatrixLowerBound(M, np.array([0.0, 0.0, self.TWO_PI]))
        heuristics.MatrixLowerBound(M)  # fine without periods

    def test_rejects_mismatched_period_size(self):
        with pytest.raises(ValueError):
            heuristics.MatrixLowerBound(np.eye(3), np.array([0.0, self.TWO_PI]))


# ---------------------------------------------------------------------------
# Product of factor bounds
# ---------------------------------------------------------------------------


class TestProductLowerBound:
    TWO_PI = 2.0 * np.pi
    SE2_BOUND = np.diag([0.5, 0.5, 2.0])
    SE2_PERIODS = np.array([0.0, 0.0, TWO_PI])
    PLANE_BOUND = np.array([[2.0, 0.6], [0.6, 1.5]])

    def block_diagonal(self):
        M = np.zeros((6, 6))
        M[:3, :3] = self.SE2_BOUND
        M[3:5, 3:5] = self.PLANE_BOUND
        M[5, 5] = 3.0
        return M

    def factors(self):
        return [(self.SE2_BOUND, self.SE2_PERIODS), self.PLANE_BOUND,
                (np.array([[3.0]]), np.array([self.TWO_PI]))]

    def test_block_diagonal_with_concatenated_periods(self):
        h = heuristics.product_lower_bound(self.factors())
        assert isinstance(h, heuristics.MatrixLowerBound)
        np.testing.assert_allclose(h.matrix(), self.block_diagonal(), atol=1e-12)
        np.testing.assert_array_equal(h.periods, [0.0, 0.0, self.TWO_PI, 0.0, 0.0, self.TWO_PI])

    def test_aperiodic_factors_carry_no_periods(self):
        h = heuristics.product_lower_bound([self.PLANE_BOUND, (np.eye(3), None)])
        assert h.periods.size == 0
        np.testing.assert_allclose(h.matrix()[2:, 2:], np.eye(3), atol=1e-12)

    def test_equals_the_matrix_lower_bound_of_the_block_diagonal(self):
        h = heuristics.product_lower_bound(self.factors())
        periods = np.array([0.0, 0.0, self.TWO_PI, 0.0, 0.0, self.TWO_PI])
        reference = heuristics.MatrixLowerBound(self.block_diagonal(), periods)
        rng = np.random.default_rng(17)
        for _ in range(100):
            a, b = rng.uniform(-4.0, 4.0, 6), rng.uniform(-4.0, 4.0, 6)
            assert h(a, b) == reference(a, b)

    def test_wraps_every_periodic_factor(self):
        h = heuristics.product_lower_bound(self.factors())
        a, b = np.zeros(6), np.zeros(6)
        a[[2, 5]] = np.pi - 0.05
        b[[2, 5]] = -np.pi + 0.05
        assert h(a, b) == pytest.approx(0.1 * np.sqrt(2.0 + 3.0), abs=1e-12)

    def test_rejects_malformed_factors(self):
        with pytest.raises(ValueError):
            heuristics.product_lower_bound([])
        with pytest.raises(ValueError):
            heuristics.product_lower_bound([np.ones((2, 3))])
        with pytest.raises(ValueError):
            heuristics.product_lower_bound([(self.SE2_BOUND, np.zeros(2))])
        with pytest.raises(ValueError):
            heuristics.product_lower_bound([(self.PLANE_BOUND, np.array([self.TWO_PI, 0.0]))])

"""Tests for geodex.ClearanceMetric."""

import numpy as np
import pytest

import geodex


class TestClearanceMetric:
    def test_far_from_obstacle(self):
        """Far from obstacle, conformal factor -> 1, inner product ~ base."""
        base = geodex.ConstantSPDMetric(np.eye(2))
        sdf = lambda q: 100.0  # very far from any obstacle
        metric = geodex.ClearanceMetric(base, sdf, kappa=5.0, beta=3.0)
        p = np.array([0.0, 0.0])
        u = np.array([1.0, 0.0])
        v = np.array([1.0, 0.0])
        base_val = base.inner(p, u, v)
        clearance_val = metric.inner(p, u, v)
        # c(q) = 1 + 5*exp(-3*100) ~ 1.0
        assert clearance_val == pytest.approx(base_val, rel=1e-6)

    def test_near_obstacle(self):
        """Near obstacle, conformal factor > 1, inner product increases."""
        base = geodex.ConstantSPDMetric(np.eye(2))
        sdf = lambda q: 0.1  # close to obstacle surface
        metric = geodex.ClearanceMetric(base, sdf, kappa=5.0, beta=3.0)
        p = np.array([0.0, 0.0])
        u = np.array([1.0, 0.0])
        base_val = base.inner(p, u, u)
        clearance_val = metric.inner(p, u, u)
        assert clearance_val > base_val * 1.5  # should be noticeably scaled up

    def test_properties(self):
        base = geodex.ConstantSPDMetric(np.eye(2))
        metric = geodex.ClearanceMetric(base, lambda q: 1.0, kappa=7.0, beta=2.0)
        assert metric.kappa == pytest.approx(7.0)
        assert metric.beta == pytest.approx(2.0)

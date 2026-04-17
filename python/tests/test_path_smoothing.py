"""Tests for geodex.smooth_path and related types."""

import numpy as np
import pytest

import geodex


class TestPathSmoothingSettings:
    def test_defaults(self):
        s = geodex.PathSmoothingSettings()
        assert s.max_shortcut_attempts == 200
        assert s.lbfgs_max_iterations == 200
        assert s.armijo_max_backtracks == 30
        assert s.grad_tol == pytest.approx(1e-8)

    def test_set_fields(self):
        s = geodex.PathSmoothingSettings()
        s.max_shortcut_attempts = 100
        assert s.max_shortcut_attempts == 100


class TestSmoothPath:
    def test_euclidean_detour(self):
        """Smooth a path with a detour in Euclidean space."""
        manifold = geodex.Euclidean(2)
        # Path: (0,0) -> (1,2) -> (2,0) -- detour through (1,2)
        path = [np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.array([2.0, 0.0])]
        validity_fn = lambda q: True  # no obstacles
        settings = geodex.PathSmoothingSettings()
        settings.max_shortcut_attempts = 50
        settings.lbfgs_max_iterations = 50
        result = geodex.smooth_path(manifold, validity_fn, path, settings)
        assert isinstance(result, geodex.PathSmoothingResult)
        assert len(result.path) >= 2
        assert result.energy >= 0.0
        assert result.distance >= 0.0

    def test_straight_path_unchanged(self):
        """A straight path should have minimal smoothing effect."""
        manifold = geodex.Euclidean(2)
        path = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        validity_fn = lambda q: True
        settings = geodex.PathSmoothingSettings()
        settings.max_shortcut_attempts = 10
        settings.lbfgs_max_iterations = 10
        result = geodex.smooth_path(manifold, validity_fn, path, settings)
        # Should still be roughly along x-axis
        for pt in result.path:
            assert abs(pt[1]) < 0.5  # y-coordinate stays small

"""Tests for geodex.collision module."""

import numpy as np
import pytest

import geodex


class TestCircleSDF:
    def test_outside(self):
        sdf = geodex.collision.CircleSDF(0.0, 0.0, 1.0)
        assert sdf(2.0, 0.0) == pytest.approx(1.0, abs=1e-10)

    def test_inside(self):
        sdf = geodex.collision.CircleSDF(0.0, 0.0, 1.0)
        assert sdf(0.0, 0.0) == pytest.approx(-1.0, abs=1e-10)

    def test_on_boundary(self):
        sdf = geodex.collision.CircleSDF(0.0, 0.0, 1.0)
        assert sdf(1.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_properties(self):
        sdf = geodex.collision.CircleSDF(1.0, 2.0, 3.0)
        assert sdf.cx == pytest.approx(1.0)
        assert sdf.cy == pytest.approx(2.0)
        assert sdf.radius == pytest.approx(3.0)


class TestCircleSmoothSDF:
    def test_smooth_distance(self):
        c1 = geodex.collision.CircleSDF(0.0, 0.0, 1.0)
        c2 = geodex.collision.CircleSDF(5.0, 0.0, 1.0)
        smooth = geodex.collision.CircleSmoothSDF([c1, c2], beta=20.0)
        # Far from both -> positive
        assert smooth(2.5, 5.0) > 0.0

    def test_is_free(self):
        c1 = geodex.collision.CircleSDF(0.0, 0.0, 1.0)
        c2 = geodex.collision.CircleSDF(5.0, 0.0, 1.0)
        smooth = geodex.collision.CircleSmoothSDF([c1, c2])
        assert smooth.is_free(2.5, 0.0)
        assert not smooth.is_free(0.0, 0.0)


class TestPolygonFootprint:
    def test_rectangle(self):
        fp = geodex.collision.PolygonFootprint.rectangle(1.0, 0.5, 8)
        assert fp.sample_count() == 32  # 4 edges * 8 samples
        assert fp.bounding_radius() > 0.0
        assert fp.bounding_radius() == pytest.approx(np.sqrt(1.0 + 0.25), abs=0.1)

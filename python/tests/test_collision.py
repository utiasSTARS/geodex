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


class TestDistanceGridSlack:
    def test_interpolated_field_stays_within_its_slack(self):
        # A distance transform of one lethal node at cell (10, 10), resolution 0.1.
        h = 0.1
        rows, cols = np.mgrid[0:30, 0:30]
        data = (np.hypot(cols - 10, rows - 10) * h).ravel().tolist()
        grid = geodex.collision.DistanceGrid(30, 30, h, data)
        assert grid.lipschitz_slack() == pytest.approx(np.sqrt(2.0) * h, abs=1e-15)
        rng = np.random.default_rng(3)
        for p, q in rng.uniform(0.0, 29.0 * h, size=(2000, 2, 2)):
            change = abs(grid.distance_at(*p) - grid.distance_at(*q))
            assert change <= np.linalg.norm(p - q) + grid.lipschitz_slack() + 1e-12

    def test_signed_grid_doubles_the_slack(self):
        # The signed transform of an occupancy grid, distance to the nearest occupied
        # node minus distance to the nearest free one.
        h = 0.1
        rows, cols = np.mgrid[0:20, 0:20]
        occupied = (cols >= 8) & (cols < 12)
        nodes = np.stack([cols.ravel(), rows.ravel()], axis=1)
        occ, free = nodes[occupied.ravel()], nodes[~occupied.ravel()]
        d_occ = np.min(np.hypot(*(nodes[:, None, :] - occ[None]).transpose(2, 0, 1)), axis=1)
        d_free = np.min(np.hypot(*(nodes[:, None, :] - free[None]).transpose(2, 0, 1)), axis=1)
        grid = geodex.collision.DistanceGrid(20, 20, h, ((d_occ - d_free) * h).tolist())
        assert grid.lipschitz_slack() == pytest.approx(2.0 * np.sqrt(2.0) * h, abs=1e-15)

    def test_reset_hands_back_the_array_to_fill(self):
        grid = geodex.collision.DistanceGrid(2, 2, 1.0, [1.0] * 4)
        assert grid.lipschitz_slack() == pytest.approx(np.sqrt(2.0))
        values = grid.reset(3, 2, 0.5)
        assert values.shape == (6,)
        values[:] = -1.0
        assert (grid.width(), grid.height(), grid.resolution()) == (3, 2, 0.5)
        assert grid.distance_at(0.2, 0.2) == pytest.approx(-1.0)
        assert grid.lipschitz_slack() == pytest.approx(2.0 * np.sqrt(2.0) * 0.5)
        with pytest.raises(ValueError):
            grid.reset(0, 2, 0.5)


class TestFootprintAccessors:
    def test_samples_and_gap(self):
        fp = geodex.collision.PolygonFootprint.rectangle(1.0, 0.5, 4)
        n = fp.sample_count_raw()
        assert n == 16
        pts = np.array([(fp.body_x(i), fp.body_y(i)) for i in range(n)])
        assert np.max(np.hypot(pts[:, 0], pts[:, 1])) == pytest.approx(fp.bounding_radius())
        assert fp.max_sample_gap() == pytest.approx(2.0 / 4)
        with pytest.raises(IndexError):
            fp.body_x(n)

    def test_capped_clearance_matches_below_the_cap(self):
        grid = geodex.collision.DistanceGrid(100, 100, 0.1, [3.0] * 10000)
        fp = geodex.collision.PolygonFootprint.rectangle(0.2, 0.2, 2)
        checker = geodex.collision.FootprintGridChecker(grid, fp, 0.5)
        q = np.array([5.0, 5.0, 0.0])
        full = checker(q)
        assert checker.min_distance_capped(q, 10.0) == pytest.approx(full)
        assert checker.min_distance_capped(q, 0.1) >= 0.1


class TestMemoizedSDF:
    def test_repeats_come_from_the_table(self):
        calls = []

        def sdf(q):
            calls.append(1)
            return float(q[0])

        memo = geodex.collision.MemoizedSDF(sdf)
        q = np.array([1.5, 2.0, 0.3])
        assert memo(q) == pytest.approx(1.5)
        assert memo(q) == pytest.approx(1.5)
        assert len(calls) == 1
        memo.clear()
        assert memo(q) == pytest.approx(1.5)
        assert len(calls) == 2

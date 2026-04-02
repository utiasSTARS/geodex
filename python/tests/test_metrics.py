import numpy as np
import pytest

import geodex


class TestKineticEnergyMetric:
    def setup_method(self):
        # Identity mass matrix (should match Euclidean metric).
        self.metric = geodex.KineticEnergyMetric(lambda q: np.eye(len(q)))

    def test_inner_identity(self):
        p = np.array([1.0, 2.0, 3.0])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert self.metric.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)
        assert self.metric.inner(p, u, u) == pytest.approx(1.0, abs=1e-12)

    def test_norm_identity(self):
        p = np.array([0.0, 0.0])
        v = np.array([3.0, 4.0])
        assert self.metric.norm(p, v) == pytest.approx(5.0, abs=1e-12)

    def test_inner_with_mass_matrix(self):
        M = np.diag([2.0, 3.0])
        metric = geodex.KineticEnergyMetric(lambda q: M)
        p = np.array([0.0, 0.0])
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        # u^T M v = 0
        assert metric.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)
        # u^T M u = 2
        assert metric.inner(p, u, u) == pytest.approx(2.0, abs=1e-12)
        # v^T M v = 3
        assert metric.inner(p, v, v) == pytest.approx(3.0, abs=1e-12)

    def test_norm_with_mass_matrix(self):
        M = np.diag([4.0, 9.0])
        metric = geodex.KineticEnergyMetric(lambda q: M)
        p = np.array([0.0, 0.0])
        v = np.array([1.0, 1.0])
        # ||v|| = sqrt(v^T M v) = sqrt(4 + 9) = sqrt(13)
        assert metric.norm(p, v) == pytest.approx(np.sqrt(13.0), abs=1e-12)

    def test_configuration_dependent(self):
        # Mass matrix that depends on configuration.
        metric = geodex.KineticEnergyMetric(lambda q: np.diag(q**2))
        p1 = np.array([1.0, 2.0])
        p2 = np.array([3.0, 4.0])
        v = np.array([1.0, 1.0])
        # At p1: v^T diag(1,4) v = 5
        assert metric.inner(p1, v, v) == pytest.approx(5.0, abs=1e-12)
        # At p2: v^T diag(9,16) v = 25
        assert metric.inner(p2, v, v) == pytest.approx(25.0, abs=1e-12)

    def test_repr(self):
        assert "KineticEnergyMetric" in repr(self.metric)


class TestJacobiMetric:
    def setup_method(self):
        self.mass_fn = lambda q: np.eye(len(q))
        self.pot_fn = lambda q: 0.5 * np.sum(q**2)
        self.H = 10.0
        self.metric = geodex.JacobiMetric(self.mass_fn, self.pot_fn, self.H)

    def test_inner_scaling(self):
        p = np.array([1.0, 1.0])
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        # P(p) = 0.5*(1+1) = 1.0
        # 2*(H - P) = 2*(10 - 1) = 18
        # inner = 18 * u^T I v = 0
        assert self.metric.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)
        # inner(u, u) = 18 * 1 = 18
        assert self.metric.inner(p, u, u) == pytest.approx(18.0, abs=1e-12)

    def test_norm(self):
        p = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        # P(0) = 0, so 2*(10-0) = 20
        # norm = sqrt(20 * 1) = sqrt(20)
        assert self.metric.norm(p, v) == pytest.approx(np.sqrt(20.0), abs=1e-12)

    def test_repr(self):
        assert "JacobiMetric" in repr(self.metric)


class TestPullbackMetric:
    def setup_method(self):
        # 2D -> 3D Jacobian (constant).
        self.J = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.jac_fn = lambda q: self.J
        self.task_fn = lambda q: np.eye(3)
        self.metric = geodex.PullbackMetric(self.jac_fn, self.task_fn)

    def test_inner_pullback(self):
        p = np.array([0.0, 0.0])
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        # G = J^T I J = [[1,0,1],[0,1,1]]^T [[1,0],[0,1],[1,1]] = [[2,1],[1,2]]
        # u^T G v = [1,0] [[2,1],[1,2]] [0,1] = 1
        assert self.metric.inner(p, u, v) == pytest.approx(1.0, abs=1e-12)
        # u^T G u = [1,0] [[2,1],[1,2]] [1,0] = 2
        assert self.metric.inner(p, u, u) == pytest.approx(2.0, abs=1e-12)

    def test_regularization(self):
        metric_reg = geodex.PullbackMetric(self.jac_fn, self.task_fn, regularization=0.5)
        p = np.array([0.0, 0.0])
        u = np.array([1.0, 0.0])
        # Without reg: u^T G u = 2
        # With reg: 2 + 0.5 * u^T u = 2 + 0.5 = 2.5
        assert metric_reg.inner(p, u, u) == pytest.approx(2.5, abs=1e-12)

    def test_repr(self):
        assert "PullbackMetric" in repr(self.metric)


class TestConstantSPDMetric:
    def setup_method(self):
        self.A = np.diag([2.0, 3.0, 5.0])
        self.metric = geodex.ConstantSPDMetric(self.A)

    def test_inner(self):
        p = np.array([1.0, 2.0, 3.0])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert self.metric.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)
        assert self.metric.inner(p, u, u) == pytest.approx(2.0, abs=1e-12)

    def test_norm(self):
        p = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 1.0, 1.0])
        # ||v|| = sqrt(2 + 3 + 5) = sqrt(10)
        assert self.metric.norm(p, v) == pytest.approx(np.sqrt(10.0), abs=1e-12)

    def test_point_independent(self):
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([100.0, 200.0, 300.0])
        assert self.metric.inner(p1, u, v) == pytest.approx(
            self.metric.inner(p2, u, v), abs=1e-12
        )

    def test_repr(self):
        assert "ConstantSPDMetric" in repr(self.metric)


class TestWeightedMetric:
    def setup_method(self):
        self.base = geodex.KineticEnergyMetric(lambda q: np.eye(len(q)))
        self.alpha = 3.0
        self.metric = geodex.WeightedMetric(self.base, self.alpha)

    def test_inner_scaling(self):
        p = np.array([0.0, 0.0])
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        # base inner = 0, so weighted = 0
        assert self.metric.inner(p, u, v) == pytest.approx(0.0, abs=1e-12)
        # base inner(u,u) = 1, so weighted = 3
        assert self.metric.inner(p, u, u) == pytest.approx(3.0, abs=1e-12)

    def test_norm_scaling(self):
        p = np.array([0.0, 0.0])
        v = np.array([1.0, 0.0])
        # norm = sqrt(alpha * base_inner(v,v)) = sqrt(3)
        assert self.metric.norm(p, v) == pytest.approx(np.sqrt(3.0), abs=1e-12)

    def test_alpha_property(self):
        assert self.metric.alpha == pytest.approx(3.0)

    def test_weighted_constant_spd(self):
        base = geodex.ConstantSPDMetric(np.eye(2))
        weighted = geodex.WeightedMetric(base, 5.0)
        p = np.array([0.0, 0.0])
        u = np.array([1.0, 1.0])
        assert weighted.inner(p, u, u) == pytest.approx(10.0, abs=1e-12)

    def test_repr(self):
        assert "WeightedMetric" in repr(self.metric)

    def test_invalid_base_raises(self):
        with pytest.raises(Exception):
            geodex.WeightedMetric("not_a_metric", 1.0)

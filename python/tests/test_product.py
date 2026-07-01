import numpy as np
import pytest

import geodex


class TestProduct:
    def test_r3_x_se2(self):
        P = geodex.Product([geodex.Euclidean(3), geodex.SE2()])
        assert P.dim() == 6
        a = P.random_point()
        b = P.random_point()
        assert a.shape == (6,)
        v = P.log(a, b)
        ar = P.exp(a, v)
        assert P.distance(ar, b) < 1e-8
        np.testing.assert_allclose(P.geodesic(a, b, 0.0), a, atol=1e-10)

    def test_distance_decomposition(self):
        E = geodex.Euclidean(3)
        S = geodex.SE2()
        P = geodex.Product([E, S])
        a = P.random_point()
        b = P.random_point()
        de = E.distance(a[:3], b[:3])
        ds = S.distance(a[3:6], b[3:6])
        assert P.distance(a, b) == pytest.approx(np.hypot(de, ds), abs=1e-8)

    def test_r3_x_so3_point_size_differs_from_dim(self):
        P = geodex.Product([geodex.Euclidean(3), geodex.SO3()])
        assert P.dim() == 6  # 3 + 3 intrinsic
        a = P.random_point()
        assert a.shape == (7,)  # 3 + 4 ambient (quaternion block)
        assert abs(np.linalg.norm(a[3:7]) - 1.0) < 1e-9
        b = P.random_point()
        v = P.log(a, b)
        assert v.shape == (6,)  # 3 + 3 tangent
        assert P.distance(P.exp(a, v), b) < 1e-7

    def test_discrete_geodesic_returns_ndarray(self):
        P = geodex.Product([geodex.Euclidean(2), geodex.SE2()])
        a = P.random_point()
        b = P.random_point()
        r = geodex.discrete_geodesic(P, a, b)
        assert r.final_distance < 1e-2
        assert isinstance(r.path, np.ndarray)
        assert r.path.shape[1] == a.size
        assert isinstance(r.waypoints, list)

    def test_so3_through_configuration_space(self):
        # A genuine-Lie manifold composes with a custom metric via ConfigurationSpace.
        so3 = geodex.SO3()
        cs = geodex.ConfigurationSpace(so3, geodex.ConstantSPDMetric(np.eye(3)))
        q0 = so3.random_point()
        q1 = so3.random_point()
        assert cs.distance(q0, q1) > 0.0

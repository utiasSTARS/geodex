"""Tests for the geodex.pinocchio submodule.

Skipped wholesale when geodex was built without `-DGEODEX_PINOCCHIO=ON` or
the Pinocchio shared library is not on the runtime loader path.
"""

from pathlib import Path

import numpy as np
import pytest

import geodex

# `geodex.pinocchio` is exposed as an attribute (a nanobind submodule), not a
# real importable subpackage. Skip the file when geodex was built without
# Pinocchio bindings or when the Pinocchio shared library isn't on the loader
# path at import time.
if not hasattr(geodex, "pinocchio"):
    pytest.skip("geodex was built without Pinocchio bindings", allow_module_level=True)
pinocchio = geodex.pinocchio


REPO = Path(__file__).resolve().parent.parent.parent
PANDA_URDF = REPO / "examples" / "manipulator_planning" / "data" / "panda" / "urdf" / "panda.urdf"
BAXTER_URDF = REPO / "tests" / "fixtures" / "pinocchio" / "baxter_both_arms.urdf"

if not PANDA_URDF.is_file():
    pytest.skip(f"Panda URDF fixture missing at {PANDA_URDF}", allow_module_level=True)


# ---------------------------------------------------------------------------
# MassMatrix
# ---------------------------------------------------------------------------


class TestMassMatrix:
    def test_loads_panda(self):
        mm = pinocchio.MassMatrix(str(PANDA_URDF))
        assert mm is not None

    def test_shape_matches_nq(self):
        mm = pinocchio.MassMatrix(str(PANDA_URDF))
        nq = pinocchio.model_nq(str(PANDA_URDF))
        M = mm(np.zeros(nq))
        assert M.shape == (nq, nq)

    def test_spd_at_random_q(self):
        nq = pinocchio.model_nq(str(PANDA_URDF))
        mm = pinocchio.MassMatrix(str(PANDA_URDF))
        rng = np.random.default_rng(0)
        for _ in range(5):
            q = rng.uniform(-1.0, 1.0, size=nq)
            M = mm(q)
            np.testing.assert_allclose(M, M.T, atol=1e-12)
            evals = np.linalg.eigvalsh(M)
            assert evals.min() > 1e-9

    def test_factory_returns_mass_matrix(self):
        mm = pinocchio.mass_matrix(str(PANDA_URDF))
        assert isinstance(mm, pinocchio.MassMatrix)


class TestModelNq:
    def test_panda_is_seven(self):
        assert pinocchio.model_nq(str(PANDA_URDF)) == 7


class TestJointLimits:
    def test_returns_pair_of_size_nq(self):
        lo, hi = pinocchio.joint_limits(str(PANDA_URDF))
        nq = pinocchio.model_nq(str(PANDA_URDF))
        assert lo.shape == (nq,)
        assert hi.shape == (nq,)
        assert (lo <= hi).all()


# ---------------------------------------------------------------------------
# FrameJacobian
# ---------------------------------------------------------------------------


class TestFrameJacobian:
    def test_panda_full_shape(self):
        nq = pinocchio.model_nq(str(PANDA_URDF))
        J_fn = pinocchio.frame_jacobian(str(PANDA_URDF))
        J = J_fn(np.zeros(nq))
        assert J.shape == (6, nq)

    def test_panda_position_shape(self):
        nq = pinocchio.model_nq(str(PANDA_URDF))
        J_fn = pinocchio.frame_position_jacobian(str(PANDA_URDF))
        J = J_fn(np.zeros(nq))
        assert J.shape == (3, nq)

    def test_baxter_stacked_shape(self):
        if not BAXTER_URDF.is_file():
            pytest.skip("Baxter URDF fixture missing")
        nq = pinocchio.model_nq(str(BAXTER_URDF))
        J_fn = pinocchio.stacked_jacobian(
            str(BAXTER_URDF), ["left_hand_link", "right_hand_link"]
        )
        J = J_fn(np.zeros(nq))
        assert J.shape == (12, nq)

    def test_unknown_frame_raises(self):
        with pytest.raises(Exception):
            pinocchio.frame_jacobian(str(PANDA_URDF), "not_a_frame_name")


# ---------------------------------------------------------------------------
# PullbackOptions + make_pullback_metric*
# ---------------------------------------------------------------------------


class TestPullbackOptions:
    def test_defaults(self):
        opts = pinocchio.PullbackOptions()
        assert opts.ee_frames == []
        assert list(opts.task_weights) == pytest.approx([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])

    def test_field_mutation(self):
        opts = pinocchio.PullbackOptions()
        opts.ee_frames = ["panda_link8"]
        opts.task_weights = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        assert opts.ee_frames == ["panda_link8"]
        assert list(opts.task_weights) == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


class TestMakePullbackMetric:
    def test_unregularized_returns_pullback_metric(self):
        opts = pinocchio.PullbackOptions()
        metric = pinocchio.make_pullback_metric(str(PANDA_URDF), opts)
        assert isinstance(metric, geodex.PullbackMetric)

    def test_iso_returns_affine_combined(self):
        opts = pinocchio.PullbackOptions()
        metric = pinocchio.make_pullback_metric_iso(str(PANDA_URDF), opts, 0.5)
        assert isinstance(metric, geodex.AffineCombinedMetric)
        assert metric.size == 2
        assert metric.coeffs == pytest.approx([1.0, 0.5])

    def test_ke_returns_affine_combined(self):
        opts = pinocchio.PullbackOptions()
        metric = pinocchio.make_pullback_metric_ke(str(PANDA_URDF), opts, 0.3)
        assert isinstance(metric, geodex.AffineCombinedMetric)
        assert metric.size == 2
        assert metric.coeffs == pytest.approx([1.0, 0.3])

    def test_iso_dominates_unregularized(self):
        opts = pinocchio.PullbackOptions()
        nq = pinocchio.model_nq(str(PANDA_URDF))
        unreg = pinocchio.make_pullback_metric(str(PANDA_URDF), opts)
        iso = pinocchio.make_pullback_metric_iso(str(PANDA_URDF), opts, 1.0)
        rng = np.random.default_rng(0)
        for _ in range(5):
            q = rng.uniform(-0.5, 0.5, size=nq)
            v = rng.standard_normal(nq)
            i_unreg = unreg.inner(q, v, v)
            i_iso = iso.inner(q, v, v)
            # iso = pullback + 1.0 * ||v||² ≥ pullback (since lambda=1 here)
            assert i_iso >= i_unreg - 1e-9
            assert i_iso == pytest.approx(i_unreg + v.dot(v), rel=1e-9)

    def test_iso_lambda_must_be_positive(self):
        opts = pinocchio.PullbackOptions()
        with pytest.raises(Exception):
            pinocchio.make_pullback_metric_iso(str(PANDA_URDF), opts, 0.0)

    def test_ke_beta_must_be_positive(self):
        opts = pinocchio.PullbackOptions()
        with pytest.raises(Exception):
            pinocchio.make_pullback_metric_ke(str(PANDA_URDF), opts, -0.1)

    def test_pullback_composes_with_configuration_space(self):
        opts = pinocchio.PullbackOptions()
        nq = pinocchio.model_nq(str(PANDA_URDF))
        metric = pinocchio.make_pullback_metric_ke(str(PANDA_URDF), opts, 0.5)
        cs = geodex.ConfigurationSpace(geodex.Euclidean(nq), metric)
        v = np.zeros(nq)
        v[0] = 1.0
        # Sanity: inner product positive at the origin
        assert cs.inner(np.zeros(nq), v, v) > 0.0

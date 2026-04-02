/// @file test_metrics.cpp
/// @brief Tests for extracted and new metric types.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <geodex/metrics/constant_spd.hpp>
#include <geodex/metrics/jacobi.hpp>
#include <geodex/metrics/kinetic_energy.hpp>
#include <geodex/metrics/pullback.hpp>
#include <geodex/metrics/se2_left_invariant.hpp>
#include <geodex/metrics/weighted.hpp>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check that a metric is symmetric: inner(p, u, v) == inner(p, v, u).
template <typename Metric, typename Point, typename Tangent>
void check_symmetry(const Metric& m, const Point& p, const Tangent& u, const Tangent& v) {
  EXPECT_NEAR(m.inner(p, u, v), m.inner(p, v, u), 1e-12);
}

/// Check bilinearity: inner(p, alpha*u + beta*w, v) == alpha*inner(p,u,v) + beta*inner(p,w,v).
template <typename Metric, typename Point, typename Tangent>
void check_bilinearity(const Metric& m, const Point& p, const Tangent& u, const Tangent& v,
                       const Tangent& w, double alpha, double beta) {
  Tangent lhs_arg = alpha * u + beta * w;
  double lhs = m.inner(p, lhs_arg, v);
  double rhs = alpha * m.inner(p, u, v) + beta * m.inner(p, w, v);
  EXPECT_NEAR(lhs, rhs, 1e-10);
}

/// Check positive-definiteness: inner(p, v, v) > 0 for v != 0.
template <typename Metric, typename Point, typename Tangent>
void check_positive_definite(const Metric& m, const Point& p, const Tangent& v) {
  EXPECT_GT(m.inner(p, v, v), 0.0);
}

// ---------------------------------------------------------------------------
// ConstantSPDMetric
// ---------------------------------------------------------------------------

TEST(ConstantSPDMetric, InnerProductProperties) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(2, 2) = 4.0;
  A(0, 1) = 0.5;
  A(1, 0) = 0.5;
  geodex::ConstantSPDMetric<3> metric{A};

  Eigen::Vector3d p(1, 0, 0);
  Eigen::Vector3d u(0, 1, 0.5);
  Eigen::Vector3d v(0, 0.3, 0.7);
  Eigen::Vector3d w(0, -0.2, 0.4);

  check_symmetry(metric, p, u, v);
  check_bilinearity(metric, p, u, v, w, 2.5, -1.3);
  check_positive_definite(metric, p, u);
}

TEST(ConstantSPDMetric, WorksWithDim2) {
  Eigen::Matrix2d A;
  A << 2.0, 0.5, 0.5, 3.0;
  geodex::ConstantSPDMetric<2> metric{A};

  Eigen::Vector2d p(1.0, 2.0);
  Eigen::Vector2d u(1.0, 0.5);
  Eigen::Vector2d v(0.3, -0.7);

  // u^T A v = [1, 0.5] * [[2, 0.5], [0.5, 3]] * [0.3, -0.7]
  double expected = u.dot(A * v);
  EXPECT_NEAR(metric.inner(p, u, v), expected, 1e-12);
}

// ---------------------------------------------------------------------------
// SE2LeftInvariantMetric
// ---------------------------------------------------------------------------

TEST(SE2LeftInvariantMetric, InnerProductProperties) {
  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};

  Eigen::Vector3d p(1.0, 2.0, 0.5);
  Eigen::Vector3d u(1.0, 0.5, 0.3);
  Eigen::Vector3d v(0.3, -0.7, 0.1);
  Eigen::Vector3d w(-0.2, 0.4, -0.5);

  check_symmetry(metric, p, u, v);
  check_bilinearity(metric, p, u, v, w, 2.5, -1.3);
  check_positive_definite(metric, p, u);
}

// ---------------------------------------------------------------------------
// KineticEnergyMetric
// ---------------------------------------------------------------------------

TEST(KineticEnergyMetric, InnerProductProperties) {
  auto mass_fn = [](const Eigen::Vector2d& q) {
    double c2 = std::cos(q[1]);
    Eigen::Matrix2d M;
    M(0, 0) = 2.0 + c2;
    M(0, 1) = 0.5 + 0.3 * c2;
    M(1, 0) = M(0, 1);
    M(1, 1) = 1.0;
    return M;
  };

  geodex::KineticEnergyMetric metric{mass_fn};

  Eigen::Vector2d q(0.5, 1.0);
  Eigen::Vector2d u(1.0, 0.5);
  Eigen::Vector2d v(0.3, -0.7);
  Eigen::Vector2d w(-0.2, 0.4);

  check_symmetry(metric, q, u, v);
  check_bilinearity(metric, q, u, v, w, 2.5, -1.3);
  check_positive_definite(metric, q, u);
}

TEST(KineticEnergyMetric, NormConsistent) {
  auto mass_fn = [](const Eigen::Vector2d& /*q*/) { return Eigen::Matrix2d::Identity(); };
  geodex::KineticEnergyMetric metric{mass_fn};

  Eigen::Vector2d q(0.0, 0.0);
  Eigen::Vector2d v(3.0, 4.0);

  EXPECT_NEAR(metric.norm(q, v), 5.0, 1e-12);
}

// ---------------------------------------------------------------------------
// JacobiMetric
// ---------------------------------------------------------------------------

TEST(JacobiMetric, InnerProductProperties) {
  auto mass_fn = [](const Eigen::Vector2d& /*q*/) {
    Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
    M(0, 0) = 2.0;
    return M;
  };
  auto pot_fn = [](const Eigen::Vector2d& q) { return 0.5 * q.squaredNorm(); };

  double H = 10.0;
  geodex::JacobiMetric metric{mass_fn, pot_fn, H};

  Eigen::Vector2d q(1.0, 1.0);
  Eigen::Vector2d u(1.0, 0.5);
  Eigen::Vector2d v(0.3, -0.7);
  Eigen::Vector2d w(-0.2, 0.4);

  check_symmetry(metric, q, u, v);
  check_bilinearity(metric, q, u, v, w, 2.5, -1.3);
  check_positive_definite(metric, q, u);
}

TEST(JacobiMetric, ScalesWithEnergy) {
  auto mass_fn = [](const Eigen::Vector2d& /*q*/) { return Eigen::Matrix2d::Identity(); };
  auto pot_fn = [](const Eigen::Vector2d& /*q*/) { return 1.0; };

  geodex::JacobiMetric metric1{mass_fn, pot_fn, 3.0};   // factor = 2*(3-1) = 4
  geodex::JacobiMetric metric2{mass_fn, pot_fn, 6.0};   // factor = 2*(6-1) = 10

  Eigen::Vector2d q(0, 0);
  Eigen::Vector2d v(1, 0);

  EXPECT_NEAR(metric1.inner(q, v, v), 4.0, 1e-12);
  EXPECT_NEAR(metric2.inner(q, v, v), 10.0, 1e-12);
}

// ---------------------------------------------------------------------------
// PullbackMetric
// ---------------------------------------------------------------------------

TEST(PullbackMetric, InnerProductProperties) {
  // 2D -> 3D Jacobian
  auto jac_fn = [](const Eigen::Vector2d& /*q*/) {
    Eigen::Matrix<double, 3, 2> J;
    J << 1.0, 0.0, 0.0, 1.0, 0.5, 0.5;
    return J;
  };
  auto task_metric_fn = [](const Eigen::Vector2d& /*q*/) {
    return Eigen::Matrix3d::Identity();
  };

  geodex::PullbackMetric metric{jac_fn, task_metric_fn};

  Eigen::Vector2d q(1.0, 2.0);
  Eigen::Vector2d u(1.0, 0.5);
  Eigen::Vector2d v(0.3, -0.7);
  Eigen::Vector2d w(-0.2, 0.4);

  check_symmetry(metric, q, u, v);
  check_bilinearity(metric, q, u, v, w, 2.5, -1.3);
  check_positive_definite(metric, q, u);
}

TEST(PullbackMetric, WithRegularization) {
  auto jac_fn = [](const Eigen::Vector2d& /*q*/) {
    // Rank-1 Jacobian (singular)
    Eigen::Matrix<double, 1, 2> J;
    J << 1.0, 0.0;
    return J;
  };
  auto task_metric_fn = [](const Eigen::Vector2d& /*q*/) {
    Eigen::Matrix<double, 1, 1> G;
    G << 1.0;
    return G;
  };

  // Without regularization: J^T G J = [[1,0],[0,0]], not positive definite
  geodex::PullbackMetric metric_noreg{jac_fn, task_metric_fn, 0.0};
  Eigen::Vector2d q(0, 0);
  Eigen::Vector2d v_null(0, 1);  // in null space of J
  EXPECT_NEAR(metric_noreg.inner(q, v_null, v_null), 0.0, 1e-12);

  // With regularization: J^T G J + lambda*I is positive definite
  geodex::PullbackMetric metric_reg{jac_fn, task_metric_fn, 0.1};
  EXPECT_GT(metric_reg.inner(q, v_null, v_null), 0.0);
}

// ---------------------------------------------------------------------------
// WeightedMetric
// ---------------------------------------------------------------------------

TEST(WeightedMetric, ScalesBaseMetric) {
  geodex::SE2LeftInvariantMetric base{1.0, 1.0, 1.0};
  geodex::WeightedMetric metric{base, 3.0};

  Eigen::Vector3d p(0, 0, 0);
  Eigen::Vector3d v(1, 0, 0);

  EXPECT_NEAR(metric.inner(p, v, v), 3.0, 1e-12);
  EXPECT_NEAR(metric.norm(p, v), std::sqrt(3.0), 1e-12);
}

TEST(WeightedMetric, InnerProductProperties) {
  Eigen::Matrix2d A;
  A << 2.0, 0.5, 0.5, 3.0;
  geodex::ConstantSPDMetric<2> base{A};
  geodex::WeightedMetric metric{base, 2.5};

  Eigen::Vector2d p(1.0, 2.0);
  Eigen::Vector2d u(1.0, 0.5);
  Eigen::Vector2d v(0.3, -0.7);
  Eigen::Vector2d w(-0.2, 0.4);

  check_symmetry(metric, p, u, v);
  check_bilinearity(metric, p, u, v, w, 2.5, -1.3);
  check_positive_definite(metric, p, u);
}

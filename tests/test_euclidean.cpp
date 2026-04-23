#include <cmath>

#include <limits>
#include <planar_manipulator_metric.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "geodex/geodex.hpp"

using namespace geodex;

// Compile-time concept checks
static_assert(RiemannianManifold<Euclidean<3>>);
static_assert(RiemannianManifold<Euclidean<Eigen::Dynamic>>);
static_assert(HasInjectivityRadius<Euclidean<3>>);
static_assert(HasInjectivityRadius<Euclidean<Eigen::Dynamic>>);
static_assert(RiemannianManifold<Euclidean<3, ConstantSPDMetric<3>>>);
static_assert(RiemannianManifold<Euclidean<2, PlanarManipulatorMetric>>);
static_assert(HasInjectivityRadius<Euclidean<2, PlanarManipulatorMetric>>);

// ---------------------------------------------------------------------------
// Fixed-dimension R^3
// ---------------------------------------------------------------------------

class EuclideanR3Test : public ::testing::Test {
 protected:
  Euclidean<3> manifold;
};

TEST_F(EuclideanR3Test, Dim) { EXPECT_EQ(manifold.dim(), 3); }

TEST_F(EuclideanR3Test, RandomPointDimension) {
  auto p = manifold.random_point();
  EXPECT_EQ(p.size(), 3);
}

TEST_F(EuclideanR3Test, ExpLogRoundTrip) {
  Eigen::Vector3d p(1.0, 2.0, 3.0);
  Eigen::Vector3d v(0.5, -0.3, 0.1);

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-12);
}

TEST_F(EuclideanR3Test, ExpIsTranslation) {
  Eigen::Vector3d p(1.0, 0.0, 0.0);
  Eigen::Vector3d v(0.0, 1.0, 0.0);

  auto q = manifold.exp(p, v);
  Eigen::Vector3d expected(1.0, 1.0, 0.0);
  EXPECT_NEAR((q - expected).norm(), 0.0, 1e-12);
}

TEST_F(EuclideanR3Test, LogIsDifference) {
  Eigen::Vector3d p(1.0, 2.0, 3.0);
  Eigen::Vector3d q(4.0, 5.0, 6.0);

  auto v = manifold.log(p, q);
  Eigen::Vector3d expected(3.0, 3.0, 3.0);
  EXPECT_NEAR((v - expected).norm(), 0.0, 1e-12);
}

TEST_F(EuclideanR3Test, DistanceCorrectness) {
  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d q(3.0, 4.0, 0.0);

  EXPECT_NEAR(manifold.distance(p, q), 5.0, 1e-12);
}

TEST_F(EuclideanR3Test, DistanceSymmetry) {
  Eigen::Vector3d p(1.0, 2.0, 3.0);
  Eigen::Vector3d q(4.0, 5.0, 6.0);

  EXPECT_NEAR(manifold.distance(p, q), manifold.distance(q, p), 1e-12);
}

TEST_F(EuclideanR3Test, DistanceZeroSamePoint) {
  Eigen::Vector3d p(1.0, 2.0, 3.0);
  EXPECT_NEAR(manifold.distance(p, p), 0.0, 1e-12);
}

TEST_F(EuclideanR3Test, GeodesicEndpoints) {
  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d q(1.0, 2.0, 3.0);

  auto start = manifold.geodesic(p, q, 0.0);
  auto end = manifold.geodesic(p, q, 1.0);

  EXPECT_NEAR((start - p).norm(), 0.0, 1e-12);
  EXPECT_NEAR((end - q).norm(), 0.0, 1e-12);
}

TEST_F(EuclideanR3Test, GeodesicMidpoint) {
  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d q(2.0, 4.0, 6.0);

  auto mid = manifold.geodesic(p, q, 0.5);
  Eigen::Vector3d expected(1.0, 2.0, 3.0);
  EXPECT_NEAR((mid - expected).norm(), 0.0, 1e-12);
}

TEST_F(EuclideanR3Test, InnerProduct) {
  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d u(1.0, 0.0, 0.0);
  Eigen::Vector3d v(0.0, 1.0, 0.0);

  EXPECT_NEAR(manifold.inner(p, u, v), 0.0, 1e-12);
  EXPECT_NEAR(manifold.inner(p, u, u), 1.0, 1e-12);
}

TEST_F(EuclideanR3Test, Norm) {
  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d v(3.0, 4.0, 0.0);

  EXPECT_NEAR(manifold.norm(p, v), 5.0, 1e-12);
}

TEST_F(EuclideanR3Test, InjectivityRadius) {
  EXPECT_EQ(manifold.injectivity_radius(), std::numeric_limits<double>::infinity());
}

// ---------------------------------------------------------------------------
// Dynamic-dimension
// ---------------------------------------------------------------------------

class EuclideanDynamicTest : public ::testing::Test {
 protected:
  Euclidean<> manifold{5};
};

TEST_F(EuclideanDynamicTest, Dim) { EXPECT_EQ(manifold.dim(), 5); }

TEST_F(EuclideanDynamicTest, RandomPointDimension) {
  auto p = manifold.random_point();
  EXPECT_EQ(p.size(), 5);
}

TEST_F(EuclideanDynamicTest, ExpLogRoundTrip) {
  Eigen::VectorXd p = Eigen::VectorXd::LinSpaced(5, 0.0, 4.0);
  Eigen::VectorXd v = Eigen::VectorXd::Ones(5) * 0.5;

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-12);
}

TEST_F(EuclideanDynamicTest, DistanceCorrectness) {
  Eigen::VectorXd p = Eigen::VectorXd::Zero(5);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(5);
  q[0] = 3.0;
  q[1] = 4.0;

  EXPECT_NEAR(manifold.distance(p, q), 5.0, 1e-12);
}

TEST_F(EuclideanDynamicTest, GeodesicMidpoint) {
  Eigen::VectorXd p = Eigen::VectorXd::Zero(5);
  Eigen::VectorXd q = Eigen::VectorXd::Ones(5) * 2.0;

  auto mid = manifold.geodesic(p, q, 0.5);
  Eigen::VectorXd expected = Eigen::VectorXd::Ones(5);
  EXPECT_NEAR((mid - expected).norm(), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Anisotropic metric on R^3
// ---------------------------------------------------------------------------

class EuclideanAnisotropicTest : public ::testing::Test {
 protected:
  Eigen::Matrix3d A = Eigen::Vector3d(4.0, 1.0, 1.0).asDiagonal();
  Euclidean<3, ConstantSPDMetric<3>> manifold{ConstantSPDMetric<3>{A}};
  Euclidean<3> standard;
};

TEST_F(EuclideanAnisotropicTest, IdentityMatchesStandard) {
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Euclidean<3, ConstantSPDMetric<3>> iso{ConstantSPDMetric<3>{I}};

  Eigen::Vector3d p(1.0, 2.0, 3.0);
  Eigen::Vector3d q(4.0, 5.0, 6.0);

  EXPECT_NEAR(iso.distance(p, q), standard.distance(p, q), 1e-12);
}

TEST_F(EuclideanAnisotropicTest, StretchedAxisLargerDistance) {
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  // Move along x (stretched axis, weight=4)
  Eigen::Vector3d qx(1.0, 0.0, 0.0);
  // Move along y (weight=1)
  Eigen::Vector3d qy(0.0, 1.0, 0.0);

  // d(p, qx) = sqrt(1*4*1) = 2, d(p, qy) = sqrt(1*1*1) = 1
  EXPECT_NEAR(manifold.distance(p, qx), 2.0, 1e-12);
  EXPECT_NEAR(manifold.distance(p, qy), 1.0, 1e-12);
  EXPECT_GT(manifold.distance(p, qx), standard.distance(p, qx));
}

TEST_F(EuclideanAnisotropicTest, DistanceSymmetry) {
  Eigen::Vector3d p(1.0, 2.0, 3.0);
  Eigen::Vector3d q(4.0, 5.0, 6.0);
  EXPECT_NEAR(manifold.distance(p, q), manifold.distance(q, p), 1e-12);
}

TEST_F(EuclideanAnisotropicTest, ExpLogUnchanged) {
  Eigen::Vector3d p(1.0, 2.0, 3.0);
  Eigen::Vector3d v(0.5, -0.3, 0.1);

  // exp/log are metric-independent for Euclidean space
  auto q = manifold.exp(p, v);
  EXPECT_NEAR((q - (p + v)).norm(), 0.0, 1e-12);

  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-12);
}

TEST_F(EuclideanAnisotropicTest, GeodesicUnchanged) {
  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d q(2.0, 4.0, 6.0);

  // Geodesic is still linear interpolation
  auto mid = manifold.geodesic(p, q, 0.5);
  Eigen::Vector3d expected(1.0, 2.0, 3.0);
  EXPECT_NEAR((mid - expected).norm(), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Mass-inertia matrix metric (2-link planar arm)
// ---------------------------------------------------------------------------

class PlanarManipulatorMetricTest : public ::testing::Test {
 protected:
  PlanarManipulatorMetric metric;
  Euclidean<2, PlanarManipulatorMetric> manifold{metric};
};

TEST_F(PlanarManipulatorMetricTest, MassMatrixSPD) {
  // Test SPD at several configurations
  std::vector<Eigen::Vector2d> configs = {
      {0.0, 0.0}, {1.0, 0.5}, {-1.0, M_PI}, {0.5, -M_PI / 2}, {2.0, M_PI / 3}};

  for (const auto& q : configs) {
    Eigen::Matrix2d M = metric.mass_matrix(q);
    // Symmetric
    EXPECT_NEAR(M(0, 1), M(1, 0), 1e-12);
    // Positive eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(M);
    EXPECT_GT(es.eigenvalues()(0), 0.0) << "at q = (" << q[0] << ", " << q[1] << ")";
    EXPECT_GT(es.eigenvalues()(1), 0.0) << "at q = (" << q[0] << ", " << q[1] << ")";
  }
}

TEST_F(PlanarManipulatorMetricTest, CouplingVariesWithQ2) {
  // Off-diagonal coupling is maximal at q2=0, minimal at q2=±pi
  Eigen::Vector2d q_zero(0.0, 0.0);
  Eigen::Vector2d q_pi(0.0, M_PI);

  double coupling_zero = std::abs(metric.mass_matrix(q_zero)(0, 1));
  double coupling_pi = std::abs(metric.mass_matrix(q_pi)(0, 1));

  EXPECT_GT(coupling_zero, coupling_pi);
}

TEST_F(PlanarManipulatorMetricTest, M22IsConstant) {
  // M22 = I2 + m2*lc2^2 — independent of q
  double m22_a = metric.mass_matrix(Eigen::Vector2d(0.0, 0.0))(1, 1);
  double m22_b = metric.mass_matrix(Eigen::Vector2d(1.0, 1.5))(1, 1);
  double m22_c = metric.mass_matrix(Eigen::Vector2d(-0.5, M_PI))(1, 1);

  EXPECT_NEAR(m22_a, m22_b, 1e-12);
  EXPECT_NEAR(m22_b, m22_c, 1e-12);
}

TEST_F(PlanarManipulatorMetricTest, DistanceSymmetry) {
  Eigen::Vector2d p(0.5, 1.0);
  Eigen::Vector2d q(2.0, -0.5);

  EXPECT_NEAR(manifold.distance(p, q), manifold.distance(q, p), 1e-10);
}

TEST_F(PlanarManipulatorMetricTest, DistanceZeroSamePoint) {
  Eigen::Vector2d p(1.0, 0.5);
  EXPECT_NEAR(manifold.distance(p, p), 0.0, 1e-12);
}

TEST_F(PlanarManipulatorMetricTest, ExpLogUnchanged) {
  Eigen::Vector2d p(1.0, 0.5);
  Eigen::Vector2d v(0.3, -0.2);

  auto q = manifold.exp(p, v);
  EXPECT_NEAR((q - (p + v)).norm(), 0.0, 1e-12);

  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-12);
}

TEST_F(PlanarManipulatorMetricTest, InjectivityRadius) {
  EXPECT_EQ(manifold.injectivity_radius(), std::numeric_limits<double>::infinity());
}

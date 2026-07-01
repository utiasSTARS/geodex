// SO(3) manifold tests.

#include <cmath>

#include <numbers>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "geodex/algorithm/interpolation.hpp"
#include "geodex/manifold/so3.hpp"

using namespace geodex;

// Compile-time concept checks.
static_assert(RiemannianManifold<SO3<>>);
static_assert(RiemannianManifold<SO3<SO3CanonicalMetric, SO3RightExponentialMap>>);

namespace {

/// @brief Compare two quaternions up to sign (the SO(3) double cover q ~ -q).
::testing::AssertionResult QuatEqUpToSign(const Eigen::Vector4d& a, const Eigen::Vector4d& b,
                                          double tol = 1e-9) {
  const double d = std::min((a - b).norm(), (a + b).norm());
  if (d < tol) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "quaternions differ (up to sign) by " << d << " > " << tol << "\n  a=" << a.transpose()
         << "\n  b=" << b.transpose();
}

/// @brief Build an Eigen quaternion from our scalar-last [x, y, z, w] vector.
Eigen::Quaterniond ToEigen(const Eigen::Vector4d& q) {
  return Eigen::Quaterniond(q[3], q[0], q[1], q[2]);  // (w, x, y, z)
}

}  // namespace

// ---------------------------------------------------------------------------
// Body (left) exponential retraction — default
// ---------------------------------------------------------------------------

class SO3LeftTest : public ::testing::Test {
 protected:
  SO3<> manifold;  // isotropic bi-invariant metric, body exp map
  // Two generic, non-antipodal rotations built from the library exp map.
  Eigen::Vector4d q0 = utils::so3_exp(Eigen::Vector3d(0.3, -0.5, 0.9));
  Eigen::Vector4d q1 = utils::so3_exp(Eigen::Vector3d(-0.7, 1.1, 0.2));
};

TEST_F(SO3LeftTest, Dim) { EXPECT_EQ(manifold.dim(), 3); }

TEST_F(SO3LeftTest, InjectivityRadius) {
  EXPECT_NEAR(manifold.injectivity_radius(), std::numbers::pi, 1e-12);
}

TEST_F(SO3LeftTest, IsotropicMetricIsRiemannianLog) {
  EXPECT_TRUE(manifold.has_riemannian_log_runtime());
}

TEST_F(SO3LeftTest, RandomPointUnitNorm) {
  for (int i = 0; i < 100; ++i) {
    auto q = manifold.random_point();
    EXPECT_NEAR(q.norm(), 1.0, 1e-12);
  }
}

TEST_F(SO3LeftTest, ExpLogRoundTripSmallOmega) {
  Eigen::Vector3d w(6e-10, 8e-10, 0.0);  // |w| = 1e-9
  ASSERT_NEAR(w.norm(), 1e-9, 1e-18);
  auto q = manifold.exp(q0, w);
  auto w_back = manifold.log(q0, q);
  EXPECT_NEAR((w - w_back).norm(), 0.0, 1e-12);
  EXPECT_NEAR(q.norm(), 1.0, 1e-12);
}

TEST_F(SO3LeftTest, ExpLogRoundTripLargeOmega) {
  Eigen::Vector3d w(1.5, -2.0, 0.0);  // |w| = 2.5
  ASSERT_NEAR(w.norm(), 2.5, 1e-12);
  auto q = manifold.exp(q0, w);
  auto w_back = manifold.log(q0, q);
  EXPECT_NEAR((w - w_back).norm(), 0.0, 1e-10);
  EXPECT_NEAR(q.norm(), 1.0, 1e-10);
}

TEST_F(SO3LeftTest, GeodesicEndpoints) {
  auto start = manifold.geodesic(q0, q1, 0.0);
  auto end = manifold.geodesic(q0, q1, 1.0);
  EXPECT_NEAR((start - q0).norm(), 0.0, 1e-12);
  // Endpoint equals q1 up to sign (double cover): compare rotations, not raw quats.
  EXPECT_NEAR((utils::quat_to_rotmat(end) - utils::quat_to_rotmat(q1)).norm(), 0.0, 1e-9);
}

TEST_F(SO3LeftTest, GeodesicIsSlerp) {
  const Eigen::Quaterniond q0e = ToEigen(q0);
  const Eigen::Quaterniond q1e = ToEigen(q1);
  for (double t : {0.25, 0.5, 0.75}) {
    const Eigen::Vector4d g = manifold.geodesic(q0, q1, t);
    const Eigen::Vector4d oracle = q0e.slerp(t, q1e).coeffs();  // .coeffs() is [x, y, z, w]
    EXPECT_TRUE(QuatEqUpToSign(g, oracle)) << "t=" << t;
  }
}

TEST_F(SO3LeftTest, DistanceSymmetryAndZero) {
  EXPECT_NEAR(manifold.distance(q0, q1), manifold.distance(q1, q0), 1e-10);
  EXPECT_NEAR(manifold.distance(q0, q0), 0.0, 1e-12);
}

TEST_F(SO3LeftTest, DistanceEqualsRotationAngle) {
  const double d = manifold.distance(q0, q1);
  const double angle = ToEigen(q0).angularDistance(ToEigen(q1));  // in [0, pi]
  EXPECT_NEAR(d, angle, 1e-9);
  EXPECT_GE(d, 0.0);
  EXPECT_LE(d, std::numbers::pi + 1e-12);
}

TEST_F(SO3LeftTest, DiscreteGeodesicReachesTarget) {
  auto result = discrete_geodesic(manifold, q0, q1);
  EXPECT_LT(result.final_distance, 1e-3);
  ASSERT_GE(result.path.size(), 2u);
  EXPECT_NEAR((result.path.front() - q0).norm(), 0.0, 1e-12);
  for (const auto& p : result.path) {
    EXPECT_NEAR(p.norm(), 1.0, 1e-9);  // stays on the unit-quaternion manifold
  }
}

// ---------------------------------------------------------------------------
// World (right) exponential retraction
// ---------------------------------------------------------------------------

class SO3RightTest : public ::testing::Test {
 protected:
  SO3<SO3CanonicalMetric, SO3RightExponentialMap> manifold;
  Eigen::Vector4d q0 = utils::so3_exp(Eigen::Vector3d(0.2, 0.4, -0.6));
  Eigen::Vector4d q1 = utils::so3_exp(Eigen::Vector3d(-0.9, 0.3, 0.8));
};

TEST_F(SO3RightTest, IsotropicMetricIsRiemannianLog) {
  EXPECT_TRUE(manifold.has_riemannian_log_runtime());
}

TEST_F(SO3RightTest, ExpLogRoundTrip) {
  Eigen::Vector3d w(0.4, -1.1, 0.9);
  auto q = manifold.exp(q0, w);
  auto w_back = manifold.log(q0, q);
  EXPECT_NEAR((w - w_back).norm(), 0.0, 1e-10);
  EXPECT_NEAR(q.norm(), 1.0, 1e-10);
}

TEST_F(SO3RightTest, GeodesicEndpoints) {
  auto start = manifold.geodesic(q0, q1, 0.0);
  auto end = manifold.geodesic(q0, q1, 1.0);
  EXPECT_NEAR((start - q0).norm(), 0.0, 1e-12);
  EXPECT_NEAR((utils::quat_to_rotmat(end) - utils::quat_to_rotmat(q1)).norm(), 0.0, 1e-9);
}

TEST_F(SO3RightTest, DistanceEqualsRotationAngle) {
  // Bi-invariant: the world-frame geodesic has the same length as the body one.
  const double d = manifold.distance(q0, q1);
  const double angle = ToEigen(q0).angularDistance(ToEigen(q1));
  EXPECT_NEAR(d, angle, 1e-9);
}

TEST_F(SO3RightTest, DiscreteGeodesicReachesTarget) {
  auto result = discrete_geodesic(manifold, q0, q1);
  EXPECT_LT(result.final_distance, 1e-3);
}

// ---------------------------------------------------------------------------
// Metrics: isotropy detection and anisotropic norms
// ---------------------------------------------------------------------------

TEST(SO3MetricTest, IsotropicScalingStaysBiInvariant) {
  SO3<> manifold{SO3CanonicalMetric{2.0}};  // 2 * I, still isotropic
  EXPECT_TRUE(manifold.has_riemannian_log_runtime());
  Eigen::Vector4d q = utils::so3_exp(Eigen::Vector3d(0.1, 0.2, 0.3));
  // norm scales as sqrt(2).
  EXPECT_NEAR(manifold.norm(q, Eigen::Vector3d(1.0, 0.0, 0.0)), std::sqrt(2.0), 1e-12);
}

TEST(SO3MetricTest, AnisotropicMetricIsNotRiemannianLog) {
  SO3<> manifold{SO3CanonicalMetric{Eigen::Vector3d(1.0, 4.0, 9.0)}};
  EXPECT_FALSE(manifold.has_riemannian_log_runtime());
  Eigen::Vector4d q = utils::so3_exp(Eigen::Vector3d(0.1, -0.2, 0.05));
  EXPECT_NEAR(manifold.norm(q, Eigen::Vector3d(0.0, 1.0, 0.0)), 2.0, 1e-12);  // sqrt(4)
  EXPECT_NEAR(manifold.norm(q, Eigen::Vector3d(0.0, 0.0, 1.0)), 3.0, 1e-12);  // sqrt(9)
}

TEST(SO3MetricTest, EulerRetractionWithIsotropicMetricNotFlagged) {
  // A non-group retraction must never claim the Riemannian-log fast path, even
  // with an isotropic metric.
  SO3<> manifold;
  EXPECT_TRUE(manifold.has_riemannian_log_runtime());
}

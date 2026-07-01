#include <cmath>

#include <algorithm>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/algorithm/interpolation.hpp"
#include "geodex/manifold/se3.hpp"

using namespace geodex;
using namespace geodex::utils;

using Point7 = Eigen::Matrix<double, 7, 1>;
using Tangent6 = Eigen::Matrix<double, 6, 1>;

// Compile-time concept checks (both retractions).
static_assert(RiemannianManifold<SE3<>>);
static_assert(RiemannianManifold<SE3<SE3InvariantMetric, SE3RightExponentialMap>>);
static_assert(Retraction<SE3LeftExponentialMap, Point7, Tangent6>);
static_assert(Retraction<SE3RightExponentialMap, Point7, Tangent6>);

namespace {

/// @brief Build an SE(3) element from a translation and an axis-angle rotation.
Point7 make_pose(const Eigen::Vector3d& t, const Eigen::Vector3d& omega) {
  Point7 g;
  g.head<3>() = t;
  g.tail<4>() = so3_exp(omega);
  return g;
}

/// @brief Compare two unit quaternions up to the double-cover sign.
bool quat_close(const Eigen::Vector4d& a, const Eigen::Vector4d& b, double tol) {
  return std::min((a - b).norm(), (a + b).norm()) < tol;
}

}  // namespace

// ---------------------------------------------------------------------------
// Left (body) exponential retraction — the default
// ---------------------------------------------------------------------------

class SE3LeftTest : public ::testing::Test {
 protected:
  SE3<> manifold;
  // A non-trivial base pose (non-zero translation, non-trivial rotation).
  Point7 g0 = make_pose({1.0, 2.0, 3.0}, {0.3, -0.4, 0.5});
};

TEST_F(SE3LeftTest, Dim) { EXPECT_EQ(manifold.dim(), 6); }

TEST_F(SE3LeftTest, ExpLogRoundTripSmallOmega) {
  Tangent6 xi;
  xi << 0.3, -0.4, 0.2, 0.01, -0.02, 0.005;
  auto q = manifold.exp(g0, xi);
  auto back = manifold.log(g0, q);
  EXPECT_NEAR((xi - back).norm(), 0.0, 1e-10);
}

TEST_F(SE3LeftTest, ExpLogRoundTripZeroOmega) {
  // Pure translation twist: omega == 0.
  Tangent6 xi;
  xi << 0.5, -0.3, 0.7, 0.0, 0.0, 0.0;
  auto q = manifold.exp(g0, xi);
  auto back = manifold.log(g0, q);
  EXPECT_NEAR((xi - back).norm(), 0.0, 1e-10);
}

TEST_F(SE3LeftTest, ExpLogRoundTripLargeOmega) {
  // |omega| = sqrt(5.5) ~= 2.345 < pi, so the log recovers the same twist.
  Tangent6 xi;
  xi << 1.0, 2.0, -0.5, 1.5, -1.5, 1.0;
  auto q = manifold.exp(g0, xi);
  auto back = manifold.log(g0, q);
  EXPECT_NEAR((xi - back).norm(), 0.0, 1e-9);
}

TEST_F(SE3LeftTest, GeodesicEndpoints) {
  Point7 g1 = make_pose({4.0, 5.0, 6.0}, {0.5, -0.3, 0.4});

  auto start = manifold.geodesic(g0, g1, 0.0);
  auto end = manifold.geodesic(g0, g1, 1.0);

  // t = 0 recovers g0 exactly.
  EXPECT_NEAR((start - g0).norm(), 0.0, 1e-12);
  // t = 1: translation exact, rotation up to the quaternion double-cover sign.
  EXPECT_NEAR((end.head<3>() - g1.head<3>()).norm(), 0.0, 1e-9);
  EXPECT_TRUE(quat_close(end.tail<4>(), g1.tail<4>(), 1e-9));
}

TEST_F(SE3LeftTest, PureTranslation) {
  // xi = [v; 0] moves the translation by R(g0) * v, rotation unchanged.
  Eigen::Vector3d v(0.5, -0.3, 0.7);
  Tangent6 xi;
  xi << v, Eigen::Vector3d::Zero();

  auto q = manifold.exp(g0, xi);
  Eigen::Vector3d expected_t = g0.head<3>() + quat_rotate(g0.tail<4>(), v);
  EXPECT_NEAR((q.head<3>() - expected_t).norm(), 0.0, 1e-12);
  EXPECT_TRUE(quat_close(q.tail<4>(), g0.tail<4>(), 1e-12));
}

TEST_F(SE3LeftTest, PureRotation) {
  // xi = [0; omega] leaves translation unchanged; rotation composes on the right
  // by so3_exp(omega) (body-frame / left retraction).
  Eigen::Vector3d omega(0.2, -0.1, 0.3);
  Tangent6 xi;
  xi << Eigen::Vector3d::Zero(), omega;

  auto q = manifold.exp(g0, xi);
  EXPECT_NEAR((q.head<3>() - g0.head<3>()).norm(), 0.0, 1e-12);
  Eigen::Vector4d expected_q = quat_normalize(quat_mul(g0.tail<4>(), so3_exp(omega)));
  EXPECT_TRUE(quat_close(q.tail<4>(), expected_q, 1e-12));
}

TEST_F(SE3LeftTest, DistanceSymmetry) {
  Point7 g1 = make_pose({4.0, 5.0, 6.0}, {0.5, -0.3, 0.4});
  EXPECT_NEAR(manifold.distance(g0, g1), manifold.distance(g1, g0), 1e-9);
}

TEST_F(SE3LeftTest, DistanceZeroSamePoint) {
  EXPECT_NEAR(manifold.distance(g0, g0), 0.0, 1e-12);
}

TEST_F(SE3LeftTest, DiscreteGeodesicReachesTarget) {
  Point7 g1 = make_pose({4.0, 5.0, 6.0}, {0.5, -0.3, 0.4});

  auto result = discrete_geodesic(manifold, g0, g1);
  ASSERT_FALSE(result.path.empty());
  const Point7 last = result.path.back();

  EXPECT_EQ(result.status, InterpolationStatus::Converged);
  EXPECT_NEAR((last.head<3>() - g1.head<3>()).norm(), 0.0, 1e-6);
  EXPECT_TRUE(quat_close(last.tail<4>(), g1.tail<4>(), 1e-4));
  EXPECT_LT(result.final_distance, 1e-4);
}

// ---------------------------------------------------------------------------
// Right (spatial) exponential retraction
// ---------------------------------------------------------------------------

class SE3RightTest : public ::testing::Test {
 protected:
  SE3<SE3InvariantMetric, SE3RightExponentialMap> manifold;
  Point7 g0 = make_pose({1.0, 2.0, 3.0}, {0.3, -0.4, 0.5});
};

TEST_F(SE3RightTest, ExpLogRoundTrip) {
  Tangent6 xi;
  xi << 0.3, -0.4, 0.2, 0.1, -0.15, 0.2;
  auto q = manifold.exp(g0, xi);
  auto back = manifold.log(g0, q);
  EXPECT_NEAR((xi - back).norm(), 0.0, 1e-10);
}

TEST_F(SE3RightTest, GeodesicEndpoints) {
  Point7 g1 = make_pose({4.0, 5.0, 6.0}, {0.5, -0.3, 0.4});
  auto start = manifold.geodesic(g0, g1, 0.0);
  auto end = manifold.geodesic(g0, g1, 1.0);
  EXPECT_NEAR((start - g0).norm(), 0.0, 1e-12);
  EXPECT_NEAR((end.head<3>() - g1.head<3>()).norm(), 0.0, 1e-9);
  EXPECT_TRUE(quat_close(end.tail<4>(), g1.tail<4>(), 1e-9));
}

// ---------------------------------------------------------------------------
// Left vs Right retractions differ
// ---------------------------------------------------------------------------

TEST(SE3RetractionTest, LeftDiffersFromRight) {
  SE3<SE3InvariantMetric, SE3LeftExponentialMap> left;
  SE3<SE3InvariantMetric, SE3RightExponentialMap> right;

  // Non-central base pose and a twist with both linear and angular parts.
  Point7 g0 = make_pose({1.0, 2.0, 3.0}, {0.3, -0.4, 0.5});
  Tangent6 xi;
  xi << 0.5, -0.3, 0.2, 0.4, 0.1, -0.2;

  auto ql = left.exp(g0, xi);
  auto qr = right.exp(g0, xi);
  EXPECT_GT((ql - qr).norm(), 1e-3);
}

// ---------------------------------------------------------------------------
// Random point
// ---------------------------------------------------------------------------

TEST(SE3RandomTest, QuaternionUnitAndTranslationInBounds) {
  SE3<> manifold;
  for (int i = 0; i < 100; ++i) {
    auto p = manifold.random_point();
    // Quaternion part (indices 3..6) is a unit quaternion.
    EXPECT_NEAR(p.tail<4>().norm(), 1.0, 1e-12);
    // Translation part within the default box [0, 10]^3.
    for (int k = 0; k < 3; ++k) {
      EXPECT_GE(p[k], 0.0);
      EXPECT_LE(p[k], 10.0);
    }
  }
}

TEST(SE3RandomTest, RespectsCustomBounds) {
  SE3<> manifold(Eigen::Vector3d(-2.0, -2.0, -2.0), Eigen::Vector3d(2.0, 2.0, 2.0));
  for (int i = 0; i < 100; ++i) {
    auto p = manifold.random_point();
    EXPECT_NEAR(p.tail<4>().norm(), 1.0, 1e-12);
    for (int k = 0; k < 3; ++k) {
      EXPECT_GE(p[k], -2.0);
      EXPECT_LE(p[k], 2.0);
    }
  }
}

// ---------------------------------------------------------------------------
// Anisotropic invariant metric
// ---------------------------------------------------------------------------

TEST(SE3MetricTest, RotationalWeightScalesNorm) {
  SE3<> iso;
  SE3<> aniso{SE3InvariantMetric{1.0, 9.0}};  // w_trans = 1, w_rot = 9

  Point7 p = make_pose({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  Tangent6 pure_rot;
  pure_rot << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;

  // ||omega||_aniso = sqrt(9) * ||omega||_iso = 3 * ||omega||_iso.
  EXPECT_NEAR(aniso.norm(p, pure_rot), 3.0 * iso.norm(p, pure_rot), 1e-12);
}

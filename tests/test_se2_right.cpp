/// @file test_se2_right.cpp
/// @brief Tests for the right-invariant (world-frame) SE(2) retraction
///        `SE2RightExponentialMap`.

#include <cmath>

#include <numbers>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/algorithm/interpolation.hpp"
#include "geodex/manifold/se2.hpp"

using namespace geodex;
using namespace geodex::utils;

/// Right-invariant / world-frame SE(2).
using SE2R = SE2<SE2LeftInvariantMetric, SE2RightExponentialMap>;
/// Default body-frame SE(2), used for left-vs-right contrast checks.
using SE2L = SE2<>;

// Compile-time concept checks.
static_assert(Retraction<SE2RightExponentialMap, Eigen::Vector3d, Eigen::Vector3d>);
static_assert(RiemannianManifold<SE2R>);

namespace {

/// @brief Generic midpoint distance mirroring algorithm/distance.hpp, computed
/// purely from the manifold's own exp/log/norm.
template <typename M>
double generic_midpoint_distance(const M& m, const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  const Eigen::Vector3d v_ab = m.log(a, b);
  const Eigen::Vector3d mid = m.exp(a, 0.5 * v_ab);
  const Eigen::Vector3d v_ma = m.log(mid, a);
  const Eigen::Vector3d v_mb = m.log(mid, b);
  return m.norm(mid, Eigen::Vector3d(v_mb - v_ma));
}

}  // namespace

// ---------------------------------------------------------------------------
// Basic manifold interface
// ---------------------------------------------------------------------------

class SE2RightTest : public ::testing::Test {
 protected:
  SE2R manifold;
};

TEST_F(SE2RightTest, Dim) { EXPECT_EQ(manifold.dim(), 3); }

// ---------------------------------------------------------------------------
// exp/log roundtrip
// ---------------------------------------------------------------------------

TEST_F(SE2RightTest, ExpLogRoundTrip) {
  Eigen::Vector3d p(2.0, 3.0, 0.5);
  Eigen::Vector3d xi(0.3, -0.4, 0.2);

  auto q = manifold.exp(p, xi);
  auto xi_back = manifold.log(p, q);
  EXPECT_NEAR((xi - xi_back).norm(), 0.0, 1e-10);
}

TEST_F(SE2RightTest, ExpLogRoundTripLargeOmega) {
  Eigen::Vector3d p(1.0, 1.0, -1.0);
  Eigen::Vector3d xi(1.0, 2.0, 2.5);

  auto q = manifold.exp(p, xi);
  auto xi_back = manifold.log(p, q);
  EXPECT_NEAR((xi - xi_back).norm(), 0.0, 1e-10);
}

TEST_F(SE2RightTest, ExpLogRoundTripZeroOmega) {
  Eigen::Vector3d p(1.0, 2.0, 0.7);
  Eigen::Vector3d xi(0.5, -0.3, 0.0);

  auto q = manifold.exp(p, xi);
  auto xi_back = manifold.log(p, q);
  EXPECT_NEAR((xi - xi_back).norm(), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Geodesic endpoints
// ---------------------------------------------------------------------------

TEST_F(SE2RightTest, GeodesicEndpoints) {
  Eigen::Vector3d p(1.0, 2.0, 0.3);
  Eigen::Vector3d q(4.0, 5.0, -1.0);

  auto start = manifold.geodesic(p, q, 0.0);
  auto end = manifold.geodesic(p, q, 1.0);

  EXPECT_NEAR((start - p).norm(), 0.0, 1e-12);
  EXPECT_NEAR(end[0], q[0], 1e-10);
  EXPECT_NEAR(end[1], q[1], 1e-10);
  EXPECT_NEAR(std::abs(wrap_to_pi(end[2] - q[2])), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Left (body) vs right (world) really differ
// ---------------------------------------------------------------------------

TEST(SE2LeftVsRight, DifferForGeneralTwist) {
  SE2L left;
  SE2R right;

  // Non-identity base and a twist mixing translation + rotation.
  Eigen::Vector3d p(1.0, 2.0, 0.5);
  Eigen::Vector3d xi(0.3, -0.4, 0.7);

  auto qL = left.exp(p, xi);
  auto qR = right.exp(p, xi);
  EXPECT_GT((qL - qR).norm(), 1e-3);
}

// A pure spatial translation adds directly in the world frame, independent of the
// base orientation.
TEST(SE2LeftVsRight, RightPureTranslationIsWorldFrame) {
  SE2R right;
  Eigen::Vector3d g(2.0, 3.0, std::numbers::pi / 3.0);  // rotated base
  Eigen::Vector3d xi(1.0, 0.0, 0.0);                    // world +x translation

  auto q = right.exp(g, xi);
  EXPECT_NEAR(q[0], g[0] + 1.0, 1e-10);  // moved +1 in world x
  EXPECT_NEAR(q[1], g[1], 1e-10);        // world y unchanged
  EXPECT_NEAR(q[2], g[2], 1e-12);        // orientation unchanged
}

// ---------------------------------------------------------------------------
// Right-invariance sanity: a spatial rotation about the world origin rotates
// the base pose's POSITION about the origin (not an in-place body rotation).
// ---------------------------------------------------------------------------

TEST(SE2RightInvariance, SpatialRotationRotatesPositionAboutOrigin) {
  SE2R right;
  SE2L left;

  Eigen::Vector3d g(2.0, 0.0, 0.4);                     // base off the origin
  Eigen::Vector3d xi(0.0, 0.0, std::numbers::pi / 2.0);  // pure spatial rotation

  auto qR = right.exp(g, xi);
  // Position rotated +90 deg about the world origin: (2,0) -> (0,2).
  EXPECT_NEAR(qR[0], 0.0, 1e-10);
  EXPECT_NEAR(qR[1], 2.0, 1e-10);
  EXPECT_NEAR(qR[2], wrap_to_pi(g[2] + std::numbers::pi / 2.0), 1e-12);
  // The x,y position actually moved (distinct from an in-place rotation).
  EXPECT_GT(std::hypot(qR[0] - g[0], qR[1] - g[1]), 1e-3);

  // Body-frame map: same spatial twist leaves the position in place.
  auto qL = left.exp(g, xi);
  EXPECT_NEAR(qL[0], g[0], 1e-12);
  EXPECT_NEAR(qL[1], g[1], 1e-12);
}

// ---------------------------------------------------------------------------
// Distance
// ---------------------------------------------------------------------------

TEST(SE2RightDistance, Symmetry) {
  SE2R right;
  Eigen::Vector3d p(2.0, 3.0, 0.4);
  Eigen::Vector3d q(4.0, 1.0, -0.6);

  EXPECT_NEAR(right.distance(p, q), right.distance(q, p), 1e-10);
}

TEST(SE2RightDistance, ZeroSamePoint) {
  SE2R right;
  Eigen::Vector3d p(1.0, 2.0, 0.5);
  EXPECT_NEAR(right.distance(p, p), 0.0, 1e-12);
}

// SE2R must NOT use the fused body-frame `distance_midpoint`; it should match the
// generic midpoint computed from its own (world-frame) exp/log/norm.
TEST(SE2RightDistance, MatchesGenericMidpoint) {
  SE2R right;
  Eigen::Vector3d p(2.0, 3.0, 0.4);
  Eigen::Vector3d q(4.0, 1.0, -0.6);

  EXPECT_NEAR(right.distance(p, q), generic_midpoint_distance(right, p, q), 1e-9);
}

// With rotation present, the world-frame distance differs from the body-frame
// distance for the same endpoints and identical metric weights.
TEST(SE2RightDistance, DiffersFromLeftWithRotation) {
  SE2R right;
  SE2L left;
  Eigen::Vector3d p(2.0, 3.0, 0.4);
  Eigen::Vector3d q(4.0, 1.0, -0.6);

  EXPECT_GT(std::abs(right.distance(p, q) - left.distance(p, q)), 1e-3);
}

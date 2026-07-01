#include <cmath>

#include <numbers>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/algorithm/interpolation.hpp"
#include "geodex/manifold/so2.hpp"

using namespace geodex;
using namespace geodex::utils;

namespace {

/// @brief Build a 1-vector holding the angle `x`.
Eigen::Matrix<double, 1, 1> a1(double x) {
  Eigen::Matrix<double, 1, 1> m;
  m[0] = x;
  return m;
}

constexpr double kPi = std::numbers::pi;

}  // namespace

// Compile-time concept checks
static_assert(RiemannianManifold<SO2<>>);

// ---------------------------------------------------------------------------
// Exponential map retraction
// ---------------------------------------------------------------------------

class SO2ExpTest : public ::testing::Test {
 protected:
  SO2<> manifold;
};

TEST_F(SO2ExpTest, Dim) { EXPECT_EQ(manifold.dim(), 1); }

TEST_F(SO2ExpTest, ExpLogRoundTrip) {
  auto p = a1(0.5);
  auto v = a1(0.3);

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-12);
}

TEST_F(SO2ExpTest, ExpLogRoundTripWrapsAcrossPi) {
  // p + v = 3.5 > pi, so exp must wrap into [-pi, pi); log must recover v.
  auto p = a1(3.0);
  auto v = a1(0.5);

  auto q = manifold.exp(p, v);
  EXPECT_GE(q[0], -kPi);
  EXPECT_LT(q[0], kPi);

  auto v_back = manifold.log(p, q);
  EXPECT_NEAR(v_back[0], 0.5, 1e-12);
}

TEST_F(SO2ExpTest, GeodesicEndpoints) {
  auto p = a1(1.0);
  auto q = a1(2.0);

  auto start = manifold.geodesic(p, q, 0.0);
  auto end = manifold.geodesic(p, q, 1.0);

  EXPECT_NEAR(std::abs(wrap_to_pi(start[0] - p[0])), 0.0, 1e-12);
  EXPECT_NEAR(std::abs(wrap_to_pi(end[0] - q[0])), 0.0, 1e-12);
}

TEST_F(SO2ExpTest, GeodesicTakesShortWayThroughPi) {
  // From 3.0 to -3.0 the short arc (length ~0.283) passes through +/-pi, not 0.
  auto p = a1(3.0);
  auto q = a1(-3.0);

  auto mid = manifold.geodesic(p, q, 0.5);

  // Midpoint sits at +/-pi, far from 0.
  EXPECT_NEAR(std::abs(mid[0]), kPi, 1e-6);
  EXPECT_GT(std::abs(mid[0]), 3.0);

  // Endpoints are still recovered (modulo wrap).
  EXPECT_NEAR(std::abs(wrap_to_pi(manifold.geodesic(p, q, 1.0)[0] - q[0])), 0.0, 1e-12);
}

TEST_F(SO2ExpTest, DistanceShortArc) {
  // Distance across the wrap boundary equals the short arc length ~0.283.
  auto p = a1(3.0);
  auto q = a1(-3.0);
  const double short_arc = std::abs(wrap_to_pi(q[0] - p[0]));
  EXPECT_NEAR(manifold.distance(p, q), short_arc, 1e-10);
}

TEST_F(SO2ExpTest, DistanceSymmetry) {
  auto p = a1(1.0);
  auto q = a1(-2.5);
  EXPECT_NEAR(manifold.distance(p, q), manifold.distance(q, p), 1e-10);
}

TEST_F(SO2ExpTest, DistanceZeroSamePoint) {
  auto p = a1(0.7);
  EXPECT_NEAR(manifold.distance(p, p), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Anisotropic (scaled) metric
// ---------------------------------------------------------------------------

TEST(SO2AnisotropicTest, WeightScalesDistanceBySqrt) {
  SO2<> iso_manifold;
  SO2<> weighted_manifold{SO2CanonicalMetric{4.0}};

  auto p = a1(0.5);
  auto q = a1(2.0);

  const double d_iso = iso_manifold.distance(p, q);
  const double d_weighted = weighted_manifold.distance(p, q);

  // w = 4 scales the norm (hence distance) by sqrt(4) = 2.
  EXPECT_NEAR(d_weighted, 2.0 * d_iso, 1e-9);
}

TEST(SO2AnisotropicTest, NormScalesBySqrtWeight) {
  SO2<> iso_manifold;
  SO2<> weighted_manifold{SO2CanonicalMetric{4.0}};

  auto p = a1(0.0);
  auto v = a1(1.0);

  EXPECT_NEAR(iso_manifold.norm(p, v), 1.0, 1e-12);
  EXPECT_NEAR(weighted_manifold.norm(p, v), 2.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Random point
// ---------------------------------------------------------------------------

TEST(SO2RandomTest, RandomPointInBounds) {
  SO2<> manifold;
  for (int i = 0; i < 200; ++i) {
    auto p = manifold.random_point();
    EXPECT_GE(p[0], -kPi);
    EXPECT_LT(p[0], kPi);
  }
}

// ---------------------------------------------------------------------------
// discrete_geodesic smoke test
// ---------------------------------------------------------------------------

TEST(SO2InterpolationTest, DiscreteGeodesicReachesTarget) {
  SO2<> manifold;
  auto start = a1(0.0);
  auto target = a1(2.0);

  auto result = discrete_geodesic(manifold, start, target);

  EXPECT_GT(result.path.size(), 1u);
  EXPECT_EQ(result.status, InterpolationStatus::Converged);

  const auto& reached = result.path.back();
  EXPECT_NEAR(std::abs(wrap_to_pi(reached[0] - target[0])), 0.0, 1e-3);
}

TEST(SO2InterpolationTest, DiscreteGeodesicWrapsToTarget) {
  // Target reachable most cheaply by crossing the wrap boundary.
  SO2<> manifold;
  auto start = a1(2.8);
  auto target = a1(-2.8);

  auto result = discrete_geodesic(manifold, start, target);

  EXPECT_EQ(result.status, InterpolationStatus::Converged);
  const auto& reached = result.path.back();
  EXPECT_NEAR(std::abs(wrap_to_pi(reached[0] - target[0])), 0.0, 1e-3);
}

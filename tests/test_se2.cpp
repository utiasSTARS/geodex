#include <cmath>

#include <numbers>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/geodex.hpp"

using namespace geodex;
using namespace geodex::utils;

// Compile-time concept checks
static_assert(RiemannianManifold<SE2<>>);
static_assert(RiemannianManifold<SE2<SE2LeftInvariantMetric, SE2EulerRetraction>>);

// ---------------------------------------------------------------------------
// Exponential map retraction
// ---------------------------------------------------------------------------

class SE2ExpTest : public ::testing::Test {
 protected:
  SE2<> manifold;
};

TEST_F(SE2ExpTest, Dim) { EXPECT_EQ(manifold.dim(), 3); }

TEST_F(SE2ExpTest, ExpLogRoundTrip) {
  Eigen::Vector3d p(2.0, 3.0, 0.5);
  Eigen::Vector3d v(0.3, -0.4, 0.2);

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-10);
}

TEST_F(SE2ExpTest, ExpLogRoundTripLargeOmega) {
  Eigen::Vector3d p(1.0, 1.0, -1.0);
  Eigen::Vector3d v(1.0, 2.0, 2.5);

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-10);
}

TEST_F(SE2ExpTest, ExpLogRoundTripZeroOmega) {
  Eigen::Vector3d p(1.0, 2.0, 0.7);
  Eigen::Vector3d v(0.5, -0.3, 0.0);

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-10);
}

TEST_F(SE2ExpTest, DistanceSymmetry) {
  Eigen::Vector3d p(1.0, 2.0, 0.3);
  Eigen::Vector3d q(4.0, 5.0, -1.0);

  EXPECT_NEAR(manifold.distance(p, q), manifold.distance(q, p), 1e-10);
}

TEST_F(SE2ExpTest, DistanceZeroSamePoint) {
  Eigen::Vector3d p(1.0, 2.0, 0.5);
  EXPECT_NEAR(manifold.distance(p, p), 0.0, 1e-12);
}

TEST_F(SE2ExpTest, GeodesicEndpoints) {
  Eigen::Vector3d p(1.0, 2.0, 0.3);
  Eigen::Vector3d q(4.0, 5.0, -1.0);

  auto start = manifold.geodesic(p, q, 0.0);
  auto end = manifold.geodesic(p, q, 1.0);

  EXPECT_NEAR((start - p).norm(), 0.0, 1e-12);
  // end theta may differ by wrapping, check individually
  EXPECT_NEAR(end[0], q[0], 1e-10);
  EXPECT_NEAR(end[1], q[1], 1e-10);
  EXPECT_NEAR(std::abs(wrap_to_pi(end[2] - q[2])), 0.0, 1e-10);
}

TEST_F(SE2ExpTest, PureTranslation) {
  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d v(1.0, 0.0, 0.0);

  auto q = manifold.exp(p, v);
  EXPECT_NEAR(q[0], 1.0, 1e-12);
  EXPECT_NEAR(q[1], 0.0, 1e-12);
  EXPECT_NEAR(q[2], 0.0, 1e-12);
}

TEST_F(SE2ExpTest, PureTranslationRotatedBase) {
  // At theta = pi/2, forward (vx=1) should move in +y direction
  Eigen::Vector3d p(0.0, 0.0, std::numbers::pi / 2.0);
  Eigen::Vector3d v(1.0, 0.0, 0.0);

  auto q = manifold.exp(p, v);
  EXPECT_NEAR(q[0], 0.0, 1e-10);
  EXPECT_NEAR(q[1], 1.0, 1e-10);
  EXPECT_NEAR(q[2], std::numbers::pi / 2.0, 1e-12);
}

TEST_F(SE2ExpTest, PureRotation) {
  Eigen::Vector3d p(3.0, 4.0, 0.0);
  Eigen::Vector3d v(0.0, 0.0, 1.0);

  auto q = manifold.exp(p, v);
  EXPECT_NEAR(q[0], 3.0, 1e-12);
  EXPECT_NEAR(q[1], 4.0, 1e-12);
  EXPECT_NEAR(q[2], 1.0, 1e-12);
}

TEST_F(SE2ExpTest, ThetaWrapping) {
  Eigen::Vector3d p(0.0, 0.0, 3.0);
  Eigen::Vector3d v(0.0, 0.0, 1.0);

  auto q = manifold.exp(p, v);
  // 3.0 + 1.0 = 4.0, wrapped: 4.0 - 2*pi ≈ -2.283
  double expected = wrap_to_pi(4.0);
  EXPECT_NEAR(q[2], expected, 1e-12);
}

TEST_F(SE2ExpTest, ThetaWrappingNegative) {
  Eigen::Vector3d p(0.0, 0.0, -3.0);
  Eigen::Vector3d v(0.0, 0.0, -1.0);

  auto q = manifold.exp(p, v);
  // -3.0 + -1.0 = -4.0, wrapped
  double expected = wrap_to_pi(-4.0);
  EXPECT_NEAR(q[2], expected, 1e-12);
}

// ---------------------------------------------------------------------------
// Anisotropic metric
// ---------------------------------------------------------------------------

TEST(SE2AnisotropicTest, LateralMotionCostsMore) {
  SE2<> iso_manifold;
  SE2<> aniso_manifold{SE2LeftInvariantMetric{1.0, 100.0, 0.5}};

  // Pure forward motion (vx only)
  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d v_forward(1.0, 0.0, 0.0);
  Eigen::Vector3d v_lateral(0.0, 1.0, 0.0);

  double norm_fwd_iso = iso_manifold.norm(p, v_forward);
  double norm_lat_iso = iso_manifold.norm(p, v_lateral);
  EXPECT_NEAR(norm_fwd_iso, norm_lat_iso, 1e-12);

  double norm_fwd_aniso = aniso_manifold.norm(p, v_forward);
  double norm_lat_aniso = aniso_manifold.norm(p, v_lateral);
  EXPECT_GT(norm_lat_aniso, 5.0 * norm_fwd_aniso);
}

TEST(SE2AnisotropicTest, AnisotropicDistanceDiffers) {
  SE2<> iso_manifold;
  SE2<> aniso_manifold{SE2LeftInvariantMetric{1.0, 100.0, 0.5}};

  Eigen::Vector3d p(0.0, 0.0, 0.0);
  Eigen::Vector3d q(0.0, 1.0, 0.0);  // pure lateral displacement

  double d_iso = iso_manifold.distance(p, q);
  double d_aniso = aniso_manifold.distance(p, q);
  EXPECT_GT(d_aniso, 2.0 * d_iso);
}

// ---------------------------------------------------------------------------
// Euler retraction
// ---------------------------------------------------------------------------

class SE2EulerTest : public ::testing::Test {
 protected:
  SE2<SE2LeftInvariantMetric, SE2EulerRetraction> manifold;
};

TEST_F(SE2EulerTest, ExpLogRoundTrip) {
  Eigen::Vector3d p(2.0, 3.0, 0.5);
  Eigen::Vector3d v(0.3, -0.4, 0.2);

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-10);
}

TEST_F(SE2EulerTest, DistanceSymmetry) {
  Eigen::Vector3d p(1.0, 2.0, 0.3);
  Eigen::Vector3d q(4.0, 5.0, -1.0);

  EXPECT_NEAR(manifold.distance(p, q), manifold.distance(q, p), 1e-10);
}

TEST_F(SE2EulerTest, GeodesicEndpoints) {
  Eigen::Vector3d p(1.0, 2.0, 0.3);
  Eigen::Vector3d q(4.0, 5.0, -1.0);

  auto start = manifold.geodesic(p, q, 0.0);
  auto end = manifold.geodesic(p, q, 1.0);

  EXPECT_NEAR((start - p).norm(), 0.0, 1e-12);
  EXPECT_NEAR(end[0], q[0], 1e-10);
  EXPECT_NEAR(end[1], q[1], 1e-10);
  EXPECT_NEAR(std::abs(wrap_to_pi(end[2] - q[2])), 0.0, 1e-10);
}

TEST_F(SE2EulerTest, PureTranslation) {
  Eigen::Vector3d p(1.0, 2.0, 0.0);
  Eigen::Vector3d v(3.0, 4.0, 0.0);

  auto q = manifold.exp(p, v);
  EXPECT_NEAR(q[0], 4.0, 1e-12);
  EXPECT_NEAR(q[1], 6.0, 1e-12);
  EXPECT_NEAR(q[2], 0.0, 1e-12);
}

TEST_F(SE2EulerTest, ThetaWrapping) {
  Eigen::Vector3d p(0.0, 0.0, 3.0);
  Eigen::Vector3d v(0.0, 0.0, 1.0);

  auto q = manifold.exp(p, v);
  double expected = wrap_to_pi(4.0);
  EXPECT_NEAR(q[2], expected, 1e-12);
}

// ---------------------------------------------------------------------------
// Random point
// ---------------------------------------------------------------------------

TEST(SE2RandomTest, RandomPointInBounds) {
  SE2<> manifold;
  for (int i = 0; i < 100; ++i) {
    auto p = manifold.random_point();
    EXPECT_GE(p[0], 0.0);
    EXPECT_LE(p[0], 10.0);
    EXPECT_GE(p[1], 0.0);
    EXPECT_LE(p[1], 10.0);
    EXPECT_GE(p[2], -std::numbers::pi);
    EXPECT_LE(p[2], std::numbers::pi);
  }
}

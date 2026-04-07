#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <geodex/geodex.hpp>
#include <numbers>

using namespace geodex;

// Compile-time concept checks
static_assert(RiemannianManifold<Sphere<>>);
static_assert(RiemannianManifold<Sphere<2, ConstantSPDMetric<3>>>);
static_assert(RiemannianManifold<Sphere<2, SphereRoundMetric, SphereProjectionRetraction>>);
static_assert(Retraction<SphereExponentialMap, Eigen::Vector3d, Eigen::Vector3d>);
static_assert(Retraction<SphereProjectionRetraction, Eigen::Vector3d, Eigen::Vector3d>);

// HasInjectivityRadius is now unconditional on Sphere (topological).
// The default metric (ConstantSPDMetric<3> identity) gives the canonical π.
static_assert(HasInjectivityRadius<Sphere<>>);
static_assert(HasInjectivityRadius<Sphere<2, SphereRoundMetric, SphereProjectionRetraction>>);

// Helper: create a known point pair with known geodesic distance.
// North pole to a point at latitude θ on the great circle through x-axis.
static Eigen::Vector3d point_at_theta(double theta) {
  return Eigen::Vector3d(std::sin(theta), 0.0, std::cos(theta));
}

// ---------------------------------------------------------------------------
// Sphere with round metric + true exp/log
// ---------------------------------------------------------------------------

class SphereRoundTest : public ::testing::Test {
 protected:
  Sphere<> sphere;
  Eigen::Vector3d north{0.0, 0.0, 1.0};
};

TEST_F(SphereRoundTest, Dim) { EXPECT_EQ(sphere.dim(), 2); }

TEST_F(SphereRoundTest, RandomPointOnSphere) {
  auto p = sphere.random_point();
  EXPECT_NEAR(p.norm(), 1.0, 1e-12);
}

TEST_F(SphereRoundTest, ExpLogRoundTrip) {
  Eigen::Vector3d p = north;
  Eigen::Vector3d v(0.3, 0.4, 0.0);  // tangent at north pole (z-component = 0)

  auto q = sphere.exp(p, v);
  EXPECT_NEAR(q.norm(), 1.0, 1e-12);

  auto v_back = sphere.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-10);
}

TEST_F(SphereRoundTest, DistanceKnown) {
  double theta = std::numbers::pi / 3.0;
  auto q = point_at_theta(theta);
  EXPECT_NEAR(sphere.distance(north, q), theta, 1e-12);
}

TEST_F(SphereRoundTest, DistanceSymmetry) {
  auto p = point_at_theta(0.5);
  auto q = point_at_theta(1.2);
  EXPECT_NEAR(sphere.distance(p, q), sphere.distance(q, p), 1e-12);
}

TEST_F(SphereRoundTest, GeodesicEndpoints) {
  auto q = point_at_theta(1.0);
  auto start = sphere.geodesic(north, q, 0.0);
  auto end = sphere.geodesic(north, q, 1.0);
  EXPECT_NEAR((start - north).norm(), 0.0, 1e-12);
  EXPECT_NEAR((end - q).norm(), 0.0, 1e-12);
}

TEST_F(SphereRoundTest, LogZeroDistance) {
  auto v = sphere.log(north, north);
  EXPECT_NEAR(v.norm(), 0.0, 1e-10);
}

TEST_F(SphereRoundTest, DistanceAntipodal) {
  Eigen::Vector3d south = -north;
  EXPECT_NEAR(sphere.distance(north, south), std::numbers::pi, 1e-12);
}

TEST_F(SphereRoundTest, LogAntipodalReturnsZero) {
  Eigen::Vector3d south = -north;
  auto v = sphere.log(north, south);
  EXPECT_NEAR(v.norm(), 0.0, 1e-10);
}

TEST_F(SphereRoundTest, InjectivityRadius) {
  EXPECT_NEAR(sphere.injectivity_radius(), std::numbers::pi, 1e-12);
}

// ---------------------------------------------------------------------------
// Midpoint distance
// ---------------------------------------------------------------------------

TEST_F(SphereRoundTest, MidpointDistanceMatchesExact) {
  auto q = point_at_theta(1.0);
  double exact = sphere.distance(north, q);
  double midpoint = distance_midpoint(sphere, north, q);
  EXPECT_NEAR(midpoint, exact, 1e-10);
}

TEST_F(SphereRoundTest, MidpointDistanceVariousAngles) {
  for (double theta : {0.1, 0.5, 1.0, 2.0, 3.0}) {
    auto q = point_at_theta(theta);
    double exact = sphere.distance(north, q);
    double midpoint = distance_midpoint(sphere, north, q);
    EXPECT_NEAR(midpoint, exact, 1e-6) << "Failed for theta=" << theta;
  }
}

// ---------------------------------------------------------------------------
// Sphere with projection retraction
// ---------------------------------------------------------------------------

TEST(SphereProjection, RetractionReturnsSpherePoint) {
  Sphere<2, SphereRoundMetric, SphereProjectionRetraction> sphere;
  Eigen::Vector3d p(0.0, 0.0, 1.0);
  Eigen::Vector3d v(0.3, 0.4, 0.0);

  auto q = sphere.exp(p, v);
  EXPECT_NEAR(q.norm(), 1.0, 1e-12);
}

TEST(SphereProjection, InverseRetractIsTangent) {
  Sphere<2, SphereRoundMetric, SphereProjectionRetraction> sphere;
  Eigen::Vector3d p(0.0, 0.0, 1.0);
  Eigen::Vector3d q = point_at_theta(0.5);

  auto v = sphere.log(p, q);
  // v should be tangent to p: v·p ≈ 0
  EXPECT_NEAR(v.dot(p), 0.0, 1e-12);
}

TEST(SphereProjection, AntipodalReturnsZero) {
  Sphere<2, SphereRoundMetric, SphereProjectionRetraction> sphere;
  Eigen::Vector3d p(0.0, 0.0, 1.0);
  Eigen::Vector3d south(0.0, 0.0, -1.0);

  auto v = sphere.log(p, south);
  EXPECT_NEAR(v.norm(), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Sphere with anisotropic metric
// ---------------------------------------------------------------------------

TEST(SphereAnisotropic, ConceptSatisfied) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 0) = 2.0;
  Sphere<2, ConstantSPDMetric<3>> sphere{ConstantSPDMetric<3>{A}};
  EXPECT_EQ(sphere.dim(), 2);
}

TEST(SphereAnisotropic, DistanceDiffersFromRound) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 0) = 4.0;  // stretch x-direction

  Sphere<> round_sphere;
  Sphere<2, ConstantSPDMetric<3>> aniso_sphere{ConstantSPDMetric<3>{A}};

  Eigen::Vector3d p(0.0, 0.0, 1.0);
  auto q = point_at_theta(0.5);

  double d_round = round_sphere.distance(p, q);
  double d_aniso = aniso_sphere.distance(p, q);

  // Anisotropic metric stretches x-axis, so distance along x should be larger.
  EXPECT_GT(d_aniso, d_round);
}

TEST(SphereAnisotropic, MidpointDistance) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 0) = 2.0;

  Sphere<2, ConstantSPDMetric<3>> sphere{ConstantSPDMetric<3>{A}};

  Eigen::Vector3d p(0.0, 0.0, 1.0);
  auto q = point_at_theta(0.5);

  double d_exact = sphere.distance(p, q);
  double d_mid = distance_midpoint(sphere, p, q);

  // For anisotropic metric with round exp/log, midpoint distance should be
  // a reasonable approximation (not exact).
  EXPECT_NEAR(d_mid, d_exact, 0.5);
}

TEST(SphereAnisotropic, DistanceAntipodal) {
  // At antipodal points, log is undefined (cut locus).
  // The manifold handles this by returning π (geodesic distance on the round sphere).
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 0) = 2.0;

  Sphere<2, ConstantSPDMetric<3>> sphere{ConstantSPDMetric<3>{A}};

  Eigen::Vector3d north(0.0, 0.0, 1.0);
  Eigen::Vector3d south(0.0, 0.0, -1.0);

  double d = sphere.distance(north, south);
  EXPECT_NEAR(d, std::numbers::pi, 1e-10);
}

#include <cmath>

#include <numbers>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/geodex.hpp"

using namespace geodex;

// ---------------------------------------------------------------------------
// Compile-time concept checks for static n-dim Sphere instantiations
// ---------------------------------------------------------------------------

static_assert(RiemannianManifold<Sphere<2>>);
static_assert(RiemannianManifold<Sphere<3>>);
static_assert(RiemannianManifold<Sphere<4>>);
static_assert(RiemannianManifold<Sphere<Eigen::Dynamic>>);

static_assert(HasInjectivityRadius<Sphere<3>>);
static_assert(HasInjectivityRadius<Sphere<Eigen::Dynamic>>);

// ---------------------------------------------------------------------------
// Dimension and Ambient checks
// ---------------------------------------------------------------------------

TEST(SphereNDim, DimensionReportedCorrectly) {
  Sphere<3> s3;
  Sphere<4> s4;
  Sphere<Eigen::Dynamic> s_dyn(5);

  EXPECT_EQ(s3.dim(), 3);
  EXPECT_EQ(s4.dim(), 4);
  EXPECT_EQ(s_dyn.dim(), 5);
}

TEST(SphereNDim, AmbientIsDimPlusOne) {
  Sphere<3> s3;
  auto p3 = s3.random_point();
  EXPECT_EQ(p3.size(), 4);  // S^3 lives in R^4

  Sphere<Eigen::Dynamic> s_dyn(4);
  auto p_dyn = s_dyn.random_point();
  EXPECT_EQ(p_dyn.size(), 5);  // S^4 lives in R^5
}

// ---------------------------------------------------------------------------
// Random sampling produces unit vectors
// ---------------------------------------------------------------------------

TEST(SphereNDim, RandomPointIsUnitVectorS3) {
  Sphere<3> s;
  for (int i = 0; i < 100; ++i) {
    auto p = s.random_point();
    EXPECT_NEAR(p.norm(), 1.0, 1e-12);
  }
}

TEST(SphereNDim, RandomPointIsUnitVectorDynamic) {
  Sphere<Eigen::Dynamic> s(5);
  for (int i = 0; i < 100; ++i) {
    auto p = s.random_point();
    EXPECT_NEAR(p.norm(), 1.0, 1e-12);
  }
}

// ---------------------------------------------------------------------------
// Exp/log round trip
// ---------------------------------------------------------------------------

TEST(SphereNDim, ExpLogRoundTripS3) {
  Sphere<3> s;
  Eigen::Vector4d p(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector4d q(0.0, 1.0, 0.0, 0.0);

  auto v = s.log(p, q);
  auto q_recovered = s.exp(p, v);
  EXPECT_LT((q_recovered - q).norm(), 1e-10);
}

TEST(SphereNDim, ExpLogRoundTripS4) {
  Sphere<4> s;
  Eigen::Vector<double, 5> p;
  Eigen::Vector<double, 5> q;
  p << 1.0, 0.0, 0.0, 0.0, 0.0;
  q << 0.0, 0.0, 1.0, 0.0, 0.0;

  auto v = s.log(p, q);
  auto q_recovered = s.exp(p, v);
  EXPECT_LT((q_recovered - q).norm(), 1e-10);
}

// ---------------------------------------------------------------------------
// Geodesic distances
// ---------------------------------------------------------------------------

TEST(SphereNDim, DistanceOrthogonalPointsS3) {
  // Distance between orthogonal unit vectors on S^3 should be π/2.
  Sphere<3> s;
  Eigen::Vector4d p(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector4d q(0.0, 1.0, 0.0, 0.0);
  EXPECT_NEAR(s.distance(p, q), std::numbers::pi / 2.0, 1e-10);
}

TEST(SphereNDim, DistanceAntipodalS3) {
  Sphere<3> s;
  Eigen::Vector4d p(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector4d q(-1.0, 0.0, 0.0, 0.0);
  EXPECT_NEAR(s.distance(p, q), std::numbers::pi, 1e-10);
}

// ---------------------------------------------------------------------------
// Injectivity radius
// ---------------------------------------------------------------------------

TEST(SphereNDim, InjectivityRadiusIsPi) {
  Sphere<3> s3;
  Sphere<4> s4;
  Sphere<Eigen::Dynamic> s_dyn(5);

  EXPECT_NEAR(s3.injectivity_radius(), std::numbers::pi, 1e-15);
  EXPECT_NEAR(s4.injectivity_radius(), std::numbers::pi, 1e-15);
  EXPECT_NEAR(s_dyn.injectivity_radius(), std::numbers::pi, 1e-15);
}

// ---------------------------------------------------------------------------
// Projection onto tangent space
// ---------------------------------------------------------------------------

TEST(SphereNDim, ProjectRemovesRadialComponentS3) {
  Sphere<3> s;
  Eigen::Vector4d p(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector4d v(3.0, 2.0, 1.0, 0.5);  // arbitrary ambient vector

  auto v_tangent = s.project(p, v);
  // Tangent vectors are orthogonal to p.
  EXPECT_NEAR(v_tangent.dot(p), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// has_riemannian_log for default Sphere<3>
// ---------------------------------------------------------------------------

TEST(SphereNDim, DefaultSphereHasRiemannianLogS3) {
  Sphere<3> s;
  EXPECT_TRUE(is_riemannian_log(s));
}

TEST(SphereNDim, SphereWithAnisotropicMetricIsNotRiemannianLogS3) {
  Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
  A(0, 0) = 10.0;
  Sphere<3, ConstantSPDMetric<4>> s{ConstantSPDMetric<4>{A}};
  EXPECT_FALSE(is_riemannian_log(s));
}

// ---------------------------------------------------------------------------
// Sphere with ProjectionRetraction in n-dim
// ---------------------------------------------------------------------------

TEST(SphereNDim, ProjectionRetractionS3) {
  Sphere<3, SphereRoundMetric, SphereProjectionRetraction> s;
  Eigen::Vector4d p(1.0, 0.0, 0.0, 0.0);
  Eigen::Vector4d v(0.0, 0.5, 0.0, 0.0);

  auto q = s.exp(p, v);
  EXPECT_NEAR(q.norm(), 1.0, 1e-12);

  // Projection retraction is not exact, but `is_riemannian_log` should be false.
  EXPECT_FALSE(is_riemannian_log(s));
}

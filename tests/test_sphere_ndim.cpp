#include <cmath>

#include <numbers>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/geodex.hpp"
#include "geodex/utils/normal.hpp"

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
// Inverse normal CDF and the unit-cube map on higher spheres
// ---------------------------------------------------------------------------

TEST(NormalQuantile, RoundTripThroughCdf) {
  for (double p = 0.001; p < 0.999; p += 0.0137) {
    const double x = utils::normal_quantile(p);
    const double cdf = 0.5 * std::erfc(-x / std::numbers::sqrt2);
    EXPECT_NEAR(cdf, p, 1e-9);
  }
  EXPECT_NEAR(utils::normal_quantile(0.5), 0.0, 1e-9);
  EXPECT_NEAR(utils::normal_quantile(0.975), 1.959963985, 1e-6);
  EXPECT_NEAR(utils::normal_quantile(0.025), -1.959963985, 1e-6);
  // The unit interval's endpoints map to finite values.
  EXPECT_TRUE(std::isfinite(utils::normal_quantile(0.0)));
  EXPECT_TRUE(std::isfinite(utils::normal_quantile(1.0)));
}

TEST(SphereNDim, UnitCubeDimAndUnitNorm) {
  for (int n : {3, 4, 5, 7}) {
    Sphere<Eigen::Dynamic> s(n);
    EXPECT_EQ(s.unit_cube_dim(), n + 1);
    Eigen::VectorXd u(n + 1);
    for (int trial = 0; trial < 20; ++trial) {
      for (int i = 0; i <= n; ++i) {
        u[i] = 0.05 + 0.9 * static_cast<double>((trial * 7 + i * 3) % 100) / 100.0;
      }
      EXPECT_NEAR(s.from_unit_cube(u).norm(), 1.0, 1e-12);
    }
  }
}

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
  // Orthogonal unit vectors on S^3 lie π/2 apart.
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

  // The projection retraction is not exact, and `is_riemannian_log` is false.
  EXPECT_FALSE(is_riemannian_log(s));
}

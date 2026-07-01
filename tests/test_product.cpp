#include <cmath>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/algorithm/interpolation.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/product.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/manifold/sphere.hpp"

using namespace geodex;

// Compile-time concept checks.
static_assert(RiemannianManifold<ProductManifold<Euclidean<Eigen::Dynamic>, SE2<>>>);
static_assert(RiemannianManifold<ProductManifold<Sphere<>, Euclidean<Eigen::Dynamic>>>);

namespace {

// Build a length-5 vector.
Eigen::VectorXd V5(double a, double b, double c, double d, double e) {
  Eigen::VectorXd v(5);
  v << a, b, c, d, e;
  return v;
}

}  // namespace

// ===========================================================================
// Euclidean(2) x SE2 : point size == dim for both blocks (total 2 + 3 == 5).
// ===========================================================================

class ProductEuclSE2Test : public ::testing::Test {
 protected:
  // Product of R^2 and SE(2).
  ProductManifold<Euclidean<Eigen::Dynamic>, SE2<>> prod{Euclidean<Eigen::Dynamic>(2), SE2<>{}};
  // Standalone blocks for cross-checking.
  Euclidean<Eigen::Dynamic> eucl{2};
  SE2<> se2{};

  Eigen::VectorXd p_ = V5(0.1, 0.2, 1.0, 1.0, 0.2);
  Eigen::VectorXd q_ = V5(0.5, -0.3, 3.0, 2.0, -0.5);
};

TEST_F(ProductEuclSE2Test, DimAndPointSize) {
  EXPECT_EQ(prod.dim(), 5);
  EXPECT_EQ(prod.random_point().size(), 5);
}

TEST_F(ProductEuclSE2Test, ExpLogRoundTrip) {
  Eigen::VectorXd v = V5(0.3, -0.4, 0.6, -0.2, 0.5);
  Eigen::VectorXd q2 = prod.exp(p_, v);
  ASSERT_EQ(q2.size(), 5);
  Eigen::VectorXd v_back = prod.log(p_, q2);
  ASSERT_EQ(v_back.size(), 5);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-9);
}

TEST_F(ProductEuclSE2Test, GeodesicEndpoints) {
  EXPECT_NEAR((prod.geodesic(p_, q_, 0.0) - p_).norm(), 0.0, 1e-9);
  EXPECT_NEAR((prod.geodesic(p_, q_, 1.0) - q_).norm(), 0.0, 1e-9);
}

TEST_F(ProductEuclSE2Test, DistanceIsBlockPythagoras) {
  Eigen::VectorXd pe = p_.head(2), qe = q_.head(2);
  Eigen::Vector3d ps = p_.tail(3), qs = q_.tail(3);

  const double dprod = prod.distance(p_, q_);
  const double de = eucl.distance(pe, qe);
  const double ds = se2.distance(ps, qs);
  EXPECT_NEAR(dprod * dprod, de * de + ds * ds, 1e-9);
}

TEST_F(ProductEuclSE2Test, BlocksMatchStandalone) {
  Eigen::VectorXd v = V5(0.3, -0.4, 0.6, -0.2, 0.5);
  Eigen::VectorXd pe = p_.head(2);
  Eigen::Vector3d ps = p_.tail(3);
  Eigen::VectorXd ve = v.head(2);
  Eigen::Vector3d vs = v.tail(3);

  Eigen::VectorXd q2 = prod.exp(p_, v);
  EXPECT_NEAR((q2.head(2) - eucl.exp(pe, ve)).norm(), 0.0, 1e-12);
  EXPECT_NEAR((q2.tail(3) - se2.exp(ps, vs)).norm(), 0.0, 1e-12);

  Eigen::VectorXd qe = q_.head(2);
  Eigen::Vector3d qs = q_.tail(3);
  Eigen::VectorXd lp = prod.log(p_, q_);
  EXPECT_NEAR((lp.head(2) - eucl.log(pe, qe)).norm(), 0.0, 1e-12);
  EXPECT_NEAR((lp.tail(3) - se2.log(ps, qs)).norm(), 0.0, 1e-12);
}

TEST_F(ProductEuclSE2Test, InnerIsBlockSum) {
  Eigen::VectorXd u = V5(1.0, 2.0, -1.0, 0.5, 0.3);
  Eigen::VectorXd v = V5(0.3, -0.4, 0.6, -0.2, 0.5);
  Eigen::VectorXd pe = p_.head(2);
  Eigen::Vector3d ps = p_.tail(3);

  const double ip = prod.inner(p_, u, v);
  const double ie = eucl.inner(pe, u.head(2), v.head(2));
  Eigen::Vector3d us = u.tail(3), vs = v.tail(3);
  const double is = se2.inner(ps, us, vs);
  EXPECT_NEAR(ip, ie + is, 1e-12);
  EXPECT_NEAR(prod.norm(p_, v), std::sqrt(prod.inner(p_, v, v)), 1e-12);
}

TEST_F(ProductEuclSE2Test, DiscreteGeodesicConverges) {
  auto r = discrete_geodesic(prod, p_, q_);
  EXPECT_EQ(r.status, InterpolationStatus::Converged);
  EXPECT_LT(r.final_distance, 1e-3);
  ASSERT_GE(r.path.size(), 2u);
  EXPECT_NEAR((r.path.front() - p_).norm(), 0.0, 1e-12);
  EXPECT_LT(prod.distance(r.path.back(), q_), 1e-2);
}

// ===========================================================================
// Sphere(S^2) x Euclidean(2) : point size != dim (sphere ambient 3 vs dim 2).
// Total point size 3 + 2 == 5; intrinsic dim 2 + 2 == 4.
// ===========================================================================

class ProductSphereEuclTest : public ::testing::Test {
 protected:
  ProductManifold<Sphere<>, Euclidean<Eigen::Dynamic>> prod{Sphere<>{},
                                                            Euclidean<Eigen::Dynamic>(2)};
  Sphere<> sph{};
  Euclidean<Eigen::Dynamic> eucl{2};

  // A base point (sphere block unit-norm) and a small ambient tangent at it.
  Eigen::VectorXd p_ = prod.random_point();
  Eigen::VectorXd v_ = 0.3 * prod.project(p_, V5(0.7, -0.4, 0.9, 0.5, -0.6));
};

TEST_F(ProductSphereEuclTest, DimAndSizes) {
  EXPECT_EQ(prod.dim(), 4);       // intrinsic tangent dimension
  EXPECT_EQ(p_.size(), 5);        // ambient point size (3 + 2)
  // Tangent is stored ambiently (3 + 2 == 5); dim() reports the intrinsic 4.
  EXPECT_EQ(prod.log(p_, p_).size(), 5);
}

TEST_F(ProductSphereEuclTest, SphereBlockStaysUnitNorm) {
  EXPECT_NEAR(p_.head(3).norm(), 1.0, 1e-9);
  // Sphere tangent segment is orthogonal to the sphere base point.
  EXPECT_NEAR(v_.head(3).dot(p_.head(3)), 0.0, 1e-12);

  Eigen::VectorXd q = prod.exp(p_, v_);
  EXPECT_NEAR(q.head(3).norm(), 1.0, 1e-9);
}

TEST_F(ProductSphereEuclTest, ExpLogRoundTrip) {
  Eigen::VectorXd q = prod.exp(p_, v_);
  Eigen::VectorXd v_back = prod.log(p_, q);
  EXPECT_NEAR((v_ - v_back).norm(), 0.0, 1e-9);
}

TEST_F(ProductSphereEuclTest, DistanceIsBlockPythagoras) {
  Eigen::VectorXd q = prod.exp(p_, v_);
  Eigen::Vector3d psph = p_.head(3), qsph = q.head(3);
  Eigen::VectorXd pe = p_.tail(2), qe = q.tail(2);

  const double dprod = prod.distance(p_, q);
  const double dsph = sph.distance(psph, qsph);
  const double de = eucl.distance(pe, qe);
  EXPECT_NEAR(dprod, std::sqrt(dsph * dsph + de * de), 1e-9);
}

// ===========================================================================
// make_product factory.
// ===========================================================================

TEST(ProductFactory, MakeProductDeducesTypes) {
  auto prod = make_product(Euclidean<Eigen::Dynamic>(2), SE2<>{});
  static_assert(RiemannianManifold<decltype(prod)>);
  EXPECT_EQ(prod.dim(), 5);
}

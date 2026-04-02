#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <geodex/geodex.hpp>
#include <numbers>

using namespace geodex;

// Compile-time concept checks
static_assert(RiemannianManifold<Torus<2>>);
static_assert(RiemannianManifold<Torus<Eigen::Dynamic>>);
static_assert(HasInjectivityRadius<Torus<2>>);
static_assert(HasInjectivityRadius<Torus<Eigen::Dynamic>>);

// ---------------------------------------------------------------------------
// Fixed-dimension T^2
// ---------------------------------------------------------------------------

class TorusT2Test : public ::testing::Test {
 protected:
  Torus<2> manifold;
};

TEST_F(TorusT2Test, Dim) { EXPECT_EQ(manifold.dim(), 2); }

TEST_F(TorusT2Test, RandomPointInRange) {
  for (int i = 0; i < 100; ++i) {
    auto p = manifold.random_point();
    EXPECT_EQ(p.size(), 2);
    for (int j = 0; j < 2; ++j) {
      EXPECT_GE(p[j], 0.0);
      EXPECT_LT(p[j], 2.0 * std::numbers::pi);
    }
  }
}

TEST_F(TorusT2Test, ExpLogRoundTrip) {
  Eigen::Vector2d p(1.0, 2.0);
  Eigen::Vector2d v(0.3, -0.5);

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-12);
}

TEST_F(TorusT2Test, ExpWraps) {
  Eigen::Vector2d p(6.0, 0.1);
  Eigen::Vector2d v(1.0, 0.0);

  auto q = manifold.exp(p, v);
  // 6.0 + 1.0 = 7.0 → 7.0 - 2π ≈ 0.717
  double expected = 7.0 - 2.0 * std::numbers::pi;
  EXPECT_NEAR(q[0], expected, 1e-12);
  EXPECT_NEAR(q[1], 0.1, 1e-12);
}

TEST_F(TorusT2Test, LogShortestPath) {
  // From 0.1 to 5.9: going forward = 5.8, going backward through 0 = 2π - 5.8 ≈ 0.483
  // Shortest path should go backward (negative direction).
  Eigen::Vector2d p(0.1, 0.0);
  Eigen::Vector2d q(5.9, 0.0);

  auto v = manifold.log(p, q);
  // Expected: 5.9 - 0.1 = 5.8, wrapped to [-π,π) → 5.8 - 2π ≈ -0.483
  double expected = 5.8 - 2.0 * std::numbers::pi;
  EXPECT_NEAR(v[0], expected, 1e-12);
  EXPECT_NEAR(v[1], 0.0, 1e-12);
}

TEST_F(TorusT2Test, LogShortestPathForward) {
  // From 1.0 to 2.0: forward = 1.0, backward = 2π - 1.0 ≈ 5.28
  // Shortest should go forward.
  Eigen::Vector2d p(1.0, 1.0);
  Eigen::Vector2d q(2.0, 1.5);

  auto v = manifold.log(p, q);
  EXPECT_NEAR(v[0], 1.0, 1e-12);
  EXPECT_NEAR(v[1], 0.5, 1e-12);
}

TEST_F(TorusT2Test, DistanceWrapping) {
  Eigen::Vector2d p(0.1, 0.0);
  Eigen::Vector2d q(5.9, 0.0);

  // Shortest distance for first component: |5.8 - 2π| ≈ 0.483
  double expected = std::abs(5.8 - 2.0 * std::numbers::pi);
  EXPECT_NEAR(manifold.distance(p, q), expected, 1e-12);
}

TEST_F(TorusT2Test, DistanceSymmetry) {
  Eigen::Vector2d p(0.5, 1.0);
  Eigen::Vector2d q(4.0, 5.0);

  EXPECT_NEAR(manifold.distance(p, q), manifold.distance(q, p), 1e-12);
}

TEST_F(TorusT2Test, DistanceZeroSamePoint) {
  Eigen::Vector2d p(1.0, 2.0);
  EXPECT_NEAR(manifold.distance(p, p), 0.0, 1e-12);
}

TEST_F(TorusT2Test, GeodesicEndpoints) {
  Eigen::Vector2d p(1.0, 2.0);
  Eigen::Vector2d q(2.0, 3.0);

  auto start = manifold.geodesic(p, q, 0.0);
  auto end = manifold.geodesic(p, q, 1.0);

  EXPECT_NEAR((start - p).norm(), 0.0, 1e-12);
  EXPECT_NEAR((end - q).norm(), 0.0, 1e-12);
}

TEST_F(TorusT2Test, GeodesicMidpoint) {
  Eigen::Vector2d p(1.0, 1.0);
  Eigen::Vector2d q(2.0, 2.0);

  auto mid = manifold.geodesic(p, q, 0.5);
  EXPECT_NEAR(mid[0], 1.5, 1e-12);
  EXPECT_NEAR(mid[1], 1.5, 1e-12);
}

TEST_F(TorusT2Test, GeodesicWrapsCorrectly) {
  // Geodesic crossing the 0/2π boundary
  Eigen::Vector2d p(0.1, 0.0);
  Eigen::Vector2d q(5.9, 0.0);

  // At t=0.5, midpoint should be near 0/2π boundary
  auto mid = manifold.geodesic(p, q, 0.5);
  // log(p, q)[0] ≈ -0.483, so midpoint = wrap(0.1 + 0.5 * (-0.483)) ≈ wrap(-0.142) ≈ 6.141
  double expected_0 = wrap_to_2pi(0.1 + 0.5 * (5.8 - 2.0 * std::numbers::pi));
  EXPECT_NEAR(mid[0], expected_0, 1e-12);
}

TEST_F(TorusT2Test, InnerProduct) {
  Eigen::Vector2d p(0.0, 0.0);
  Eigen::Vector2d u(1.0, 0.0);
  Eigen::Vector2d v(0.0, 1.0);

  EXPECT_NEAR(manifold.inner(p, u, v), 0.0, 1e-12);
  EXPECT_NEAR(manifold.inner(p, u, u), 1.0, 1e-12);
}

TEST_F(TorusT2Test, Norm) {
  Eigen::Vector2d p(0.0, 0.0);
  Eigen::Vector2d v(3.0, 4.0);

  EXPECT_NEAR(manifold.norm(p, v), 5.0, 1e-12);
}

TEST_F(TorusT2Test, InjectivityRadius) {
  EXPECT_NEAR(manifold.injectivity_radius(), std::numbers::pi, 1e-12);
}

// ---------------------------------------------------------------------------
// Dynamic-dimension
// ---------------------------------------------------------------------------

class TorusDynamicTest : public ::testing::Test {
 protected:
  Torus<> manifold{3};
};

TEST_F(TorusDynamicTest, Dim) { EXPECT_EQ(manifold.dim(), 3); }

TEST_F(TorusDynamicTest, RandomPointInRange) {
  for (int i = 0; i < 100; ++i) {
    auto p = manifold.random_point();
    EXPECT_EQ(p.size(), 3);
    for (int j = 0; j < 3; ++j) {
      EXPECT_GE(p[j], 0.0);
      EXPECT_LT(p[j], 2.0 * std::numbers::pi);
    }
  }
}

TEST_F(TorusDynamicTest, ExpLogRoundTrip) {
  Eigen::VectorXd p(3);
  p << 1.0, 2.0, 3.0;
  Eigen::VectorXd v(3);
  v << 0.3, -0.5, 0.1;

  auto q = manifold.exp(p, v);
  auto v_back = manifold.log(p, q);
  EXPECT_NEAR((v - v_back).norm(), 0.0, 1e-12);
}

TEST_F(TorusDynamicTest, DistanceSymmetry) {
  Eigen::VectorXd p(3);
  p << 0.5, 1.0, 2.0;
  Eigen::VectorXd q(3);
  q << 4.0, 5.0, 0.5;

  EXPECT_NEAR(manifold.distance(p, q), manifold.distance(q, p), 1e-12);
}

TEST_F(TorusDynamicTest, LogShortestPath) {
  Eigen::VectorXd p(3);
  p << 0.1, 0.1, 0.1;
  Eigen::VectorXd q(3);
  q << 5.9, 5.9, 5.9;

  auto v = manifold.log(p, q);
  double expected = 5.8 - 2.0 * std::numbers::pi;
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(v[i], expected, 1e-12);
  }
}

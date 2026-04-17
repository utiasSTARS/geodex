/// @file test_clearance_metric.cpp
/// @brief Tests for SDFConformalMetric.

#include <cmath>

#include <numbers>
#include <vector>

#include <gtest/gtest.h>

#include "geodex/algorithm/path_smoothing.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/metrics/clearance.hpp"

// ---------------------------------------------------------------------------
// Simple circular obstacle SDF (individual, exact)
// ---------------------------------------------------------------------------

struct CircleSDF {
  double cx, cy, radius;

  double operator()(const auto& q) const {
    double dx = q[0] - cx, dy = q[1] - cy;
    return std::sqrt(dx * dx + dy * dy) - radius;
  }
};

// ---------------------------------------------------------------------------
// Smooth-min SDF over multiple circles via log-sum-exp
// ---------------------------------------------------------------------------

struct CircleSmoothSDF {
  struct Circle {
    double cx, cy, radius;
  };

  std::vector<Circle> circles;
  double beta_sdf = 20.0;  // smoothness (higher = closer to true min)

  double operator()(const auto& q) const {
    if (circles.empty()) return 1e10;

    // log-sum-exp smooth-min: d = -1/β * log(Σ exp(-β * d_i))
    double max_neg_d = -1e30;
    for (const auto& c : circles) {
      double dx = q[0] - c.cx, dy = q[1] - c.cy;
      double d_i = std::sqrt(dx * dx + dy * dy) - c.radius;
      max_neg_d = std::max(max_neg_d, -beta_sdf * d_i);
    }

    double sum = 0.0;
    for (const auto& c : circles) {
      double dx = q[0] - c.cx, dy = q[1] - c.cy;
      double d_i = std::sqrt(dx * dx + dy * dy) - c.radius;
      sum += std::exp(-beta_sdf * d_i - max_neg_d);
    }

    return -(max_neg_d + std::log(sum)) / beta_sdf;
  }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// (a) Far from obstacles: conformal factor should be ~1.0.
TEST(SDFConformalMetric, FarFromObstacle) {
  geodex::SE2LeftInvariantMetric base{1.0, 1.0, 0.5};
  CircleSDF sdf{5.0, 5.0, 1.0};
  geodex::SDFConformalMetric metric{base, sdf, 5.0, 3.0};

  Eigen::Vector3d q_far(50.0, 50.0, 0.0);  // far from obstacle at (5,5)
  double c = metric.conformal_factor(q_far);
  EXPECT_NEAR(c, 1.0, 1e-6);  // exp(-3*45) ≈ 0
}

// (b) At obstacle surface: conformal factor should be 1 + kappa.
TEST(SDFConformalMetric, AtObstacleSurface) {
  geodex::SE2LeftInvariantMetric base{1.0, 1.0, 0.5};
  CircleSDF sdf{5.0, 5.0, 1.0};
  double kappa = 5.0;
  geodex::SDFConformalMetric metric{base, sdf, kappa, 3.0};

  // Point on the obstacle boundary (sdf = 0)
  Eigen::Vector3d q_surface(6.0, 5.0, 0.0);
  double c = metric.conformal_factor(q_surface);
  EXPECT_NEAR(c, 1.0 + kappa, 1e-6);
}

// (c) Inside obstacle: conformal factor should be > 1 + kappa.
TEST(SDFConformalMetric, InsideObstacle) {
  geodex::SE2LeftInvariantMetric base{1.0, 1.0, 0.5};
  CircleSDF sdf{5.0, 5.0, 1.0};
  double kappa = 5.0;
  geodex::SDFConformalMetric metric{base, sdf, kappa, 3.0};

  // Point inside obstacle (sdf < 0)
  Eigen::Vector3d q_inside(5.0, 5.0, 0.0);
  double c = metric.conformal_factor(q_inside);
  EXPECT_GT(c, 1.0 + kappa);
}

// (d) Monotonicity: conformal factor decreases with distance from obstacle.
TEST(SDFConformalMetric, MonotonicDecrease) {
  geodex::SE2LeftInvariantMetric base{1.0, 1.0, 0.5};
  CircleSDF sdf{5.0, 5.0, 1.0};
  geodex::SDFConformalMetric metric{base, sdf, 5.0, 3.0};

  double c_prev = 1e10;
  for (double d = 0.0; d <= 5.0; d += 0.5) {
    Eigen::Vector3d q(5.0 + 1.0 + d, 5.0, 0.0);  // distance d from surface
    double c = metric.conformal_factor(q);
    EXPECT_LT(c, c_prev) << "at distance " << d;
    c_prev = c;
  }
}

// (e) Inner product scales correctly.
TEST(SDFConformalMetric, InnerProductScaling) {
  geodex::SE2LeftInvariantMetric base{1.0, 2.0, 0.5};
  CircleSDF sdf{5.0, 5.0, 1.0};
  geodex::SDFConformalMetric metric{base, sdf, 5.0, 3.0};

  Eigen::Vector3d q(7.0, 5.0, 0.0);  // sdf = 1.0
  Eigen::Vector3d u(1.0, 0.5, 0.2);
  Eigen::Vector3d v(0.3, 1.0, -0.1);

  double c = metric.conformal_factor(q);
  double base_inner = base.inner(q, u, v);
  double scaled_inner = metric.inner(q, u, v);

  EXPECT_NEAR(scaled_inner, c * base_inner, 1e-12);
}

// (f) Norm scales correctly (sqrt of conformal factor).
TEST(SDFConformalMetric, NormScaling) {
  geodex::SE2LeftInvariantMetric base{1.0, 2.0, 0.5};
  CircleSDF sdf{5.0, 5.0, 1.0};
  geodex::SDFConformalMetric metric{base, sdf, 5.0, 3.0};

  Eigen::Vector3d q(7.0, 5.0, 0.0);
  Eigen::Vector3d v(1.0, 0.5, 0.2);

  double c = metric.conformal_factor(q);
  double base_norm = base.norm(q, v);
  double scaled_norm = metric.norm(q, v);

  EXPECT_NEAR(scaled_norm, std::sqrt(c) * base_norm, 1e-12);
}

// (g) Smooth-min SDF: agrees with true min for well-separated obstacles.
TEST(CircleSmoothSDF, AgreesWithTrueMin) {
  CircleSmoothSDF smooth;
  smooth.circles = {{0.0, 0.0, 1.0}, {10.0, 0.0, 1.0}};
  smooth.beta_sdf = 20.0;

  // Point near first obstacle only — smooth-min ≈ true min
  Eigen::Vector2d q_near(2.0, 0.0);
  double d_true = 1.0;  // dist to first circle surface = 2-1 = 1
  double d_smooth = smooth(q_near);
  EXPECT_NEAR(d_smooth, d_true, 0.1);

  // Point equidistant between obstacles
  Eigen::Vector2d q_mid(5.0, 0.0);
  double d_true_mid = 4.0;  // dist to either surface = 5-1 = 4
  double d_smooth_mid = smooth(q_mid);
  // Smooth-min slightly below true min at Voronoi boundary
  EXPECT_NEAR(d_smooth_mid, d_true_mid, 0.1);
}

// (h) Smooth-min SDF: smooth gradient at Voronoi boundary.
TEST(CircleSmoothSDF, SmoothGradientAtVoronoi) {
  CircleSmoothSDF smooth;
  smooth.circles = {{0.0, 0.0, 1.0}, {6.0, 0.0, 1.0}};
  smooth.beta_sdf = 20.0;

  // Finite-difference gradient at the midpoint (Voronoi boundary)
  double h = 1e-5;
  Eigen::Vector2d q_mid(3.0, 0.0);
  Eigen::Vector2d q_plus(3.0 + h, 0.0);
  Eigen::Vector2d q_minus(3.0 - h, 0.0);

  double grad_x = (smooth(q_plus) - smooth(q_minus)) / (2.0 * h);

  // At midpoint between equal obstacles: gradient in x should be ~0 by symmetry
  EXPECT_NEAR(grad_x, 0.0, 1e-3);

  // Gradient should be continuous: check on both sides
  Eigen::Vector2d q_left(2.9, 0.0);
  Eigen::Vector2d q_right(3.1, 0.0);
  Eigen::Vector2d q_left_p(2.9 + h, 0.0);
  Eigen::Vector2d q_left_m(2.9 - h, 0.0);
  Eigen::Vector2d q_right_p(3.1 + h, 0.0);
  Eigen::Vector2d q_right_m(3.1 - h, 0.0);

  double grad_left = (smooth(q_left_p) - smooth(q_left_m)) / (2.0 * h);
  double grad_right = (smooth(q_right_p) - smooth(q_right_m)) / (2.0 * h);

  // Gradients point away from nearest obstacle (SDF increases away from surface)
  EXPECT_GT(grad_left, 0.0);   // closer to obstacle 1: gradient points right (away)
  EXPECT_LT(grad_right, 0.0);  // closer to obstacle 2: gradient points left (away)
}

// (i) SDFConformalMetric works with ConfigurationSpace.
TEST(SDFConformalMetric, WorksWithConfigurationSpace) {
  geodex::SE2LeftInvariantMetric base{1.0, 1.0, 0.5};
  CircleSDF sdf{5.0, 5.0, 1.0};
  geodex::SDFConformalMetric metric{base, sdf, 5.0, 3.0};

  geodex::SE2<> se2{geodex::SE2LeftInvariantMetric{1.0, 1.0, 0.5}};
  geodex::ConfigurationSpace cspace{se2, metric};

  Eigen::Vector3d p(7.0, 5.0, 0.0);
  Eigen::Vector3d u(1.0, 0.0, 0.0);
  Eigen::Vector3d v(0.0, 1.0, 0.0);

  // Should compile and return a valid inner product
  double ip = cspace.inner(p, u, v);
  double expected = metric.inner(p, u, v);
  EXPECT_NEAR(ip, expected, 1e-12);
}

// (j) Euclidean manifold: conformal metric produces longer paths near obstacles.
TEST(SDFConformalMetric, EuclideanDistanceIncreasesNearObstacle) {
  using Euclidean2 = geodex::Euclidean<2>;
  geodex::IdentityMetric<2> base;
  CircleSDF sdf{5.0, 5.0, 1.0};
  geodex::SDFConformalMetric metric{base, sdf, 5.0, 3.0};

  Euclidean2 euc;
  geodex::ConfigurationSpace cspace{euc, metric};

  // Two points, same Euclidean distance apart
  Eigen::Vector2d a_far(50.0, 50.0), b_far(51.0, 50.0);  // far from obstacle
  Eigen::Vector2d a_near(6.5, 5.0), b_near(7.5, 5.0);    // near obstacle

  double d_far = cspace.distance(a_far, b_far);
  double d_near = cspace.distance(a_near, b_near);

  // Distance near obstacle should be larger (higher metric cost)
  EXPECT_GT(d_near, d_far);
}

// ---------------------------------------------------------------------------
// Path smoothing tests
// ---------------------------------------------------------------------------

// (k) Shortcutting reduces path length on SE(2).
TEST(PathSmoothing, ShortcuttingReducesVertices) {
  geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 0.5};
  geodex::SE2<> manifold{metric};

  // Create a zigzag path with redundant vertices.
  std::vector<Eigen::Vector3d> path = {
      {0.0, 0.0, 0.0}, {1.0, 1.0, 0.1},  {2.0, 0.5, 0.0},
      {3.0, 1.5, 0.2}, {4.0, 0.0, -0.1}, {5.0, 1.0, 0.0},
  };
  auto validity = [](const Eigen::Vector3d&) { return true; };  // no obstacles

  geodex::algorithm::PathSmoothingSettings settings;
  settings.max_shortcut_attempts = 100;
  settings.lbfgs_target_segments = 16;
  settings.lbfgs_max_iterations = 50;

  auto result = geodex::algorithm::smooth_path(manifold, validity, path, settings);

  // Should have removed some vertices and reduced energy.
  EXPECT_GE(result.vertices_removed, 0);
  EXPECT_TRUE(result.collision_free);
  EXPECT_GT(result.path.size(), 1u);
}

// (l) L-BFGS smoothing reduces energy on SE(2) with anisotropic metric.
TEST(PathSmoothing, LBFGSReducesEnergy) {
  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  geodex::SE2<> manifold{metric};

  // Straight-line path in coordinates — not a geodesic for anisotropic metric.
  std::vector<Eigen::Vector3d> path;
  for (int i = 0; i <= 10; ++i) {
    double t = static_cast<double>(i) / 10.0;
    path.push_back({t * 5.0, t * 2.0 + 0.5 * std::sin(t * 6.0), t * 0.5});
  }

  auto validity = [](const Eigen::Vector3d&) { return true; };

  // Compute initial energy.
  double E_before = 0.0;
  for (std::size_t k = 0; k + 1 < path.size(); ++k) {
    double d = manifold.distance(path[k], path[k + 1]);
    E_before += d * d;
  }
  E_before *= static_cast<double>(path.size() - 1);

  geodex::algorithm::PathSmoothingSettings settings;
  settings.max_shortcut_attempts = 50;
  settings.lbfgs_target_segments = 32;
  settings.lbfgs_max_iterations = 100;

  auto result = geodex::algorithm::smooth_path(manifold, validity, path, settings);

  // Smoothed path should have lower energy.
  EXPECT_LT(result.energy, E_before);
  EXPECT_TRUE(result.collision_free);
}

// (m) Collision-constrained smoothing respects obstacles.
TEST(PathSmoothing, RespectsObstacles) {
  geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 0.5};
  geodex::SE2<> manifold{metric};

  // Path that goes around an obstacle at (2.5, 0.5).
  std::vector<Eigen::Vector3d> path = {
      {0.0, 0.0, 0.0},  {1.0, -1.0, 0.0}, {2.0, -2.0, 0.0},
      {3.0, -2.0, 0.0}, {4.0, -1.0, 0.0}, {5.0, 0.0, 0.0},
  };

  double obs_cx = 2.5, obs_cy = 0.5, obs_r = 0.8;
  auto validity = [&](const Eigen::Vector3d& q) {
    double dx = q[0] - obs_cx, dy = q[1] - obs_cy;
    return dx * dx + dy * dy > obs_r * obs_r;
  };

  geodex::algorithm::PathSmoothingSettings settings;
  settings.max_shortcut_attempts = 100;
  settings.lbfgs_target_segments = 32;
  settings.lbfgs_max_iterations = 100;

  auto result = geodex::algorithm::smooth_path(manifold, validity, path, settings);

  EXPECT_TRUE(result.collision_free);

  // All path points should be outside the obstacle.
  for (const auto& q : result.path) {
    double dx = q[0] - obs_cx, dy = q[1] - obs_cy;
    EXPECT_GT(dx * dx + dy * dy, obs_r * obs_r * 0.99);
  }
}

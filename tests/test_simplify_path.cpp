/// @file test_simplify_path.cpp
/// @brief Tests for `geodex::algorithm::simplify_path` —
///        energy-aware shortcutting + collision-constrained L-BFGS smoothing on
///        a `RiemannianManifold`.

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <vector>

#include "geodex/algorithm/simplify_path.hpp"
#include "geodex/manifold/euclidean.hpp"

namespace ga = geodex::algorithm;

namespace {

auto always_valid() {
  return [](const Eigen::Vector2d& /*p*/) { return true; };
}

Eigen::Vector2d v2(double x, double y) { return Eigen::Vector2d(x, y); }

}  // namespace

TEST(SimplifyPath, ShortcutRemovesCollinearVertices) {
  // Three collinear waypoints with uneven spacing: subpath energy (5) > shortcut energy (4).
  // Shortcut fires and the middle vertex is removed.
  geodex::Euclidean<2> manifold;
  std::vector<Eigen::Vector2d> path = {v2(0.0, 0.0), v2(0.5, 0.0), v2(2.0, 0.0)};

  ga::SimplifyPathSettings settings;
  settings.max_iter_per_level = 0;  // disable phase 2

  const auto result = ga::simplify_path(manifold, always_valid(), path, settings);
  EXPECT_GE(result.vertices_removed, 1);
  EXPECT_TRUE(result.collision_free);
}

TEST(SimplifyPath, RespectsValidityFn) {
  // U-shaped detour around a "wall" near (1, 0). The direct shortcut from start
  // to goal must be rejected because its midpoint sits in the forbidden region.
  geodex::Euclidean<2> manifold;
  std::vector<Eigen::Vector2d> path = {v2(0.0, 0.0), v2(1.0, 0.5), v2(2.0, 0.0)};

  auto wall_validity = [](const Eigen::Vector2d& p) {
    return !(p[0] > 0.9 && p[0] < 1.1 && p[1] < 0.4);
  };

  ga::SimplifyPathSettings settings;
  settings.max_iter_per_level = 0;
  settings.smooth_target_segments = 2;
  settings.edge_collision_samples = 5;

  const auto result = ga::simplify_path(manifold, wall_validity, path, settings);
  EXPECT_EQ(result.vertices_removed, 0);
  EXPECT_TRUE(result.collision_free);
  ASSERT_GE(result.path.size(), 3u);
}

TEST(SimplifyPath, EnergyNonIncreasingAcrossPhases) {
  // Zigzag path. Energy must not increase from input through shortcutting and
  // then through smoothing.
  geodex::Euclidean<2> manifold;
  std::vector<Eigen::Vector2d> path = {v2(0.0, 0.0), v2(0.5, 1.0), v2(1.0, -0.3),
                                       v2(1.5, 1.2), v2(2.0, 0.0)};

  // Direct discrete energy: K * sum_k ||p_{k+1}-p_k||² (left-endpoint inner with identity metric).
  double E_in = 0.0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const Eigen::Vector2d d = path[i + 1] - path[i];
    E_in += d.dot(d);
  }
  E_in *= static_cast<int>(path.size()) - 1;

  ga::SimplifyPathSettings settings;
  settings.smooth_target_segments = 16;
  settings.max_iter_per_level = 200;

  const auto result = ga::simplify_path(manifold, always_valid(), path, settings);
  EXPECT_LE(result.energy, E_in + 1e-9);
  EXPECT_TRUE(result.collision_free);
}

TEST(SimplifyPath, MaxDisplacementClampLimitsDrift) {
  // With a tight trust region the smoothing phase can barely move the upsampled
  // waypoints. Compare against a manual upsample reference.
  geodex::Euclidean<2> manifold;
  std::vector<Eigen::Vector2d> path = {v2(0.0, 0.0), v2(0.5, 0.4), v2(1.0, 0.0),
                                       v2(1.5, 0.4), v2(2.0, 0.0)};

  ga::SimplifyPathSettings settings;
  settings.max_shortcut_attempts = 0;  // skip phase 1 — only test phase 2 trust region
  settings.smooth_target_segments = 8;
  settings.max_iter_per_level = 200;
  settings.max_displacement = 1e-3;

  const auto result = ga::simplify_path(manifold, always_valid(), path, settings);
  ASSERT_GT(result.path.size(), 4u);

  // Reproduce the upsample procedure: midpoint insertion via manifold.geodesic.
  // For Euclidean this is the linear midpoint.
  std::vector<Eigen::Vector2d> ref = path;
  while (static_cast<int>(ref.size()) - 1 < settings.smooth_target_segments) {
    std::vector<Eigen::Vector2d> next;
    next.reserve(2 * ref.size() - 1);
    for (std::size_t i = 0; i < ref.size(); ++i) {
      next.push_back(ref[i]);
      if (i + 1 < ref.size()) next.push_back(0.5 * (ref[i] + ref[i + 1]));
    }
    ref = std::move(next);
  }
  ASSERT_EQ(ref.size(), result.path.size());
  for (std::size_t i = 1; i + 1 < result.path.size(); ++i) {
    const double drift = (result.path[i] - ref[i]).norm();
    EXPECT_LE(drift, settings.max_displacement + 1e-9)
        << "waypoint " << i << " drifted " << drift;
  }
}

TEST(SimplifyPath, EndpointsArePreserved) {
  geodex::Euclidean<2> manifold;
  std::vector<Eigen::Vector2d> path = {v2(-0.7, 0.4), v2(0.0, 0.5), v2(0.9, -0.3)};

  const auto result = ga::simplify_path(manifold, always_valid(), path);
  ASSERT_GE(result.path.size(), 2u);
  EXPECT_TRUE(result.path.front().isApprox(path.front(), 1e-12));
  EXPECT_TRUE(result.path.back().isApprox(path.back(), 1e-12));
}

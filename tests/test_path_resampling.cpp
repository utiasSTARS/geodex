/// @file test_path_resampling.cpp
/// @brief Tests for uniform arc-length path resampling used in the Nav2 planner.

#include <cmath>

#include <numbers>
#include <vector>

#include <gtest/gtest.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/PathGeometric.h>

#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/manifold/se2.hpp"

namespace ob = ompl::base;
namespace og = ompl::geometric;

using SE2Manifold = geodex::SE2<>;
using AnisotropicSE2 = geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap>;
using StateSpace = geodex::integration::ompl::GeodexStateSpace<SE2Manifold>;
using AnisotropicStateSpace = geodex::integration::ompl::GeodexStateSpace<AnisotropicSE2>;
template <typename M>
using StateType = geodex::integration::ompl::GeodexState<M>;

// ---------------------------------------------------------------------------
// Resampling function extracted from Nav2 planner for testability
// ---------------------------------------------------------------------------

/// Resample an OMPL PathGeometric at uniform Euclidean xy spacing.
/// Uses geodesic interpolation (exp/log) for intermediate poses.
/// Returns a vector of (x, y, theta) waypoints.
template <typename ManifoldT>
static std::vector<Eigen::Vector3d> resamplePath(const og::PathGeometric& solution,
                                                 const ob::SpaceInformationPtr& si,
                                                 double waypoint_spacing) {
  using ST = geodex::integration::ompl::GeodexState<ManifoldT>;

  size_t n = solution.getStateCount();
  if (n < 2) {
    if (n == 1) {
      const auto* s = solution.getState(0)->as<ST>();
      return {{s->values[0], s->values[1], s->values[2]}};
    }
    return {};
  }

  // Compute cumulative Euclidean xy arc-length
  std::vector<double> cumDist(n, 0.0);
  for (size_t i = 1; i < n; ++i) {
    const auto* prev = solution.getState(i - 1)->as<ST>();
    const auto* curr = solution.getState(i)->as<ST>();
    double dx = curr->values[0] - prev->values[0];
    double dy = curr->values[1] - prev->values[1];
    cumDist[i] = cumDist[i - 1] + std::hypot(dx, dy);
  }
  double totalLength = cumDist.back();

  // Resample at uniform xy spacing with geodesic interpolation
  std::vector<Eigen::Vector3d> poses;
  const auto* first = solution.getState(0)->as<ST>();
  poses.push_back({first->values[0], first->values[1], first->values[2]});

  double targetDist = waypoint_spacing;
  size_t segIdx = 0;
  while (targetDist < totalLength - 1e-6) {
    while (segIdx + 1 < n - 1 && cumDist[segIdx + 1] < targetDist) {
      ++segIdx;
    }

    double segStart = cumDist[segIdx];
    double segEnd = cumDist[segIdx + 1];
    double segLength = segEnd - segStart;

    if (segLength > 1e-9) {
      double t = (targetDist - segStart) / segLength;
      ob::State* interp = si->allocState();
      si->getStateSpace()->interpolate(solution.getState(segIdx), solution.getState(segIdx + 1), t,
                                       interp);
      const auto* s = interp->as<ST>();
      poses.push_back({s->values[0], s->values[1], s->values[2]});
      si->freeState(interp);
    }

    targetDist += waypoint_spacing;
  }

  // Always include goal
  const auto* last = solution.getState(n - 1)->as<ST>();
  poses.push_back({last->values[0], last->values[1], last->values[2]});

  return poses;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::pair<std::shared_ptr<StateSpace>, ob::SpaceInformationPtr> makeIsotropicSetup() {
  geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 0.5};
  SE2Manifold manifold{metric, geodex::SE2ExponentialMap{},
                       Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                       Eigen::Vector3d(100.0, 100.0, std::numbers::pi)};
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, 0.0);
  bounds.setHigh(0, 100.0);
  bounds.setLow(1, 0.0);
  bounds.setHigh(1, 100.0);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);
  auto space = std::make_shared<StateSpace>(manifold, bounds);
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();
  return {space, si};
}

static std::pair<std::shared_ptr<AnisotropicStateSpace>, ob::SpaceInformationPtr>
makeAnisotropicSetup() {
  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  AnisotropicSE2 manifold{metric, geodex::SE2ExponentialMap{},
                          Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                          Eigen::Vector3d(100.0, 100.0, std::numbers::pi)};
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, 0.0);
  bounds.setHigh(0, 100.0);
  bounds.setLow(1, 0.0);
  bounds.setHigh(1, 100.0);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);
  auto space = std::make_shared<AnisotropicStateSpace>(manifold, bounds);
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();
  return {space, si};
}

template <typename M>
static void setState(ob::State* state, double x, double y, double theta) {
  auto* s = state->as<geodex::integration::ompl::GeodexState<M>>();
  s->values[0] = x;
  s->values[1] = y;
  s->values[2] = theta;
}

/// Compute Euclidean xy distance between consecutive poses.
static std::vector<double> xySpacings(const std::vector<Eigen::Vector3d>& poses) {
  std::vector<double> spacings;
  for (size_t i = 1; i < poses.size(); ++i) {
    double dx = poses[i][0] - poses[i - 1][0];
    double dy = poses[i][1] - poses[i - 1][1];
    spacings.push_back(std::hypot(dx, dy));
  }
  return spacings;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// A straight 10m path should produce ~10/0.1 = 100 interior waypoints + start + goal
TEST(PathResampling, StraightPath_CorrectCount) {
  auto [space, si] = makeIsotropicSetup();

  og::PathGeometric path(si);
  auto* s0 = space->allocState();
  auto* s1 = space->allocState();
  setState<SE2Manifold>(s0, 10.0, 20.0, 0.0);
  setState<SE2Manifold>(s1, 20.0, 20.0, 0.0);  // 10m along x
  path.append(s0);
  path.append(s1);

  // Densify (like the Nav2 planner does)
  path.interpolate();

  auto poses = resamplePath<SE2Manifold>(path, si, 0.1);

  // 10m / 0.1m = 100 intervals → 101 waypoints, plus possible goal
  // Allow some tolerance for floating point
  EXPECT_GE(poses.size(), 100u);
  EXPECT_LE(poses.size(), 103u);

  // First and last poses should match start and goal
  EXPECT_NEAR(poses.front()[0], 10.0, 1e-6);
  EXPECT_NEAR(poses.front()[1], 20.0, 1e-6);
  EXPECT_NEAR(poses.back()[0], 20.0, 1e-6);
  EXPECT_NEAR(poses.back()[1], 20.0, 1e-6);

  space->freeState(s0);
  space->freeState(s1);
}

// All interior spacings should be approximately equal to waypoint_spacing
TEST(PathResampling, StraightPath_UniformSpacing) {
  auto [space, si] = makeIsotropicSetup();

  og::PathGeometric path(si);
  auto* s0 = space->allocState();
  auto* s1 = space->allocState();
  setState<SE2Manifold>(s0, 5.0, 5.0, 0.0);
  setState<SE2Manifold>(s1, 25.0, 5.0, 0.0);  // 20m along x
  path.append(s0);
  path.append(s1);
  path.interpolate();

  auto poses = resamplePath<SE2Manifold>(path, si, 0.5);
  auto spacings = xySpacings(poses);

  // All interior spacings (except possibly the last) should be ~0.5m
  for (size_t i = 0; i + 1 < spacings.size(); ++i) {
    EXPECT_NEAR(spacings[i], 0.5, 0.05) << "Spacing at index " << i << " is " << spacings[i];
  }
  // Last spacing can be shorter (remainder)
  EXPECT_LE(spacings.back(), 0.5 + 0.05);

  space->freeState(s0);
  space->freeState(s1);
}

// Short path (2m) and long path (50m) should both have ~0.1m spacing
TEST(PathResampling, ShortAndLongPaths_SameSpacing) {
  auto [space, si] = makeIsotropicSetup();
  double target = 0.1;

  auto testPath = [&](double x0, double x1) {
    og::PathGeometric path(si);
    auto* s0 = space->allocState();
    auto* s1 = space->allocState();
    setState<SE2Manifold>(s0, x0, 10.0, 0.0);
    setState<SE2Manifold>(s1, x1, 10.0, 0.0);
    path.append(s0);
    path.append(s1);
    path.interpolate();
    auto poses = resamplePath<SE2Manifold>(path, si, target);
    space->freeState(s0);
    space->freeState(s1);
    return poses;
  };

  auto short_poses = testPath(10.0, 12.0);  // 2m
  auto long_poses = testPath(10.0, 60.0);   // 50m

  auto short_spacings = xySpacings(short_poses);
  auto long_spacings = xySpacings(long_poses);

  // Both should have uniform ~0.1m spacing
  for (size_t i = 0; i + 1 < short_spacings.size(); ++i) {
    EXPECT_NEAR(short_spacings[i], target, 0.02)
        << "Short path spacing[" << i << "] = " << short_spacings[i];
  }
  for (size_t i = 0; i + 1 < long_spacings.size(); ++i) {
    EXPECT_NEAR(long_spacings[i], target, 0.02)
        << "Long path spacing[" << i << "] = " << long_spacings[i];
  }

  // Count check: 2m → ~20 waypoints, 50m → ~500 waypoints
  EXPECT_GE(short_poses.size(), 19u);
  EXPECT_LE(short_poses.size(), 23u);
  EXPECT_GE(long_poses.size(), 498u);
  EXPECT_LE(long_poses.size(), 503u);
}

// Anisotropic metric (wy=100) should NOT affect physical xy spacing
TEST(PathResampling, AnisotropicMetric_UniformXYSpacing) {
  auto [space, si] = makeAnisotropicSetup();

  // Path along x (low-weight direction)
  {
    og::PathGeometric path(si);
    auto* s0 = space->allocState();
    auto* s1 = space->allocState();
    setState<AnisotropicSE2>(s0, 10.0, 20.0, 0.0);
    setState<AnisotropicSE2>(s1, 20.0, 20.0, 0.0);  // 10m along x
    path.append(s0);
    path.append(s1);
    path.interpolate();

    auto poses = resamplePath<AnisotropicSE2>(path, si, 0.1);
    auto spacings = xySpacings(poses);

    for (size_t i = 0; i + 1 < spacings.size(); ++i) {
      EXPECT_NEAR(spacings[i], 0.1, 0.02) << "X-path spacing[" << i << "] = " << spacings[i];
    }
    EXPECT_GE(poses.size(), 99u);

    space->freeState(s0);
    space->freeState(s1);
  }

  // Path along y (high-weight direction) — should still be 0.1m physical
  {
    og::PathGeometric path(si);
    auto* s0 = space->allocState();
    auto* s1 = space->allocState();
    setState<AnisotropicSE2>(s0, 20.0, 10.0, 0.0);
    setState<AnisotropicSE2>(s1, 20.0, 20.0, 0.0);  // 10m along y
    path.append(s0);
    path.append(s1);
    path.interpolate();

    auto poses = resamplePath<AnisotropicSE2>(path, si, 0.1);
    auto spacings = xySpacings(poses);

    for (size_t i = 0; i + 1 < spacings.size(); ++i) {
      EXPECT_NEAR(spacings[i], 0.1, 0.02) << "Y-path spacing[" << i << "] = " << spacings[i];
    }
    EXPECT_GE(poses.size(), 99u);

    space->freeState(s0);
    space->freeState(s1);
  }
}

// Diagonal path should also have uniform xy spacing
TEST(PathResampling, DiagonalPath_UniformSpacing) {
  auto [space, si] = makeIsotropicSetup();

  og::PathGeometric path(si);
  auto* s0 = space->allocState();
  auto* s1 = space->allocState();
  setState<SE2Manifold>(s0, 10.0, 10.0, 0.0);
  setState<SE2Manifold>(s1, 20.0, 20.0, std::numbers::pi / 4.0);
  path.append(s0);
  path.append(s1);
  path.interpolate();

  // xy distance = sqrt(100+100) ≈ 14.14m
  auto poses = resamplePath<SE2Manifold>(path, si, 0.2);
  auto spacings = xySpacings(poses);

  for (size_t i = 0; i + 1 < spacings.size(); ++i) {
    EXPECT_NEAR(spacings[i], 0.2, 0.05) << "Diagonal spacing[" << i << "] = " << spacings[i];
  }

  space->freeState(s0);
  space->freeState(s1);
}

// Multi-segment path (like an RRT* solution with multiple waypoints)
TEST(PathResampling, MultiSegmentPath_UniformSpacing) {
  auto [space, si] = makeIsotropicSetup();

  og::PathGeometric path(si);
  // Simulate an RRT*-like path with 4 waypoints
  std::vector<std::tuple<double, double, double>> waypoints = {
      {5.0, 5.0, 0.0},
      {15.0, 5.0, 0.0},    // 10m east
      {15.0, 15.0, 1.57},  // 10m north
      {25.0, 15.0, 0.0},   // 10m east
  };

  for (auto& [x, y, th] : waypoints) {
    auto* s = space->allocState();
    setState<SE2Manifold>(s, x, y, th);
    path.append(s);
    space->freeState(s);
  }
  path.interpolate();

  // Total xy length ≈ 30m, spacing 0.5 → ~60 waypoints
  auto poses = resamplePath<SE2Manifold>(path, si, 0.5);
  auto spacings = xySpacings(poses);

  for (size_t i = 0; i + 1 < spacings.size(); ++i) {
    EXPECT_NEAR(spacings[i], 0.5, 0.1) << "Multi-segment spacing[" << i << "] = " << spacings[i];
  }

  // Roughly 30m / 0.5m = 60 waypoints (geodesic SE(2) paths with rotation
  // can be slightly longer in xy than straight-line segments)
  EXPECT_GE(poses.size(), 55u);
  EXPECT_LE(poses.size(), 70u);
}

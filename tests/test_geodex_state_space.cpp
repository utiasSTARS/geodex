/// @file test_geodex_state_space.cpp
/// @brief Tests for GeodexStateSpace OMPL integration.

#include <cmath>

#include <numbers>

#include <gtest/gtest.h>
#include <ompl/base/DiscreteMotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorBounds.h>

#include "geodex/algorithm/interpolation.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/metrics/constant_spd.hpp"

namespace ob = ompl::base;
using SE2Manifold = geodex::SE2<>;
using StateSpace = geodex::integration::ompl::GeodexStateSpace<SE2Manifold>;
using StateType = geodex::integration::ompl::GeodexState<SE2Manifold>;

// Anisotropic SE2 types (car-like: expensive lateral motion).
using AnisotropicSE2 = geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap>;
using AnisotropicSE2Space = geodex::integration::ompl::GeodexStateSpace<AnisotropicSE2>;
using AnisotropicSE2State = geodex::integration::ompl::GeodexState<AnisotropicSE2>;

// Anisotropic Euclidean types.
using AnisotropicEuclidean = geodex::Euclidean<2, geodex::ConstantSPDMetric<2>>;
using AnisotropicEuclideanSpace = geodex::integration::ompl::GeodexStateSpace<AnisotropicEuclidean>;
using AnisotropicEuclideanState = geodex::integration::ompl::GeodexState<AnisotropicEuclidean>;

/// Helper: create SE(2) OMPL bounds.
static ob::RealVectorBounds makeSE2Bounds(double x_lo, double x_hi, double y_lo, double y_hi) {
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, x_lo);
  bounds.setHigh(0, x_hi);
  bounds.setLow(1, y_lo);
  bounds.setHigh(1, y_hi);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);
  return bounds;
}

/// Helper: create a large SE(2) state space (mimicking Willow Garage scale).
static std::shared_ptr<StateSpace> makeLargeSpace() {
  geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 0.5};
  geodex::SE2<> manifold{metric, geodex::SE2ExponentialMap{},
                         Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                         Eigen::Vector3d(60.0, 48.0, std::numbers::pi)};

  return std::make_shared<StateSpace>(manifold, makeSE2Bounds(0.0, 60.0, 0.0, 48.0));
}

/// Helper: set state values.
static void setState(ob::State* state, double x, double y, double theta) {
  auto* s = state->as<StateType>();
  s->values[0] = x;
  s->values[1] = y;
  s->values[2] = theta;
}

// -----------------------------------------------------------------------
// Test (a): Without collision resolution, validSegmentCount matches OMPL default
// -----------------------------------------------------------------------

TEST(GeodexStateSpaceTest, ValidSegmentCount_DefaultMatchesOMPL) {
  auto space = makeLargeSpace();

  // Must create SpaceInformation and call setup() so longestValidSegmentLength_
  // is computed from longestValidSegmentFraction * maxExtent.
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  setState(s1, 2.0, 5.0, 0.0);
  setState(s2, 8.0, 5.0, 0.0);

  // OMPL default: ceil(distance / (longestValidSegmentFraction * maxExtent))
  double dist = space->distance(s1, s2);
  double lvs = space->getLongestValidSegmentFraction() * space->getMaximumExtent();
  unsigned int expected = static_cast<unsigned int>(std::ceil(dist / lvs));

  EXPECT_EQ(space->validSegmentCount(s1, s2), expected);

  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (b): Collision resolution increases segment count
// -----------------------------------------------------------------------

TEST(GeodexStateSpaceTest, ValidSegmentCount_CollisionResolutionIncreasesCount) {
  auto space = makeLargeSpace();
  space->setCollisionResolution(0.05);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  setState(s1, 0.0, 5.0, 0.0);
  setState(s2, 10.0, 5.0, 0.0);

  unsigned int n = space->validSegmentCount(s1, s2);
  // 10m / 0.05m = 200 segments minimum
  EXPECT_GE(n, 200u);

  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (c): Thin wall detection — the critical regression test
// -----------------------------------------------------------------------

/// Validity checker that rejects states with 5.0 < x < 5.2 (a 0.2m wall).
class ThinWallChecker : public ob::StateValidityChecker {
 public:
  using ob::StateValidityChecker::StateValidityChecker;

  bool isValid(const ob::State* state) const override {
    const auto* s = state->as<StateType>();
    double x = s->values[0];
    return x <= 5.0 || x >= 5.2;
  }
};

TEST(GeodexStateSpaceTest, CollisionCheck_ThinWallDetected) {
  auto space = makeLargeSpace();

  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker(std::make_shared<ThinWallChecker>(si));

  // Without collision resolution: wall should be missed
  si->setup();

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  setState(s1, 2.0, 5.0, 0.0);
  setState(s2, 8.0, 5.0, 0.0);

  auto mv_default = std::make_shared<ob::DiscreteMotionValidator>(si);
  // The default segment length is ~0.78m, so the 0.2m wall is likely missed
  EXPECT_TRUE(mv_default->checkMotion(s1, s2))
      << "Expected OMPL default to miss the thin wall (test setup issue if this fails)";

  // With collision resolution: wall should be detected
  space->setCollisionResolution(0.05);
  si->setup();  // recalculate internal state

  auto mv_fine = std::make_shared<ob::DiscreteMotionValidator>(si);
  EXPECT_FALSE(mv_fine->checkMotion(s1, s2))
      << "With collision resolution 0.05m, the 0.2m wall must be detected";

  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (d): Thin wall detection with rotation
// -----------------------------------------------------------------------

TEST(GeodexStateSpaceTest, CollisionCheck_ThinWallDetected_WithRotation) {
  auto space = makeLargeSpace();
  space->setCollisionResolution(0.05);

  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker(std::make_shared<ThinWallChecker>(si));
  si->setup();

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  setState(s1, 2.0, 5.0, 0.0);
  setState(s2, 8.0, 5.0, std::numbers::pi / 2.0);

  auto mv = std::make_shared<ob::DiscreteMotionValidator>(si);
  EXPECT_FALSE(mv->checkMotion(s1, s2)) << "Wall must be detected even when path includes rotation";

  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (e): Interpolated path stays collision-free on a valid edge
// -----------------------------------------------------------------------

TEST(GeodexStateSpaceTest, Interpolation_StaysCollisionFree) {
  auto space = makeLargeSpace();
  space->setCollisionResolution(0.05);

  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker(std::make_shared<ThinWallChecker>(si));
  si->setup();

  // Both states on the same side of the wall — valid edge
  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  setState(s1, 1.0, 5.0, 0.0);
  setState(s2, 4.0, 10.0, 1.0);

  auto mv = std::make_shared<ob::DiscreteMotionValidator>(si);
  EXPECT_TRUE(mv->checkMotion(s1, s2)) << "Edge that doesn't cross the wall should be valid";

  // Interpolate and verify all points are valid
  unsigned int n = space->validSegmentCount(s1, s2);
  auto* interp = space->allocState();
  for (unsigned int i = 0; i <= n; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(n);
    space->interpolate(s1, s2, t, interp);
    EXPECT_TRUE(si->isValid(interp)) << "Interpolated state at t=" << t << " is invalid";
  }

  space->freeState(interp);
  space->freeState(s1);
  space->freeState(s2);
}

// ========================================================================
// Discrete geodesic interpolation tests
// ========================================================================

// -----------------------------------------------------------------------
// Test (f): Identity metric takes the fast path (no discrete geodesic)
// -----------------------------------------------------------------------

TEST(GeodexDiscreteGeodesicTest, IdentityMetric_InterpolateUnchanged) {
  geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 1.0};  // unit weights → fast path
  geodex::SE2<> manifold{metric};

  auto bounds = makeSE2Bounds(0.0, 10.0, 0.0, 10.0);
  auto space = std::make_shared<StateSpace>(manifold, bounds);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  setState(s1, 2.0, 3.0, 0.5);
  setState(s2, 8.0, 7.0, -1.0);

  // Auto mode: identity metric should take the fast path via is_riemannian_log
  // (default is nullopt = auto)

  // Compute expected result using manifold.geodesic() directly
  auto p1 = s1->as<StateType>()->asEigen();
  auto p2 = s2->as<StateType>()->asEigen();

  for (int j = 1; j <= 10; ++j) {
    double t = j / 10.0;
    space->interpolate(s1, s2, t, result);
    auto expected = manifold.geodesic(p1, p2, t);
    auto* r = result->as<StateType>();
    for (int i = 0; i < 3; ++i) {
      EXPECT_DOUBLE_EQ(r->values[i], expected[i]) << "Mismatch at t=" << t << " dim=" << i;
    }
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (g): DisableFlag forces simple geodesic even for anisotropic metric
// -----------------------------------------------------------------------

TEST(GeodexDiscreteGeodesicTest, DisableFlag_ForcesSimpleGeodesic) {
  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  AnisotropicSE2 manifold{metric};

  auto bounds = makeSE2Bounds(0.0, 10.0, 0.0, 10.0);
  auto space = std::make_shared<AnisotropicSE2Space>(manifold, bounds);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::BaseGeodesic);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  s1->as<AnisotropicSE2State>()->values[0] = 2.0;
  s1->as<AnisotropicSE2State>()->values[1] = 3.0;
  s1->as<AnisotropicSE2State>()->values[2] = 0.0;
  s2->as<AnisotropicSE2State>()->values[0] = 8.0;
  s2->as<AnisotropicSE2State>()->values[1] = 7.0;
  s2->as<AnisotropicSE2State>()->values[2] = 0.0;

  // With discrete geodesic disabled, result should match manifold.geodesic()
  auto p1 = s1->as<AnisotropicSE2State>()->asEigen();
  auto p2 = s2->as<AnisotropicSE2State>()->asEigen();

  for (int j = 1; j <= 10; ++j) {
    double t = j / 10.0;
    space->interpolate(s1, s2, t, result);
    auto expected = manifold.geodesic(p1, p2, t);
    auto* r = result->as<AnisotropicSE2State>();
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(r->values[i], expected[i], 1e-12)
          << "Disabled discrete geodesic should match manifold.geodesic() at t=" << t;
    }
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (h): Boundary values return exact endpoints
// -----------------------------------------------------------------------

TEST(GeodexDiscreteGeodesicTest, BoundaryValues_ExactEndpoints) {
  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  AnisotropicSE2 manifold{metric};

  auto bounds = makeSE2Bounds(0.0, 10.0, 0.0, 10.0);
  auto space = std::make_shared<AnisotropicSE2Space>(manifold, bounds);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  s1->as<AnisotropicSE2State>()->values[0] = 2.0;
  s1->as<AnisotropicSE2State>()->values[1] = 3.0;
  s1->as<AnisotropicSE2State>()->values[2] = 0.5;
  s2->as<AnisotropicSE2State>()->values[0] = 8.0;
  s2->as<AnisotropicSE2State>()->values[1] = 7.0;
  s2->as<AnisotropicSE2State>()->values[2] = -1.0;

  // t=0 should return exact start
  space->interpolate(s1, s2, 0.0, result);
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(result->as<AnisotropicSE2State>()->values[i],
                     s1->as<AnisotropicSE2State>()->values[i])
        << "t=0 should return exact start at dim=" << i;
  }

  // t=1 should return exact end
  space->interpolate(s1, s2, 1.0, result);
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(result->as<AnisotropicSE2State>()->values[i],
                     s2->as<AnisotropicSE2State>()->values[i])
        << "t=1 should return exact end at dim=" << i;
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (i): Anisotropic Euclidean — monotone distance along interpolation
// -----------------------------------------------------------------------

TEST(GeodexDiscreteGeodesicTest, AnisotropicEuclidean_MonotoneDistance) {
  Eigen::Matrix2d A;
  A << 4.0, 0.0, 0.0, 1.0;  // 4x weight on x-axis
  geodex::ConstantSPDMetric<2> metric{A};
  AnisotropicEuclidean manifold{metric};

  ob::RealVectorBounds bounds(2);
  bounds.setLow(0.0);
  bounds.setHigh(10.0);
  auto space = std::make_shared<AnisotropicEuclideanSpace>(manifold, bounds);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  s1->as<AnisotropicEuclideanState>()->values[0] = 1.0;
  s1->as<AnisotropicEuclideanState>()->values[1] = 1.0;
  s2->as<AnisotropicEuclideanState>()->values[0] = 9.0;
  s2->as<AnisotropicEuclideanState>()->values[1] = 9.0;

  // distance(from, interp(t)) should be monotonically increasing
  double prev_dist = 0.0;
  const int N = 20;
  for (int j = 1; j <= N; ++j) {
    double t = static_cast<double>(j) / N;
    space->interpolate(s1, s2, t, result);
    double d = space->distance(s1, result);
    EXPECT_GE(d, prev_dist - 1e-10) << "Distance should be monotonically increasing at t=" << t;
    prev_dist = d;
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (j): Anisotropic SE2 — discrete geodesic path has lower energy
// -----------------------------------------------------------------------

TEST(GeodexDiscreteGeodesicTest, AnisotropicSE2_BetterThanRetraction) {
  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  AnisotropicSE2 manifold{metric};

  auto bounds = makeSE2Bounds(0.0, 10.0, 0.0, 10.0);
  auto space = std::make_shared<AnisotropicSE2Space>(manifold, bounds);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  s1->as<AnisotropicSE2State>()->values[0] = 2.0;
  s1->as<AnisotropicSE2State>()->values[1] = 2.0;
  s1->as<AnisotropicSE2State>()->values[2] = 0.0;
  s2->as<AnisotropicSE2State>()->values[0] = 8.0;
  s2->as<AnisotropicSE2State>()->values[1] = 8.0;
  s2->as<AnisotropicSE2State>()->values[2] = 1.0;

  auto p1 = s1->as<AnisotropicSE2State>()->asEigen();
  auto p2 = s2->as<AnisotropicSE2State>()->asEigen();

  // Compute total path length with discrete geodesic interpolation
  const int N = 20;
  auto* prev = space->allocState();
  auto* curr = space->allocState();
  space->copyState(prev, s1);

  double discrete_length = 0.0;
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);
  for (int j = 1; j <= N; ++j) {
    space->interpolate(s1, s2, static_cast<double>(j) / N, curr);
    discrete_length += space->distance(prev, curr);
    space->copyState(prev, curr);
  }

  // Compute total path length with simple retraction interpolation
  double retraction_length = 0.0;
  space->copyState(prev, s1);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::BaseGeodesic);
  for (int j = 1; j <= N; ++j) {
    space->interpolate(s1, s2, static_cast<double>(j) / N, curr);
    retraction_length += space->distance(prev, curr);
    space->copyState(prev, curr);
  }

  // Discrete geodesic path should be shorter or equal (energy-minimizing)
  EXPECT_LE(discrete_length, retraction_length + 1e-6)
      << "Discrete geodesic path should have lower or equal total length.\n"
      << "  discrete: " << discrete_length << "\n"
      << "  retraction: " << retraction_length;

  space->freeState(curr);
  space->freeState(prev);
  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (k): Anisotropic SE2 — thin wall still detected
// -----------------------------------------------------------------------

TEST(GeodexDiscreteGeodesicTest, AnisotropicSE2_ThinWallDetected) {
  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  AnisotropicSE2 manifold{metric};

  auto bounds = makeSE2Bounds(0.0, 60.0, 0.0, 48.0);
  auto space = std::make_shared<AnisotropicSE2Space>(manifold, bounds);
  space->setCollisionResolution(0.05);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);

  auto si = std::make_shared<ob::SpaceInformation>(space);
  // Thin wall at x in [5.0, 5.2]
  si->setStateValidityChecker([](const ob::State* state) {
    const auto* s = state->as<AnisotropicSE2State>();
    double x = s->values[0];
    return x <= 5.0 || x >= 5.2;
  });
  si->setup();

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  s1->as<AnisotropicSE2State>()->values[0] = 2.0;
  s1->as<AnisotropicSE2State>()->values[1] = 5.0;
  s1->as<AnisotropicSE2State>()->values[2] = 0.0;
  s2->as<AnisotropicSE2State>()->values[0] = 8.0;
  s2->as<AnisotropicSE2State>()->values[1] = 5.0;
  s2->as<AnisotropicSE2State>()->values[2] = 0.0;

  auto mv = std::make_shared<ob::DiscreteMotionValidator>(si);
  EXPECT_FALSE(mv->checkMotion(s1, s2))
      << "Thin wall must be detected with anisotropic metric + discrete geodesic";

  space->freeState(s1);
  space->freeState(s2);
}

// -----------------------------------------------------------------------
// Test (l): Convergence failure falls back gracefully
// -----------------------------------------------------------------------

TEST(GeodexDiscreteGeodesicTest, ConvFailure_GracefulFallback) {
  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  AnisotropicSE2 manifold{metric};

  auto bounds = makeSE2Bounds(0.0, 10.0, 0.0, 10.0);
  auto space = std::make_shared<AnisotropicSE2Space>(manifold, bounds);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);

  // Use very restrictive settings to force convergence issues
  geodex::InterpolationSettings settings;
  settings.max_steps = 1;  // very limited budget
  settings.step_size = 0.01;
  space->setInterpolationSettings(settings);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  s1->as<AnisotropicSE2State>()->values[0] = 1.0;
  s1->as<AnisotropicSE2State>()->values[1] = 1.0;
  s1->as<AnisotropicSE2State>()->values[2] = 0.0;
  s2->as<AnisotropicSE2State>()->values[0] = 9.0;
  s2->as<AnisotropicSE2State>()->values[1] = 9.0;
  s2->as<AnisotropicSE2State>()->values[2] = 2.0;

  // Should not crash — falls back to simple geodesic or uses partial path
  EXPECT_NO_FATAL_FAILURE({ space->interpolate(s1, s2, 0.5, result); });

  // Result should be a valid state (not NaN)
  auto* r = result->as<AnisotropicSE2State>();
  for (int i = 0; i < 3; ++i) {
    EXPECT_FALSE(std::isnan(r->values[i])) << "Result should not be NaN at dim=" << i;
    EXPECT_TRUE(std::isfinite(r->values[i])) << "Result should be finite at dim=" << i;
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}

/// @file test_geodex_informed_sampling.cpp
/// @brief Tests for direct-sampling strategies in `GeodexDirectInfSampler`.

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <ompl/base/Cost.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/util/ProlateHyperspheroid.h>

#include "geodex/heuristics/eigenvalue_lower_bound.hpp"
#include "geodex/heuristics/euclidean.hpp"
#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/integration/ompl/geodex_informed_sampler.hpp"
#include "geodex/integration/ompl/geodex_optimization_objective.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/manifold/sphere.hpp"
#include "geodex/utils/angle.hpp"

namespace ob = ompl::base;
namespace gio = geodex::integration::ompl;
namespace gh = geodex::heuristics;

using Manifold2D = geodex::Euclidean<2>;
using Manifold3D = geodex::Euclidean<3>;
using Space2D = gio::GeodexStateSpace<Manifold2D>;
using Space3D = gio::GeodexStateSpace<Manifold3D>;
using State2D = gio::GeodexState<Manifold2D>;
using State3D = gio::GeodexState<Manifold3D>;

using ManifoldSE2 = geodex::SE2<>;
using SpaceSE2 = gio::GeodexStateSpace<ManifoldSE2>;
using StateSE2 = gio::GeodexState<ManifoldSE2>;
using MlbDyn = gh::MatrixLowerBound<Eigen::Dynamic>;

namespace {

ob::RealVectorBounds makeBounds(int dim, double lo, double hi) {
  ob::RealVectorBounds b(dim);
  b.setLow(lo);
  b.setHigh(hi);
  return b;
}

ob::RealVectorBounds makeBoundsAxes(double xlo, double xhi, double ylo, double yhi) {
  ob::RealVectorBounds b(2);
  b.setLow(0, xlo);
  b.setHigh(0, xhi);
  b.setLow(1, ylo);
  b.setHigh(1, yhi);
  return b;
}

template <typename Space>
std::pair<ob::SpaceInformationPtr, ob::ProblemDefinitionPtr> makeSiAndPdef(
    std::shared_ptr<Space> space, std::initializer_list<double> start_vals,
    std::initializer_list<double> goal_vals) {
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  auto pdef = std::make_shared<ob::ProblemDefinition>(si);
  ob::ScopedState<Space> start(space);
  ob::ScopedState<Space> goal(space);
  std::size_t i = 0;
  for (double v : start_vals) start->values[i++] = v;
  i = 0;
  for (double v : goal_vals) goal->values[i++] = v;
  pdef->setStartAndGoalStates(start, goal);
  pdef->setOptimizationObjective(std::make_shared<ob::PathLengthOptimizationObjective>(si));
  return {si, pdef};
}

}  // namespace

// ============================================================================
// Euclidean PHS
// ============================================================================

TEST(GeodexInformedSampling, EuclideanPHS_SamplesInsideInformedRegion) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  gio::GeodexDirectInfSampler<gh::Euclidean> sampler(pdef, /*maxNumberCalls=*/100);

  const double max_cost = 10.0;
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(max_cost)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_LE((x - s).norm() + (x - g).norm(), max_cost + 1e-9);
  }
  space->freeState(state);
}

// ============================================================================
// EigenvalueLowerBound, cost scaled by 1/sqrt(lambda_min)
// ============================================================================

TEST(GeodexInformedSampling, EigenvalueLB_SamplesRespectScaledCost) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  // Foci at (-1, 0) and (1, 0), d_foci = 2.
  auto [si, pdef] = makeSiAndPdef(space, {-1.0, 0.0}, {1.0, 0.0});

  // lambda_min = 4 → sqrt = 2. Effective PHS transverse diameter = 10/2 = 5,
  // comfortably above d_foci = 2.
  const double lambda_min = 4.0;
  gh::EigenvalueLowerBound<gh::Euclidean> heuristic{lambda_min};
  gio::GeodexDirectInfSampler<gh::EigenvalueLowerBound<gh::Euclidean>> sampler(pdef, 100,
                                                                              heuristic);

  const double max_cost = 10.0;
  auto* state = space->allocState();
  const Eigen::Vector2d s(-1.0, 0.0);
  const Eigen::Vector2d g(1.0, 0.0);
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(max_cost)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    const double scaled = std::sqrt(lambda_min) * ((x - s).norm() + (x - g).norm());
    EXPECT_LE(scaled, max_cost + 1e-9);
  }
  space->freeState(state);
}

// ============================================================================
// MatrixLowerBound, latent-space PHS with isotropic M_lower
// ============================================================================

TEST(GeodexInformedSampling, MatrixLB_SamplesInsideInformedRegion) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler(pdef, 100, heuristic,
                                                               makeBounds(2, -10, 10));

  const double max_cost = 10.0;
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(max_cost)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    const double cost = heuristic(s, x) + heuristic(x, g);
    EXPECT_LE(cost, max_cost + 1e-9);
    EXPECT_TRUE(space->satisfiesBounds(state));
  }
  space->freeState(state);
}

// ============================================================================
// MatrixLowerBound, anisotropic M_lower exercises latent-space transform
// ============================================================================

TEST(GeodexInformedSampling, MatrixLB_AnisotropicMetricRespectsBoundsAndCost) {
  // Tight y-bound forces the clipped-AABB strategy on the long thin PHS.
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBoundsAxes(-5, 5, -2, 2));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  // Anisotropic M_lower with x stiffer than y by 4x.
  Eigen::Matrix2d M;
  M << 4.0, 0.0, 0.0, 1.0;
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler(pdef, 100, heuristic,
                                                               makeBoundsAxes(-5, 5, -2, 2));

  // h(s,x)+h(x,g) >= h_min = sqrt((g-s)^T M (g-s)) = sqrt(36*4)=12.
  // Use a slightly larger c_best to admit a non-degenerate sampling region.
  const double max_cost = 14.0;
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(max_cost)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    const double cost = heuristic(s, x) + heuristic(x, g);
    EXPECT_LE(cost, max_cost + 1e-9);
    EXPECT_TRUE(space->satisfiesBounds(state));
  }
  space->freeState(state);
}

// ============================================================================
// MatrixLowerBound, empty bounds disable clipped-AABB and PHS sampling works
// ============================================================================

TEST(GeodexInformedSampling, MatrixLB_EmptyBoundsFallsBackToPhsLatent) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> heuristic{M};
  // With empty bounds the sampler falls back to PHS with rejection on bounds.
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler(pdef, 100, heuristic,
                                                               ob::RealVectorBounds(0));

  auto* state = space->allocState();
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(10.0)));
    EXPECT_TRUE(space->satisfiesBounds(state));
  }
  space->freeState(state);
}

// ============================================================================
// Volume-ratio fallback, inadmissibly small M_lower triggers uniform sampling
// ============================================================================

TEST(GeodexInformedSampling, VolumeRatio_FallsBackToUniformOnInadmissibleMetric) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -1, 1));
  auto [si, pdef] = makeSiAndPdef(space, {-0.5, 0.0}, {0.5, 0.0});

  // An inadmissibly small M_lower gives a very large PHS in the original space.
  // det(M_lower) = 1e-8 and sqrt = 1e-4 make the effective PHS volume huge.
  Eigen::Matrix2d M = 1e-4 * Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler(pdef, 100, heuristic,
                                                               makeBounds(2, -1, 1));

  // Confirm the fallback path doesn't crash and yields in-bounds samples.
  auto* state = space->allocState();
  for (int i = 0; i < 100; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(10.0)));
    EXPECT_TRUE(space->satisfiesBounds(state));
  }
  space->freeState(state);
}

// ============================================================================
// Infinite cost, uniform fallback for all branches
// ============================================================================

TEST(GeodexInformedSampling, InfiniteCost_AllStrategiesFallBackToUniform) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto [si, pdef] = makeSiAndPdef(space, {-1.0, 0.0}, {1.0, 0.0});

  const ob::Cost inf{std::numeric_limits<double>::infinity()};

  // Euclidean
  {
    gio::GeodexDirectInfSampler<gh::Euclidean> s{pdef, 50};
    auto* state = space->allocState();
    EXPECT_TRUE(s.sampleUniform(state, inf));
    EXPECT_TRUE(space->satisfiesBounds(state));
    space->freeState(state);
  }
  // EigenvalueLB
  {
    gh::EigenvalueLowerBound<gh::Euclidean> h{2.0};
    gio::GeodexDirectInfSampler<gh::EigenvalueLowerBound<gh::Euclidean>> s{pdef, 50, h};
    auto* state = space->allocState();
    EXPECT_TRUE(s.sampleUniform(state, inf));
    EXPECT_TRUE(space->satisfiesBounds(state));
    space->freeState(state);
  }
  // MatrixLB
  {
    Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
    gh::MatrixLowerBound<2> h{M};
    gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> s{pdef, 50, h, makeBounds(2, -5, 5)};
    auto* state = space->allocState();
    EXPECT_TRUE(s.sampleUniform(state, inf));
    EXPECT_TRUE(space->satisfiesBounds(state));
    space->freeState(state);
  }
}

// ============================================================================
// Below-minimum-transverse-diameter, fallback to uniform without crashing
// ============================================================================

TEST(GeodexInformedSampling, DegeneratePHS_FallsBackToUniformWithoutCrashing) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  // d_foci = 6. Dividing a cost just above 6 by sqrt(lambda_min) = 2 pulls it below the
  // minimum transverse diameter.
  gh::EigenvalueLowerBound<gh::Euclidean> heuristic{4.0};
  gio::GeodexDirectInfSampler<gh::EigenvalueLowerBound<gh::Euclidean>> sampler{pdef, 50, heuristic};

  // c_best/sqrt(lambda_min) = 6.5/2 = 3.25 < d_foci=6 → fallback to uniform.
  const ob::Cost low_cost{6.5};
  auto* state = space->allocState();
  EXPECT_TRUE(sampler.sampleUniform(state, low_cost));
  EXPECT_TRUE(space->satisfiesBounds(state));
  space->freeState(state);
}

TEST(GeodexInformedSampling, ExactMinimumCost_SamplesFocalSegment) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);

  {
    gio::GeodexDirectInfSampler<gh::Euclidean> sampler{pdef, 20};
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(6.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_LE((x - s).norm() + (x - g).norm(), 6.0 + 1e-12);
  }

  {
    gh::EigenvalueLowerBound<gh::Euclidean> heuristic{4.0};
    gio::GeodexDirectInfSampler<gh::EigenvalueLowerBound<gh::Euclidean>> sampler{pdef, 20,
                                                                                 heuristic};
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(12.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_LE(heuristic(s, x) + heuristic(x, g), 12.0 + 1e-12);
  }

  {
    Eigen::Matrix2d M;
    M << 4.0, 0.0, 0.0, 1.0;
    gh::MatrixLowerBound<2> heuristic{M};
    gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler{pdef, 20, heuristic,
                                                                 makeBounds(2, -10, 10)};
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(12.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_NEAR(x.y(), 0.0, 1e-12);
    EXPECT_LE(heuristic(s, x) + heuristic(x, g), 12.0 + 1e-12);
  }

  space->freeState(state);
}

// ============================================================================
// Custom heuristic, rejection sampling works for non-trait callables
// ============================================================================

namespace {
struct CustomHeuristic {
  template <typename A, typename B>
  auto operator()(const A& a, const B& b) const -> double {
    // Inflated Euclidean, admissible for any metric with h <= 1.5*||.||.
    return 1.5 * (a - b).norm();
  }
};
}  // namespace

TEST(GeodexInformedSampling, CustomHeuristic_FallsBackToRejectionSampling) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto [si, pdef] = makeSiAndPdef(space, {-2.0, 0.0}, {2.0, 0.0});

  CustomHeuristic h{};
  gio::GeodexDirectInfSampler<CustomHeuristic> sampler{pdef, 200, h};

  // 1.5 * (||x-s|| + ||x-g||) <= 12 → ||x-s|| + ||x-g|| <= 8
  const double max_cost = 12.0;
  auto* state = space->allocState();
  const Eigen::Vector2d s(-2.0, 0.0);
  const Eigen::Vector2d g(2.0, 0.0);
  int succeeded = 0;
  for (int i = 0; i < 200; ++i) {
    if (sampler.sampleUniform(state, ob::Cost(max_cost))) {
      ++succeeded;
      Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
      EXPECT_LE(1.5 * ((x - s).norm() + (x - g).norm()), max_cost + 1e-9);
      EXPECT_TRUE(space->satisfiesBounds(state));
    }
  }
  // Most attempts should succeed for this problem.
  EXPECT_GT(succeeded, 150);
  space->freeState(state);
}

// ============================================================================
// hasInformedMeasure, true for trait-recognized heuristics and false otherwise
// ============================================================================

TEST(GeodexInformedSampling, HasInformedMeasure_DispatchesByTrait) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto [si, pdef] = makeSiAndPdef(space, {-1.0, 0.0}, {1.0, 0.0});

  gio::GeodexDirectInfSampler<gh::Euclidean> euc{pdef, 50};
  EXPECT_TRUE(euc.hasInformedMeasure());

  gh::EigenvalueLowerBound<gh::Euclidean> elb{2.0};
  gio::GeodexDirectInfSampler<gh::EigenvalueLowerBound<gh::Euclidean>> elb_s{pdef, 50, elb};
  EXPECT_TRUE(elb_s.hasInformedMeasure());

  Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> mlb{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> mlb_s{pdef, 50, mlb};
  EXPECT_TRUE(mlb_s.hasInformedMeasure());

  CustomHeuristic ch{};
  gio::GeodexDirectInfSampler<CustomHeuristic> ch_s{pdef, 50, ch};
  EXPECT_FALSE(ch_s.hasInformedMeasure());
}

// ============================================================================
// getInformedMeasure, sane volumes for finite and infinite costs
// ============================================================================

TEST(GeodexInformedSampling, GetInformedMeasure_ReturnsSaneVolumes) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto [si, pdef] = makeSiAndPdef(space, {-1.0, 0.0}, {1.0, 0.0});

  // For Euclidean a finite cost above the minimum transverse diameter gives a finite measure.
  gio::GeodexDirectInfSampler<gh::Euclidean> euc{pdef, 50};
  EXPECT_GT(euc.getInformedMeasure(ob::Cost(5.0)), 0.0);
  EXPECT_LT(euc.getInformedMeasure(ob::Cost(5.0)), 100.0);  // < space measure (10x10)
  // Below min trans diameter (foci distance = 2) → 0.
  EXPECT_DOUBLE_EQ(euc.getInformedMeasure(ob::Cost(1.0)), 0.0);
  // Infinite cost → space measure.
  EXPECT_DOUBLE_EQ(euc.getInformedMeasure(ob::Cost(std::numeric_limits<double>::infinity())),
                   space->getMeasure());

  // MatrixLB volumes are positive, finite and below the space measure.
  Eigen::Matrix2d M = 4.0 * Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> mlb{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> mlb_s{pdef, 50, mlb};
  // The latent foci distance is sqrt((g-s)^T M (g-s)) = 4, and c_best must exceed 4.
  EXPECT_GT(mlb_s.getInformedMeasure(ob::Cost(8.0)), 0.0);
  EXPECT_DOUBLE_EQ(mlb_s.getInformedMeasure(ob::Cost(2.0)), 0.0);
}

// ============================================================================
// Deck-group lifts on SE(2)
// ============================================================================

namespace {

// SE(2) over [-5, 5]^2 with theta spanning exactly one period.
std::shared_ptr<SpaceSE2> makeSE2Space() {
  ob::RealVectorBounds b(3);
  b.setLow(0, -5.0);
  b.setHigh(0, 5.0);
  b.setLow(1, -5.0);
  b.setHigh(1, 5.0);
  b.setLow(2, -std::numbers::pi);
  b.setHigh(2, std::numbers::pi);
  return std::make_shared<SpaceSE2>(ManifoldSE2{}, b);
}

Eigen::VectorXd se2Periods() { return Eigen::Vector3d(0.0, 0.0, geodex::utils::two_pi); }

Eigen::MatrixXd identity3() { return Eigen::MatrixXd::Identity(3, 3); }

// The informed measure of a lift union is the sum of its hyperspheroid measures. Start
// theta 3 and goal theta -3 put the goal lifts (k in {-1, 0, +1} on theta) 6 + 2 pi, 6 and
// 2 pi - 6 from the start.
double liftMeasure(const double foci_distance, const double cost) {
  const double s[3] = {0.0, 0.0, 0.0};
  const double g[3] = {foci_distance, 0.0, 0.0};
  return ompl::ProlateHyperspheroid(3, s, g).getPhsMeasure(cost);
}

}  // namespace

TEST(GeodexInformedLifts, PeriodicAxisBuildsThreeLifts) {
  // Wide x and y bounds keep the union below the C-space measure, which caps it.
  ob::RealVectorBounds b(3);
  b.setLow(0, -50.0);
  b.setHigh(0, 50.0);
  b.setLow(1, -50.0);
  b.setHigh(1, 50.0);
  b.setLow(2, -std::numbers::pi);
  b.setHigh(2, std::numbers::pi);
  auto space = std::make_shared<SpaceSE2>(ManifoldSE2{}, b);
  auto [si, pdef] = makeSiAndPdef(space, {0.0, 0.0, 3.0}, {0.0, 0.0, -3.0});
  MlbDyn heuristic{identity3(), se2Periods()};
  gio::GeodexDirectInfSampler<MlbDyn> sampler(pdef, 100, heuristic, space->getBounds());
  const double c = 13.0;
  const double sum = liftMeasure(6.0 + geodex::utils::two_pi, c) + liftMeasure(6.0, c) +
                     liftMeasure(geodex::utils::two_pi - 6.0, c);
  ASSERT_LT(sum, space->getMeasure());
  EXPECT_NEAR(sampler.getInformedMeasure(ob::Cost(c)), sum, 1e-9 * sum);
}

TEST(GeodexInformedLifts, EmptyDeckGroupYieldsExactlyOnePhs) {
  auto space = makeSE2Space();
  auto [si, pdef] = makeSiAndPdef(space, {0.0, 0.0, 3.0}, {0.0, 0.0, -3.0});
  MlbDyn heuristic{identity3()};  // the robot case
  gio::GeodexDirectInfSampler<MlbDyn> sampler(pdef, 100, heuristic, space->getBounds());
  // The short route through the cut is not in the informed set.
  EXPECT_EQ(sampler.getInformedMeasure(ob::Cost(1.0)), 0.0);
  EXPECT_NEAR(sampler.getInformedMeasure(ob::Cost(7.0)), liftMeasure(6.0, 7.0), 1e-12);
}

TEST(GeodexInformedLifts, LiftsNeedBoundsForTheFundamentalDomain) {
  auto space = makeSE2Space();
  auto [si, pdef] = makeSiAndPdef(space, {0.0, 0.0, 3.0}, {0.0, 0.0, -3.0});
  MlbDyn heuristic{identity3(), se2Periods()};
  gio::GeodexDirectInfSampler<MlbDyn> sampler(pdef, 100, heuristic);
  EXPECT_EQ(sampler.getInformedMeasure(ob::Cost(1.0)), 0.0);
  EXPECT_NEAR(sampler.getInformedMeasure(ob::Cost(7.0)), liftMeasure(6.0, 7.0), 1e-12);
}

TEST(GeodexInformedLifts, SamplesLandOnBothSidesOfTheCut) {
  // Start near +pi and goal near -pi are 2 eps apart through the cut. A single PHS puts
  // its foci 2 pi - 2 eps apart and does not represent this.
  constexpr double eps = 0.1;
  auto space = makeSE2Space();
  const double ts = std::numbers::pi - eps, tg = -std::numbers::pi + eps;
  auto [si, pdef] = makeSiAndPdef(space, {0.0, 0.0, ts}, {0.0, 0.0, tg});
  MlbDyn heuristic{identity3(), se2Periods()};
  gio::GeodexDirectInfSampler<MlbDyn> sampler(pdef, 100, heuristic, space->getBounds());

  const double max_cost = 4.0 * eps;
  auto* state = space->allocState();
  int above = 0, below = 0;
  for (int i = 0; i < 400; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(max_cost)));
    const double theta = state->as<StateSE2>()->values[2];
    if (theta > 0.0) ++above;
    if (theta < 0.0) ++below;
  }
  space->freeState(state);
  EXPECT_GT(above, 0);
  EXPECT_GT(below, 0);
  // The volume-ratio fallback did not run.
  EXPECT_LE(sampler.getSamplingStats().last_volume_ratio, sampler.getVolumeRatioThreshold());
}

TEST(GeodexInformedLifts, EverySampleSatisfiesTheWrappedInformedSet) {
  constexpr double eps = 0.1;
  auto space = makeSE2Space();
  const Eigen::Vector3d s(0.0, 0.0, std::numbers::pi - eps);
  const Eigen::Vector3d g(0.0, 0.0, -std::numbers::pi + eps);
  auto [si, pdef] = makeSiAndPdef(space, {s[0], s[1], s[2]}, {g[0], g[1], g[2]});
  MlbDyn heuristic{identity3(), se2Periods()};
  gio::GeodexDirectInfSampler<MlbDyn> sampler(pdef, 100, heuristic, space->getBounds());

  const double max_cost = 4.0 * eps;
  auto* state = space->allocState();
  for (int i = 0; i < 400; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(max_cost)));
    Eigen::Map<const Eigen::Vector3d> x(state->as<StateSE2>()->values);
    EXPECT_LE(heuristic(s, x) + heuristic(x, g), max_cost + 1e-9);
    EXPECT_TRUE(space->satisfiesBounds(state));
  }
  space->freeState(state);
}

TEST(GeodexInformedLifts, OverlapAcceptRejectsSomeFoldedSamples) {
  // A budget wide enough for several lifts to overlap makes the overlap accept
  // reject some folded samples. Those are the attempts without another outcome.
  auto space = makeSE2Space();
  auto [si, pdef] = makeSiAndPdef(space, {0.0, 0.0, 3.0}, {0.0, 0.0, -3.0});
  MlbDyn heuristic{identity3(), se2Periods()};
  gio::GeodexDirectInfSampler<MlbDyn> sampler(pdef, 100, heuristic, space->getBounds());
  auto* state = space->allocState();
  for (int i = 0; i < 400; ++i) sampler.sampleUniform(state, ob::Cost(8.0));
  space->freeState(state);
  const auto stats = sampler.getSamplingStats();
  EXPECT_EQ(stats.phs_rejections, 0u);
  EXPECT_GT(stats.total_attempts, stats.accepted + stats.bounds_rejections);
}

TEST(GeodexInformedLifts, MinTransverseDiameterIsTakenOverLifts) {
  // The union has positive measure below the base lift's own minimum, where a
  // single PHS reports an empty informed set.
  constexpr double eps = 0.1;
  auto space = makeSE2Space();
  auto [si, pdef] = makeSiAndPdef(space, {0.0, 0.0, std::numbers::pi - eps},
                                  {0.0, 0.0, -std::numbers::pi + eps});
  MlbDyn heuristic{identity3(), se2Periods()};
  gio::GeodexDirectInfSampler<MlbDyn> lifted(pdef, 100, heuristic, space->getBounds());
  gio::GeodexDirectInfSampler<MlbDyn> single(pdef, 100, MlbDyn{identity3()},
                                             space->getBounds());

  const ob::Cost c(4.0 * eps);
  EXPECT_GT(lifted.getInformedMeasure(c), 0.0);
  EXPECT_EQ(single.getInformedMeasure(c), 0.0);
}

TEST(GeodexInformedLifts, UnionMeasureNeverExceedsTheCSpaceMeasure) {
  auto space = makeSE2Space();
  auto [si, pdef] = makeSiAndPdef(space, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0});
  MlbDyn heuristic{identity3(), se2Periods()};
  gio::GeodexDirectInfSampler<MlbDyn> sampler(pdef, 100, heuristic, space->getBounds());
  EXPECT_LE(sampler.getInformedMeasure(ob::Cost(500.0)), space->getMeasure() + 1e-9);
}

TEST(GeodexInformedLifts, AperiodicSamplingIsBitIdentical) {
  // Empty periods reproduce the plain constructor exactly. Robot plans rely on this.
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});
  Eigen::MatrixXd M = Eigen::MatrixXd::Identity(2, 2);
  M(0, 0) = 4.0;

  // Both samplers need the same low-discrepancy scramble to be comparable.
  geodex::set_default_seed(7);
  gio::GeodexDirectInfSampler<MlbDyn> a(pdef, 1, MlbDyn{M}, makeBounds(2, -10, 10));
  geodex::set_default_seed(7);
  gio::GeodexDirectInfSampler<MlbDyn> b(pdef, 1, MlbDyn{M, Eigen::VectorXd{}},
                                        makeBounds(2, -10, 10));

  auto* sa = space->allocState();
  auto* sb = space->allocState();
  for (int i = 0; i < 50; ++i) {
    const bool oka = a.sampleUniform(sa, ob::Cost(9.0));
    const bool okb = b.sampleUniform(sb, ob::Cost(9.0));
    ASSERT_EQ(oka, okb);
    if (!oka) continue;
    for (int d = 0; d < 2; ++d) {
      EXPECT_DOUBLE_EQ(sa->as<State2D>()->values[d], sb->as<State2D>()->values[d]);
    }
  }
  space->freeState(sa);
  space->freeState(sb);
}

// ============================================================================
// Annular sampler, sampleUniform(min, max) excludes the inner region
// ============================================================================

TEST(GeodexInformedSampling, AnnularSampling_RespectsLowerBound) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  gio::GeodexDirectInfSampler<gh::Euclidean> sampler{pdef, 200};

  const ob::Cost min_cost{8.0};
  const ob::Cost max_cost{12.0};
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  int hits = 0;
  for (int i = 0; i < 50; ++i) {
    if (sampler.sampleUniform(state, min_cost, max_cost)) {
      ++hits;
      Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
      const double cost = (x - s).norm() + (x - g).norm();
      EXPECT_GE(cost + 1e-12, min_cost.value());
      EXPECT_LE(cost, max_cost.value() + 1e-9);
    }
  }
  EXPECT_GT(hits, 0);
  space->freeState(state);
}

TEST(GeodexInformedSampling, AnnularSampling_UsesSingleAttemptBudget) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto [si, pdef] = makeSiAndPdef(space, {-2.0, 0.0}, {2.0, 0.0});

  CustomHeuristic h{};
  gio::GeodexDirectInfSampler<CustomHeuristic> sampler{pdef, 11, h};

  auto* state = space->allocState();
  EXPECT_FALSE(sampler.sampleUniform(state, ob::Cost(0.0), ob::Cost(0.0)));
  EXPECT_EQ(sampler.getSamplingStats().total_attempts, 11u);
  space->freeState(state);
}

// ============================================================================
// SamplingStats counters
// ============================================================================

TEST(GeodexInformedSampling, Stats_AccumulateAcceptsAndAttempts) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  gio::GeodexDirectInfSampler<gh::Euclidean> sampler{pdef, 100};

  auto* state = space->allocState();
  for (int i = 0; i < 50; ++i) {
    sampler.sampleUniform(state, ob::Cost(10.0));
  }
  const auto stats = sampler.getSamplingStats();
  EXPECT_GT(stats.total_attempts, 0u);
  EXPECT_GT(stats.accepted, 0u);
  EXPECT_LE(stats.accepted, stats.total_attempts);
  EXPECT_EQ(stats.total_attempts, stats.accepted + stats.bounds_rejections + stats.phs_rejections);
  space->freeState(state);
}

TEST(GeodexInformedSampling, Stats_VolumeRatioAboveThresholdTriggersFallback) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -1, 1));
  auto [si, pdef] = makeSiAndPdef(space, {-0.5, 0.0}, {0.5, 0.0});

  // Inadmissibly small M_lower -> latent PHS volume swamps space measure.
  Eigen::Matrix2d M = 1e-4 * Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler{pdef, 50, heuristic,
                                                               makeBounds(2, -1, 1)};
  sampler.setVolumeRatioThreshold(0.5);

  auto* state = space->allocState();
  for (int i = 0; i < 30; ++i) sampler.sampleUniform(state, ob::Cost(10.0));
  const auto stats = sampler.getSamplingStats();
  EXPECT_GT(stats.last_volume_ratio, sampler.getVolumeRatioThreshold());
  space->freeState(state);
}

TEST(GeodexInformedSampling, VolumeRatioThresholdZeroDisablesFallback) {
  // The volume ratio of this problem is positive. With the fallback disabled, the samples
  // equal those of a threshold the ratio never reaches.
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -1, 1));
  auto [si, pdef] = makeSiAndPdef(space, {-0.5, 0.0}, {0.5, 0.0});

  Eigen::Matrix2d M = 1e-4 * Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> heuristic{M};
  double ratio = 0.0;
  auto samples = [&](const double threshold) {
    geodex::set_default_seed(7);
    gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler{pdef, 50, heuristic,
                                                                 makeBounds(2, -1, 1)};
    sampler.setVolumeRatioThreshold(threshold);
    auto* state = space->allocState();
    std::vector<double> out;
    for (int i = 0; i < 30; ++i) {
      EXPECT_TRUE(sampler.sampleUniform(state, ob::Cost(10.0)));
      out.push_back(state->as<State2D>()->values[0]);
      out.push_back(state->as<State2D>()->values[1]);
    }
    space->freeState(state);
    ratio = sampler.getSamplingStats().last_volume_ratio;
    return out;
  };
  const auto disabled = samples(0.0);
  EXPECT_GT(ratio, 0.0);
  EXPECT_EQ(disabled, samples(std::numeric_limits<double>::max()));
}

TEST(GeodexInformedSampling, VolumeRatioFallbackStillRespectsFiniteCost) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  gio::GeodexDirectInfSampler<gh::Euclidean> sampler{pdef, 10000};
  sampler.setVolumeRatioThreshold(1e-3);

  const ob::Cost max_cost{7.0};
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  for (int i = 0; i < 50; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, max_cost));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_LE((x - s).norm() + (x - g).norm(), max_cost.value() + 1e-9);
  }
  EXPECT_GT(sampler.getSamplingStats().last_volume_ratio, sampler.getVolumeRatioThreshold());
  space->freeState(state);
}

// Every returned sample is counted once, as informed (accepted) or as a plain uniform
// sample, also when the volume-ratio fallback redirects to rejection sampling.
TEST(GeodexInformedSampling, CountersSplitEveryReturnedSample) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});
  gio::GeodexDirectInfSampler<gh::Euclidean> sampler{pdef, 10000};
  sampler.setVolumeRatioThreshold(1e-3);
  auto* state = space->allocState();
  unsigned long returned = 0;
  for (int i = 0; i < 50; ++i) returned += sampler.sampleUniform(state, ob::Cost(7.0)) ? 1 : 0;
  for (int i = 0; i < 20; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(std::numeric_limits<double>::infinity())));
    ++returned;
  }
  space->freeState(state);
  const auto stats = sampler.getSamplingStats();
  EXPECT_GT(stats.last_volume_ratio, sampler.getVolumeRatioThreshold());
  EXPECT_EQ(stats.uniform_samples, 20u);
  EXPECT_EQ(stats.accepted + stats.uniform_samples, returned);
}

// Latent foci (-6, 0) and (6, 0) and cost 14 give a latent PHS of area pi * 7 * sqrt(13),
// about 79.3. Clipped to the latent bounds [-10, 10] x [-2, 2], its bounding box has area
// 14 * 4 = 56, and the sampler picks the clipped box.
TEST(GeodexInformedSampling, Stats_ReportsClippedAABBStrategyForMatrixLB) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBoundsAxes(-5, 5, -2, 2));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Matrix2d M;
  M << 4.0, 0.0, 0.0, 1.0;
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler{pdef, 50, heuristic,
                                                               makeBoundsAxes(-5, 5, -2, 2)};

  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  auto* state = space->allocState();
  for (int i = 0; i < 20; ++i) {
    ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(14.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_LE(heuristic(s, x) + heuristic(x, g), 14.0 + 1e-9);
    EXPECT_TRUE(space->satisfiesBounds(state));
  }
  EXPECT_TRUE(sampler.getSamplingStats().using_clipped_aabb);
  space->freeState(state);
}

TEST(GeodexInformedSampling, ClippedAABBStrategyNeedsBounds) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBoundsAxes(-5, 5, -2, 2));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Matrix2d M;
  M << 4.0, 0.0, 0.0, 1.0;
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler{pdef, 50, heuristic};

  auto* state = space->allocState();
  sampler.sampleUniform(state, ob::Cost(14.0));
  EXPECT_FALSE(sampler.getSamplingStats().using_clipped_aabb);
  space->freeState(state);
}

// ============================================================================
// Objective, motion cost and last-sampler tracking
// ============================================================================

TEST(GeodexOptimizationObjectiveTest, MotionCost_IsTheEndpointDistance) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  Eigen::Vector2d goal_coords(2.0, 0.0);
  gio::GeodexOptimizationObjective<Manifold2D> obj{si, goal_coords};

  auto* a = space->allocState();
  auto* b = space->allocState();
  a->as<State2D>()->values[0] = 0.0;
  a->as<State2D>()->values[1] = 0.0;
  b->as<State2D>()->values[0] = 1.0;
  b->as<State2D>()->values[1] = 0.0;
  for (int i = 0; i < 7; ++i) EXPECT_NEAR(obj.motionCost(a, b).value(), 1.0, 1e-12);
  space->freeState(a);
  space->freeState(b);
}

TEST(GeodexOptimizationObjectiveTest, MotionCost_ThreadSafeUnderConcurrentCalls) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  Eigen::Vector2d goal_coords(1.0, 0.0);
  gio::GeodexOptimizationObjective<Manifold2D> obj{si, goal_coords};

  auto* a = space->allocState();
  auto* b = space->allocState();
  a->as<State2D>()->values[0] = 0.0;
  a->as<State2D>()->values[1] = 0.0;
  b->as<State2D>()->values[0] = 1.0;
  b->as<State2D>()->values[1] = 0.0;

  const double expected = obj.motionCost(a, b).value();
  constexpr int kThreads = 4;
  constexpr int kPerThread = 1000;
  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < kPerThread; ++i) {
        if (obj.motionCost(a, b).value() != expected) ++mismatches;
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(mismatches.load(), 0);
  space->freeState(a);
  space->freeState(b);
}

TEST(GeodexOptimizationObjectiveTest, GetLastSamplerStats_TracksLatestAllocation) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef_only] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef_only->setOptimizationObjective(obj);

  // Before any sampler is allocated the stats are the defaults.
  EXPECT_EQ(obj->getLastSamplerStats().total_attempts, 0u);

  auto sampler1 = obj->allocInformedStateSampler(pdef_only, 100);
  auto* state = space->allocState();
  for (int i = 0; i < 5; ++i) sampler1->sampleUniform(state, ob::Cost(10.0));
  EXPECT_GT(obj->getLastSamplerStats().total_attempts, 0u);
  const auto attempts_after_first = obj->getLastSamplerStats().total_attempts;

  // Allocate a fresh sampler. last_sampler_ points at it.
  auto sampler2 = obj->allocInformedStateSampler(pdef_only, 100);
  // Stats from sampler2, which has not sampled yet, are zero.
  EXPECT_EQ(obj->getLastSamplerStats().total_attempts, 0u);

  // The stats come from sampler2, not from sampler1.
  for (int i = 0; i < 3; ++i) sampler2->sampleUniform(state, ob::Cost(10.0));
  EXPECT_GT(obj->getLastSamplerStats().total_attempts, 0u);
  EXPECT_LT(obj->getLastSamplerStats().total_attempts, attempts_after_first);
  space->freeState(state);
}

// ============================================================================
// Cost-bound feedback, greedy biasing and heuristic-path-cost tightening
// ============================================================================

TEST(GeodexOptimizationObjectiveTest, Feedback_HeuristicPathCostNarrowsSampling) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);

  // Tighten via heuristic-path-cost.
  obj->setHeuristicPathCost(7.0);
  EXPECT_DOUBLE_EQ(obj->getHeuristicPathCost(), 7.0);

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  // The planner's c_best of 20 is loose. The tighter HPC=7 binds.
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_LE((x - s).norm() + (x - g).norm(), 7.0 + 1e-9);
  }
  space->freeState(state);
}

TEST(GeodexOptimizationObjectiveTest, Feedback_GreedyBiasingSamplesTighterEllipsoid) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);

  obj->setGreedyBiasingRatio(1.0);  // always-greedy
  obj->setGreedyCost(7.0);
  EXPECT_DOUBLE_EQ(obj->getGreedyBiasingRatio(), 1.0);
  EXPECT_DOUBLE_EQ(obj->getGreedyCost(), 7.0);

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_LE((x - s).norm() + (x - g).norm(), 7.0 + 1e-9);
  }
  // focused_sample_count should track every greedy hit (== every call here).
  EXPECT_GT(obj->getLastSamplerStats().focused_sample_count, 0u);
  space->freeState(state);
}

TEST(GeodexOptimizationObjectiveTest, Feedback_GreedyBiasingRatioRespected) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);

  obj->setGreedyBiasingRatio(0.5);
  obj->setGreedyCost(7.0);

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  const int kCalls = 1000;
  for (int i = 0; i < kCalls; ++i) sampler->sampleUniform(state, ob::Cost(20.0));
  const auto focused = obj->getLastSamplerStats().focused_sample_count;
  // A loose statistical bound of 50% +/- 8% on 1000 trials.
  EXPECT_GT(focused, static_cast<unsigned long>(0.42 * kCalls));
  EXPECT_LT(focused, static_cast<unsigned long>(0.58 * kCalls));
  space->freeState(state);
}

TEST(GeodexOptimizationObjectiveTest, Feedback_NarrowingToHeuristicPathCostCanBeTurnedOff) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);
  EXPECT_TRUE(obj->getNarrowToHeuristicPathCost());

  obj->setHeuristicPathCost(7.0);
  obj->setNarrowToHeuristicPathCost(false);
  EXPECT_FALSE(obj->getNarrowToHeuristicPathCost());

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  // The planner's bound of 20 applies, and samples reach past the heuristic path cost.
  double widest = 0.0;
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    const double c = (x - s).norm() + (x - g).norm();
    EXPECT_LE(c, 20.0 + 1e-9);
    widest = std::max(widest, c);
  }
  EXPECT_GT(widest, 7.0);

  // Turned back on, the same sampler narrows again.
  obj->setNarrowToHeuristicPathCost(true);
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_LE((x - s).norm() + (x - g).norm(), 7.0 + 1e-9);
  }
  space->freeState(state);
}

TEST(GeodexOptimizationObjectiveTest, GetSamplerStats_SumsTheLiveSamplers) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});
  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);

  auto* state = space->allocState();
  auto first = obj->allocInformedStateSampler(pdef, 100);
  auto second = obj->allocInformedStateSampler(pdef, 100);
  for (int i = 0; i < 5; ++i) first->sampleUniform(state, ob::Cost(10.0));
  for (int i = 0; i < 3; ++i) second->sampleUniform(state, ob::Cost(10.0));
  EXPECT_EQ(obj->getSamplerStats().accepted, 8u);
  EXPECT_EQ(obj->getLastSamplerStats().accepted, 3u);

  first.reset();
  EXPECT_EQ(obj->getSamplerStats().accepted, 3u);
  space->freeState(state);
}

// Two samplers of one seeded space give independent streams, and a second space
// with the same seed repeats them.
TEST(GeodexOptimizationObjectiveTest, SeededSpaceGivesIndependentReproducibleSamplers) {
  auto sample = [](const std::uint64_t seed) {
    auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
    space->setSamplerSeed(seed);
    auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});
    auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(
        si, Eigen::Vector2d(3.0, 0.0));
    pdef->setOptimizationObjective(obj);
    auto a = obj->allocInformedStateSampler(pdef, 100);
    auto b = obj->allocInformedStateSampler(pdef, 100);
    std::vector<double> out;
    auto* state = space->allocState();
    for (auto* sampler : {a.get(), b.get()}) {
      for (int i = 0; i < 20; ++i) {
        sampler->sampleUniform(state, ob::Cost(10.0));
        out.push_back(state->as<State2D>()->values[0]);
        out.push_back(state->as<State2D>()->values[1]);
      }
    }
    space->freeState(state);
    return out;
  };
  const auto first = sample(5);
  EXPECT_EQ(first, sample(5));
  EXPECT_NE(first, sample(6));
  const std::vector<double> a(first.begin(), first.begin() + 40);
  const std::vector<double> b(first.begin() + 40, first.end());
  EXPECT_NE(a, b);
}

// Ambient coordinates of an embedded manifold are not intrinsic. The sampler
// rejection-samples on the manifold instead of sampling a hyperspheroid point.
TEST(GeodexOptimizationObjectiveTest, EmbeddedManifoldSamplesStayOnTheManifold) {
  using Sphere = geodex::Sphere<>;
  using SphereSpace = gio::GeodexStateSpace<Sphere>;
  auto space = std::make_shared<SphereSpace>(Sphere{}, makeBounds(3, -1.05, 1.05));
  auto [si, pdef] = makeSiAndPdef(space, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Sphere>>(
      si, Eigen::Vector3d(0.0, 1.0, 0.0));
  pdef->setOptimizationObjective(obj);
  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  using Direct = gio::GeodexDirectInfSampler<gh::Euclidean, Sphere::SamplerType>;
  EXPECT_FALSE(std::static_pointer_cast<Direct>(sampler)->getDirectSampling());
  EXPECT_FALSE(sampler->hasInformedMeasure());

  auto* state = space->allocState();
  const Eigen::Vector3d s(1.0, 0.0, 0.0), g(0.0, 1.0, 0.0);
  int sampled = 0;
  for (int i = 0; i < 200; ++i) {
    if (!sampler->sampleUniform(state, ob::Cost(2.0))) continue;
    ++sampled;
    Eigen::Map<const Eigen::Vector3d> x(state->as<gio::GeodexState<Sphere>>()->values);
    EXPECT_NEAR(x.norm(), 1.0, 1e-12);
    EXPECT_LE((x - s).norm() + (x - g).norm(), 2.0 + 1e-9);
  }
  EXPECT_GT(sampled, 0);
  space->freeState(state);
}

TEST(GeodexOptimizationObjectiveTest, Feedback_SharedAcrossSamplerReallocations) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);

  // Allocate a sampler, then update HPC, then allocate another.
  auto sampler1 = obj->allocInformedStateSampler(pdef, 100);
  obj->setHeuristicPathCost(7.0);
  auto sampler2 = obj->allocInformedStateSampler(pdef, 100);

  // Both samplers see the HPC update, and samples from each lie within 7.
  auto* state = space->allocState();
  const Eigen::Vector2d s(-3.0, 0.0);
  const Eigen::Vector2d g(3.0, 0.0);
  for (int i = 0; i < 50; ++i) {
    sampler1->sampleUniform(state, ob::Cost(20.0));
    Eigen::Map<const Eigen::Vector2d> x1(state->as<State2D>()->values);
    EXPECT_LE((x1 - s).norm() + (x1 - g).norm(), 7.0 + 1e-9);
    sampler2->sampleUniform(state, ob::Cost(20.0));
    Eigen::Map<const Eigen::Vector2d> x2(state->as<State2D>()->values);
    EXPECT_LE((x2 - s).norm() + (x2 - g).norm(), 7.0 + 1e-9);
  }
  space->freeState(state);
}

TEST(GeodexOptimizationObjectiveTest, ComputeGreedyCost_MaxOverPathStates) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  Eigen::Vector2d goal_coords(3.0, 0.0);
  gio::GeodexOptimizationObjective<Manifold2D> obj{si, goal_coords};

  // A 4-state path (-3,0) → (0,2) → (1,1) → (3,0).
  std::vector<ob::State*> path;
  for (auto pt : {Eigen::Vector2d(-3, 0), Eigen::Vector2d(0, 2), Eigen::Vector2d(1, 1),
                  Eigen::Vector2d(3, 0)}) {
    auto* s = space->allocState();
    s->as<State2D>()->values[0] = pt[0];
    s->as<State2D>()->values[1] = pt[1];
    path.push_back(s);
  }

  // Heuristic is the default Euclidean. Greedy cost = max_p (||p-s|| + ||p-g||).
  const Eigen::Vector2d start(-3, 0);
  const Eigen::Vector2d goal(3, 0);
  double expected = 0.0;
  for (const auto& pt : {Eigen::Vector2d(-3, 0), Eigen::Vector2d(0, 2), Eigen::Vector2d(1, 1),
                         Eigen::Vector2d(3, 0)}) {
    expected = std::max(expected, (pt - start).norm() + (pt - goal).norm());
  }
  EXPECT_NEAR(obj.computeGreedyCost(path), expected, 1e-12);

  for (auto* s : path) space->freeState(s);
}

TEST(GeodexOptimizationObjectiveTest, ComputeHeuristicPathCost_SumOverEdges) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  Eigen::Vector2d goal_coords(3.0, 0.0);
  gio::GeodexOptimizationObjective<Manifold2D> obj{si, goal_coords};

  // A 3-state straight path along the x-axis, (-3,0) → (0,0) → (3,0). Sum = 6.
  std::vector<ob::State*> path;
  for (double x : {-3.0, 0.0, 3.0}) {
    auto* s = space->allocState();
    s->as<State2D>()->values[0] = x;
    s->as<State2D>()->values[1] = 0.0;
    path.push_back(s);
  }
  EXPECT_NEAR(obj.computeHeuristicPathCost(path), 6.0, 1e-12);

  // Empty / single-state path → +inf.
  std::vector<ob::State*> empty;
  EXPECT_TRUE(std::isinf(obj.computeHeuristicPathCost(empty)));
  std::vector<ob::State*> single{path[0]};
  EXPECT_TRUE(std::isinf(obj.computeHeuristicPathCost(single)));

  for (auto* s : path) space->freeState(s);
}

namespace {

// Build an exact-solution PathGeometric on `space` from a list of (x, y) pairs
// and register it on `pdef`. pdef owns the returned PathPtr.
std::shared_ptr<ompl::geometric::PathGeometric> addExactSolution(
    const std::shared_ptr<Space2D>& space, const ob::SpaceInformationPtr& si,
    const ob::ProblemDefinitionPtr& pdef,
    std::initializer_list<std::pair<double, double>> waypoints) {
  auto path = std::make_shared<ompl::geometric::PathGeometric>(si);
  for (auto [x, y] : waypoints) {
    auto* s = space->allocState();
    s->as<State2D>()->values[0] = x;
    s->as<State2D>()->values[1] = y;
    path->append(s);
    space->freeState(s);
  }
  pdef->addSolutionPath(path);
  return path;
}

}  // namespace

TEST(GeodexInformedSamplerSelfRefresh, AutoUpdatesOnFirstExactSolution) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);

  EXPECT_TRUE(std::isinf(obj->getHeuristicPathCost()));
  EXPECT_TRUE(std::isinf(obj->getGreedyCost()));

  // Add an exact solution, a straight path along the x-axis. HPC = 6, GC = 6.
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  space->freeState(state);

  // Auto-refresh should have populated both bounds from the path.
  EXPECT_NEAR(obj->getHeuristicPathCost(), 6.0, 1e-12);
  EXPECT_NEAR(obj->getGreedyCost(), 6.0, 1e-12);
}

TEST(GeodexInformedSamplerSelfRefresh, RefreshesOnCheaperPathOnly) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);

  // The first solution is a triangular detour through (0, 4) with cost 10.
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 4.0}, {3.0, 0.0}});

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  EXPECT_NEAR(obj->getHeuristicPathCost(), 10.0, 1e-12);
  EXPECT_NEAR(obj->getGreedyCost(), 10.0, 1e-12);

  // A small improvement still moves both bounds.
  const double small = 2.0 * std::hypot(3.0, 3.85);
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 3.85}, {3.0, 0.0}});
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  EXPECT_NEAR(obj->getHeuristicPathCost(), small, 1e-12);
  EXPECT_NEAR(obj->getGreedyCost(), small, 1e-12);

  // A more expensive path leaves the bounds alone.
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 4.5}, {3.0, 0.0}});
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  EXPECT_NEAR(obj->getHeuristicPathCost(), small, 1e-12);
  EXPECT_NEAR(obj->getGreedyCost(), small, 1e-12);

  // A cheaper path moves them again.
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  EXPECT_NEAR(obj->getHeuristicPathCost(), 6.0, 1e-12);
  EXPECT_NEAR(obj->getGreedyCost(), 6.0, 1e-12);

  space->freeState(state);
}

TEST(GeodexInformedSamplerSelfRefresh, MatrixLB_AutoRefreshTightensSampling) {
  // With M = diag(4, 1) heuristic distances along the x-axis are 2× the Euclidean
  // distance. The sampler picks up the path-derived bound and constrains samples to it.
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto [si, pdef] = makeSiAndPdef(space, {-2.0, 0.0}, {2.0, 0.0});

  Eigen::Matrix2d M_lower;
  M_lower << 4.0, 0.0, 0.0, 1.0;
  gh::MatrixLowerBound<2> heuristic{M_lower};

  Eigen::Vector2d goal_coords(2.0, 0.0);
  auto obj =
      std::make_shared<gio::GeodexOptimizationObjective<Manifold2D, gh::MatrixLowerBound<2>>>(
          si, goal_coords, heuristic);
  pdef->setOptimizationObjective(obj);

  // For the straight x-axis path HPC = 2 * 2 = 4 under the L^T diff norm with M = diag(4,1).
  addExactSolution(space, si, pdef, {{-2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}});

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  space->freeState(state);

  EXPECT_NEAR(obj->getHeuristicPathCost(), 8.0, 1e-12);   // 4 + 4
  EXPECT_NEAR(obj->getGreedyCost(), 8.0, 1e-12);
}

TEST(GeodexInformedSamplerSelfRefresh, NoFeedbackChannel_DoesNothing) {
  // A sampler without a feedback channel ignores the solution in pdef. The auto-refresh
  // path returns early on null feedback.
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});

  // Built directly without feedback, the sampler is a stateless informed sampler.
  gh::Euclidean heuristic;
  ob::RealVectorBounds bounds(2);
  bounds.setLow(-10);
  bounds.setHigh(10);
  gio::GeodexDirectInfSampler<gh::Euclidean> sampler{pdef, 100, heuristic, bounds, nullptr};
  auto* state = space->allocState();
  ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(20.0)));
  // The test checks only that sampleUniform succeeds through the early-return guard.
  space->freeState(state);
}

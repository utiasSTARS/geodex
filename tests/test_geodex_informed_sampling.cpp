/// @file test_geodex_informed_sampling.cpp
/// @brief Tests for direct-sampling strategies in `GeodexDirectInfSampler`.

#include <atomic>
#include <cmath>
#include <cstdint>
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

#include "geodex/heuristics/eigenvalue_lower_bound.hpp"
#include "geodex/heuristics/euclidean.hpp"
#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/integration/ompl/geodex_informed_sampler.hpp"
#include "geodex/integration/ompl/geodex_optimization_objective.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/manifold/euclidean.hpp"

namespace ob = ompl::base;
namespace gio = geodex::integration::ompl;
namespace gh = geodex::heuristics;

using Manifold2D = geodex::Euclidean<2>;
using Manifold3D = geodex::Euclidean<3>;
using Space2D = gio::GeodexStateSpace<Manifold2D>;
using Space3D = gio::GeodexStateSpace<Manifold3D>;
using State2D = gio::GeodexState<Manifold2D>;
using State3D = gio::GeodexState<Manifold3D>;

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
// Euclidean PHS — regression after the if-constexpr restructure
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
// EigenvalueLowerBound — cost gets scaled by 1/sqrt(lambda_min)
// ============================================================================

TEST(GeodexInformedSampling, EigenvalueLB_SamplesRespectScaledCost) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  // Foci at (-1, 0), (1, 0): d_foci = 2.
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
// MatrixLowerBound — latent-space PHS with isotropic M_lower
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
// MatrixLowerBound — anisotropic M_lower exercises latent-space transform
// ============================================================================

TEST(GeodexInformedSampling, MatrixLB_AnisotropicMetricRespectsBoundsAndCost) {
  // Tight y-bound forces the clipped-AABB strategy on the long thin PHS.
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBoundsAxes(-5, 5, -2, 2));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  // Anisotropic M_lower: x is stiffer than y by 4x.
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
// MatrixLowerBound — empty bounds disables clipped-AABB; PHS sampling still works
// ============================================================================

TEST(GeodexInformedSampling, MatrixLB_EmptyBoundsFallsBackToPhsLatent) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> heuristic{M};
  // Pass empty bounds — sampler should fall back to PHS-with-rejection on bounds.
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
// Volume-ratio fallback — inadmissibly small M_lower triggers uniform sampling
// ============================================================================

TEST(GeodexInformedSampling, VolumeRatio_FallsBackToUniformOnInadmissibleMetric) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -1, 1));
  auto [si, pdef] = makeSiAndPdef(space, {-0.5, 0.0}, {0.5, 0.0});

  // Inadmissibly small M_lower → very large PHS in original space.
  // det(M_lower) = 1e-8 → sqrt = 1e-4, makes the PHS's effective volume huge.
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
// Infinite cost — uniform fallback for all branches
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
// Below-minimum-transverse-diameter — fall back to uniform without crashing
// ============================================================================

TEST(GeodexInformedSampling, DegeneratePHS_FallsBackToUniformWithoutCrashing) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  // d_foci = 6; cost just above 6 with EigenvalueLB scaling gets pulled below
  // the minimum transverse diameter via division by sqrt(lambda_min) = 2.
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
// Custom heuristic — rejection sampling still works for non-trait callables
// ============================================================================

namespace {
struct CustomHeuristic {
  template <typename A, typename B>
  auto operator()(const A& a, const B& b) const -> double {
    // Inflated Euclidean — still admissible for any metric h <= 1.5*||.||.
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
// hasInformedMeasure — true for trait-recognized heuristics, false otherwise
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
// getInformedMeasure — returns sane volumes for finite/infinite costs
// ============================================================================

TEST(GeodexInformedSampling, GetInformedMeasure_ReturnsSaneVolumes) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto [si, pdef] = makeSiAndPdef(space, {-1.0, 0.0}, {1.0, 0.0});

  // Euclidean — finite cost above min trans diameter gives finite measure.
  gio::GeodexDirectInfSampler<gh::Euclidean> euc{pdef, 50};
  EXPECT_GT(euc.getInformedMeasure(ob::Cost(5.0)), 0.0);
  EXPECT_LT(euc.getInformedMeasure(ob::Cost(5.0)), 100.0);  // < space measure (10x10)
  // Below min trans diameter (foci distance = 2) → 0.
  EXPECT_DOUBLE_EQ(euc.getInformedMeasure(ob::Cost(1.0)), 0.0);
  // Infinite cost → space measure.
  EXPECT_DOUBLE_EQ(euc.getInformedMeasure(ob::Cost(std::numeric_limits<double>::infinity())),
                   space->getMeasure());

  // MatrixLB volumes: positive, finite, below space measure.
  Eigen::Matrix2d M = 4.0 * Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> mlb{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> mlb_s{pdef, 50, mlb};
  // Latent foci-distance = sqrt((g-s)^T M (g-s)) = 4. Need c_best > 4.
  EXPECT_GT(mlb_s.getInformedMeasure(ob::Cost(8.0)), 0.0);
  EXPECT_DOUBLE_EQ(mlb_s.getInformedMeasure(ob::Cost(2.0)), 0.0);
}

// ============================================================================
// Annular sampler — sampleUniform(min, max) excludes the inner region
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
// Diagnostics — SamplingStats counters
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

TEST(GeodexInformedSampling, Stats_ResetClearsCounters) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  gio::GeodexDirectInfSampler<gh::Euclidean> sampler{pdef, 100};
  auto* state = space->allocState();
  for (int i = 0; i < 10; ++i) sampler.sampleUniform(state, ob::Cost(10.0));
  EXPECT_GT(sampler.getSamplingStats().total_attempts, 0u);

  sampler.resetSamplingStats();
  const auto reset = sampler.getSamplingStats();
  EXPECT_EQ(reset.total_attempts, 0u);
  EXPECT_EQ(reset.accepted, 0u);
  EXPECT_EQ(reset.bounds_rejections, 0u);
  EXPECT_EQ(reset.phs_rejections, 0u);
  EXPECT_EQ(reset.uniform_fallback_count, 0u);
  EXPECT_EQ(reset.focused_sample_count, 0u);
  space->freeState(state);
}

TEST(GeodexInformedSampling, Stats_VolumeRatioFallbackIncrementsCounter) {
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
  EXPECT_GT(stats.uniform_fallback_count, 0u);
  EXPECT_GT(stats.last_volume_ratio, sampler.getVolumeRatioThreshold());
  space->freeState(state);
}

TEST(GeodexInformedSampling, VolumeRatioThresholdZeroDisablesFallback) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -1, 1));
  auto [si, pdef] = makeSiAndPdef(space, {-0.5, 0.0}, {0.5, 0.0});

  Eigen::Matrix2d M = 1e-4 * Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler{pdef, 50, heuristic,
                                                               makeBounds(2, -1, 1)};
  sampler.setVolumeRatioThreshold(0.0);

  auto* state = space->allocState();
  for (int i = 0; i < 30; ++i) sampler.sampleUniform(state, ob::Cost(10.0));
  const auto stats = sampler.getSamplingStats();
  EXPECT_EQ(stats.uniform_fallback_count, 0u);
  space->freeState(state);
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
  EXPECT_GT(sampler.getSamplingStats().uniform_fallback_count, 0u);
  space->freeState(state);
}

TEST(GeodexInformedSampling, Stats_ReportsClippedAABBVolumeForMatrixLB) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBoundsAxes(-5, 5, -2, 2));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Matrix2d M;
  M << 4.0, 0.0, 0.0, 1.0;
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> sampler{pdef, 50, heuristic,
                                                               makeBoundsAxes(-5, 5, -2, 2)};

  auto* state = space->allocState();
  for (int i = 0; i < 20; ++i) sampler.sampleUniform(state, ob::Cost(14.0));
  const auto stats = sampler.getSamplingStats();
  EXPECT_GT(stats.phs_volume, 0.0);
  // clipped_aabb_volume is set whenever the strategy decision runs.
  EXPECT_GE(stats.clipped_aabb_volume, 0.0);
  space->freeState(state);
}

TEST(GeodexInformedSampling, Sampler_LatentBoundsAABBExposed) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -3, 3));
  auto [si, pdef] = makeSiAndPdef(space, {-1.0, 0.0}, {1.0, 0.0});

  Eigen::Matrix2d M = Eigen::Matrix2d::Identity();
  gh::MatrixLowerBound<2> heuristic{M};
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> mlb_with_bounds{pdef, 50, heuristic,
                                                                      makeBounds(2, -3, 3)};
  Eigen::VectorXd lo, hi;
  EXPECT_TRUE(mlb_with_bounds.getLatentBoundsAABB(lo, hi));
  EXPECT_EQ(lo.size(), 2);
  EXPECT_EQ(hi.size(), 2);
  EXPECT_LT(lo[0], hi[0]);
  EXPECT_LT(lo[1], hi[1]);

  // Without bounds the accessor returns false.
  gio::GeodexDirectInfSampler<gh::MatrixLowerBound<2>> mlb_no_bounds{pdef, 50, heuristic};
  Eigen::VectorXd lo2, hi2;
  EXPECT_FALSE(mlb_no_bounds.getLatentBoundsAABB(lo2, hi2));

  // Euclidean has no latent bounds at all.
  gio::GeodexDirectInfSampler<gh::Euclidean> euc{pdef, 50};
  Eigen::VectorXd lo3, hi3;
  EXPECT_FALSE(euc.getLatentBoundsAABB(lo3, hi3));
}

// ============================================================================
// Objective — motion-cost call counter and last-sampler tracking
// ============================================================================

TEST(GeodexOptimizationObjectiveTest, MotionCostCallCount_IncrementsPerCall) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -5, 5));
  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  Eigen::Vector2d goal_coords(2.0, 0.0);
  gio::GeodexOptimizationObjective<Manifold2D> obj{si, goal_coords};
  EXPECT_EQ(obj.getMotionCostCallCount(), 0u);

  auto* a = space->allocState();
  auto* b = space->allocState();
  a->as<State2D>()->values[0] = 0.0;
  a->as<State2D>()->values[1] = 0.0;
  b->as<State2D>()->values[0] = 1.0;
  b->as<State2D>()->values[1] = 0.0;
  for (int i = 0; i < 7; ++i) (void)obj.motionCost(a, b);
  EXPECT_EQ(obj.getMotionCostCallCount(), 7u);
  space->freeState(a);
  space->freeState(b);
}

TEST(GeodexOptimizationObjectiveTest, MotionCostCallCount_ThreadSafeUnderConcurrentCalls) {
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

  constexpr int kThreads = 4;
  constexpr int kPerThread = 1000;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < kPerThread; ++i) (void)obj.motionCost(a, b);
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(obj.getMotionCostCallCount(), static_cast<std::uint64_t>(kThreads * kPerThread));
  space->freeState(a);
  space->freeState(b);
}

TEST(GeodexOptimizationObjectiveTest, GetLastSamplerStats_TracksLatestAllocation) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef_only] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef_only->setOptimizationObjective(obj);

  // No sampler allocated yet → default stats.
  EXPECT_EQ(obj->getLastSamplerStats().total_attempts, 0u);

  auto sampler1 = obj->allocInformedStateSampler(pdef_only, 100);
  auto* state = space->allocState();
  for (int i = 0; i < 5; ++i) sampler1->sampleUniform(state, ob::Cost(10.0));
  EXPECT_GT(obj->getLastSamplerStats().total_attempts, 0u);
  const auto attempts_after_first = obj->getLastSamplerStats().total_attempts;

  // Allocate a fresh sampler — last_sampler_ now points at it.
  auto sampler2 = obj->allocInformedStateSampler(pdef_only, 100);
  // Stats from sampler2 (just allocated, no calls) should be zero.
  EXPECT_EQ(obj->getLastSamplerStats().total_attempts, 0u);

  // Make sure the older sampler's stats aren't mistakenly returned.
  for (int i = 0; i < 3; ++i) sampler2->sampleUniform(state, ob::Cost(10.0));
  EXPECT_GT(obj->getLastSamplerStats().total_attempts, 0u);
  EXPECT_LT(obj->getLastSamplerStats().total_attempts, attempts_after_first);
  space->freeState(state);
}

// ============================================================================
// Cost-bound feedback — greedy biasing + heuristic-path-cost tightening
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
  // Planner's c_best is loose at 20; the tighter HPC=7 should bind.
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
    Eigen::Map<const Eigen::Vector2d> x(state->as<State2D>()->values);
    EXPECT_LE((x - s).norm() + (x - g).norm(), 7.0 + 1e-9);
  }
  space->freeState(state);
}

TEST(GeodexOptimizationObjectiveTest, Feedback_GreedyBiasingDrawsTighterEllipsoid) {
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
  // Loose statistical bound: 50% +/- 8% on 1000 trials.
  EXPECT_GT(focused, static_cast<unsigned long>(0.42 * kCalls));
  EXPECT_LT(focused, static_cast<unsigned long>(0.58 * kCalls));
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

  // Both samplers should see the HPC update: samples from each are within 7.
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

  // Build a 4-state path: (-3,0) → (0,2) → (1,1) → (3,0).
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

  // 3-state straight path along x-axis: (-3,0) → (0,0) → (3,0). Sum = 6.
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
// and register it on `pdef`. Returned PathPtr is owned by pdef.
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

  // Add an exact solution: straight path along the x-axis. HPC = 6, GC = 6.
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  space->freeState(state);

  // Auto-refresh should have populated both bounds from the path.
  EXPECT_NEAR(obj->getHeuristicPathCost(), 6.0, 1e-12);
  EXPECT_NEAR(obj->getGreedyCost(), 6.0, 1e-12);
}

TEST(GeodexInformedSamplerSelfRefresh, ThresholdGate_NoOpOnSubFivePctImprovement) {
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});

  Eigen::Vector2d goal_coords(3.0, 0.0);
  auto obj = std::make_shared<gio::GeodexOptimizationObjective<Manifold2D>>(si, goal_coords);
  pdef->setOptimizationObjective(obj);

  // First solution: HPC = 8 (a triangular detour through (0, 4)).
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 4.0}, {3.0, 0.0}});

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  const double hpc_after_first = obj->getHeuristicPathCost();
  const double gc_after_first = obj->getGreedyCost();
  EXPECT_NEAR(hpc_after_first, 10.0, 1e-12);  // 5 + 5

  // Second solution: only ~3% better (HPC = 9.7). Threshold (5%) gates the write.
  // The early-return short-circuits both bounds, so neither HPC nor GC moves.
  addExactSolution(space, si, pdef,
                   {{-3.0, 0.0}, {0.0, 3.85}, {3.0, 0.0}});  // 2 * sqrt(3^2 + 3.85^2) ≈ 9.764
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  EXPECT_NEAR(obj->getHeuristicPathCost(), hpc_after_first, 1e-12)
      << "sub-threshold improvement should not update the bound";
  EXPECT_NEAR(obj->getGreedyCost(), gc_after_first, 1e-12)
      << "sub-threshold gate should leave GC unchanged too";

  // Third solution: a strictly better straight path (HPC = 6). >5% better, write through.
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  EXPECT_NEAR(obj->getHeuristicPathCost(), 6.0, 1e-12);
  EXPECT_NEAR(obj->getGreedyCost(), 6.0, 1e-12);

  space->freeState(state);
}

TEST(GeodexInformedSamplerSelfRefresh, MatrixLB_AutoRefreshTightensSampling) {
  // Anisotropic metric: M = diag(4, 1) → heuristic distances along x-axis are
  // 2× the Euclidean distance. Verify the sampler picks up the path-derived
  // bound and constrains samples accordingly.
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

  // Straight x-axis path: HPC = 2 * 2 = 4 under L^T diff norm with M = diag(4,1).
  addExactSolution(space, si, pdef, {{-2.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}});

  auto sampler = obj->allocInformedStateSampler(pdef, 100);
  auto* state = space->allocState();
  ASSERT_TRUE(sampler->sampleUniform(state, ob::Cost(20.0)));
  space->freeState(state);

  EXPECT_NEAR(obj->getHeuristicPathCost(), 8.0, 1e-12);   // 4 + 4
  EXPECT_NEAR(obj->getGreedyCost(), 8.0, 1e-12);
}

TEST(GeodexInformedSamplerSelfRefresh, NoFeedbackChannel_DoesNothing) {
  // Construct a sampler with a null feedback channel by using an objective and
  // immediately stripping the feedback shared_ptr. The auto-refresh code path
  // must early-return on null feedback regardless of pdef state.
  auto space = std::make_shared<Space2D>(Manifold2D{}, makeBounds(2, -10, 10));
  auto [si, pdef] = makeSiAndPdef(space, {-3.0, 0.0}, {3.0, 0.0});
  addExactSolution(space, si, pdef, {{-3.0, 0.0}, {0.0, 0.0}, {3.0, 0.0}});

  // Direct construction with no feedback: behaves as a stateless informed sampler.
  gh::Euclidean heuristic;
  ob::RealVectorBounds bounds(2);
  bounds.setLow(-10);
  bounds.setHigh(10);
  gio::GeodexDirectInfSampler<gh::Euclidean> sampler{pdef, 100, heuristic, bounds, nullptr};
  auto* state = space->allocState();
  ASSERT_TRUE(sampler.sampleUniform(state, ob::Cost(20.0)));
  // No assertion on feedback (we don't have one); the test verifies sample
  // returns true without crashing the early-return guard.
  space->freeState(state);
}

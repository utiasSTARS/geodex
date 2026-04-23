/// @file bench_ompl_interpolation.cpp
/// @brief Benchmarks for GeodexStateSpace interpolation with discrete geodesic caching.

#include <numbers>

#include <Eigen/Core>
#include <benchmark/benchmark.h>
#include <ompl/base/DiscreteMotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorBounds.h>

#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/metrics/constant_spd.hpp"

namespace ob = ompl::base;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ob::RealVectorBounds makeSE2Bounds() {
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, 0.0);
  bounds.setHigh(0, 10.0);
  bounds.setLow(1, 0.0);
  bounds.setHigh(1, 10.0);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);
  return bounds;
}

template <typename StateType>
static void setStateValues(ob::State* s, double v0, double v1, double v2) {
  auto* st = s->as<StateType>();
  st->values[0] = v0;
  st->values[1] = v1;
  st->values[2] = v2;
}

// ---------------------------------------------------------------------------
// BM_Interpolate_SE2_Identity: baseline, zero-overhead fast path
// ---------------------------------------------------------------------------

static void BM_Interpolate_SE2_Identity(benchmark::State& state) {
  using M = geodex::SE2<>;
  using SS = geodex::integration::ompl::GeodexStateSpace<M>;
  using ST = geodex::integration::ompl::GeodexState<M>;

  geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 1.0};
  M manifold{metric};
  auto space = std::make_shared<SS>(manifold, makeSE2Bounds());
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  setStateValues<ST>(s1, 2.0, 3.0, 0.5);
  setStateValues<ST>(s2, 8.0, 7.0, -1.0);

  const int N = state.range(0);
  for (auto _ : state) {
    for (int j = 1; j <= N; ++j) {
      space->interpolate(s1, s2, static_cast<double>(j) / N, result);
    }
    benchmark::DoNotOptimize(result->as<ST>()->values[0]);
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}
BENCHMARK(BM_Interpolate_SE2_Identity)->Arg(10)->Arg(50)->Arg(100);

// ---------------------------------------------------------------------------
// BM_Interpolate_SE2_Anisotropic_CacheHot: amortized cost with cache
// ---------------------------------------------------------------------------

static void BM_Interpolate_SE2_Anisotropic_CacheHot(benchmark::State& state) {
  using M = geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap>;
  using SS = geodex::integration::ompl::GeodexStateSpace<M>;
  using ST = geodex::integration::ompl::GeodexState<M>;

  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  M manifold{metric};
  auto space = std::make_shared<SS>(manifold, makeSE2Bounds());
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  setStateValues<ST>(s1, 2.0, 3.0, 0.0);
  setStateValues<ST>(s2, 8.0, 7.0, 1.0);

  const int N = state.range(0);
  for (auto _ : state) {
    // Same (s1,s2) pair → cache hit after first call
    for (int j = 1; j <= N; ++j) {
      space->interpolate(s1, s2, static_cast<double>(j) / N, result);
    }
    benchmark::DoNotOptimize(result->as<ST>()->values[0]);
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}
BENCHMARK(BM_Interpolate_SE2_Anisotropic_CacheHot)->Arg(10)->Arg(50)->Arg(100);

// ---------------------------------------------------------------------------
// BM_Interpolate_SE2_Anisotropic_CacheCold: cache miss every iteration
// ---------------------------------------------------------------------------

static void BM_Interpolate_SE2_Anisotropic_CacheCold(benchmark::State& state) {
  using M = geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap>;
  using SS = geodex::integration::ompl::GeodexStateSpace<M>;
  using ST = geodex::integration::ompl::GeodexState<M>;

  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  M manifold{metric};
  auto space = std::make_shared<SS>(manifold, makeSE2Bounds());
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  setStateValues<ST>(s1, 2.0, 3.0, 0.0);

  int counter = 0;
  for (auto _ : state) {
    // Vary s2 to force cache miss every iteration
    double offset = 0.01 * (counter++ % 100);
    setStateValues<ST>(s2, 8.0 + offset, 7.0, 1.0);
    space->interpolate(s1, s2, 0.5, result);
    benchmark::DoNotOptimize(result->as<ST>()->values[0]);
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}
BENCHMARK(BM_Interpolate_SE2_Anisotropic_CacheCold);

// ---------------------------------------------------------------------------
// BM_Interpolate_Euclidean_Anisotropic_CacheHot
// ---------------------------------------------------------------------------

static void BM_Interpolate_Euclidean_Anisotropic_CacheHot(benchmark::State& state) {
  using M = geodex::Euclidean<2, geodex::ConstantSPDMetric<2>>;
  using SS = geodex::integration::ompl::GeodexStateSpace<M>;
  using ST = geodex::integration::ompl::GeodexState<M>;

  Eigen::Matrix2d A;
  A << 4.0, 0.0, 0.0, 1.0;
  geodex::ConstantSPDMetric<2> metric{A};
  M manifold{metric};

  ob::RealVectorBounds bounds(2);
  bounds.setLow(0.0);
  bounds.setHigh(10.0);
  auto space = std::make_shared<SS>(manifold, bounds);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::RiemannianGeodesic);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  auto* result = space->allocState();
  s1->as<ST>()->values[0] = 1.0;
  s1->as<ST>()->values[1] = 1.0;
  s2->as<ST>()->values[0] = 9.0;
  s2->as<ST>()->values[1] = 9.0;

  const int N = state.range(0);
  for (auto _ : state) {
    for (int j = 1; j <= N; ++j) {
      space->interpolate(s1, s2, static_cast<double>(j) / N, result);
    }
    benchmark::DoNotOptimize(result->as<ST>()->values[0]);
  }

  space->freeState(result);
  space->freeState(s1);
  space->freeState(s2);
}
BENCHMARK(BM_Interpolate_Euclidean_Anisotropic_CacheHot)->Arg(10)->Arg(50)->Arg(100);

// ---------------------------------------------------------------------------
// BM_MotionValidation_SE2_Anisotropic: full collision check cycle
// ---------------------------------------------------------------------------

static void BM_MotionValidation_SE2_Anisotropic(benchmark::State& state) {
  using M = geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap>;
  using SS = geodex::integration::ompl::GeodexStateSpace<M>;
  using ST = geodex::integration::ompl::GeodexState<M>;

  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  M manifold{metric};
  auto space = std::make_shared<SS>(manifold, makeSE2Bounds());
  space->setCollisionResolution(0.1);

  const bool use_discrete = state.range(0);
  space->setInterpolationMode(use_discrete
                                  ? geodex::integration::ompl::InterpolationMode::RiemannianGeodesic
                                  : geodex::integration::ompl::InterpolationMode::BaseGeodesic);

  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  auto mv = std::make_shared<ob::DiscreteMotionValidator>(si);

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  setStateValues<ST>(s1, 2.0, 3.0, 0.0);
  setStateValues<ST>(s2, 8.0, 7.0, 1.0);

  for (auto _ : state) {
    bool valid = mv->checkMotion(s1, s2);
    benchmark::DoNotOptimize(valid);
  }

  space->freeState(s1);
  space->freeState(s2);
}
BENCHMARK(BM_MotionValidation_SE2_Anisotropic)
    ->Arg(0)   // discrete geodesic OFF
    ->Arg(1);  // discrete geodesic ON

// ---------------------------------------------------------------------------
// BM_ValidSegmentCount: segment counting overhead
// ---------------------------------------------------------------------------

static void BM_ValidSegmentCount_SE2(benchmark::State& state) {
  using M = geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap>;
  using SS = geodex::integration::ompl::GeodexStateSpace<M>;
  using ST = geodex::integration::ompl::GeodexState<M>;

  geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
  M manifold{metric};
  auto space = std::make_shared<SS>(manifold, makeSE2Bounds());
  space->setCollisionResolution(0.1);

  auto si = std::make_shared<ob::SpaceInformation>(space);
  si->setStateValidityChecker([](const ob::State*) { return true; });
  si->setup();

  auto* s1 = space->allocState();
  auto* s2 = space->allocState();
  setStateValues<ST>(s1, 2.0, 3.0, 0.0);
  setStateValues<ST>(s2, 8.0, 7.0, 1.0);

  for (auto _ : state) {
    unsigned int n = space->validSegmentCount(s1, s2);
    benchmark::DoNotOptimize(n);
  }

  space->freeState(s1);
  space->freeState(s2);
}
BENCHMARK(BM_ValidSegmentCount_SE2);

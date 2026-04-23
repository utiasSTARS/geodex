/// @file bench_retractions.cpp
/// @brief Benchmarks for retraction speed and accuracy comparison.

#include <cmath>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include "geodex/geodex.hpp"

using namespace geodex;

// ---------------------------------------------------------------------------
// SE(2) test data
// ---------------------------------------------------------------------------

static const Eigen::Vector3d se2_p(2.0, 3.0, 0.5);
static const Eigen::Vector3d se2_q(7.0, 5.0, 2.0);
static const Eigen::Vector3d se2_v(0.5, 0.3, 0.4);  // moderate angle

// Retraction instances
static SE2ExponentialMap se2_exp_map;
static SE2EulerRetraction se2_euler;

// ---------------------------------------------------------------------------
// Sphere test data
// ---------------------------------------------------------------------------

static const Eigen::Vector3d sph_p(0.0, 0.0, 1.0);  // north pole
static const Eigen::Vector3d sph_v(0.3, 0.4, 0.0);  // tangent at north pole
static const Eigen::Vector3d sph_q = Eigen::Vector3d(std::sin(1.0), 0.0, std::cos(1.0));  // theta=1

static SphereExponentialMap sphere_exp_map;
static SphereProjectionRetraction sphere_proj;

// ===========================================================================
// SE(2) retract speed
// ===========================================================================

static void BM_SE2_Retract_Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_exp_map.retract(se2_p, se2_v));
  }
}
BENCHMARK(BM_SE2_Retract_Exp);

static void BM_SE2_Retract_Euler(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_euler.retract(se2_p, se2_v));
  }
}
BENCHMARK(BM_SE2_Retract_Euler);

// ===========================================================================
// SE(2) inverse_retract speed
// ===========================================================================

static void BM_SE2_InvRetract_Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_exp_map.inverse_retract(se2_p, se2_q));
  }
}
BENCHMARK(BM_SE2_InvRetract_Exp);

static void BM_SE2_InvRetract_Euler(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_euler.inverse_retract(se2_p, se2_q));
  }
}
BENCHMARK(BM_SE2_InvRetract_Euler);

// ===========================================================================
// Sphere retract speed
// ===========================================================================

static void BM_Sphere_Retract_Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_exp_map.retract(sph_p, sph_v));
  }
}
BENCHMARK(BM_Sphere_Retract_Exp);

static void BM_Sphere_Retract_Proj(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_proj.retract(sph_p, sph_v));
  }
}
BENCHMARK(BM_Sphere_Retract_Proj);

// ===========================================================================
// Sphere inverse_retract speed
// ===========================================================================

static void BM_Sphere_InvRetract_Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_exp_map.inverse_retract(sph_p, sph_q));
  }
}
BENCHMARK(BM_Sphere_InvRetract_Exp);

static void BM_Sphere_InvRetract_Proj(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_proj.inverse_retract(sph_p, sph_q));
  }
}
BENCHMARK(BM_Sphere_InvRetract_Proj);

// ===========================================================================
// Sphere retraction accuracy
// ===========================================================================

static void BM_Sphere_Accuracy_Proj(benchmark::State& state) {
  const double magnitudes[] = {0.01, 0.1, 0.5, 1.0, 2.0};
  double max_error = 0.0;

  for (auto _ : state) {
    max_error = 0.0;
    for (double mag : magnitudes) {
      Eigen::Vector3d v_scaled = sph_v.normalized() * mag;
      auto ref = sphere_exp_map.retract(sph_p, v_scaled);
      auto approx = sphere_proj.retract(sph_p, v_scaled);
      double err = (ref - approx).norm();
      max_error = std::max(max_error, err);
    }
    benchmark::DoNotOptimize(max_error);
  }
  state.counters["max_error"] = max_error;
}
BENCHMARK(BM_Sphere_Accuracy_Proj);

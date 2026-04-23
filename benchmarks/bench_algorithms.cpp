/// @file bench_algorithms.cpp
/// @brief Benchmarks for geodex algorithms: discrete_geodesic and distance_midpoint.

#include <cmath>

#include <numbers>
#include <vector>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include "geodex/geodex.hpp"
#include "geodex/metrics/clearance.hpp"

#include "planar_manipulator_metric.hpp"

using namespace geodex;

// ---------------------------------------------------------------------------
// discrete_geodesic benchmarks
// ---------------------------------------------------------------------------

static const InterpolationSettings settings{
    .step_size = 0.5,
    .convergence_tol = 1e-4,
    .max_steps = 200,
    .fd_epsilon = 1e-4,
    .distortion_ratio = 1.5,
};

static void BM_DiscreteGeodesic_Sphere(benchmark::State& state) {
  Sphere<> sphere;
  Eigen::Vector3d start(0.0, 0.0, 1.0);
  Eigen::Vector3d target(std::sin(1.0), 0.0, std::cos(1.0));

  InterpolationCache<Sphere<>> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    auto result = discrete_geodesic(sphere, start, target, settings, &ws);
    total_iters += result.iterations;
    total_halvings += result.distortion_halvings;
    ++total_calls;
    benchmark::DoNotOptimize(result);
  }
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_Sphere);

static void BM_DiscreteGeodesic_SphereAniso(benchmark::State& state) {
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 0) = 4.0;
  using AnisoSphere = Sphere<2, ConstantSPDMetric<3>>;
  AnisoSphere sphere{ConstantSPDMetric<3>{A}};
  Eigen::Vector3d start(0.0, 0.0, 1.0);
  Eigen::Vector3d target(std::sin(0.8), 0.0, std::cos(0.8));

  InterpolationCache<AnisoSphere> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    auto result = discrete_geodesic(sphere, start, target, settings, &ws);
    total_iters += result.iterations;
    total_halvings += result.distortion_halvings;
    ++total_calls;
    benchmark::DoNotOptimize(result);
  }
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_SphereAniso);

static void BM_DiscreteGeodesic_Torus2(benchmark::State& state) {
  Torus<2> torus;
  Eigen::Vector2d start(0.5, 0.5);
  Eigen::Vector2d target(5.0, 5.0);

  InterpolationCache<Torus<2>> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    auto result = discrete_geodesic(torus, start, target, settings, &ws);
    total_iters += result.iterations;
    total_halvings += result.distortion_halvings;
    ++total_calls;
    benchmark::DoNotOptimize(result);
  }
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_Torus2);

static void BM_DiscreteGeodesic_SE2(benchmark::State& state) {
  SE2<> se2;
  Eigen::Vector3d start(1.0, 1.0, 0.0);
  Eigen::Vector3d target(8.0, 8.0, std::numbers::pi / 2.0);

  InterpolationCache<SE2<>> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    auto result = discrete_geodesic(se2, start, target, settings, &ws);
    total_iters += result.iterations;
    total_halvings += result.distortion_halvings;
    ++total_calls;
    benchmark::DoNotOptimize(result);
  }
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_SE2);

static void BM_DiscreteGeodesic_CSpace_KE(benchmark::State& state) {
  PlanarManipulatorMetric arm_metric;
  auto ke =
      KineticEnergyMetric{[&](const Eigen::Vector2d& q) { return arm_metric.mass_matrix(q); }};
  using CSpace = ConfigurationSpace<Torus<2>, decltype(ke)>;
  CSpace cspace{Torus<2>{}, std::move(ke)};

  Eigen::Vector2d start(0.5, 0.5);
  Eigen::Vector2d target(2.5, 2.5);

  InterpolationCache<CSpace> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    auto result = discrete_geodesic(cspace, start, target, settings, &ws);
    total_iters += result.iterations;
    total_halvings += result.distortion_halvings;
    ++total_calls;
    benchmark::DoNotOptimize(result);
  }
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_CSpace_KE);

// ---------------------------------------------------------------------------
// Anisotropic SE2 — exercises the FD fallback when the left-invariant weights
// produce a Riemannian geodesic that disagrees with the Lie-group exp.
// ---------------------------------------------------------------------------

static void BM_DiscreteGeodesic_SE2_Anisotropic(benchmark::State& state) {
  using Se2Aniso = SE2<SE2LeftInvariantMetric>;
  Se2Aniso se2(SE2LeftInvariantMetric{1.0, 1.0, 5.0});
  Eigen::Vector3d start(1.0, 1.0, 0.0);
  Eigen::Vector3d target(8.0, 8.0, std::numbers::pi / 2.0);

  InterpolationCache<Se2Aniso> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    auto result = discrete_geodesic(se2, start, target, settings, &ws);
    total_iters += result.iterations;
    total_halvings += result.distortion_halvings;
    ++total_calls;
    benchmark::DoNotOptimize(result);
  }
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_SE2_Anisotropic);

// ---------------------------------------------------------------------------
// SE(2) + SDFConformal — exercises the FD path on a spatially-varying metric.
// Compares the new midpoint FD surrogate against the forced via-log fallback
// (tau=0), which reproduces the pre-fix behavior. Same (start, target, step)
// across the two entries so wall time / iterations / halvings / fallbacks are
// directly comparable.
// ---------------------------------------------------------------------------

namespace {

auto make_sdf_conformal_cspace() {
  auto sdf = [](const Eigen::Vector3d& q) {
    const double r = std::sqrt(q[0] * q[0] + q[1] * q[1]);
    return r - 1.0;  // unit circle at origin, positive outside
  };
  SE2LeftInvariantMetric base{1.0, 1.0, 0.5};
  SE2<SE2LeftInvariantMetric, SE2ExponentialMap> se2{base};
  SDFConformalMetric clearance{base, sdf, 2.0, 2.0};
  using CSpace = ConfigurationSpace<decltype(se2), decltype(clearance)>;
  return CSpace{se2, std::move(clearance)};
}

const Eigen::Vector3d kSDFStart{-2.0, 1.5, 0.0};
const Eigen::Vector3d kSDFTarget{2.0, 1.5, 0.0};

}  // namespace

static void BM_DiscreteGeodesic_SE2_SDFConformal_Midpoint(benchmark::State& state) {
  auto cspace = make_sdf_conformal_cspace();
  using CSpace = decltype(cspace);

  InterpolationSettings s = settings;
  s.step_size = 0.2;
  s.max_steps = 200;

  InterpolationCache<CSpace> ws;
  long long total_iters = 0, total_halvings = 0, total_fallbacks = 0, total_calls = 0;
  for (auto _ : state) {
    auto result = discrete_geodesic(cspace, kSDFStart, kSDFTarget, s, &ws);
    total_iters += result.iterations;
    total_halvings += result.distortion_halvings;
    total_fallbacks += result.fd_midpoint_fallbacks;
    ++total_calls;
    benchmark::DoNotOptimize(result);
  }
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
  state.counters["fallbacks/call"] = static_cast<double>(total_fallbacks) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_SE2_SDFConformal_Midpoint);

static void BM_DiscreteGeodesic_SE2_SDFConformal_ViaLog(benchmark::State& state) {
  auto cspace = make_sdf_conformal_cspace();
  using CSpace = decltype(cspace);

  InterpolationSettings s = settings;
  s.step_size = 0.2;
  s.max_steps = 200;
  s.fd_midpoint_guard_tau = 0.0;  // force via-log on every FD sample

  InterpolationCache<CSpace> ws;
  long long total_iters = 0, total_halvings = 0, total_fallbacks = 0, total_calls = 0;
  for (auto _ : state) {
    auto result = discrete_geodesic(cspace, kSDFStart, kSDFTarget, s, &ws);
    total_iters += result.iterations;
    total_halvings += result.distortion_halvings;
    total_fallbacks += result.fd_midpoint_fallbacks;
    ++total_calls;
    benchmark::DoNotOptimize(result);
  }
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
  state.counters["fallbacks/call"] = static_cast<double>(total_fallbacks) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_SE2_SDFConformal_ViaLog);

// ---------------------------------------------------------------------------
// Batch steer workload — simulates an RRT* steer loop with N random endpoints
// and a single reused workspace. This is the primary "real workload"
// before/after comparison.
// ---------------------------------------------------------------------------

static void BM_DiscreteGeodesic_Sphere_BatchSteer(benchmark::State& state) {
  Sphere<> sphere;
  const int N = 256;
  std::vector<Eigen::Vector3d> starts(N), targets(N);
  for (int i = 0; i < N; ++i) {
    starts[i] = sphere.random_point();
    targets[i] = sphere.random_point();
  }

  InterpolationCache<Sphere<>> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    for (int i = 0; i < N; ++i) {
      auto result = discrete_geodesic(sphere, starts[i], targets[i], settings, &ws);
      total_iters += result.iterations;
      total_halvings += result.distortion_halvings;
      ++total_calls;
      benchmark::DoNotOptimize(result);
    }
  }
  state.SetItemsProcessed(state.iterations() * N);
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_Sphere_BatchSteer);

static void BM_DiscreteGeodesic_Torus7_BatchSteer(benchmark::State& state) {
  using Torus7 = Torus<7>;
  Torus7 torus;
  const int N = 256;
  std::vector<Eigen::Vector<double, 7>> starts(N), targets(N);
  for (int i = 0; i < N; ++i) {
    starts[i] = torus.random_point();
    targets[i] = torus.random_point();
  }

  InterpolationCache<Torus7> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    for (int i = 0; i < N; ++i) {
      auto result = discrete_geodesic(torus, starts[i], targets[i], settings, &ws);
      total_iters += result.iterations;
      total_halvings += result.distortion_halvings;
      ++total_calls;
      benchmark::DoNotOptimize(result);
    }
  }
  state.SetItemsProcessed(state.iterations() * N);
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_Torus7_BatchSteer);

static void BM_DiscreteGeodesic_SE2_BatchSteer(benchmark::State& state) {
  SE2<> se2;
  const int N = 256;
  std::vector<Eigen::Vector3d> starts(N), targets(N);
  for (int i = 0; i < N; ++i) {
    starts[i] = se2.random_point();
    targets[i] = se2.random_point();
  }

  InterpolationCache<SE2<>> ws;
  long long total_iters = 0, total_halvings = 0, total_calls = 0;
  for (auto _ : state) {
    for (int i = 0; i < N; ++i) {
      auto result = discrete_geodesic(se2, starts[i], targets[i], settings, &ws);
      total_iters += result.iterations;
      total_halvings += result.distortion_halvings;
      ++total_calls;
      benchmark::DoNotOptimize(result);
    }
  }
  state.SetItemsProcessed(state.iterations() * N);
  state.counters["iters/call"] = static_cast<double>(total_iters) / total_calls;
  state.counters["halvings/call"] = static_cast<double>(total_halvings) / total_calls;
}
BENCHMARK(BM_DiscreteGeodesic_SE2_BatchSteer);

// ---------------------------------------------------------------------------
// Batch distance_midpoint throughput
// ---------------------------------------------------------------------------

static void BM_DistanceMidpoint_Sphere_Batch(benchmark::State& state) {
  Sphere<> sphere;
  const int N = 1000;

  // Pre-generate random point pairs.
  std::vector<Eigen::Vector3d> as(N), bs(N);
  for (int i = 0; i < N; ++i) {
    as[i] = sphere.random_point();
    bs[i] = sphere.random_point();
  }

  for (auto _ : state) {
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
      total += distance_midpoint(sphere, as[i], bs[i]);
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_DistanceMidpoint_Sphere_Batch);

static void BM_DistanceMidpoint_Torus2_Batch(benchmark::State& state) {
  Torus<2> torus;
  const int N = 1000;

  std::vector<Eigen::Vector2d> as(N), bs(N);
  for (int i = 0; i < N; ++i) {
    as[i] = torus.random_point();
    bs[i] = torus.random_point();
  }

  for (auto _ : state) {
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
      total += distance_midpoint(torus, as[i], bs[i]);
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_DistanceMidpoint_Torus2_Batch);

static void BM_DistanceMidpoint_SE2_Batch(benchmark::State& state) {
  SE2<> se2;
  const int N = 1000;

  std::vector<Eigen::Vector3d> as(N), bs(N);
  for (int i = 0; i < N; ++i) {
    as[i] = se2.random_point();
    bs[i] = se2.random_point();
  }

  for (auto _ : state) {
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
      total += distance_midpoint(se2, as[i], bs[i]);
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_DistanceMidpoint_SE2_Batch);

static void BM_DistanceMidpoint_Torus7_Batch(benchmark::State& state) {
  Torus<7> torus;
  const int N = 1000;

  std::vector<Eigen::Vector<double, 7>> as(N), bs(N);
  for (int i = 0; i < N; ++i) {
    as[i] = torus.random_point();
    bs[i] = torus.random_point();
  }

  for (auto _ : state) {
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
      total += distance_midpoint(torus, as[i], bs[i]);
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_DistanceMidpoint_Torus7_Batch);

// ---------------------------------------------------------------------------
// Dimension scaling: distance_midpoint on dynamic Torus
// ---------------------------------------------------------------------------

static void BM_DistanceMidpoint_TorusDynamic(benchmark::State& state) {
  const int d = static_cast<int>(state.range(0));
  Torus<Eigen::Dynamic> torus(d);

  Eigen::VectorXd p = Eigen::VectorXd::Constant(d, 0.5);
  Eigen::VectorXd q = Eigen::VectorXd::Constant(d, 4.0);

  for (auto _ : state) {
    benchmark::DoNotOptimize(distance_midpoint(torus, p, q));
  }
}
BENCHMARK(BM_DistanceMidpoint_TorusDynamic)->Arg(2)->Arg(5)->Arg(7)->Arg(10)->Arg(20)->Arg(50);

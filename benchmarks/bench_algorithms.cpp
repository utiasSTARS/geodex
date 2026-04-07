/// @file bench_algorithms.cpp
/// @brief Benchmarks for geodex algorithms: discrete_geodesic and distance_midpoint.

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <cmath>
#include <geodex/geodex.hpp>
#include <numbers>
#include <vector>

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

  InterpolationWorkspace<Sphere<>> ws;
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

  InterpolationWorkspace<AnisoSphere> ws;
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

  InterpolationWorkspace<Torus<2>> ws;
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

  InterpolationWorkspace<SE2<>> ws;
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
  auto ke = KineticEnergyMetric{[&](const Eigen::Vector2d& q) { return arm_metric.mass_matrix(q); }};
  using CSpace = ConfigurationSpace<Torus<2>, decltype(ke)>;
  CSpace cspace{Torus<2>{}, std::move(ke)};

  Eigen::Vector2d start(0.5, 0.5);
  Eigen::Vector2d target(2.5, 2.5);

  InterpolationWorkspace<CSpace> ws;
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

  InterpolationWorkspace<Se2Aniso> ws;
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

  InterpolationWorkspace<Sphere<>> ws;
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

  InterpolationWorkspace<Torus7> ws;
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

  InterpolationWorkspace<SE2<>> ws;
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

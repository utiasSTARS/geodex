/// @file bench_metrics.cpp
/// @brief Benchmarks for metric inner product evaluation across all metric types.

#include <cmath>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include "geodex/geodex.hpp"

#include "planar_manipulator_metric.hpp"

using namespace geodex;

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

// 2D
static const Eigen::Vector2d q2(1.0, 1.5);
static const Eigen::Vector2d u2(0.3, 0.4);
static const Eigen::Vector2d v2(0.1, -0.2);

// 3D
static const Eigen::Vector3d q3(1.0, 2.0, 0.5);
static const Eigen::Vector3d u3(0.3, 0.4, 0.1);
static const Eigen::Vector3d v3(0.1, -0.2, 0.5);

// 7D
static const Eigen::Vector<double, 7> q7 =
    (Eigen::Vector<double, 7>() << 1, 1.5, 2, 2.5, 3, 3.5, 4).finished();
static const Eigen::Vector<double, 7> u7 =
    (Eigen::Vector<double, 7>() << 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8).finished();
static const Eigen::Vector<double, 7> v7 =
    (Eigen::Vector<double, 7>() << 0.1, -0.4, 0.2, -0.1, 0.3, 0.5, -0.2).finished();

// ===========================================================================
// Point-independent metrics
// ===========================================================================

static void BM_Inner_EuclideanStandard2(benchmark::State& state) {
  EuclideanStandardMetric<2> metric;
  for (auto _ : state) {
    benchmark::DoNotOptimize(metric.inner(q2, u2, v2));
  }
}
BENCHMARK(BM_Inner_EuclideanStandard2);

static void BM_Inner_EuclideanStandard7(benchmark::State& state) {
  EuclideanStandardMetric<7> metric;
  for (auto _ : state) {
    benchmark::DoNotOptimize(metric.inner(q7, u7, v7));
  }
}
BENCHMARK(BM_Inner_EuclideanStandard7);

static void BM_Inner_TorusFlat2(benchmark::State& state) {
  TorusFlatMetric<2> metric;
  for (auto _ : state) {
    benchmark::DoNotOptimize(metric.inner(q2, u2, v2));
  }
}
BENCHMARK(BM_Inner_TorusFlat2);

static void BM_Inner_TorusFlat7(benchmark::State& state) {
  TorusFlatMetric<7> metric;
  for (auto _ : state) {
    benchmark::DoNotOptimize(metric.inner(q7, u7, v7));
  }
}
BENCHMARK(BM_Inner_TorusFlat7);

static void BM_Inner_ConstantSPD3(benchmark::State& state) {
  Eigen::Matrix3d A;
  A << 2.0, 0.5, 0.0, 0.5, 3.0, 0.1, 0.0, 0.1, 1.5;
  ConstantSPDMetric<3> metric{A};
  for (auto _ : state) {
    benchmark::DoNotOptimize(metric.inner(q3, u3, v3));
  }
}
BENCHMARK(BM_Inner_ConstantSPD3);

static void BM_Inner_ConstantSPD7(benchmark::State& state) {
  Eigen::Matrix<double, 7, 7> A = Eigen::Matrix<double, 7, 7>::Identity();
  A(0, 1) = 0.5;
  A(1, 0) = 0.5;
  A(2, 2) = 4.0;
  ConstantSPDMetric<7> metric{A};
  for (auto _ : state) {
    benchmark::DoNotOptimize(metric.inner(q7, u7, v7));
  }
}
BENCHMARK(BM_Inner_ConstantSPD7);

static void BM_Inner_SE2LeftInvariant(benchmark::State& state) {
  SE2LeftInvariantMetric metric(1.0, 100.0, 0.5);
  for (auto _ : state) {
    benchmark::DoNotOptimize(metric.inner(q3, u3, v3));
  }
}
BENCHMARK(BM_Inner_SE2LeftInvariant);

// ===========================================================================
// Point-dependent metrics
// ===========================================================================

static void BM_Inner_KineticEnergy_2Link(benchmark::State& state) {
  PlanarManipulatorMetric arm;
  auto ke = KineticEnergyMetric{[&](const Eigen::Vector2d& q) { return arm.mass_matrix(q); }};
  for (auto _ : state) {
    benchmark::DoNotOptimize(ke.inner(q2, u2, v2));
  }
}
BENCHMARK(BM_Inner_KineticEnergy_2Link);

static void BM_Inner_Jacobi_2Link(benchmark::State& state) {
  PlanarManipulatorMetric arm;
  auto jacobi = JacobiMetric{
      [&](const Eigen::Vector2d& q) { return arm.mass_matrix(q); },
      [](const Eigen::Vector2d& q) {
        // Simple gravity potential: P(q) = -m*g*(l1*cos(q1) + l2*cos(q1+q2))
        return -9.81 * (std::cos(q[0]) + 0.5 * std::cos(q[0] + q[1]));
      },
      20.0  // total energy H
  };
  for (auto _ : state) {
    benchmark::DoNotOptimize(jacobi.inner(q2, u2, v2));
  }
}
BENCHMARK(BM_Inner_Jacobi_2Link);

static void BM_Inner_Pullback_2Link(benchmark::State& state) {
  // 2-link arm: 2D config space -> 2D task space (end-effector position)
  auto pullback = PullbackMetric{
      [](const Eigen::Vector2d& q) -> Eigen::Matrix2d {
        // Jacobian of 2-link arm end-effector
        double c1 = std::cos(q[0]), s1 = std::sin(q[0]);
        double c12 = std::cos(q[0] + q[1]), s12 = std::sin(q[0] + q[1]);
        Eigen::Matrix2d J;
        J(0, 0) = -s1 - 0.5 * s12;
        J(0, 1) = -0.5 * s12;
        J(1, 0) = c1 + 0.5 * c12;
        J(1, 1) = 0.5 * c12;
        return J;
      },
      [](const Eigen::Vector2d& /*q*/) -> Eigen::Matrix2d { return Eigen::Matrix2d::Identity(); },
      0.01  // regularization lambda
  };
  for (auto _ : state) {
    benchmark::DoNotOptimize(pullback.inner(q2, u2, v2));
  }
}
BENCHMARK(BM_Inner_Pullback_2Link);

static void BM_Inner_Weighted_ConstantSPD3(benchmark::State& state) {
  Eigen::Matrix3d A;
  A << 2.0, 0.5, 0.0, 0.5, 3.0, 0.1, 0.0, 0.1, 1.5;
  WeightedMetric weighted{ConstantSPDMetric<3>{A}, 2.5};
  for (auto _ : state) {
    benchmark::DoNotOptimize(weighted.inner(q3, u3, v3));
  }
}
BENCHMARK(BM_Inner_Weighted_ConstantSPD3);

// ===========================================================================
// ConfigurationSpace overhead: KE metric vs flat metric on Torus<2>
// ===========================================================================

static void BM_Distance_Torus2_Flat(benchmark::State& state) {
  Torus<2> torus;
  Eigen::Vector2d p(0.5, 0.5);
  Eigen::Vector2d q(2.5, 2.5);
  for (auto _ : state) {
    benchmark::DoNotOptimize(p.data());
    benchmark::DoNotOptimize(q.data());
    auto d = distance_midpoint(torus, p, q);
    benchmark::DoNotOptimize(d);
  }
}
BENCHMARK(BM_Distance_Torus2_Flat);

static void BM_Distance_CSpace_Torus2_KE(benchmark::State& state) {
  PlanarManipulatorMetric arm;
  auto ke = KineticEnergyMetric{[&](const Eigen::Vector2d& q) { return arm.mass_matrix(q); }};
  ConfigurationSpace cspace{Torus<2>{}, std::move(ke)};

  Eigen::Vector2d p(0.5, 0.5);
  Eigen::Vector2d q(2.5, 2.5);
  for (auto _ : state) {
    benchmark::DoNotOptimize(p.data());
    benchmark::DoNotOptimize(q.data());
    auto d = distance_midpoint(cspace, p, q);
    benchmark::DoNotOptimize(d);
  }
}
BENCHMARK(BM_Distance_CSpace_Torus2_KE);

// ===========================================================================
// Norm benchmarks (metric.norm includes sqrt overhead)
// ===========================================================================

static void BM_Norm_ConstantSPD3(benchmark::State& state) {
  Eigen::Matrix3d A;
  A << 2.0, 0.5, 0.0, 0.5, 3.0, 0.1, 0.0, 0.1, 1.5;
  ConstantSPDMetric<3> metric{A};
  for (auto _ : state) {
    benchmark::DoNotOptimize(metric.norm(q3, u3));
  }
}
BENCHMARK(BM_Norm_ConstantSPD3);

static void BM_Norm_KineticEnergy_2Link(benchmark::State& state) {
  PlanarManipulatorMetric arm;
  auto ke = KineticEnergyMetric{[&](const Eigen::Vector2d& q) { return arm.mass_matrix(q); }};
  for (auto _ : state) {
    benchmark::DoNotOptimize(ke.norm(q2, u2));
  }
}
BENCHMARK(BM_Norm_KineticEnergy_2Link);

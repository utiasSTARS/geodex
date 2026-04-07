/// @file bench_manifold_ops.cpp
/// @brief Micro-benchmarks for manifold primitive operations.

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <geodex/geodex.hpp>

using namespace geodex;

// ---------------------------------------------------------------------------
// Fixed test data per manifold type
// ---------------------------------------------------------------------------

// Sphere S^2
static const Eigen::Vector3d sphere_p = Eigen::Vector3d(0.0, 0.0, 1.0);  // north pole
static const Eigen::Vector3d sphere_q =
    Eigen::Vector3d(std::sin(1.0), 0.0, std::cos(1.0));  // theta=1 on great circle
static const Eigen::Vector3d sphere_v(0.3, 0.4, 0.0);    // tangent at north pole
static const Eigen::Vector3d sphere_u(0.1, -0.2, 0.0);

// SE(2)
static const Eigen::Vector3d se2_p(1.0, 1.0, 0.0);
static const Eigen::Vector3d se2_q(5.0, 3.0, 1.0);
static const Eigen::Vector3d se2_v(0.5, 0.3, 0.2);
static const Eigen::Vector3d se2_u(0.1, -0.4, 0.1);

// Euclidean R^2
static const Eigen::Vector2d euc2_p(1.0, 2.0);
static const Eigen::Vector2d euc2_q(4.0, 6.0);
static const Eigen::Vector2d euc2_v(0.5, 0.3);
static const Eigen::Vector2d euc2_u(0.1, -0.4);

// Euclidean R^7
static const Eigen::Vector<double, 7> euc7_p =
    (Eigen::Vector<double, 7>() << 1, 2, 3, 4, 5, 6, 7).finished();
static const Eigen::Vector<double, 7> euc7_q =
    (Eigen::Vector<double, 7>() << 4, 6, 2, 8, 1, 3, 9).finished();
static const Eigen::Vector<double, 7> euc7_v =
    (Eigen::Vector<double, 7>() << 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8).finished();
static const Eigen::Vector<double, 7> euc7_u =
    (Eigen::Vector<double, 7>() << 0.1, -0.4, 0.2, -0.1, 0.3, 0.5, -0.2).finished();

// Torus T^2
static const Eigen::Vector2d tor2_p(0.5, 0.5);
static const Eigen::Vector2d tor2_q(5.0, 4.0);
static const Eigen::Vector2d tor2_v(0.3, 0.4);
static const Eigen::Vector2d tor2_u(0.1, -0.2);

// Torus T^7
static const Eigen::Vector<double, 7> tor7_p =
    (Eigen::Vector<double, 7>() << 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5).finished();
static const Eigen::Vector<double, 7> tor7_q =
    (Eigen::Vector<double, 7>() << 4.0, 5.0, 0.5, 3.0, 1.0, 4.5, 2.0).finished();
static const Eigen::Vector<double, 7> tor7_v =
    (Eigen::Vector<double, 7>() << 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8).finished();
static const Eigen::Vector<double, 7> tor7_u =
    (Eigen::Vector<double, 7>() << 0.1, -0.4, 0.2, -0.1, 0.3, 0.5, -0.2).finished();

// ---------------------------------------------------------------------------
// Manifold instances
// ---------------------------------------------------------------------------

static Sphere<> sphere;
static Sphere<2, SphereRoundMetric, SphereProjectionRetraction> sphere_proj;
static Sphere<2, ConstantSPDMetric<3>> sphere_aniso{
    ConstantSPDMetric<3>{(Eigen::Matrix3d() << 10, 0, 0, 0, 1, 0, 0, 0, 1).finished()}};
static Euclidean<2> euclidean2;
static Euclidean<7> euclidean7;
static Torus<2> torus2;
static Torus<7> torus7;
static SE2<> se2_exp;

// ===========================================================================
// Sphere (default: round metric + exponential map)
// ===========================================================================

static void BM_Exp_Sphere(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere.exp(sphere_p, sphere_v));
  }
}
BENCHMARK(BM_Exp_Sphere);

static void BM_Log_Sphere(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere.log(sphere_p, sphere_q));
  }
}
BENCHMARK(BM_Log_Sphere);

static void BM_Distance_Sphere(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere.distance(sphere_p, sphere_q));
  }
}
BENCHMARK(BM_Distance_Sphere);

static void BM_Inner_Sphere(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere.inner(sphere_p, sphere_u, sphere_v));
  }
}
BENCHMARK(BM_Inner_Sphere);

static void BM_Norm_Sphere(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere.norm(sphere_p, sphere_v));
  }
}
BENCHMARK(BM_Norm_Sphere);

static void BM_Geodesic_Sphere(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere.geodesic(sphere_p, sphere_q, 0.5));
  }
}
BENCHMARK(BM_Geodesic_Sphere);

static void BM_RandomPoint_Sphere(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere.random_point());
  }
}
BENCHMARK(BM_RandomPoint_Sphere);

// ===========================================================================
// Sphere (projection retraction)
// ===========================================================================

static void BM_Exp_SphereProj(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_proj.exp(sphere_p, sphere_v));
  }
}
BENCHMARK(BM_Exp_SphereProj);

static void BM_Log_SphereProj(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_proj.log(sphere_p, sphere_q));
  }
}
BENCHMARK(BM_Log_SphereProj);

static void BM_Distance_SphereProj(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_proj.distance(sphere_p, sphere_q));
  }
}
BENCHMARK(BM_Distance_SphereProj);

// ===========================================================================
// Sphere (anisotropic: ConstantSPDMetric)
// ===========================================================================

static void BM_Exp_SphereAniso(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_aniso.exp(sphere_p, sphere_v));
  }
}
BENCHMARK(BM_Exp_SphereAniso);

static void BM_Distance_SphereAniso(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_aniso.distance(sphere_p, sphere_q));
  }
}
BENCHMARK(BM_Distance_SphereAniso);

static void BM_Inner_SphereAniso(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(sphere_aniso.inner(sphere_p, sphere_u, sphere_v));
  }
}
BENCHMARK(BM_Inner_SphereAniso);

// ===========================================================================
// Euclidean R^2
// ===========================================================================

static void BM_Exp_Euclidean2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(euclidean2.exp(euc2_p, euc2_v));
  }
}
BENCHMARK(BM_Exp_Euclidean2);

static void BM_Log_Euclidean2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(euclidean2.log(euc2_p, euc2_q));
  }
}
BENCHMARK(BM_Log_Euclidean2);

static void BM_Distance_Euclidean2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(euclidean2.distance(euc2_p, euc2_q));
  }
}
BENCHMARK(BM_Distance_Euclidean2);

static void BM_Inner_Euclidean2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(euclidean2.inner(euc2_p, euc2_u, euc2_v));
  }
}
BENCHMARK(BM_Inner_Euclidean2);

// ===========================================================================
// Euclidean R^7
// ===========================================================================

static void BM_Exp_Euclidean7(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(euclidean7.exp(euc7_p, euc7_v));
  }
}
BENCHMARK(BM_Exp_Euclidean7);

static void BM_Log_Euclidean7(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(euclidean7.log(euc7_p, euc7_q));
  }
}
BENCHMARK(BM_Log_Euclidean7);

static void BM_Distance_Euclidean7(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(euclidean7.distance(euc7_p, euc7_q));
  }
}
BENCHMARK(BM_Distance_Euclidean7);

static void BM_Inner_Euclidean7(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(euclidean7.inner(euc7_p, euc7_u, euc7_v));
  }
}
BENCHMARK(BM_Inner_Euclidean7);

// ===========================================================================
// Torus T^2
// ===========================================================================

static void BM_Exp_Torus2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus2.exp(tor2_p, tor2_v));
  }
}
BENCHMARK(BM_Exp_Torus2);

static void BM_Log_Torus2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus2.log(tor2_p, tor2_q));
  }
}
BENCHMARK(BM_Log_Torus2);

static void BM_Distance_Torus2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus2.distance(tor2_p, tor2_q));
  }
}
BENCHMARK(BM_Distance_Torus2);

static void BM_Inner_Torus2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus2.inner(tor2_p, tor2_u, tor2_v));
  }
}
BENCHMARK(BM_Inner_Torus2);

static void BM_Geodesic_Torus2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus2.geodesic(tor2_p, tor2_q, 0.5));
  }
}
BENCHMARK(BM_Geodesic_Torus2);

static void BM_RandomPoint_Torus2(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus2.random_point());
  }
}
BENCHMARK(BM_RandomPoint_Torus2);

// ===========================================================================
// Torus T^7
// ===========================================================================

static void BM_Exp_Torus7(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus7.exp(tor7_p, tor7_v));
  }
}
BENCHMARK(BM_Exp_Torus7);

static void BM_Log_Torus7(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus7.log(tor7_p, tor7_q));
  }
}
BENCHMARK(BM_Log_Torus7);

static void BM_Distance_Torus7(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus7.distance(tor7_p, tor7_q));
  }
}
BENCHMARK(BM_Distance_Torus7);

static void BM_Inner_Torus7(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(torus7.inner(tor7_p, tor7_u, tor7_v));
  }
}
BENCHMARK(BM_Inner_Torus7);

// ===========================================================================
// SE(2) — Exponential map
// ===========================================================================

static void BM_Exp_SE2Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_exp.exp(se2_p, se2_v));
  }
}
BENCHMARK(BM_Exp_SE2Exp);

static void BM_Log_SE2Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_exp.log(se2_p, se2_q));
  }
}
BENCHMARK(BM_Log_SE2Exp);

static void BM_Distance_SE2Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_exp.distance(se2_p, se2_q));
  }
}
BENCHMARK(BM_Distance_SE2Exp);

static void BM_Inner_SE2Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_exp.inner(se2_p, se2_u, se2_v));
  }
}
BENCHMARK(BM_Inner_SE2Exp);

static void BM_Geodesic_SE2Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_exp.geodesic(se2_p, se2_q, 0.5));
  }
}
BENCHMARK(BM_Geodesic_SE2Exp);

static void BM_RandomPoint_SE2Exp(benchmark::State& state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(se2_exp.random_point());
  }
}
BENCHMARK(BM_RandomPoint_SE2Exp);

// ===========================================================================
// Dimension scaling: Torus (dynamic dimension)
// ===========================================================================

static void BM_Distance_TorusDynamic(benchmark::State& state) {
  const int d = static_cast<int>(state.range(0));
  Torus<Eigen::Dynamic> torus(d);

  Eigen::VectorXd p = Eigen::VectorXd::Constant(d, 0.5);
  Eigen::VectorXd q = Eigen::VectorXd::Constant(d, 4.0);

  for (auto _ : state) {
    benchmark::DoNotOptimize(torus.distance(p, q));
  }
}
BENCHMARK(BM_Distance_TorusDynamic)->Arg(2)->Arg(5)->Arg(7)->Arg(10)->Arg(20)->Arg(50);

static void BM_Exp_TorusDynamic(benchmark::State& state) {
  const int d = static_cast<int>(state.range(0));
  Torus<Eigen::Dynamic> torus(d);

  Eigen::VectorXd p = Eigen::VectorXd::Constant(d, 0.5);
  Eigen::VectorXd v = Eigen::VectorXd::Constant(d, 0.3);

  for (auto _ : state) {
    benchmark::DoNotOptimize(torus.exp(p, v));
  }
}
BENCHMARK(BM_Exp_TorusDynamic)->Arg(2)->Arg(5)->Arg(7)->Arg(10)->Arg(20)->Arg(50);

static void BM_Log_TorusDynamic(benchmark::State& state) {
  const int d = static_cast<int>(state.range(0));
  Torus<Eigen::Dynamic> torus(d);

  Eigen::VectorXd p = Eigen::VectorXd::Constant(d, 0.5);
  Eigen::VectorXd q = Eigen::VectorXd::Constant(d, 4.0);

  for (auto _ : state) {
    benchmark::DoNotOptimize(torus.log(p, q));
  }
}
BENCHMARK(BM_Log_TorusDynamic)->Arg(2)->Arg(5)->Arg(7)->Arg(10)->Arg(20)->Arg(50);

// ===========================================================================
// Dimension scaling: Euclidean (dynamic dimension)
// ===========================================================================

static void BM_Distance_EuclideanDynamic(benchmark::State& state) {
  const int d = static_cast<int>(state.range(0));
  Euclidean<Eigen::Dynamic> euc(d);

  Eigen::VectorXd p = Eigen::VectorXd::Constant(d, 1.0);
  Eigen::VectorXd q = Eigen::VectorXd::Constant(d, 4.0);

  for (auto _ : state) {
    benchmark::DoNotOptimize(euc.distance(p, q));
  }
}
BENCHMARK(BM_Distance_EuclideanDynamic)->Arg(2)->Arg(5)->Arg(7)->Arg(10)->Arg(20)->Arg(50);

static void BM_Exp_EuclideanDynamic(benchmark::State& state) {
  const int d = static_cast<int>(state.range(0));
  Euclidean<Eigen::Dynamic> euc(d);

  Eigen::VectorXd p = Eigen::VectorXd::Constant(d, 1.0);
  Eigen::VectorXd v = Eigen::VectorXd::Constant(d, 0.3);

  for (auto _ : state) {
    benchmark::DoNotOptimize(euc.exp(p, v));
  }
}
BENCHMARK(BM_Exp_EuclideanDynamic)->Arg(2)->Arg(5)->Arg(7)->Arg(10)->Arg(20)->Arg(50);

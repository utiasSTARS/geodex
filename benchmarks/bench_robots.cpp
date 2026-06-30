/// @file benchmarks/bench_robots.cpp
/// @brief Microbenchmarks comparing the precompiled Panda CRBA in
///        geodex::robots against Pinocchio's runtime CRBA across the call
///        shapes that drive the runbench Panda trial profile (single eval,
///        KineticEnergy::inner, KineticEnergy::inner_matrix,
///        distance_midpoint).

#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "geodex/algorithm/distance.hpp"
#include "geodex/integration/pinocchio/mass_matrix.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/robots/mass_matrix.hpp"

namespace {

const std::string kPandaUrdf = GEODEX_PANDA_URDF;

using PandaMM = geodex::robots::MassMatrix<geodex::robots::Robot::Panda>;
using PandaVec = PandaMM::Vec;  // Eigen::Matrix<double, 7, 1>

// ---------------- Sample buffers shared across runs ---------------------

class QSet {
 public:
  explicit QSet(int n = 1024) : qs_(n) {
    const auto [lo, hi] = PandaMM::joint_limits();
    std::mt19937 rng(2024);
    for (auto& q : qs_) {
      for (int i = 0; i < PandaMM::Nq; ++i) {
        std::uniform_real_distribution<double> u(lo[i], hi[i]);
        q[i] = u(rng);
      }
    }
  }
  // Fixed-size accessor for the geodex_robots / fixed-size benches.
  const PandaVec& at(std::size_t i) const { return qs_[i % qs_.size()]; }
  // Dynamic-size accessor (for Pinocchio benches whose API takes VectorXd).
  Eigen::VectorXd at_dyn(std::size_t i) const {
    return Eigen::VectorXd(qs_[i % qs_.size()]);
  }
  std::size_t size() const { return qs_.size(); }

 private:
  std::vector<PandaVec> qs_;
};

QSet& qset() {
  static QSet q;
  return q;
}

// ---------------- Layered drill-downs ------------------------------------
// Three call shapes, each adding one layer of overhead:
//
//   forward_zero (raw)       the CppAD::CG-generated symbol with no Eigen,
//                            no wrapper — pure scalar math (with hoisted
//                            libmvec trig on Linux x86_64).
//   panda_crba (wrapper)     the extern "C" adaptor that builds the in/out
//                            pointer arrays and zero-inits the atomic struct.
//   MassMatrix::operator()   the public path: function-pointer dispatch +
//                            mirror loop into a `mutable Eigen::MatrixXd&`.
//
// Comparing the three tells us how much overhead each layer adds on top of
// the generated math. The struct LangCAtomicFun is layout-compatible with
// the one defined inside panda_crba.cpp (3 pointer-sized fields, passed by
// value); the C ABI does not encode struct names in the symbol.
extern "C" {
struct LangCAtomicFunBench {
  void* libModel;
  void* forward;
  void* reverse;
};
void panda_crba_forward_zero(double const* const* in, double* const* out,
                             LangCAtomicFunBench atomic);
void panda_crba(const double q[7], double M_upper[28]);
}  // extern "C"

static void BM_CRBA_Panda_RawForwardZero(benchmark::State& state) {
  alignas(64) double M_upper[28];
  std::size_t i = 0;
  for (auto _ : state) {
    const auto& q = qset().at(i++);
    const double* in[1] = {q.data()};
    double* out[1] = {M_upper};
    LangCAtomicFunBench atomic = {};
    panda_crba_forward_zero(in, out, atomic);
    benchmark::DoNotOptimize(M_upper);
  }
}
BENCHMARK(BM_CRBA_Panda_RawForwardZero);

static void BM_CRBA_Panda_ExternCWrapper(benchmark::State& state) {
  alignas(64) double M_upper[28];
  std::size_t i = 0;
  for (auto _ : state) {
    const auto& q = qset().at(i++);
    panda_crba(q.data(), M_upper);
    benchmark::DoNotOptimize(M_upper);
  }
}
BENCHMARK(BM_CRBA_Panda_ExternCWrapper);

// ---------------- Single-call CRBA ---------------------------------------

static void BM_CRBA_Panda_Pinocchio(benchmark::State& state) {
  ::pinocchio::Model model;
  ::pinocchio::urdf::buildModel(kPandaUrdf, model);
  ::pinocchio::Data data(model);
  std::size_t i = 0;
  for (auto _ : state) {
    const Eigen::VectorXd q = qset().at_dyn(i++);
    ::pinocchio::crba(model, data, q);
    benchmark::DoNotOptimize(data.M);
  }
}
BENCHMARK(BM_CRBA_Panda_Pinocchio);

static void BM_CRBA_Panda_Robots(benchmark::State& state) {
  PandaMM mm{};
  std::size_t i = 0;
  for (auto _ : state) {
    const auto& q = qset().at(i++);
    const auto& M = mm(q);
    benchmark::DoNotOptimize(M);
  }
}
BENCHMARK(BM_CRBA_Panda_Robots);

// ---------------- KineticEnergyMetric::inner(q,u,v) ----------------------

static void BM_KineticInner_Panda_Pinocchio(benchmark::State& state) {
  geodex::integration::pinocchio::MassMatrix mass{kPandaUrdf};
  geodex::KineticEnergyMetric ke{std::ref(mass)};
  std::mt19937 rng(99);
  PandaVec u, v;
  for (int i = 0; i < PandaMM::Nq; ++i) {
    u[i] = std::uniform_real_distribution<double>(-1, 1)(rng);
    v[i] = std::uniform_real_distribution<double>(-1, 1)(rng);
  }
  std::size_t i = 0;
  for (auto _ : state) {
    const auto& q = qset().at(i++);
    const double inner = ke.inner(q, u, v);
    benchmark::DoNotOptimize(inner);
  }
}
BENCHMARK(BM_KineticInner_Panda_Pinocchio);

static void BM_KineticInner_Panda_Robots(benchmark::State& state) {
  PandaMM mm{};
  geodex::KineticEnergyMetric ke{std::ref(mm)};
  std::mt19937 rng(99);
  PandaVec u, v;
  for (int i = 0; i < PandaMM::Nq; ++i) {
    u[i] = std::uniform_real_distribution<double>(-1, 1)(rng);
    v[i] = std::uniform_real_distribution<double>(-1, 1)(rng);
  }
  std::size_t i = 0;
  for (auto _ : state) {
    const auto& q = qset().at(i++);
    const double inner = ke.inner(q, u, v);
    benchmark::DoNotOptimize(inner);
  }
}
BENCHMARK(BM_KineticInner_Panda_Robots);

// ---------------- KineticEnergyMetric::inner_matrix (batched, d=7) -------
// Mirrors the existing optimization at kinetic_energy.hpp:60 — one M(q) call
// produces the entire d×d tensor U^T M V via a single matmul.

static void BM_KineticInnerMatrix_Panda_Pinocchio(benchmark::State& state) {
  geodex::integration::pinocchio::MassMatrix mass{kPandaUrdf};
  geodex::KineticEnergyMetric ke{std::ref(mass)};
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
  std::size_t i = 0;
  for (auto _ : state) {
    const auto& q = qset().at(i++);
    const Eigen::MatrixXd G = ke.inner_matrix(q, I, I);
    benchmark::DoNotOptimize(G);
  }
}
BENCHMARK(BM_KineticInnerMatrix_Panda_Pinocchio);

static void BM_KineticInnerMatrix_Panda_Robots(benchmark::State& state) {
  PandaMM mm{};
  geodex::KineticEnergyMetric ke{std::ref(mm)};
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
  std::size_t i = 0;
  for (auto _ : state) {
    const auto& q = qset().at(i++);
    const Eigen::MatrixXd G = ke.inner_matrix(q, I, I);
    benchmark::DoNotOptimize(G);
  }
}
BENCHMARK(BM_KineticInnerMatrix_Panda_Robots);

// ---------------- distance_midpoint via KineticEnergyMetric --------------
// The actual hot path from the Panda runbench profile: each motion-validator
// step calls space->distance(s1, s2) → distance_midpoint → metric.norm →
// one M(q) eval.

template <typename MassFn>
static void run_distance_midpoint(benchmark::State& state, MassFn mass_fn) {
  using Manifold = geodex::ConfigurationSpace<
      geodex::Euclidean<7>, geodex::KineticEnergyMetric<MassFn>>;
  Manifold manifold{geodex::Euclidean<7>{},
                    geodex::KineticEnergyMetric<MassFn>{mass_fn}};

  std::size_t i = 0;
  for (auto _ : state) {
    const auto& a = qset().at(i++);
    const auto& b = qset().at(i++);
    const double d = geodex::distance_midpoint(manifold, a, b);
    benchmark::DoNotOptimize(d);
  }
}

static void BM_DistanceMidpoint_Panda_Pinocchio(benchmark::State& state) {
  geodex::integration::pinocchio::MassMatrix mass{kPandaUrdf};
  auto fn = [m = &mass](const Eigen::VectorXd& q) -> const Eigen::MatrixXd& {
    return (*m)(q);
  };
  run_distance_midpoint(state, fn);
}
BENCHMARK(BM_DistanceMidpoint_Panda_Pinocchio);

static void BM_DistanceMidpoint_Panda_Robots(benchmark::State& state) {
  PandaMM mm{};
  // std::ref keeps the fixed-size signature of mm intact: PandaMM::operator()
  // takes/returns Matrix<double, 7, 7>, so KineticEnergyMetric::inner sees
  // fixed-size types end-to-end.
  run_distance_midpoint(state, std::ref(mm));
}
BENCHMARK(BM_DistanceMidpoint_Panda_Robots);

}  // namespace

BENCHMARK_MAIN();

/// @file test_precompute_matrix_lower_bound.cpp
/// @brief Tests for `geodex::algorithm::precompute_matrix_lower_bound`.

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "geodex/algorithm/precompute_matrix_lower_bound.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/metrics/kinetic_energy.hpp"

namespace ga = geodex::algorithm;
namespace gh = geodex::heuristics;

namespace {

// 2-link arm mass matrix:
//   M(q) = [[a + 2b cos(q2), c + b cos(q2)],
//           [c + b cos(q2), c]]
// SPD when a > 2|b| + c and c > 0.
struct TwoLinkArmMass {
  double a = 2.0;
  double b = 0.5;
  double c = 1.0;
  Eigen::Matrix2d operator()(const Eigen::Vector2d& q) const {
    const double cq2 = std::cos(q[1]);
    Eigen::Matrix2d M;
    M << a + 2.0 * b * cq2, c + b * cq2,
         c + b * cq2,       c;
    return M;
  }
};

// Build a ConfigurationSpace<Euclidean<2>, KineticEnergyMetric<TwoLinkArmMass>> with
// joint bounds [-pi, pi] x [-pi, pi].
auto make_two_link_manifold() {
  geodex::Euclidean<2, geodex::KineticEnergyMetric<TwoLinkArmMass>> base{
      geodex::KineticEnergyMetric<TwoLinkArmMass>{TwoLinkArmMass{}}};
  Eigen::VectorXd lo(2);
  lo << -3.14159265358979, -3.14159265358979;
  Eigen::VectorXd hi(2);
  hi << 3.14159265358979, 3.14159265358979;
  base.set_sampling_bounds(lo, hi);
  return base;
}

// Configuration space with a constant SPD metric (M(q) = A everywhere).
auto make_constant_manifold(const Eigen::Matrix3d& A) {
  auto mass_fn = [A](const Eigen::Vector3d& /*q*/) -> Eigen::Matrix3d { return A; };
  geodex::Euclidean<3, geodex::KineticEnergyMetric<decltype(mass_fn)>> base{
      geodex::KineticEnergyMetric<decltype(mass_fn)>{std::move(mass_fn)}};
  Eigen::VectorXd lo(3);
  lo << -1.0, -1.0, -1.0;
  Eigen::VectorXd hi(3);
  hi << 1.0, 1.0, 1.0;
  base.set_sampling_bounds(lo, hi);
  return base;
}

}  // namespace

TEST(PrecomputeMatrixLB, ConstantMetricTerminatesAtFirstIteration) {
  // For a constant SPD metric A, initializing M_lower = M(q_center) = A makes
  // L^{-1} M(q) L^{-T} = I everywhere, so the very first outer iteration finds
  // lambda_min = 1 and the loop exits immediately.
  Eigen::Matrix3d A;
  A << 4.0, 0.5, 0.0,
       0.5, 2.0, 0.0,
       0.0, 0.0, 1.0;
  const auto manifold = make_constant_manifold(A);

  const auto result = ga::precompute_matrix_lower_bound(manifold);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.n_outer_iters, 0);
  EXPECT_NEAR(result.lambda_min_certificate, 1.0, 1e-6);
}

TEST(PrecomputeMatrixLB, ConstantMetricRecoversTheMatrix) {
  // M_lower must equal A (no Loewner-meet updates were needed).
  Eigen::Matrix3d A;
  A << 5.0, 1.0, 0.0,
       1.0, 3.0, 0.5,
       0.0, 0.5, 2.0;
  const auto manifold = make_constant_manifold(A);

  const auto result = ga::precompute_matrix_lower_bound(manifold);
  EXPECT_TRUE(result.M_lower.isApprox(A, 1e-10));
}

TEST(PrecomputeMatrixLB, TwoLinkArmLoewnerBoundIsPSDAtRandomSamples) {
  const auto manifold = make_two_link_manifold();
  const auto result = ga::precompute_matrix_lower_bound(manifold);
  ASSERT_TRUE(result.converged);

  // For 30 random configurations, M(q) - M_lower must be PSD modulo numerical slack.
  std::mt19937 rng(2026);
  std::uniform_real_distribution<double> qd(-3.14159, 3.14159);
  TwoLinkArmMass mass;
  for (int trial = 0; trial < 30; ++trial) {
    Eigen::Vector2d q(qd(rng), qd(rng));
    const Eigen::Matrix2d M = mass(q);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M - result.M_lower,
                                                          Eigen::EigenvaluesOnly);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1e-6);
  }
}

TEST(PrecomputeMatrixLB, CertificateLambdaMinCloseToOne) {
  const auto manifold = make_two_link_manifold();
  ga::PrecomputeMatrixLowerBoundSettings settings;
  settings.tol = 1e-4;
  const auto result = ga::precompute_matrix_lower_bound(manifold, settings);
  EXPECT_TRUE(result.converged);
  EXPECT_GE(result.lambda_min_certificate, 1.0 - settings.tol);
  EXPECT_LE(result.lambda_min_certificate, 1.0 + 1e-3);  // never exceeds 1 by much
}

TEST(PrecomputeMatrixLB, DeterministicWithSeed) {
  const auto manifold = make_two_link_manifold();
  ga::PrecomputeMatrixLowerBoundSettings settings;
  settings.seed = 7;
  settings.max_outer = 5;  // bound runtime — algorithm itself is deterministic per seed
  const auto r1 = ga::precompute_matrix_lower_bound(manifold, settings);
  const auto r2 = ga::precompute_matrix_lower_bound(manifold, settings);
  EXPECT_TRUE(r1.M_lower.isApprox(r2.M_lower, 0.0));
  EXPECT_EQ(r1.n_outer_iters, r2.n_outer_iters);
  EXPECT_EQ(r1.n_metric_evals, r2.n_metric_evals);
  EXPECT_EQ(r1.lambda_min_certificate, r2.lambda_min_certificate);
}

TEST(PrecomputeMatrixLB, AcceptsConfigurationSpaceWrapper) {
  // The algorithm must accept ConfigurationSpace<Euclidean<...>, KineticEnergyMetric<...>>
  // — bounds and inner_matrix forward correctly through the wrapper.
  TwoLinkArmMass mass;
  geodex::Euclidean<2> base;
  Eigen::VectorXd lo(2);
  lo << -1.0, -1.0;
  Eigen::VectorXd hi(2);
  hi << 1.0, 1.0;
  base.set_sampling_bounds(lo, hi);
  geodex::ConfigurationSpace cs{std::move(base),
                                geodex::KineticEnergyMetric<TwoLinkArmMass>{mass}};

  const auto result = ga::precompute_matrix_lower_bound(cs);
  EXPECT_TRUE(result.converged);
  // M_lower is SPD → all eigenvalues positive.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(result.M_lower, Eigen::EigenvaluesOnly);
  EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

TEST(PrecomputeMatrixLB, BoundIntegratesWithMatrixLowerBoundHeuristic) {
  // The result should plug into heuristics::MatrixLowerBound and produce a valid
  // distance lower bound between two configurations.
  const auto manifold = make_two_link_manifold();
  const auto result = ga::precompute_matrix_lower_bound(manifold);
  ASSERT_TRUE(result.converged);

  gh::MatrixLowerBound<> h(result.M_lower);
  Eigen::Vector2d a(0.0, 0.0);
  Eigen::Vector2d b(1.0, 0.5);

  const double h_val = h(a, b);
  EXPECT_GE(h_val, 0.0);
  // Sanity: h(a,a) = 0.
  EXPECT_NEAR(h(a, a), 0.0, 1e-12);
}

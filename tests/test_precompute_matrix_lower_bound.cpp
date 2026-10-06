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
#include "geodex/manifold/se2.hpp"
#include "geodex/metrics/clearance.hpp"
#include "geodex/metrics/se2_left_invariant.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/utils/angle.hpp"

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
  settings.max_outer = 5;  // the algorithm is deterministic per seed
  const auto r1 = ga::precompute_matrix_lower_bound(manifold, settings);
  const auto r2 = ga::precompute_matrix_lower_bound(manifold, settings);
  EXPECT_TRUE(r1.M_lower.isApprox(r2.M_lower, 0.0));
  EXPECT_EQ(r1.n_outer_iters, r2.n_outer_iters);
  EXPECT_EQ(r1.n_metric_evals, r2.n_metric_evals);
  EXPECT_EQ(r1.lambda_min_certificate, r2.lambda_min_certificate);
}

TEST(PrecomputeMatrixLB, AcceptsConfigurationSpaceWrapper) {
  // The algorithm accepts ConfigurationSpace<Euclidean<...>, KineticEnergyMetric<...>>.
  // Bounds and inner_matrix forward through the wrapper.
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

// ---------------------------------------------------------------------------
// SE(2) coordinate metric
// ---------------------------------------------------------------------------

namespace {

// Car-like SE(2) over a 30 x 12 world with theta spanning a full turn.
auto make_car_like_se2(const double turning_radius = 1.5, const double lateral_penalty = 20.0) {
  return geodex::SE2<>{
      geodex::SE2LeftInvariantMetric::car_like(turning_radius, lateral_penalty),
      geodex::SE2LeftExponentialMap{}, Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
      Eigen::Vector3d(30.0, 12.0, std::numbers::pi)};
}

}  // namespace

TEST(PrecomputeMatrixLB, SE2CarLikeMeetsToIsotropicTranslationBlock) {
  // G(theta) rotates its translation eigenframe, so the meet over theta collapses
  // that block to min(wx, wy) * I while the decoupled theta axis keeps wtheta.
  constexpr double turning_radius = 1.5, lateral_penalty = 20.0;
  const double wx = 1.0, wy = lateral_penalty, wtheta = turning_radius * turning_radius;
  const auto manifold = make_car_like_se2(turning_radius, lateral_penalty);

  const auto result = ga::precompute_matrix_lower_bound(manifold);
  ASSERT_TRUE(result.converged);

  Eigen::Matrix3d expected = Eigen::Vector3d(std::min(wx, wy), std::min(wx, wy), wtheta)
                                 .asDiagonal()
                                 .toDenseMatrix();
  EXPECT_TRUE(result.M_lower.isApprox(expected, 1e-6))
      << "M_lower =\n" << result.M_lower << "\nexpected =\n" << expected;
  EXPECT_NEAR(result.lambda_min_certificate, 1.0, 1e-4);
  EXPECT_LE(result.n_outer_iters, 2);
}

TEST(PrecomputeMatrixLB, SE2BoundIsLoewnerBelowTheCoordinateMetric) {
  // The certificate must hold against the coordinate metric at every theta.
  const auto manifold = make_car_like_se2();
  ga::PrecomputeMatrixLowerBoundSettings settings;
  const auto result = ga::precompute_matrix_lower_bound(manifold, settings);
  ASSERT_TRUE(result.converged);

  // lambda_min >= 1 - tol means G >= (1 - tol) M_lower, so the residual bottoms
  // out at -tol * lambda_max(M_lower).
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> bound_solver(result.M_lower,
                                                              Eigen::EigenvaluesOnly);
  const double slack = -settings.tol * bound_solver.eigenvalues().maxCoeff();

  for (int k = 0; k < 64; ++k) {
    const double theta = -std::numbers::pi + k * (geodex::utils::two_pi / 64.0);
    const Eigen::Matrix3d G = manifold.coordinate_metric(Eigen::Vector3d(3.0, 4.0, theta));
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(G - result.M_lower,
                                                          Eigen::EigenvaluesOnly);
    EXPECT_GE(solver.eigenvalues().minCoeff(), slack) << "theta = " << theta;
  }
}

TEST(PrecomputeMatrixLB, SE2BodyMetricWouldBeInadmissible) {
  // A lateral coordinate chord at theta = pi/2 is a longitudinal body motion costing wx,
  // but the body bound charges wy.
  constexpr double turning_radius = 1.5, lateral_penalty = 20.0;
  const auto manifold = make_car_like_se2(turning_radius, lateral_penalty);
  const Eigen::Matrix3d G_at_half_pi =
      manifold.coordinate_metric(Eigen::Vector3d(0.0, 0.0, std::numbers::pi / 2.0));

  Eigen::Matrix3d body =
      Eigen::Vector3d(1.0, lateral_penalty, turning_radius * turning_radius)
          .asDiagonal()
          .toDenseMatrix();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(G_at_half_pi - body,
                                                        Eigen::EigenvaluesOnly);
  EXPECT_LT(solver.eigenvalues().minCoeff(), -1.0);
}

TEST(PrecomputeMatrixLB, SE2IsotropicMetricIsAlreadyConstantInCoordinates) {
  // wx == wy makes the rotating block isotropic, so G(theta) is constant.
  geodex::SE2<> manifold{geodex::SE2LeftInvariantMetric{2.0, 2.0, 3.0},
                         geodex::SE2LeftExponentialMap{},
                         Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                         Eigen::Vector3d(10.0, 10.0, std::numbers::pi)};

  const auto result = ga::precompute_matrix_lower_bound(manifold);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.n_outer_iters, 0);
  Eigen::Matrix3d expected = Eigen::Vector3d(2.0, 2.0, 3.0).asDiagonal().toDenseMatrix();
  EXPECT_TRUE(result.M_lower.isApprox(expected, 1e-10));
}

TEST(PrecomputeMatrixLB, SE2CoordinateMetricNeverCouplesThetaToTranslation) {
  // Per-axis wrapping is the exact deck-group minimum only while theta stays
  // decoupled. G(theta) is block diagonal, so every iterate inherits that.
  const auto manifold = make_car_like_se2();
  const auto result = ga::precompute_matrix_lower_bound(manifold);
  ASSERT_TRUE(result.converged);
  EXPECT_NEAR(result.M_lower(0, 2), 0.0, 1e-12);
  EXPECT_NEAR(result.M_lower(1, 2), 0.0, 1e-12);
}

TEST(PrecomputeMatrixLB, SE2DerivedHeuristicNeverExceedsThePlannerDistance) {
  // The heuristic must never exceed the distance the planner charges for an edge,
  // including across the theta cut.
  const auto manifold = make_car_like_se2();
  const auto result = ga::precompute_matrix_lower_bound(manifold);
  ASSERT_TRUE(result.converged);
  gh::MatrixLowerBound<> h{result.M_lower, Eigen::VectorXd(manifold.periods())};

  std::mt19937 rng(31337);
  std::uniform_real_distribution<double> xd(0.0, 30.0);
  std::uniform_real_distribution<double> yd(0.0, 12.0);
  std::uniform_real_distribution<double> td(-std::numbers::pi, std::numbers::pi);
  // A quarter of the pairs straddle the cut, where the unwrapped bound blows up.
  std::uniform_real_distribution<double> near_pi(std::numbers::pi - 0.4, std::numbers::pi);

  int violations = 0, cut_pairs = 0;
  double worst_cut_ratio = 0.0;
  for (int trial = 0; trial < 4000; ++trial) {
    const bool at_cut = (trial % 4 == 0);
    Eigen::Vector3d a(xd(rng), yd(rng), at_cut ? near_pi(rng) : td(rng));
    Eigen::Vector3d b(xd(rng), yd(rng), at_cut ? -near_pi(rng) : td(rng));
    const double d = manifold.distance(a, b);
    if (h(a, b) > d * (1.0 + 1e-9)) ++violations;
    if (at_cut) {
      ++cut_pairs;
      if (d > 1e-9) worst_cut_ratio = std::max(worst_cut_ratio, h(a, b) / d);
    }
  }
  EXPECT_EQ(violations, 0);
  EXPECT_GT(cut_pairs, 0);
  EXPECT_LE(worst_cut_ratio, 1.0);
}

TEST(PrecomputeMatrixLB, SE2UnwrappedBoundIsInadmissibleAtTheCut) {
  // Without periods the same bound overshoots at the cut.
  const auto manifold = make_car_like_se2();
  const auto result = ga::precompute_matrix_lower_bound(manifold);
  ASSERT_TRUE(result.converged);
  gh::MatrixLowerBound<> unwrapped{result.M_lower};

  constexpr double eps = 0.02;
  const Eigen::Vector3d a(10.0, 5.0, std::numbers::pi - eps);
  const Eigen::Vector3d b(10.0, 5.0, -std::numbers::pi + eps);
  EXPECT_GT(unwrapped(a, b), manifold.distance(a, b));
}

// ---------------------------------------------------------------------------
// Metric overlays on SE(2)
// ---------------------------------------------------------------------------

namespace {

// Conformal clearance overlay around a single obstacle at the world center.
auto make_clearance_cspace(const double kappa = 1.5, const double beta = 1.5) {
  auto sdf = [](const Eigen::Vector3d& q) {
    return std::hypot(q[0] - 15.0, q[1] - 6.0) - 1.0;
  };
  // Same base weights as the manifold, so the overlay is a pure conformal scale.
  geodex::SDFConformalMetric clearance{geodex::SE2LeftInvariantMetric::car_like(1.5, 20.0), sdf,
                                       kappa, beta};
  return geodex::ConfigurationSpace{make_car_like_se2(), clearance};
}

}  // namespace

TEST(PrecomputeMatrixLB, ConfigurationSpacePullsTheOverlayThroughTheBaseFrame) {
  // The frame belongs to SE(2) and the metric to the overlay, so the coordinate
  // metric is J^T (c(q) M_base) J, not the base manifold's own coordinate metric.
  const auto cs = make_clearance_cspace();
  const Eigen::Vector3d q(4.0, 3.0, std::numbers::pi / 2.0);

  const Eigen::Matrix3d J = cs.base().coordinate_jacobian(q);
  const Eigen::MatrixXd expected = J.transpose() * cs.metric().inner_matrix(
                                                      q, Eigen::MatrixXd::Identity(3, 3),
                                                      Eigen::MatrixXd::Identity(3, 3)) *
                                   J;
  EXPECT_TRUE(cs.coordinate_metric(q).isApprox(expected, 1e-12));
  // The overlay scales the base coordinate metric by c(q) > 1.
  const double c = cs.metric().conformal_factor(q);
  EXPECT_GT(c, 1.0);
  EXPECT_TRUE(cs.coordinate_metric(q).isApprox(c * cs.base().coordinate_metric(q), 1e-12));
}

TEST(PrecomputeMatrixLB, ConformalOverlayStillCertifiesAConstantBound) {
  // c(q) = 1 + kappa exp(-beta sdf) is bounded below by 1, so the overlay only
  // ever raises the base metric and a constant Loewner bound exists. It is the
  // free-space bound, loose near obstacles but admissible everywhere.
  const auto cs = make_clearance_cspace();
  const auto result = ga::precompute_matrix_lower_bound(cs);
  ASSERT_TRUE(result.converged);

  // The overlay bottoms out at c = 1 far from the obstacle, so the certified bound
  // is the free-space one, min(wx, wy) on translation, modulo the solver tolerance.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(result.M_lower, Eigen::EigenvaluesOnly);
  EXPECT_GE(solver.eigenvalues().minCoeff(), 1.0 - 1e-5);

  gh::MatrixLowerBound<> h{result.M_lower, Eigen::VectorXd(cs.periods())};
  std::mt19937 rng(555);
  std::uniform_real_distribution<double> xd(0.0, 30.0), yd(0.0, 12.0);
  std::uniform_real_distribution<double> td(-std::numbers::pi, std::numbers::pi);
  for (int trial = 0; trial < 2000; ++trial) {
    const Eigen::Vector3d a(xd(rng), yd(rng), td(rng));
    const Eigen::Vector3d b(xd(rng), yd(rng), td(rng));
    EXPECT_LE(h(a, b), cs.distance(a, b) * (1.0 + 1e-9));
  }
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

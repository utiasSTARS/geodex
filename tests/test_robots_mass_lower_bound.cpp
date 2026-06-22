// Verifies the precompiled Loewner lower bounds shipped in
// src/robots/generated/<robot>_bound.hpp against the exact generated CRBA.
//
// The core guarantee is admissibility: for every q in the joint-limit box,
// M(q) >= M_lower in the Loewner order, where M(q) is robots::MassMatrix<R>
// (the same precompiled CRBA the planner evaluates) and M_lower is
// robots::MassLowerBound<R>::matrix(). This is what makes the
// MatrixLowerBound heuristic an admissible lower bound on geodesic distance.
//
// Needs no Pinocchio — it consumes only the committed generated sources.

#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "geodex/robots/mass_lower_bound.hpp"
#include "geodex/robots/mass_matrix.hpp"

namespace {

using geodex::robots::MassLowerBound;
using geodex::robots::MassMatrix;
using geodex::robots::Robot;

// Min eigenvalue of M(q) - M_lower below this counts as a domination failure.
// A real inadmissible bound produces violations on the order of the mass-matrix
// entries themselves (1e-3 and up), far below this floor; the floor only
// absorbs eigensolver / FP noise and the precompute's finite search accuracy.
constexpr double kDominationFloor = -1e-6;

template <Robot R>
void verify_bound() {
  using LB = MassLowerBound<R>;
  using Mat = typename LB::Mat;

  const Mat M_lower = LB::matrix();

  // Structure: symmetric and SPD.
  EXPECT_LT((M_lower - M_lower.transpose()).norm(), 1e-12) << "M_lower not symmetric";
  Eigen::LLT<Mat> llt(M_lower);
  ASSERT_EQ(llt.info(), Eigen::Success) << "M_lower not SPD";

  // Provenance: the shipped bound should be a converged certificate near 1.
  EXPECT_TRUE(LB::converged) << "bound precompute did not converge";
  EXPECT_GE(LB::certificate, 0.99) << "certificate too low: " << LB::certificate;

  // Admissibility sweep: M(q) - M_lower must be PSD across the box.
  MassMatrix<R> mm;
  const auto [lo, hi] = MassMatrix<R>::joint_limits();
  std::mt19937 rng(12345);
  typename MassMatrix<R>::Vec q;

  double worst_min_eig = std::numeric_limits<double>::infinity();
  constexpr int kSamples = 500;
  for (int s = 0; s < kSamples; ++s) {
    for (int i = 0; i < LB::Nv; ++i) {
      std::uniform_real_distribution<double> dist(lo[i], hi[i]);
      q[i] = dist(rng);
    }
    const Mat Mq = mm(q);  // copy out: operator() reuses an internal buffer
    Eigen::SelfAdjointEigenSolver<Mat> es(Mq - M_lower, Eigen::EigenvaluesOnly);
    ASSERT_EQ(es.info(), Eigen::Success);
    worst_min_eig = std::min(worst_min_eig, es.eigenvalues().minCoeff());
  }
  EXPECT_GE(worst_min_eig, kDominationFloor)
      << "M(q) does not dominate M_lower over the box; worst min eigenvalue of "
         "M(q) - M_lower = "
      << worst_min_eig;
}

}  // namespace

TEST(RobotsMassLowerBound, PandaIsAdmissible) { verify_bound<Robot::Panda>(); }
TEST(RobotsMassLowerBound, Ur5IsAdmissible) { verify_bound<Robot::Ur5>(); }
TEST(RobotsMassLowerBound, FetchIsAdmissible) { verify_bound<Robot::Fetch>(); }
TEST(RobotsMassLowerBound, BaxterIsAdmissible) { verify_bound<Robot::Baxter>(); }
TEST(RobotsMassLowerBound, Pr2IsAdmissible) { verify_bound<Robot::Pr2>(); }

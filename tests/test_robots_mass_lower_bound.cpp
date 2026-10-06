// Checks the precompiled Loewner lower bounds in src/robots/generated/<robot>_bound.hpp
// against the generated CRBA at every vertex of each joint box, on its faces and inside it.
//
// A bound is admissible when M(q) >= M_lower in the Loewner order for every q in the joint
// box. M(q) is robots::MassMatrix<R>, the CRBA the planner evaluates, and M_lower is
// robots::MassLowerBound<R>::matrix().
//
// The test does not need Pinocchio. It reads only the committed generated sources.

#include <limits>
#include <random>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "geodex/robots/mass_lower_bound.hpp"
#include "geodex/robots/mass_matrix.hpp"
#include "geodex/utils/random.hpp"

namespace {

using geodex::robots::MassLowerBound;
using geodex::robots::MassMatrix;
using geodex::robots::Robot;

// The tolerance covers the rounding in evaluating M(q). Each bound has a proved margin of
// at least 0.3 percent.
constexpr double kTolerance = 1e-9;

template <Robot R>
void verify_bound() {
  using LB = MassLowerBound<R>;
  using Mat = typename LB::Mat;
  constexpr int n = LB::Nv;

  const Mat M_lower = LB::matrix();

  // M_lower is symmetric and SPD.
  EXPECT_LT((M_lower - M_lower.transpose()).norm(), 1e-12) << "M_lower not symmetric";
  Eigen::LLT<Mat> llt(M_lower);
  ASSERT_EQ(llt.info(), Eigen::Success) << "M_lower not SPD";
  const Mat Ci = Mat(llt.matrixL()).inverse();

  // The header records a proof with a certificate of at least 1.
  EXPECT_TRUE(LB::converged) << "bound was not proved";
  EXPECT_GE(LB::certificate, 1.0) << "certificate " << LB::certificate;

  // lambda_min(C^-1 M(q) C^-T) >= 1 at every vertex of the joint box, on its faces and
  // inside it.
  MassMatrix<R> mm;
  const auto [lo, hi] = MassMatrix<R>::joint_limits();
  std::mt19937_64 rng(12345);
  double worst = std::numeric_limits<double>::infinity();
  auto check = [&](const typename MassMatrix<R>::Vec& q) {
    const Mat Mq = mm(q);  // operator() reuses an internal buffer.
    Eigen::SelfAdjointEigenSolver<Mat> es(Ci * Mq * Ci.transpose(), Eigen::EigenvaluesOnly);
    ASSERT_EQ(es.info(), Eigen::Success);
    worst = std::min(worst, es.eigenvalues().minCoeff());
  };
  typename MassMatrix<R>::Vec q;
  for (long v = 0; v < (1L << n); ++v) {
    for (int i = 0; i < n; ++i) q[i] = (v >> i) & 1 ? hi[i] : lo[i];
    check(q);
  }
  for (int s = 0; s < 4000; ++s) {
    for (int i = 0; i < n; ++i) q[i] = geodex::utils::uniform_real(rng, lo[i], hi[i]);
    const auto face = static_cast<int>(geodex::utils::uniform_index(rng, n));
    q[face] = s % 2 ? hi[face] : lo[face];
    check(q);
  }
  for (int s = 0; s < 4000; ++s) {
    for (int i = 0; i < n; ++i) q[i] = geodex::utils::uniform_real(rng, lo[i], hi[i]);
    check(q);
  }
  EXPECT_GE(worst, 1.0 - kTolerance)
      << "M(q) does not dominate M_lower over the box; smallest lambda_min of "
         "M_lower^-1 M(q) = "
      << worst;
}

}  // namespace

TEST(RobotsMassLowerBound, EveryRegisteredRobotIsAdmissible) {
  for (const Robot r : geodex::robots::registered_robots()) {
    SCOPED_TRACE(std::string(geodex::robots::name(r)));
    geodex::robots::visit(r, []<Robot R>() { verify_bound<R>(); });
  }
}

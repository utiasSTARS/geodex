#include <random>
#include <string>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "geodex/integration/pinocchio/mass_matrix.hpp"

namespace {

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

std::string panda_urdf() { return GEODEX_PANDA_URDF; }

Eigen::VectorXd uniform_in_limits(std::mt19937& rng, const Eigen::VectorXd& lo,
                                  const Eigen::VectorXd& hi) {
  Eigen::VectorXd q(lo.size());
  for (int i = 0; i < lo.size(); ++i) {
    std::uniform_real_distribution<double> u(lo[i], hi[i]);
    q[i] = u(rng);
  }
  return q;
}

}  // namespace

TEST(PinocchioMassMatrix, LoadsPandaURDF) {
  geodex::integration::pinocchio::MassMatrix mass{panda_urdf()};
  EXPECT_EQ(mass.model().nq, 7);
  EXPECT_EQ(mass.model().nv, 7);
}

TEST(PinocchioMassMatrix, MassMatrixSPD) {
  geodex::integration::pinocchio::MassMatrix mass{panda_urdf()};
  const int nq = mass.model().nq;
  const auto [lo, hi] = geodex::integration::pinocchio::joint_limits(panda_urdf());

  std::mt19937 rng(42);
  for (int trial = 0; trial < 20; ++trial) {
    const Eigen::VectorXd q = uniform_in_limits(rng, lo, hi);
    const Eigen::MatrixXd& M = mass(q);
    ASSERT_EQ(M.rows(), nq);
    ASSERT_EQ(M.cols(), nq);
    const double asym = (M - M.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(asym, 1e-12) << "trial " << trial;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 1e-9) << "trial " << trial;
  }
}

TEST(PinocchioMassMatrix, MatchesDirectCRBACall) {
  geodex::integration::pinocchio::MassMatrix mass{panda_urdf()};

  ::pinocchio::Model ref_model;
  ::pinocchio::urdf::buildModel(panda_urdf(), ref_model);
  ::pinocchio::Data ref_data(ref_model);

  std::mt19937 rng(7);
  const auto [lo, hi] = geodex::integration::pinocchio::joint_limits(panda_urdf());
  for (int trial = 0; trial < 5; ++trial) {
    const Eigen::VectorXd q = uniform_in_limits(rng, lo, hi);
    const Eigen::MatrixXd M_ours = mass(q);

    ::pinocchio::crba(ref_model, ref_data, q);
    ref_data.M.triangularView<Eigen::StrictlyLower>() =
        ref_data.M.transpose().triangularView<Eigen::StrictlyLower>();

    EXPECT_LT((M_ours - ref_data.M).cwiseAbs().maxCoeff(), 1e-12)
        << "trial " << trial;
  }
}

TEST(PinocchioMassMatrix, JointLimitsNonZero) {
  const auto [lo, hi] = geodex::integration::pinocchio::joint_limits(panda_urdf());
  EXPECT_EQ(lo.size(), 7);
  EXPECT_EQ(hi.size(), 7);
  for (int i = 0; i < lo.size(); ++i) {
    EXPECT_LT(lo[i], hi[i]) << "joint " << i;
    EXPECT_GT(hi[i] - lo[i], 0.1) << "joint " << i;
  }
}

TEST(PinocchioMassMatrix, ModelNq) {
  EXPECT_EQ(geodex::integration::pinocchio::model_nq(panda_urdf()), 7);
}

TEST(PinocchioMassMatrix, MassFunctionReturnsMassMatrix) {
  auto mass = geodex::integration::pinocchio::mass_function(panda_urdf());
  EXPECT_EQ(mass.model().nq, 7);
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
  const Eigen::MatrixXd& M = mass(q);
  EXPECT_EQ(M.rows(), 7);
  EXPECT_EQ(M.cols(), 7);
}

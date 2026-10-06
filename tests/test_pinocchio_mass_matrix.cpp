#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

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

TEST(PinocchioMassMatrixFromXML, LoadsPandaURDF) {
  const auto mass = geodex::integration::pinocchio::MassMatrix::from_xml(read_file(panda_urdf()));
  EXPECT_EQ(mass.model().nq, 7);
  EXPECT_EQ(mass.model().nv, 7);
}

TEST(PinocchioMassMatrixFromXML, MatchesPathConstructor) {
  const auto from_xml =
      geodex::integration::pinocchio::MassMatrix::from_xml(read_file(panda_urdf()));
  const geodex::integration::pinocchio::MassMatrix from_path{panda_urdf()};
  const auto [lo, hi] = geodex::integration::pinocchio::joint_limits(panda_urdf());

  std::mt19937 rng(42);
  for (int trial = 0; trial < 20; ++trial) {
    const Eigen::VectorXd q = uniform_in_limits(rng, lo, hi);
    const Eigen::MatrixXd M_xml = from_xml(q);
    const Eigen::MatrixXd M_path = from_path(q);
    EXPECT_EQ((M_xml - M_path).cwiseAbs().maxCoeff(), 0.0) << "trial " << trial;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M_xml);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 1e-9) << "trial " << trial;
  }
}

TEST(PinocchioMassMatrixFromXML, ReducedModelLocksInactiveJoints) {
  const std::string xml = read_file(panda_urdf());
  const std::vector<std::string> active{"panda_joint1", "panda_joint2", "panda_joint3",
                                        "panda_joint4", "panda_joint5", "panda_joint6"};
  const auto reduced = geodex::integration::pinocchio::MassMatrix::from_xml(xml, active);
  const auto full = geodex::integration::pinocchio::MassMatrix::from_xml(xml);
  ASSERT_EQ(reduced.model().nq, 6);
  EXPECT_EQ(geodex::integration::pinocchio::model_nq_from_xml(xml), 7);
  EXPECT_EQ(geodex::integration::pinocchio::model_nq_from_xml(xml, active), 6);

  // The locked joint sits at its neutral value 0. The reduced mass matrix equals the
  // leading block of the full one at q7 = 0.
  const auto [lo, hi] = geodex::integration::pinocchio::joint_limits(panda_urdf());
  std::mt19937 rng(7);
  for (int trial = 0; trial < 10; ++trial) {
    Eigen::VectorXd q = uniform_in_limits(rng, lo, hi);
    q[6] = 0.0;
    const Eigen::MatrixXd M_full = full(q);
    const Eigen::MatrixXd M_red = reduced(q.head(6));
    EXPECT_LT((M_red - M_full.topLeftCorner(6, 6)).cwiseAbs().maxCoeff(), 1e-12)
        << "trial " << trial;
  }
}

TEST(PinocchioMassMatrixFromXML, AllJointsActiveKeepsFullModel) {
  const std::string xml = read_file(panda_urdf());
  std::vector<std::string> active;
  for (int i = 1; i <= 7; ++i) active.push_back("panda_joint" + std::to_string(i));
  EXPECT_EQ(geodex::integration::pinocchio::model_nq_from_xml(xml, active), 7);
}

#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "geodex/integration/pinocchio/jacobian.hpp"

namespace {

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

std::string panda_urdf() { return GEODEX_PANDA_URDF; }
std::string baxter_urdf() {
  return std::string(kFixturesDir) + "/pinocchio/baxter_both_arms.urdf";
}

Eigen::VectorXd random_config(std::mt19937& rng, int nq, double bound = 1.0) {
  std::uniform_real_distribution<double> u(-bound, bound);
  Eigen::VectorXd q(nq);
  for (int i = 0; i < nq; ++i) {
    q[i] = u(rng);
  }
  return q;
}

}  // namespace

TEST(FrameJacobian, PandaShape6x7) {
  auto J = geodex::integration::pinocchio::frame_jacobian(panda_urdf(), "panda_link7");
  Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
  const Eigen::MatrixXd& Jq = J(q);
  EXPECT_EQ(Jq.rows(), 6);
  EXPECT_EQ(Jq.cols(), 7);
}

TEST(FrameJacobian, PositionShape3x7) {
  auto J = geodex::integration::pinocchio::frame_position_jacobian(panda_urdf(),
                                                                   "panda_link7");
  Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
  const Eigen::MatrixXd& Jq = J(q);
  EXPECT_EQ(Jq.rows(), 3);
  EXPECT_EQ(Jq.cols(), 7);
}

TEST(FrameJacobian, StackedMultiEE) {
  auto J = geodex::integration::pinocchio::stacked_jacobian(
      baxter_urdf(), {"left_hand_link", "right_hand_link"});
  const int nv = J.model().nv;
  ASSERT_GT(nv, 0);
  std::mt19937 rng(3);
  const Eigen::VectorXd q = random_config(rng, J.model().nq, 0.5);
  const Eigen::MatrixXd& Jq = J(q);
  EXPECT_EQ(Jq.rows(), 12);
  EXPECT_EQ(Jq.cols(), nv);
  EXPECT_EQ(J.frame_ids().size(), 2u);
}

TEST(FrameJacobian, StackedPositionMultiEE) {
  auto J = geodex::integration::pinocchio::stacked_position_jacobian(
      baxter_urdf(), {"left_hand_link", "right_hand_link"});
  const int nv = J.model().nv;
  std::mt19937 rng(4);
  const Eigen::VectorXd q = random_config(rng, J.model().nq, 0.5);
  const Eigen::MatrixXd& Jq = J(q);
  EXPECT_EQ(Jq.rows(), 6);
  EXPECT_EQ(Jq.cols(), nv);
}

TEST(FrameJacobian, AutoDetectsLastBodyFrame) {
  auto J = geodex::integration::pinocchio::frame_jacobian(panda_urdf());
  ASSERT_EQ(J.frame_ids().size(), 1u);
  const auto fid = J.frame_ids()[0];
  const auto& frame = J.model().frames[fid];
  EXPECT_EQ(frame.type, ::pinocchio::BODY);
  EXPECT_EQ(frame.parentJoint,
            static_cast<::pinocchio::JointIndex>(J.model().njoints - 1));

  Eigen::VectorXd q = Eigen::VectorXd::Zero(7);
  q[0] = 0.4;
  const Eigen::MatrixXd& Jq = J(q);
  EXPECT_GT(Jq.cwiseAbs().maxCoeff(), 0.1);
}

TEST(FrameJacobian, MatchesPinocchioDirectCall) {
  auto J = geodex::integration::pinocchio::frame_jacobian(panda_urdf(), "panda_link7");

  ::pinocchio::Model ref_model;
  ::pinocchio::urdf::buildModel(panda_urdf(), ref_model);
  ::pinocchio::Data ref_data(ref_model);
  const auto fid = ref_model.getFrameId("panda_link7");

  std::mt19937 rng(11);
  for (int trial = 0; trial < 5; ++trial) {
    const Eigen::VectorXd q = random_config(rng, ref_model.nq, 1.2);
    const Eigen::MatrixXd Jq = J(q);

    ::pinocchio::computeJointJacobians(ref_model, ref_data, q);
    ::pinocchio::updateFramePlacements(ref_model, ref_data);
    Eigen::MatrixXd J_ref(6, ref_model.nv);
    J_ref.setZero();
    ::pinocchio::getFrameJacobian(ref_model, ref_data, fid,
                                  ::pinocchio::LOCAL_WORLD_ALIGNED, J_ref);

    EXPECT_LT((Jq - J_ref).cwiseAbs().maxCoeff(), 1e-12) << "trial " << trial;
  }
}

TEST(FrameJacobian, UnknownFrameNameThrows) {
  EXPECT_THROW(geodex::integration::pinocchio::frame_jacobian(panda_urdf(), "no_such_frame"),
               std::runtime_error);
}

#include <array>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "geodex/integration/pinocchio/mass_matrix.hpp"
#include "geodex/integration/pinocchio/pullback.hpp"

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

Eigen::MatrixXd stacked_frame_jacobian_full(::pinocchio::Model& model, ::pinocchio::Data& data,
                                            const std::vector<::pinocchio::FrameIndex>& fids,
                                            const Eigen::VectorXd& q) {
  ::pinocchio::computeJointJacobians(model, data, q);
  ::pinocchio::updateFramePlacements(model, data);
  Eigen::MatrixXd J(6 * static_cast<int>(fids.size()), model.nv);
  Eigen::MatrixXd J_single(6, model.nv);
  for (std::size_t k = 0; k < fids.size(); ++k) {
    J_single.setZero();
    ::pinocchio::getFrameJacobian(model, data, fids[k], ::pinocchio::LOCAL_WORLD_ALIGNED, J_single);
    J.middleRows(static_cast<int>(k) * 6, 6) = J_single;
  }
  return J;
}

Eigen::MatrixXd block_diag_task_weight(const std::array<double, 6>& per_axis, int num_frames) {
  const int task_dim = 6 * num_frames;
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(task_dim, task_dim);
  for (int f = 0; f < num_frames; ++f) {
    const int base = f * 6;
    for (int a = 0; a < 6; ++a) {
      W(base + a, base + a) = per_axis[a];
    }
  }
  return W;
}

}  // namespace

namespace pin_int = geodex::integration::pinocchio;

TEST(MakePullback, UnregularizedReturnsPullbackMetric) {
  pin_int::PullbackOptions opts{.ee_frames = {"panda_link7"}};
  auto metric = pin_int::make_pullback_metric(panda_urdf(), opts);
  static_assert(requires { metric.lambda(); },
                "Unregularized overload must return a PullbackMetric.");
  EXPECT_DOUBLE_EQ(metric.lambda(), 0.0);
}

TEST(MakePullback, IsotropicRegularizationReturnsAffineCombined) {
  pin_int::PullbackOptions opts{.ee_frames = {"panda_link7"}};
  auto metric = pin_int::make_pullback_metric(panda_urdf(), opts,
                                              pin_int::IsotropicRegularization{1e-2});
  static_assert(requires { metric.coeffs(); },
                "IsotropicRegularization overload must return an AffineCombinedMetric.");
  EXPECT_EQ(metric.coeffs()[0], 1.0);
  EXPECT_EQ(metric.coeffs()[1], 1e-2);
}

TEST(MakePullback, KineticEnergyRegularizationReturnsAffineCombined) {
  pin_int::PullbackOptions opts{.ee_frames = {"panda_link7"}};
  auto metric = pin_int::make_pullback_metric(panda_urdf(), opts,
                                              pin_int::KineticEnergyRegularization{0.1});
  static_assert(requires { metric.coeffs(); },
                "KineticEnergyRegularization overload must return an AffineCombinedMetric.");
  EXPECT_EQ(metric.coeffs()[0], 1.0);
  EXPECT_EQ(metric.coeffs()[1], 0.1);
}

TEST(MakePullback, LambdaIdentityRegSPD) {
  pin_int::PullbackOptions opts{.ee_frames = {"panda_link7"},
                                .task_weights = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0}};
  auto metric = pin_int::make_pullback_metric(panda_urdf(), opts,
                                              pin_int::IsotropicRegularization{1e-2});
  const int nq = 7;
  std::mt19937 rng(42);
  for (int trial = 0; trial < 20; ++trial) {
    const Eigen::VectorXd q = random_config(rng, nq, 1.0);
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nq, nq);
    const Eigen::MatrixXd M = metric.inner_matrix(q, I, I);
    ASSERT_EQ(M.rows(), nq);
    ASSERT_EQ(M.cols(), nq);
    const double asym = (M - M.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(asym, 1e-10) << "trial " << trial;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GT(solver.eigenvalues().minCoeff(), 1e-3) << "trial " << trial;
  }
}

TEST(MakePullback, KEBetaMatchesManualSum) {
  const double beta = 0.1;
  const std::array<double, 6> weights{1.0, 1.0, 1.0, 0.1, 0.1, 0.1};
  pin_int::PullbackOptions opts{.ee_frames = {"panda_link7"}, .task_weights = weights};
  auto metric = pin_int::make_pullback_metric(panda_urdf(), opts,
                                              pin_int::KineticEnergyRegularization{beta});

  ::pinocchio::Model model;
  ::pinocchio::urdf::buildModel(panda_urdf(), model);
  ::pinocchio::Data data(model);
  const auto fid = model.getFrameId("panda_link7");
  const Eigen::MatrixXd W = block_diag_task_weight(weights, /*num_frames=*/1);

  std::mt19937 rng(5);
  for (int trial = 0; trial < 5; ++trial) {
    const Eigen::VectorXd q = random_config(rng, model.nq, 1.0);
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(model.nq, model.nq);
    const Eigen::MatrixXd M_combined = metric.inner_matrix(q, I, I);

    const Eigen::MatrixXd J = stacked_frame_jacobian_full(model, data, {fid}, q);

    ::pinocchio::crba(model, data, q);
    data.M.triangularView<Eigen::StrictlyLower>() =
        data.M.transpose().triangularView<Eigen::StrictlyLower>();
    const Eigen::MatrixXd M_manual = J.transpose() * W * J + beta * data.M;

    EXPECT_LT((M_combined - M_manual).cwiseAbs().maxCoeff(), 1e-10) << "trial " << trial;
  }
}

TEST(MakePullback, MultiEEStacksJacobians) {
  const std::array<double, 6> weights{1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
  pin_int::PullbackOptions opts{.ee_frames = {"left_hand_link", "right_hand_link"},
                                .task_weights = weights};
  auto metric = pin_int::make_pullback_metric(baxter_urdf(), opts);

  ::pinocchio::Model model;
  ::pinocchio::urdf::buildModel(baxter_urdf(), model);
  ::pinocchio::Data data(model);
  const std::vector<::pinocchio::FrameIndex> fids = {model.getFrameId("left_hand_link"),
                                                     model.getFrameId("right_hand_link")};

  std::mt19937 rng(9);
  const Eigen::VectorXd q = random_config(rng, model.nq, 0.3);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(model.nq, model.nq);
  const Eigen::MatrixXd M_factory = metric.inner_matrix(q, I, I);

  const Eigen::MatrixXd J = stacked_frame_jacobian_full(model, data, fids, q);
  const Eigen::MatrixXd W = block_diag_task_weight(weights, /*num_frames=*/2);
  const Eigen::MatrixXd M_manual = J.transpose() * W * J;

  EXPECT_EQ(M_factory.rows(), model.nq);
  EXPECT_EQ(M_factory.cols(), model.nq);
  EXPECT_LT((M_factory - M_manual).cwiseAbs().maxCoeff(), 1e-10);
}

TEST(MakePullback, MultiEEPerAxisWeightsApplyToEachFrame) {
  // Assign different weights per axis and verify both EE blocks see them.
  const std::array<double, 6> weights{2.0, 3.0, 5.0, 0.7, 0.11, 0.13};
  pin_int::PullbackOptions opts{.ee_frames = {"left_hand_link", "right_hand_link"},
                                .task_weights = weights};
  auto metric = pin_int::make_pullback_metric(baxter_urdf(), opts);

  ::pinocchio::Model model;
  ::pinocchio::urdf::buildModel(baxter_urdf(), model);
  ::pinocchio::Data data(model);
  const std::vector<::pinocchio::FrameIndex> fids = {model.getFrameId("left_hand_link"),
                                                     model.getFrameId("right_hand_link")};

  std::mt19937 rng(17);
  const Eigen::VectorXd q = random_config(rng, model.nq, 0.3);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(model.nq, model.nq);
  const Eigen::MatrixXd M_factory = metric.inner_matrix(q, I, I);

  const Eigen::MatrixXd J = stacked_frame_jacobian_full(model, data, fids, q);
  const Eigen::MatrixXd W = block_diag_task_weight(weights, /*num_frames=*/2);
  const Eigen::MatrixXd M_manual = J.transpose() * W * J;

  EXPECT_LT((M_factory - M_manual).cwiseAbs().maxCoeff(), 1e-10);
}

TEST(MakePullback, PositionOnlyWeightsGiveRankDeficientMetric) {
  pin_int::PullbackOptions opts{.ee_frames = {"panda_link7"},
                                .task_weights = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0}};
  auto metric = pin_int::make_pullback_metric(panda_urdf(), opts);
  const int nq = 7;

  std::mt19937 rng(123);
  const Eigen::VectorXd q = random_config(rng, nq, 0.8);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nq, nq);
  const Eigen::MatrixXd M = metric.inner_matrix(q, I, I);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_LT(solver.eigenvalues().minCoeff(), 1e-9)
      << "Position-only task weights in 7 DoF should have rank <= 3 and thus be singular.";
  EXPECT_GT(solver.eigenvalues().maxCoeff(), 1e-3);
}

TEST(MakePullback, IsotropicRegularizationRejectsNonPositiveLambda) {
  pin_int::PullbackOptions opts{.ee_frames = {"panda_link7"}};
  EXPECT_THROW(pin_int::make_pullback_metric(panda_urdf(), opts,
                                             pin_int::IsotropicRegularization{0.0}),
               std::invalid_argument);
  EXPECT_THROW(pin_int::make_pullback_metric(panda_urdf(), opts,
                                             pin_int::IsotropicRegularization{-1.0}),
               std::invalid_argument);
}

TEST(MakePullback, KineticEnergyRegularizationRejectsNonPositiveBeta) {
  pin_int::PullbackOptions opts{.ee_frames = {"panda_link7"}};
  EXPECT_THROW(pin_int::make_pullback_metric(panda_urdf(), opts,
                                             pin_int::KineticEnergyRegularization{0.0}),
               std::invalid_argument);
}

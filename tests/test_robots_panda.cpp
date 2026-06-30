/// @file tests/test_robots_panda.cpp
/// @brief Parity, SPD, symmetry, and energy cross-checks for the precompiled
///        Panda CRBA against `pinocchio::crba` (the oracle).
///
/// Compiled when GEODEX_PINOCCHIO=ON (Pinocchio is needed only for the
/// parity oracle; the geodex_robots target itself is always built).

#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "geodex/robots/mass_matrix.hpp"

namespace {

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

std::string panda_urdf() { return GEODEX_PANDA_URDF; }

std::string pr2_urdf() { return GEODEX_PR2_URDF; }

using PandaMM = geodex::robots::MassMatrix<geodex::robots::Robot::Panda>;
using Ur5MM = geodex::robots::MassMatrix<geodex::robots::Robot::Ur5>;
using FetchMM = geodex::robots::MassMatrix<geodex::robots::Robot::Fetch>;
using BaxterMM = geodex::robots::MassMatrix<geodex::robots::Robot::Baxter>;
using Pr2MM = geodex::robots::MassMatrix<geodex::robots::Robot::Pr2>;
using PandaVec = PandaMM::Vec;  // Eigen::Matrix<double, 7, 1>
using Pr2Vec = Pr2MM::Vec;

template <typename Vec>
Vec uniform_in_limits(std::mt19937& rng, const Vec& lo, const Vec& hi) {
  Vec q;
  for (int i = 0; i < Vec::SizeAtCompileTime; ++i) {
    std::uniform_real_distribution<double> u(lo[i], hi[i]);
    q[i] = u(rng);
  }
  return q;
}

template <typename MM>
void expect_joint_limits_valid(const char* robot_name) {
  const auto [lo, hi] = MM::joint_limits();
  for (int i = 0; i < MM::Nq; ++i) {
    EXPECT_LT(lo[i], hi[i]) << robot_name << " joint " << i;
  }
}

class PandaRobotsFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    ::pinocchio::urdf::buildModel(panda_urdf(), pin_model_);
    pin_data_ = ::pinocchio::Data(pin_model_);
  }

  Eigen::MatrixXd pin_crba(const Eigen::VectorXd& q) {
    ::pinocchio::crba(pin_model_, pin_data_, q);
    pin_data_.M.triangularView<Eigen::StrictlyLower>() =
        pin_data_.M.transpose().triangularView<Eigen::StrictlyLower>();
    return pin_data_.M;
  }

  ::pinocchio::Model pin_model_;
  ::pinocchio::Data pin_data_{};
  PandaMM mm_{};
};

}  // namespace

TEST_F(PandaRobotsFixture, RegisteredAndConstants) {
  using geodex::robots::Robot;
  const auto names = geodex::robots::registered_robots();
  ASSERT_EQ(names.size(), 5u);
  EXPECT_EQ(names[0], Robot::Panda);
  EXPECT_EQ(names[1], Robot::Ur5);
  EXPECT_EQ(names[2], Robot::Fetch);
  EXPECT_EQ(names[3], Robot::Baxter);
  EXPECT_EQ(names[4], Robot::Pr2);
  static_assert(PandaMM::nq() == 7);
  static_assert(Ur5MM::nq() == 6);
  static_assert(FetchMM::nq() == 8);
  static_assert(BaxterMM::nq() == 14);
  static_assert(Pr2MM::nq() == 14);

  const auto [lo, hi] = PandaMM::joint_limits();
  static_assert(decltype(lo)::SizeAtCompileTime == 7);
  static_assert(decltype(hi)::SizeAtCompileTime == 7);
  expect_joint_limits_valid<PandaMM>("panda");
  expect_joint_limits_valid<Ur5MM>("ur5");
  expect_joint_limits_valid<FetchMM>("fetch");
  expect_joint_limits_valid<BaxterMM>("baxter");
  expect_joint_limits_valid<Pr2MM>("pr2");
  EXPECT_NEAR(lo[0], -2.8973, 1e-6);
  EXPECT_NEAR(hi[0], 2.8973, 1e-6);
}

// Note: the previous `UnknownRobotThrows` test is gone — invalid `Robot`
// values are now a compile-time error (no `RobotTraits<R>` specialization),
// which is strictly better than the old runtime throw.

TEST_F(PandaRobotsFixture, ParityVsPinocchioCRBA_Random) {
  const auto [lo, hi] = PandaMM::joint_limits();
  std::mt19937 rng(2024);

  double max_abs_err = 0.0;
  constexpr int kTrials = 1000;
  for (int t = 0; t < kTrials; ++t) {
    const PandaVec q = uniform_in_limits(rng, lo, hi);
    const Eigen::MatrixXd M_cg = mm_(q);
    const Eigen::MatrixXd M_pin = pin_crba(q);
    const double err = (M_cg - M_pin).cwiseAbs().maxCoeff();
    max_abs_err = std::max(max_abs_err, err);
    ASSERT_LT(err, 1e-12) << "trial " << t << " q=" << q.transpose();
  }
  std::cout << "Max |M_robots - M_pin| over " << kTrials << " random q: " << max_abs_err
            << std::endl;
}

TEST_F(PandaRobotsFixture, ParityVsPinocchioCRBA_Boundary) {
  const auto [lo, hi] = PandaMM::joint_limits();

  std::vector<std::pair<std::string, PandaVec>> cases = {
      {"zero", PandaVec::Zero()},
      {"lo", lo},
      {"hi", hi},
      {"mid", 0.5 * (lo + hi)},
  };
  for (const auto& [name, q] : cases) {
    const Eigen::MatrixXd M_cg = mm_(q);
    const Eigen::MatrixXd M_pin = pin_crba(q);
    const double err = (M_cg - M_pin).cwiseAbs().maxCoeff();
    EXPECT_LT(err, 1e-12) << "boundary case: " << name;
  }
}

TEST_F(PandaRobotsFixture, Symmetric) {
  std::mt19937 rng(7);
  const auto [lo, hi] = PandaMM::joint_limits();
  for (int t = 0; t < 50; ++t) {
    const PandaVec q = uniform_in_limits(rng, lo, hi);
    const Eigen::MatrixXd M = mm_(q);
    const double asym = (M - M.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(asym, 1e-14) << "trial " << t;
  }
}

TEST_F(PandaRobotsFixture, SPDViaCholesky) {
  std::mt19937 rng(11);
  const auto [lo, hi] = PandaMM::joint_limits();
  for (int t = 0; t < 50; ++t) {
    const PandaVec q = uniform_in_limits(rng, lo, hi);
    const Eigen::MatrixXd M = mm_(q);
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    EXPECT_EQ(llt.info(), Eigen::Success) << "trial " << t;
  }
}

TEST_F(PandaRobotsFixture, KineticEnergyCrossCheck) {
  std::mt19937 rng(13);
  const auto [lo, hi] = PandaMM::joint_limits();
  std::uniform_real_distribution<double> uv(-1.0, 1.0);
  for (int t = 0; t < 100; ++t) {
    const PandaVec q = uniform_in_limits(rng, lo, hi);
    PandaVec v;
    for (int i = 0; i < PandaMM::Nq; ++i) v[i] = uv(rng);

    const auto& M = mm_(q);  // const Matrix<double, 7, 7>&
    const double ke_ours = 0.5 * v.dot(M * v);
    const Eigen::VectorXd q_dyn = q;
    const Eigen::VectorXd v_dyn = v;
    const double ke_pin = ::pinocchio::computeKineticEnergy(pin_model_, pin_data_, q_dyn, v_dyn);
    EXPECT_NEAR(ke_ours, ke_pin, 1e-12 * std::max(1.0, std::abs(ke_pin))) << "trial " << t;
  }
}

TEST_F(PandaRobotsFixture, Determinism) {
  std::mt19937 rng(17);
  const auto [lo, hi] = PandaMM::joint_limits();
  const PandaVec q = uniform_in_limits(rng, lo, hi);
  const Eigen::MatrixXd M1 = mm_(q);
  const Eigen::MatrixXd M2 = mm_(q);
  const Eigen::MatrixXd M3 = mm_(q);
  EXPECT_TRUE(M1 == M2);
  EXPECT_TRUE(M2 == M3);
}

TEST_F(PandaRobotsFixture, ConcurrencySmoke_OneInstancePerThread) {
  const auto [lo, hi] = PandaMM::joint_limits();

  // Each thread owns its own MassMatrix instance (per the not-thread-safe
  // contract). Check that two threads can run simultaneously without crashing
  // or producing wrong results.
  auto worker = [&](int seed, std::vector<Eigen::MatrixXd>& out) {
    PandaMM mm{};
    std::mt19937 rng(seed);
    for (int t = 0; t < 50; ++t) {
      out.push_back(mm(uniform_in_limits(rng, lo, hi)));
    }
  };
  std::vector<Eigen::MatrixXd> a, b;
  std::thread ta(worker, 41, std::ref(a));
  std::thread tb(worker, 73, std::ref(b));
  ta.join();
  tb.join();
  ASSERT_EQ(a.size(), 50u);
  ASSERT_EQ(b.size(), 50u);

  // Spot-check parity against pinocchio::crba on the main thread.
  std::mt19937 rng(41);
  for (int t = 0; t < 50; ++t) {
    const Eigen::MatrixXd M_pin = pin_crba(uniform_in_limits(rng, lo, hi));
    EXPECT_LT((a[t] - M_pin).cwiseAbs().maxCoeff(), 1e-12);
  }
}

TEST(Pr2Robots, ParityVsPinocchioCRBA_Random) {
  ::pinocchio::Model pin_model;
  ::pinocchio::urdf::buildModel(pr2_urdf(), pin_model);
  ::pinocchio::Data pin_data(pin_model);
  ASSERT_EQ(pin_model.nq, Pr2MM::Nq);

  auto pin_crba = [&](const Eigen::VectorXd& q) {
    ::pinocchio::crba(pin_model, pin_data, q);
    pin_data.M.triangularView<Eigen::StrictlyLower>() =
        pin_data.M.transpose().triangularView<Eigen::StrictlyLower>();
    return pin_data.M;
  };

  Pr2MM mm{};
  const auto [lo, hi] = Pr2MM::joint_limits();
  std::mt19937 rng(2025);

  double max_abs_err = 0.0;
  constexpr int kTrials = 200;
  for (int t = 0; t < kTrials; ++t) {
    const Pr2Vec q = uniform_in_limits(rng, lo, hi);
    const Eigen::MatrixXd M_cg = mm(q);
    const Eigen::MatrixXd M_pin = pin_crba(q);
    const double err = (M_cg - M_pin).cwiseAbs().maxCoeff();
    max_abs_err = std::max(max_abs_err, err);
    ASSERT_LT(err, 1e-12) << "trial " << t << " q=" << q.transpose();
  }
  std::cout << "Max |M_pr2 - M_pin| over " << kTrials << " random q: " << max_abs_err << std::endl;
}

TEST(Pr2Robots, SymmetricAndSPD) {
  Pr2MM mm{};
  const auto [lo, hi] = Pr2MM::joint_limits();
  std::mt19937 rng(29);
  for (int t = 0; t < 50; ++t) {
    const Pr2Vec q = uniform_in_limits(rng, lo, hi);
    const Eigen::MatrixXd M = mm(q);
    EXPECT_LT((M - M.transpose()).cwiseAbs().maxCoeff(), 1e-14);
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    EXPECT_EQ(llt.info(), Eigen::Success) << "trial " << t;
  }
}

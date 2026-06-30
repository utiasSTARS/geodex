/// @file test_vamp_panda.cpp
/// @brief Tests for the Panda VAMP specialization + registry wiring.

#include <algorithm>
#include <array>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include "geodex/integration/vamp/registry.hpp"

#ifdef GEODEX_TEST_HAS_PINOCCHIO
#include "geodex/integration/pinocchio/mass_matrix.hpp"
#endif

namespace {

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

auto fixture_path(const std::string& relative) -> std::string {
  return std::string(kFixturesDir) + "/" + relative;
}

// Panda's "ready" pose (Franka default), well inside joint limits.
constexpr std::array<double, 7> kReadyPose{0.0, -0.7853981633974483, 0.0,
                                            -2.356194490192345, 0.0,
                                            1.5707963267948966,
                                            0.7853981633974483};

// A small twist of the base joint relative to the ready pose; keeps the arm
// in a similar self-collision-free configuration so motion validation has a
// clear interpolated path.
constexpr std::array<double, 7> kReadyPoseTwisted{0.5, -0.7853981633974483,
                                                  0.0, -2.356194490192345,
                                                  0.0, 1.5707963267948966,
                                                  0.7853981633974483};

auto make_panda_space_information() -> ompl::base::SpaceInformationPtr {
  auto space = std::make_shared<ompl::base::RealVectorStateSpace>(7);
  ompl::base::RealVectorBounds bounds(7);
  bounds.setLow(-3.14159);
  bounds.setHigh(3.14159);
  space->setBounds(bounds);
  return std::make_shared<ompl::base::SpaceInformation>(space);
}

auto make_panda_state(const ompl::base::SpaceInformationPtr& si,
                      const std::array<double, 7>& q)
    -> ompl::base::State* {
  auto* state = si->allocState();
  auto* rv = state->as<ompl::base::RealVectorStateSpace::StateType>();
  for (int i = 0; i < 7; ++i) rv->values[i] = q[i];
  return state;
}

}  // namespace

namespace vamp_int = geodex::integration::vamp;

TEST(VampPanda, RegistersInRegistry) {
  const auto names = vamp_int::registered_robots();
  EXPECT_NE(std::find(names.begin(), names.end(), "panda"), names.end());
}

TEST(VampPanda, MakeCheckerReturnsNonNull) {
  auto env = vamp_int::load_scene(fixture_path("vamp/panda/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("panda", env);
  ASSERT_NE(checker, nullptr);
}

TEST(VampPanda, UnknownRobotThrows) {
  auto env = vamp_int::load_scene(fixture_path("vamp/panda/empty.yaml"));
  EXPECT_THROW(vamp_int::make_vamp_checker("not-a-robot", env),
               std::runtime_error);
}

TEST(VampPanda, WrongDimReturnsFalse) {
  auto env = vamp_int::load_scene(fixture_path("vamp/panda/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("panda", env);
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 6));
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 8));
}

TEST(VampPanda, EmptySceneAcceptsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/panda/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("panda", env);
  EXPECT_TRUE(checker->is_valid(kReadyPose.data(), 7));
  EXPECT_TRUE(checker->is_valid(kReadyPoseTwisted.data(), 7));
}

TEST(VampPanda, EnclosureRejectsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/panda/enclosure.yaml"));
  auto checker = vamp_int::make_vamp_checker("panda", env);
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 7));
}

TEST(VampPanda, MotionValidatorAcceptsClearMotion) {
  auto si = make_panda_space_information();
  auto env = vamp_int::load_scene(fixture_path("vamp/panda/empty.yaml"));
  auto validator = vamp_int::make_vamp_motion_validator("panda", si, env);
  ASSERT_NE(validator, nullptr);

  auto* s1 = make_panda_state(si, kReadyPose);
  auto* s2 = make_panda_state(si, kReadyPoseTwisted);
  EXPECT_TRUE(validator->checkMotion(s1, s2));
  si->freeState(s1);
  si->freeState(s2);
}

TEST(VampPanda, MotionValidatorRejectsBlockedMotion) {
  auto si = make_panda_space_information();
  auto env = vamp_int::load_scene(fixture_path("vamp/panda/enclosure.yaml"));
  auto validator = vamp_int::make_vamp_motion_validator("panda", si, env);
  auto* s1 = make_panda_state(si, kReadyPose);
  auto* s2 = make_panda_state(si, kReadyPoseTwisted);
  EXPECT_FALSE(validator->checkMotion(s1, s2));
  si->freeState(s1);
  si->freeState(s2);
}

#ifdef GEODEX_TEST_HAS_PINOCCHIO
TEST(VampPanda, PinocchioVampLinkTogether) {
  // Sanity link test: catches Eigen-alignment ABI regressions when an AVX
  // VAMP TU and a non-AVX Pinocchio TU end up in the same binary.
  geodex::integration::pinocchio::MassMatrix mass(GEODEX_PANDA_URDF);
  Eigen::VectorXd q = Eigen::VectorXd::Zero(mass.model().nq);
  const Eigen::MatrixXd& M = mass(q);
  EXPECT_EQ(M.rows(), mass.model().nq);
  EXPECT_EQ(M.cols(), mass.model().nq);

  auto env = vamp_int::load_scene(fixture_path("vamp/panda/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("panda", env);
  EXPECT_TRUE(checker->is_valid(kReadyPose.data(), 7));
}
#endif

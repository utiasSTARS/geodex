/// @file test_vamp_ur5.cpp
/// @brief Tests for the UR5 VAMP specialization.

#include <algorithm>
#include <array>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include "geodex/integration/vamp/registry.hpp"

namespace {

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

auto fixture_path(const std::string& relative) -> std::string {
  return std::string(kFixturesDir) + "/" + relative;
}

// UR5 "ready" pose; all joints at canonical home values used in the
// ur_description package. Self-collision-free under VAMP's sphere model.
constexpr std::array<double, 6> kReadyPose{0.0, -1.5707963267948966,
                                           1.5707963267948966,
                                           -1.5707963267948966,
                                           -1.5707963267948966, 0.0};

// Slight twist of the base joint; keeps the arm in a safe configuration so
// motion validation has a clear interpolated path.
constexpr std::array<double, 6> kReadyPoseTwisted{0.5, -1.5707963267948966,
                                                  1.5707963267948966,
                                                  -1.5707963267948966,
                                                  -1.5707963267948966, 0.0};

auto make_ur5_space_information() -> ompl::base::SpaceInformationPtr {
  auto space = std::make_shared<ompl::base::RealVectorStateSpace>(6);
  ompl::base::RealVectorBounds bounds(6);
  bounds.setLow(-3.14159);
  bounds.setHigh(3.14159);
  space->setBounds(bounds);
  return std::make_shared<ompl::base::SpaceInformation>(space);
}

auto make_ur5_state(const ompl::base::SpaceInformationPtr& si,
                    const std::array<double, 6>& q)
    -> ompl::base::State* {
  auto* state = si->allocState();
  auto* rv = state->as<ompl::base::RealVectorStateSpace::StateType>();
  for (int i = 0; i < 6; ++i) rv->values[i] = q[i];
  return state;
}

}  // namespace

namespace vamp_int = geodex::integration::vamp;

TEST(VampUR5, RegistersInRegistry) {
  const auto names = vamp_int::registered_robots();
  EXPECT_NE(std::find(names.begin(), names.end(), "ur5"), names.end());
}

TEST(VampUR5, MakeCheckerReturnsNonNull) {
  auto env = vamp_int::load_scene(fixture_path("vamp/ur5/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("ur5", env);
  ASSERT_NE(checker, nullptr);
}

TEST(VampUR5, WrongDimReturnsFalse) {
  auto env = vamp_int::load_scene(fixture_path("vamp/ur5/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("ur5", env);
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 5));
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 7));
}

TEST(VampUR5, EmptySceneAcceptsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/ur5/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("ur5", env);
  EXPECT_TRUE(checker->is_valid(kReadyPose.data(), 6));
  EXPECT_TRUE(checker->is_valid(kReadyPoseTwisted.data(), 6));
}

TEST(VampUR5, EnclosureRejectsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/ur5/enclosure.yaml"));
  auto checker = vamp_int::make_vamp_checker("ur5", env);
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 6));
}

TEST(VampUR5, MotionValidatorAcceptsClearMotion) {
  auto si = make_ur5_space_information();
  auto env = vamp_int::load_scene(fixture_path("vamp/ur5/empty.yaml"));
  auto validator = vamp_int::make_vamp_motion_validator("ur5", si, env);
  ASSERT_NE(validator, nullptr);

  auto* s1 = make_ur5_state(si, kReadyPose);
  auto* s2 = make_ur5_state(si, kReadyPoseTwisted);
  EXPECT_TRUE(validator->checkMotion(s1, s2));
  si->freeState(s1);
  si->freeState(s2);
}

TEST(VampUR5, MotionValidatorRejectsBlockedMotion) {
  auto si = make_ur5_space_information();
  auto env = vamp_int::load_scene(fixture_path("vamp/ur5/enclosure.yaml"));
  auto validator = vamp_int::make_vamp_motion_validator("ur5", si, env);
  auto* s1 = make_ur5_state(si, kReadyPose);
  auto* s2 = make_ur5_state(si, kReadyPoseTwisted);
  EXPECT_FALSE(validator->checkMotion(s1, s2));
  si->freeState(s1);
  si->freeState(s2);
}

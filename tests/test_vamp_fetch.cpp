/// @file test_vamp_fetch.cpp
/// @brief Tests for the Fetch VAMP specialization.

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

// VAMP joint order for Fetch (from <vamp/robots/fetch.hh>):
//   {torso_lift, shoulder_pan, shoulder_lift, upperarm_roll,
//    elbow_flex, forearm_roll, wrist_flex, wrist_roll}.
//
// "Ready" pose: MBM table-pick start_state (arm_with_torso group),
// corresponding to waypoint[0] of path0001 in the table_pick_fetch dataset.
// Self-collision-free under VAMP's sphere model.
constexpr std::array<double, 8> kReadyPose{0.1, 1.32, 1.4, -0.2,
                                           1.72, 0.0, 1.66, 0.0};

// Adjacent waypoint[1] from the same MBM path. The planner already validated
// the straight-line motion ReadyPose → Twisted, so the motion validator has
// a guaranteed-clear pair to check in the empty scene.
constexpr std::array<double, 8> kReadyPoseTwisted{
    0.1594012679798468, 0.9316809504714426, 1.314200191054968,
   -0.8697318805118964, 1.825186400407607, -0.3731160184614637,
    1.308154341300254, -0.08474403886317583};

// Final goal of the same MBM path. Used for validity-only checks; the
// straight-line ReadyPose → Goal swing on upperarm_roll is ~2.6 rad and
// passes through self-collision (which is why the original MBM problem
// needs a planner, not interpolation).
constexpr std::array<double, 8> kGoalPose{
    0.3861498498445005, 0.7495198662964392, 1.517669523796908,
    2.447023673108444,  1.539420537298841, -1.510986423980533,
   -0.4066730485362175, -1.597305370780135};

auto make_fetch_space_information() -> ompl::base::SpaceInformationPtr {
  auto space = std::make_shared<ompl::base::RealVectorStateSpace>(8);
  ompl::base::RealVectorBounds bounds(8);
  bounds.setLow(-3.14159);
  bounds.setHigh(3.14159);
  space->setBounds(bounds);
  return std::make_shared<ompl::base::SpaceInformation>(space);
}

auto make_fetch_state(const ompl::base::SpaceInformationPtr& si,
                      const std::array<double, 8>& q)
    -> ompl::base::State* {
  auto* state = si->allocState();
  auto* rv = state->as<ompl::base::RealVectorStateSpace::StateType>();
  for (int i = 0; i < 8; ++i) rv->values[i] = q[i];
  return state;
}

}  // namespace

namespace vamp_int = geodex::integration::vamp;

TEST(VampFetch, RegistersInRegistry) {
  const auto names = vamp_int::registered_robots();
  EXPECT_NE(std::find(names.begin(), names.end(), "fetch"), names.end());
}

TEST(VampFetch, MakeCheckerReturnsNonNull) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fetch/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("fetch", env);
  ASSERT_NE(checker, nullptr);
}

TEST(VampFetch, WrongDimReturnsFalse) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fetch/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("fetch", env);
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 7));
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 9));
}

TEST(VampFetch, EmptySceneAcceptsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fetch/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("fetch", env);
  EXPECT_TRUE(checker->is_valid(kReadyPose.data(), 8));
  EXPECT_TRUE(checker->is_valid(kReadyPoseTwisted.data(), 8));
  EXPECT_TRUE(checker->is_valid(kGoalPose.data(), 8));
}

TEST(VampFetch, EnclosureRejectsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fetch/enclosure.yaml"));
  auto checker = vamp_int::make_vamp_checker("fetch", env);
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 8));
}

TEST(VampFetch, MotionValidatorAcceptsClearMotion) {
  auto si = make_fetch_space_information();
  auto env = vamp_int::load_scene(fixture_path("vamp/fetch/empty.yaml"));
  auto validator = vamp_int::make_vamp_motion_validator("fetch", si, env);
  ASSERT_NE(validator, nullptr);

  auto* s1 = make_fetch_state(si, kReadyPose);
  auto* s2 = make_fetch_state(si, kReadyPoseTwisted);
  EXPECT_TRUE(validator->checkMotion(s1, s2));
  si->freeState(s1);
  si->freeState(s2);
}

TEST(VampFetch, MotionValidatorRejectsBlockedMotion) {
  auto si = make_fetch_space_information();
  auto env = vamp_int::load_scene(fixture_path("vamp/fetch/enclosure.yaml"));
  auto validator = vamp_int::make_vamp_motion_validator("fetch", si, env);
  auto* s1 = make_fetch_state(si, kReadyPose);
  auto* s2 = make_fetch_state(si, kReadyPoseTwisted);
  EXPECT_FALSE(validator->checkMotion(s1, s2));
  si->freeState(s1);
  si->freeState(s2);
}

/// @file test_vamp_baxter.cpp
/// @brief Tests for the Baxter VAMP specialization.

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

// Baxter relaxed pose: both arms at 90-degree elbow flex with shoulders
// rotated slightly inward, well clear of the torso.
constexpr std::array<double, 14> kReadyPose{
    -0.5, -0.5, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0,
     0.5, -0.5, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0};

constexpr std::array<double, 14> kReadyPoseTwisted{
    -0.6, -0.5, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0,
     0.6, -0.5, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0};

auto make_baxter_space_information() -> ompl::base::SpaceInformationPtr {
  auto space = std::make_shared<ompl::base::RealVectorStateSpace>(14);
  ompl::base::RealVectorBounds bounds(14);
  bounds.setLow(-3.14159);
  bounds.setHigh(3.14159);
  space->setBounds(bounds);
  return std::make_shared<ompl::base::SpaceInformation>(space);
}

auto make_baxter_state(const ompl::base::SpaceInformationPtr& si,
                       const std::array<double, 14>& q)
    -> ompl::base::State* {
  auto* state = si->allocState();
  auto* rv = state->as<ompl::base::RealVectorStateSpace::StateType>();
  for (int i = 0; i < 14; ++i) rv->values[i] = q[i];
  return state;
}

}  // namespace

namespace vamp_int = geodex::integration::vamp;

TEST(VampBaxter, RegistersInRegistry) {
  const auto names = vamp_int::registered_robots();
  EXPECT_NE(std::find(names.begin(), names.end(), "baxter"), names.end());
}

TEST(VampBaxter, MakeCheckerReturnsNonNull) {
  auto env = vamp_int::load_scene(fixture_path("vamp/baxter/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("baxter", env);
  ASSERT_NE(checker, nullptr);
}

TEST(VampBaxter, WrongDimReturnsFalse) {
  auto env = vamp_int::load_scene(fixture_path("vamp/baxter/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("baxter", env);
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 7));
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 13));
}

TEST(VampBaxter, EmptySceneAcceptsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/baxter/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("baxter", env);
  EXPECT_TRUE(checker->is_valid(kReadyPose.data(), 14));
  EXPECT_TRUE(checker->is_valid(kReadyPoseTwisted.data(), 14));
}

TEST(VampBaxter, EnclosureRejectsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/baxter/enclosure.yaml"));
  auto checker = vamp_int::make_vamp_checker("baxter", env);
  EXPECT_FALSE(checker->is_valid(kReadyPose.data(), 14));
}

TEST(VampBaxter, MotionValidatorAcceptsClearMotion) {
  auto si = make_baxter_space_information();
  auto env = vamp_int::load_scene(fixture_path("vamp/baxter/empty.yaml"));
  auto validator = vamp_int::make_vamp_motion_validator("baxter", si, env);
  ASSERT_NE(validator, nullptr);

  auto* s1 = make_baxter_state(si, kReadyPose);
  auto* s2 = make_baxter_state(si, kReadyPoseTwisted);
  EXPECT_TRUE(validator->checkMotion(s1, s2));
  si->freeState(s1);
  si->freeState(s2);
}

TEST(VampBaxter, MotionValidatorRejectsBlockedMotion) {
  auto si = make_baxter_space_information();
  auto env = vamp_int::load_scene(fixture_path("vamp/baxter/enclosure.yaml"));
  auto validator = vamp_int::make_vamp_motion_validator("baxter", si, env);
  auto* s1 = make_baxter_state(si, kReadyPose);
  auto* s2 = make_baxter_state(si, kReadyPoseTwisted);
  EXPECT_FALSE(validator->checkMotion(s1, s2));
  si->freeState(s1);
  si->freeState(s2);
}

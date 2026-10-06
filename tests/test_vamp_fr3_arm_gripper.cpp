/// @file test_vamp_fr3_arm_gripper.cpp
/// @brief Tests the VAMP model of the FR3 with a Robotiq 2F-85, its joint box, batched
/// checks and attached spheres.

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "geodex/integration/vamp/registry.hpp"

namespace {

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

auto fixture_path(const std::string& relative) -> std::string {
  return std::string(kFixturesDir) + "/" + relative;
}

// Franka "ready" pose, inside the FR3 joint limits.
constexpr std::array<double, 7> kReady{0.0, -0.7853981633974483, 0.0, -2.356194490192345,
                                       0.0, 1.5707963267948966,  0.7853981633974483};
constexpr std::array<double, 7> kReadyTwisted{0.5, -0.7853981633974483, 0.0, -2.356194490192345,
                                              0.0, 1.5707963267948966,  0.7853981633974483};
// Joint 1 beyond its 2.9007 rad limit.
constexpr std::array<double, 7> kOutOfBox{3.0, -0.7853981633974483, 0.0, -2.356194490192345,
                                          0.0, 1.5707963267948966,  0.7853981633974483};

}  // namespace

namespace vamp_int = geodex::integration::vamp;

TEST(VampFr3ArmGripper, RegistersInRegistry) {
  const auto names = vamp_int::registered_robots();
  EXPECT_NE(std::find(names.begin(), names.end(), "fr3_arm_gripper"), names.end());
  EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
  EXPECT_EQ(vamp_int::robot_dimension("fr3_arm_gripper"), 7);
  EXPECT_THROW(vamp_int::robot_dimension("not-a-robot"), std::runtime_error);
}

TEST(VampFr3ArmGripper, SphereModel) {
  // 60 foam spheres, none wider than 93 mm in radius. At the ready pose the lowest sphere
  // belongs to the base and reaches 36 mm below its mounting plane.
  const auto spheres = vamp_int::robot_spheres("fr3_arm_gripper", kReady.data(), 7);
  ASSERT_EQ(spheres.size(), 60u);
  double lowest = 0.0;
  for (const auto& s : spheres) {
    EXPECT_GT(s[3], 0.0);
    EXPECT_LT(s[3], 0.093);
    lowest = std::min(lowest, s[2] - s[3]);
  }
  EXPECT_NEAR(lowest, -0.0364, 1e-3);
}

TEST(VampFr3ArmGripper, EmptySceneAcceptsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fr3_arm_gripper/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("fr3_arm_gripper", env);
  EXPECT_TRUE(checker->is_valid(kReady.data(), 7));
  EXPECT_TRUE(checker->is_valid(kReadyTwisted.data(), 7));
}

TEST(VampFr3ArmGripper, EnclosureRejectsReadyPose) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fr3_arm_gripper/enclosure.yaml"));
  auto checker = vamp_int::make_vamp_checker("fr3_arm_gripper", env);
  EXPECT_FALSE(checker->is_valid(kReady.data(), 7));
}

TEST(VampFr3ArmGripper, JointBoxRejectsOutOfLimits) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fr3_arm_gripper/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("fr3_arm_gripper", env);
  EXPECT_FALSE(checker->is_valid(kOutOfBox.data(), 7));
}

TEST(VampFr3ArmGripper, AllValidIsTheConjunction) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fr3_arm_gripper/empty.yaml"));
  auto checker = vamp_int::make_vamp_checker("fr3_arm_gripper", env);
  EXPECT_GE(checker->batch_width(), 1);

  // Eleven rows fill more than one SIMD rake and exercise the padded tail.
  std::vector<double> rows;
  for (int i = 0; i < 11; ++i) {
    const auto& q = (i % 2 == 0) ? kReady : kReadyTwisted;
    rows.insert(rows.end(), q.begin(), q.end());
  }
  EXPECT_TRUE(checker->all_valid(rows.data(), 7, 11));
  EXPECT_TRUE(checker->all_valid(rows.data(), 7, 0));

  rows.insert(rows.end(), kOutOfBox.begin(), kOutOfBox.end());
  EXPECT_FALSE(checker->all_valid(rows.data(), 7, 12));
  EXPECT_THROW(checker->all_valid(rows.data(), 6, 2), std::invalid_argument);
}

TEST(VampFr3ArmGripper, AttachedSpheresAreChecked) {
  auto env = vamp_int::load_scene(fixture_path("vamp/fr3_arm_gripper/empty.yaml"));
  // A 0.4 m sphere at the TCP overlaps the arm's own spheres.
  vamp_int::attach_spheres(env, {{0.0, 0.0, 0.0, 0.4}});
  auto checker = vamp_int::make_vamp_checker("fr3_arm_gripper", env);
  EXPECT_FALSE(checker->is_valid(kReady.data(), 7));

  // A checker keeps the attachment it was built with. A new checker sees the change.
  vamp_int::attach_spheres(env, {});
  EXPECT_FALSE(checker->is_valid(kReady.data(), 7));
  EXPECT_TRUE(vamp_int::make_vamp_checker("fr3_arm_gripper", env)->is_valid(kReady.data(), 7));
}

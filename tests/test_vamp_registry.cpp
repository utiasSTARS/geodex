/// @file test_vamp_registry.cpp
/// @brief The VAMP registry names every model's joints, in order, and its attachment frame.

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "geodex/integration/vamp/registry.hpp"

namespace gv = geodex::integration::vamp;

TEST(VampRegistry, JointNamesMatchTheDimensionAndAreDistinct) {
  for (const auto& name : gv::registered_robots()) {
    const auto joints = gv::robot_joint_names(name);
    EXPECT_EQ(static_cast<int>(joints.size()), gv::robot_dimension(name)) << name;
    EXPECT_EQ(std::set<std::string>(joints.begin(), joints.end()).size(), joints.size()) << name;
    EXPECT_FALSE(gv::robot_end_effector(name).empty()) << name;
  }
}

TEST(VampRegistry, KnownModels) {
  EXPECT_EQ(gv::robot_joint_names("panda"),
            (std::vector<std::string>{"panda_joint1", "panda_joint2", "panda_joint3",
                                      "panda_joint4", "panda_joint5", "panda_joint6",
                                      "panda_joint7"}));
  EXPECT_EQ(gv::robot_end_effector("panda"), "panda_grasptarget");
  EXPECT_EQ(gv::robot_joint_names("fr3_arm_gripper"),
            (std::vector<std::string>{"fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
                                      "fr3_joint5", "fr3_joint6", "fr3_joint7"}));
  EXPECT_EQ(gv::robot_end_effector("fr3_arm_gripper"), "2f85_tcp");
  const auto stretch = gv::robot_joint_names("stretch3");
  ASSERT_EQ(stretch.size(), 8u);
  EXPECT_EQ(stretch[0], "base_x_joint");
  EXPECT_EQ(stretch[1], "base_y_joint");
  EXPECT_EQ(stretch[2], "base_theta_joint");
  EXPECT_EQ(stretch[4], "joint_arm");
  EXPECT_EQ(gv::robot_joint_names("baxter").front(), "left_s0");
  EXPECT_EQ(gv::robot_joint_names("baxter").back(), "right_w2");
}

TEST(VampRegistry, UnknownRobotThrows) {
  EXPECT_THROW(gv::robot_joint_names("no_such_robot"), std::runtime_error);
  EXPECT_THROW(gv::robot_end_effector("no_such_robot"), std::runtime_error);
}

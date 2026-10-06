/// @file test_robots_registry.cpp
/// @brief Names, lookup and compile-time dispatch of the built-in robot registry.

#include <set>
#include <string>

#include <gtest/gtest.h>

#include "geodex/robots/mass_matrix.hpp"

namespace gr = geodex::robots;

TEST(RobotsRegistry, NamesAreUniqueAndRoundTrip) {
  std::set<std::string> seen;
  for (const gr::Robot r : gr::registered_robots()) {
    const std::string n(gr::name(r));
    EXPECT_FALSE(n.empty());
    EXPECT_TRUE(seen.insert(n).second) << "duplicate name " << n;
    const auto back = gr::robot_from_name(n);
    ASSERT_TRUE(back.has_value()) << n;
    EXPECT_EQ(*back, r) << n;
  }
}

TEST(RobotsRegistry, UnknownNamesAreRejected) {
  EXPECT_FALSE(gr::robot_from_name("").has_value());
  EXPECT_FALSE(gr::robot_from_name("not-a-robot").has_value());
  EXPECT_FALSE(gr::robot_from_name("fr3_gripper").has_value());
}

TEST(RobotsRegistry, VisitDispatchesToTheMatchingRobot) {
  for (const gr::Robot r : gr::registered_robots()) {
    const int nq = gr::visit(r, []<gr::Robot R>() { return gr::MassMatrix<R>::Nq; });
    const bool same = gr::visit(r, [r]<gr::Robot R>() { return R == r; });
    EXPECT_TRUE(same) << gr::name(r);
    EXPECT_GT(nq, 0) << gr::name(r);
  }
}

TEST(RobotsRegistry, Fr3ArmGripper) {
  const auto r = gr::robot_from_name("fr3_arm_gripper");
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, gr::Robot::Fr3Gripper);
  EXPECT_EQ(gr::MassMatrix<gr::Robot::Fr3Gripper>::Nq, 7);
}

TEST(RobotsRegistry, Fr3ArmGripperMassMatrixMatchesPinocchio) {
  // Pinocchio's CRBA of data/robots/fr3_gripper/fr3_arm_gripper_dynamics.urdf at the Franka
  // ready pose. The model holds the FR3 of franka_description 2.9.0 and the Robotiq 2F-85
  // on its coupling, with the fingers 40 mm apart.
  const gr::MassMatrix<gr::Robot::Fr3Gripper> mass;
  gr::MassMatrix<gr::Robot::Fr3Gripper>::Vec q;
  q << 0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966,
      0.7853981633974483;
  const auto& M = mass(q);
  EXPECT_NEAR(M(0, 0), 0.587072849706287, 1e-12);
  EXPECT_NEAR(M(1, 3), -0.7636729416010635, 1e-12);
  EXPECT_NEAR(M(6, 6), 0.000991798201424664, 1e-12);
}

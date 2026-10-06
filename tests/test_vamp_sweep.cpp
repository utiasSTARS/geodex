/// @file test_vamp_sweep.cpp
/// @brief The precomputed sphere-travel bounds hold against each kernel's own forward
/// kinematics, and every kernel's checker enforces its joint limits.

#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/integration/vamp/registry.hpp"
#include "geodex/integration/vamp/robots/generated/baxter_sweep.hh"
#include "geodex/integration/vamp/robots/generated/fr3_arm_gripper_sweep.hh"
#include "geodex/integration/vamp/robots/generated/husky_ur5e_sweep.hh"
#include "geodex/integration/vamp/robots/generated/panda_sweep.hh"
#include "geodex/integration/vamp/robots/generated/pr2_sweep.hh"
#include "geodex/integration/vamp/robots/generated/ridgeback_ur5e_sweep.hh"
#include "geodex/integration/vamp/robots/generated/stretch3_sweep.hh"
#include "geodex/integration/vamp/robots/generated/stretch4_sweep.hh"
#include "geodex/integration/vamp/robots/generated/ur5_sweep.hh"
#include "geodex/manifold/se2.hpp"
#include "geodex/utils/random.hpp"

namespace {

namespace gv = geodex::integration::vamp;
namespace gen = geodex::integration::vamp::detail::generated;

/// A kernel's sweep data with the dimension erased.
struct Sweep {
  bool planar_base;
  double base_reach, ee_base_reach;
  std::vector<double> reach, ee_reach, ee_rotation, lower, upper;
};

template <std::size_t N>
Sweep erase(const geodex::integration::vamp::detail::SweepModel<N>& s) {
  return {s.planar_base,
          s.base_reach,
          s.ee_base_reach,
          {s.reach.begin(), s.reach.end()},
          {s.ee_reach.begin(), s.ee_reach.end()},
          {s.ee_rotation.begin(), s.ee_rotation.end()},
          {s.lower.begin(), s.lower.end()},
          {s.upper.begin(), s.upper.end()}};
}

Sweep sweep_of(const std::string& name) {
  if (name == "baxter") return erase(gen::baxter_sweep);
  if (name == "fr3_arm_gripper") return erase(gen::fr3_arm_gripper_sweep);
  if (name == "husky_ur5e") return erase(gen::husky_ur5e_sweep);
  if (name == "panda") return erase(gen::panda_sweep);
  if (name == "pr2") return erase(gen::pr2_sweep);
  if (name == "ridgeback_ur5e") return erase(gen::ridgeback_ur5e_sweep);
  if (name == "stretch3") return erase(gen::stretch3_sweep);
  if (name == "stretch4") return erase(gen::stretch4_sweep);
  if (name == "ur5") return erase(gen::ur5_sweep);
  throw std::runtime_error("no sweep data for " + name);
}

/// A configuration inside the limits, with the base inside a 4 m square.
Eigen::VectorXd random_q(const Sweep& s, std::mt19937_64& rng) {
  const int n = static_cast<int>(s.lower.size());
  Eigen::VectorXd q(n);
  for (int j = 0; j < n; ++j) {
    double lo = s.lower[j], hi = s.upper[j];
    if (s.planar_base && j < 2) lo = -2.0, hi = 2.0;
    q[j] = geodex::utils::uniform_real(rng, lo, hi);
  }
  return q;
}

}  // namespace

TEST(VampSweep, TravelBoundsHoldAlongEdges) {
  std::mt19937_64 rng(5);
  for (const auto& name : gv::registered_robots()) {
    SCOPED_TRACE(name);
    const Sweep s = sweep_of(name);
    const int n = static_cast<int>(s.lower.size());
    ASSERT_EQ(n, gv::robot_dimension(name));
    double worst = 0.0;
    for (int edge = 0; edge < 40; ++edge) {
      const Eigen::VectorXd a = random_q(s, rng), b = random_q(s, rng);
      Eigen::Vector3d twist = Eigen::Vector3d::Zero();
      double bound = 0.0;
      int first = 0;
      if (s.planar_base) {
        twist = geodex::SE2LeftExponentialMap{}.inverse_retract(a.head<3>(), b.head<3>());
        bound += std::hypot(twist[0], twist[1]) + std::abs(twist[2]) * s.base_reach;
        first = 3;
      }
      for (int j = first; j < n; ++j) bound += s.reach[j] * std::abs(b[j] - a[j]);
      auto at = [&](double t) {
        Eigen::VectorXd q = a + t * (b - a);
        if (s.planar_base) q.head<3>() = geodex::SE2LeftExponentialMap{}.retract(a.head<3>(), t * twist);
        return gv::robot_spheres(name, q.data(), n);
      };
      constexpr int kSteps = 400;
      auto prev = at(0.0);
      for (int k = 1; k <= kSteps; ++k) {
        const auto cur = at(static_cast<double>(k) / kSteps);
        for (std::size_t i = 0; i < cur.size(); ++i) {
          const double moved = std::hypot(cur[i][0] - prev[i][0], cur[i][1] - prev[i][1],
                                          cur[i][2] - prev[i][2]);
          // Travel per unit of the edge parameter. Float kinematics adds a few micrometers.
          worst = std::max(worst, (moved - 2e-5) * kSteps / bound);
        }
        prev = cur;
      }
    }
    EXPECT_LE(worst, 1.0);
  }
}

TEST(VampSweep, SphereTravelIsTheEdgeBound) {
  // make_sphere_travel gives the bound that TravelBoundsHoldAlongEdges checks, from the edge's
  // tangent, and sphere_speed is the norm of its coefficients. The library's build may fuse
  // a multiply and an add, and the values agree to a few units in the last place.
  const auto env = gv::build_scene(gv::make_scene_builder());
  std::mt19937_64 rng(7);
  for (const auto& name : gv::registered_robots()) {
    SCOPED_TRACE(name);
    const Sweep s = sweep_of(name);
    const int n = static_cast<int>(s.lower.size());
    const int first = s.planar_base ? 3 : 0;
    double sum = s.planar_base ? 1.0 + s.base_reach * s.base_reach : 0.0;
    for (int j = first; j < n; ++j) sum += s.reach[j] * s.reach[j];
    EXPECT_DOUBLE_EQ(gv::sphere_speed(name, env), std::sqrt(sum));
    const gv::SphereTravel travel = gv::make_sphere_travel(name, env);
    for (int edge = 0; edge < 20; ++edge) {
      const Eigen::VectorXd a = random_q(s, rng), b = random_q(s, rng);
      Eigen::VectorXd tangent = b - a;
      double bound = 0.0;
      if (s.planar_base) {
        tangent.head<3>() =
            geodex::SE2LeftExponentialMap{}.inverse_retract(a.head<3>(), b.head<3>());
        bound += std::hypot(tangent[0], tangent[1]) + std::abs(tangent[2]) * s.base_reach;
      }
      for (int j = first; j < n; ++j) bound += s.reach[j] * std::abs(tangent[j]);
      EXPECT_DOUBLE_EQ(travel(tangent), bound);
    }
    EXPECT_THROW((void)travel(Eigen::VectorXd::Zero(n + 1)), std::invalid_argument);
  }
}

TEST(VampSweep, SphereTravelFoldsInAnAttachedBody) {
  // A body held 0.13 m from the end-effector origin moves at most `ee_reach + 0.13 ee_rotation`
  // per unit of each joint, and `ee_base_reach + 0.13` per unit of the base's turn.
  std::mt19937_64 rng(11);
  for (const auto& name : gv::registered_robots()) {
    SCOPED_TRACE(name);
    auto env = gv::build_scene(gv::make_scene_builder());
    gv::attach_spheres(env, {{0.0, 0.05, 0.12, 0.02}, {0.0, 0.0, 0.05, 0.03}});
    const double a = std::sqrt(0.05 * 0.05 + 0.12 * 0.12);
    const Sweep s = sweep_of(name);
    const int n = static_cast<int>(s.lower.size());
    const int first = s.planar_base ? 3 : 0;
    const gv::SphereTravel travel = gv::make_sphere_travel(name, env);
    for (int edge = 0; edge < 20; ++edge) {
      const Eigen::VectorXd p = random_q(s, rng), q = random_q(s, rng);
      Eigen::VectorXd tangent = q - p;
      double bound = 0.0;
      if (s.planar_base) {
        tangent.head<3>() =
            geodex::SE2LeftExponentialMap{}.inverse_retract(p.head<3>(), q.head<3>());
        bound += std::hypot(tangent[0], tangent[1]) +
                 std::abs(tangent[2]) * std::max(s.base_reach, s.ee_base_reach + a);
      }
      for (int j = first; j < n; ++j) {
        bound += std::max(s.reach[j], s.ee_reach[j] + a * s.ee_rotation[j]) * std::abs(tangent[j]);
      }
      // The attached spheres are stored in single precision.
      EXPECT_NEAR(travel(tangent), bound, 1e-6 * bound);
    }
  }
}

TEST(VampSweep, CheckersEnforceTheJointLimits) {
  const auto env = gv::build_scene(gv::make_scene_builder());
  for (const auto& name : gv::registered_robots()) {
    SCOPED_TRACE(name);
    const Sweep s = sweep_of(name);
    const int n = static_cast<int>(s.lower.size());
    const auto checker = gv::make_vamp_checker(name, env);
    for (int j = 0; j < n; ++j) {
      Eigen::VectorXd q(n);
      for (int i = 0; i < n; ++i) q[i] = 0.5 * (s.lower[i] + s.upper[i]);
      if (s.planar_base) q.head<3>().setZero();
      q[j] = s.upper[j] + 0.3;
      EXPECT_FALSE(checker->is_valid(q.data(), n)) << "joint " << j << " above its limit";
      q[j] = s.lower[j] - 0.3;
      EXPECT_FALSE(checker->is_valid(q.data(), n)) << "joint " << j << " below its limit";
    }
  }
}

TEST(VampSweep, Fr3ArmGripperLimitsAndToolCentrePoint) {
  // The joint limits of franka_description 2.9.0 for the FR3.
  const auto& s = gen::fr3_arm_gripper_sweep;
  const std::array<double, 7> lower{-2.9007, -1.8361, -2.9007, -3.077, -2.8763, 0.4398, -3.0508};
  const std::array<double, 7> upper{2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508};
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(s.lower[i], lower[i]) << "joint " << i + 1;
    EXPECT_EQ(s.upper[i], upper[i]) << "joint " << i + 1;
  }
  // The TCP lies on the joint 7 axis, 0.107 m to the flange, 0.011 m through the coupling and
  // 0.142 m into the gripper, so it moves 0.26 m per radian of joint 7 at most.
  EXPECT_NEAR(s.ee_reach[6], 0.107 + 0.011 + 0.142, 1e-12);
  EXPECT_FALSE(s.planar_base);
}

TEST(VampSweep, RobotSpheresRejectsTheWrongDimension) {
  const std::vector<double> q(3, 0.0);
  EXPECT_THROW(gv::robot_spheres("panda", q.data(), 3), std::invalid_argument);
  EXPECT_THROW(gv::robot_spheres("no_such_robot", q.data(), 3), std::runtime_error);
}

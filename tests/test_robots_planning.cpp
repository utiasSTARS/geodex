/// @file test_robots_planning.cpp
/// @brief One whole-body `robots::plan` per mobile robot through a scene with a wall
/// between start and goal, rechecked densely along the returned path.

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/integration/vamp/registry.hpp"
#include "geodex/robots/planning.hpp"

namespace gr = geodex::robots;
namespace gv = geodex::integration::vamp;
using geodex::SE2LeftInvariantMetric;
using gr::Robot;

namespace {

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

Eigen::VectorXd vec(std::initializer_list<double> v) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(v.size()));
  int i = 0;
  for (const double x : v) out[i++] = x;
  return out;
}

/// Deepest reach of a collision sphere of robot @p name at @p q into the box of wall.yaml,
/// centered at (0, 0, 0.6) with half extents (0.15, 0.8, 0.6), and 0 outside it.
double depth_in_wall(const std::string& name, const Eigen::VectorXd& q) {
  const Eigen::Vector3d center(0.0, 0.0, 0.6);
  const Eigen::Vector3d half(0.15, 0.8, 0.6);
  double depth = 0.0;
  for (const auto& s : gv::robot_spheres(name, q.data(), static_cast<int>(q.size()))) {
    const Eigen::Vector3d d = (Eigen::Vector3d(s[0], s[1], s[2]) - center).cwiseAbs() - half;
    const double signed_distance = d.cwiseMax(0.0).norm() + std::min(d.maxCoeff(), 0.0);
    depth = std::max(depth, s[3] - signed_distance);
  }
  return depth;
}

/// Checks that the waypoints pass the scene, that no state along the path reaches into the
/// wall by more than half the smoother's 5 mm check travel, and that the base stays in its
/// region.
///
/// Each edge is rechecked along its SE(2) x R^n geodesic every 0.25 mm of sphere travel.
/// Between two of the smoother's checks, a sphere moves at most 5 mm, and it comes within
/// 2.5 mm of a checked position.
template <Robot R>
void expect_valid_path(const std::vector<Eigen::VectorXd>& path, const gv::EnvHandle& env,
                       const Eigen::Vector2d& lo, const Eigen::Vector2d& hi) {
  const std::string name(gr::name(R));
  const auto in_scene = gv::make_vamp_checker(name, env);
  const gv::SphereTravel travel = gv::make_sphere_travel(name, env);
  const geodex::SE2<> base(Eigen::Vector3d(lo.x(), lo.y(), -std::numbers::pi),
                           Eigen::Vector3d(hi.x(), hi.y(), std::numbers::pi));
  const auto space = geodex::make_product(base, gr::joint_space<R, gr::ArmMetric::Euclidean>());
  for (const auto& q : path) EXPECT_TRUE(in_scene->is_valid(q.data(), static_cast<int>(q.size())));
  double depth = 0.0;
  int outside = 0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const double sweep = travel(space.log(path[i], path[i + 1]));
    const int m = std::max(1, static_cast<int>(std::ceil(sweep / 2.5e-4)));
    for (int k = 0; k <= m; ++k) {
      const Eigen::VectorXd q = space.geodesic(path[i], path[i + 1], static_cast<double>(k) / m);
      depth = std::max(depth, depth_in_wall(name, q));
      outside += (q.head<2>().array() < lo.array() - 1e-9).any() ||
                 (q.head<2>().array() > hi.array() + 1e-9).any();
    }
  }
  EXPECT_LT(depth, 0.0025);
  EXPECT_EQ(outside, 0);
}

template <Robot R>
void plan_through_wall(const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
                       const gr::RobotPlanOptions& options) {
  const auto env = gv::load_scene(std::string(kFixturesDir) + "/vamp/mobile/wall.yaml");
  geodex::planning::PlanSettings settings;
  settings.time = 10.0;
  settings.seed = 7;
  const auto result = gr::plan<R>(start, goal, env, settings, options);
  ASSERT_TRUE(result.solved) << gr::name(R);
  ASSERT_GE(result.path.size(), 2u);
  EXPECT_LT((result.path.front() - start).norm(), 1e-9);
  EXPECT_LT((result.path.back() - goal).norm(), 1e-9);
  EXPECT_TRUE(std::isfinite(result.cost));
  EXPECT_GE(result.first_solution_ms, 0.0);
  EXPECT_LE(result.first_solution_ms, result.time_ms);
  EXPECT_GE(result.first_solution_iterations, 1u);
  EXPECT_GT(result.informed_samples + result.focused_samples + result.uniform_samples, 0u);
  expect_valid_path<R>(result.path, env, options.base_region->first,
                       options.base_region->second);
}

}  // namespace

// The plan sees the shelf grown by half the smoother's 5 mm check travel. Every state along
// the returned path, rechecked every 0.25 mm of sphere travel, then clears the shelf itself.
TEST(RobotsPlanning, FixedBaseArmPathClearsTheSceneBetweenChecks) {
  const auto env = gv::load_scene(std::string(kFixturesDir) + "/smoothing/shelf_post.scene.yaml");
  const auto grown = gv::pad_scene(env, 0.5 * gv::kDefaultMaxSphereStep);
  const Eigen::VectorXd start = vec({0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785});
  const Eigen::VectorXd goal = vec({1.2, -0.4, -0.3, -1.9, 0.2, 1.9, 0.5});
  const auto checker = gv::make_vamp_checker("panda", env);
  const gv::SphereTravel travel = gv::make_sphere_travel("panda", env);
  for (const std::uint64_t seed : {1u, 2u, 3u}) {
    geodex::planning::PlanSettings settings;
    settings.iterations = 1000;
    settings.seed = seed;
    const auto r = gr::plan<Robot::Panda>(start, goal, grown, settings);
    ASSERT_TRUE(r.solved) << "seed " << seed;
    int bad = 0;
    for (std::size_t i = 0; i + 1 < r.path.size(); ++i) {
      const Eigen::VectorXd step = r.path[i + 1] - r.path[i];
      const int m = std::max(1, static_cast<int>(std::ceil(travel(step) / 2.5e-4)));
      for (int k = 0; k <= m; ++k) {
        const Eigen::VectorXd q = r.path[i] + (static_cast<double>(k) / m) * step;
        if (!checker->is_valid(q.data(), 7)) ++bad;
      }
    }
    EXPECT_EQ(bad, 0) << "seed " << seed;
  }
}

namespace {

gr::RobotPlanOptions options(const SE2LeftInvariantMetric& base, gr::ArmMetric metric) {
  gr::RobotPlanOptions o;
  o.base = base;
  o.metric = metric;
  o.base_region = {Eigen::Vector2d(-3, -3), Eigen::Vector2d(3, 3)};
  return o;
}

}  // namespace

TEST(RobotsPlan, Stretch3DifferentialDriveKineticEnergy) {
  plan_through_wall<Robot::Stretch3>(
      vec({-1.5, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0}),
      vec({1.5, 0.5, 1.57, 0.9, 0.3, 1.0, -0.3, 0.0}),
      options(SE2LeftInvariantMetric::differential_drive(), gr::ArmMetric::KineticEnergy));
}

TEST(RobotsPlan, Stretch4HolonomicEuclidean) {
  plan_through_wall<Robot::Stretch4>(
      vec({-1.5, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0}),
      vec({1.5, 0.5, -1.57, 0.9, 0.3, 1.0, 0.3, 0.0}),
      options(SE2LeftInvariantMetric::holonomic(), gr::ArmMetric::Euclidean));
}

TEST(RobotsPlan, RidgebackUr5eHolonomicKineticEnergy) {
  plan_through_wall<Robot::RidgebackUr5e>(
      vec({-1.5, 0.0, 0.0, 0.0, -1.57, 1.57, -1.57, -1.57, 0.0}),
      vec({1.5, 0.5, 3.0, 1.0, -0.8, 1.2, -1.9, -1.57, 0.3}),
      options(SE2LeftInvariantMetric::holonomic(), gr::ArmMetric::KineticEnergy));
}

TEST(RobotsPlan, HuskyUr5eDifferentialDriveKineticEnergy) {
  plan_through_wall<Robot::HuskyUr5e>(
      vec({-1.5, 0.0, 0.0, 0.0, -1.57, 1.57, -1.57, -1.57, 0.0}),
      vec({1.5, 0.5, 3.0, 1.0, -0.8, 1.2, -1.9, -1.57, 0.3}),
      options(SE2LeftInvariantMetric::differential_drive(), gr::ArmMetric::KineticEnergy));
}

TEST(RobotsPlan, FixedBaseArmByRuntimeName) {
  const auto env = gv::load_scene(std::string(kFixturesDir) + "/vamp/panda/empty.yaml");
  geodex::planning::PlanSettings settings;
  settings.time = 2.0;
  settings.seed = 7;
  const auto robot = gr::robot_from_name("panda");
  ASSERT_TRUE(robot.has_value());
  const auto result = gr::plan(*robot, vec({0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785}),
                               vec({0.5, -0.4, 0.3, -2.0, 0.2, 1.8, 0.5}), env, settings);
  ASSERT_TRUE(result.solved);
  const auto checker = gv::make_vamp_checker("panda", env);
  for (const auto& q : result.path) EXPECT_TRUE(checker->is_valid(q.data(), 7));
}

TEST(RobotsPlan, RejectsConfigurationsOfTheWrongSize) {
  const auto env = gv::load_scene(std::string(kFixturesDir) + "/vamp/panda/empty.yaml");
  const Eigen::VectorXd arm = Eigen::VectorXd::Zero(7);
  EXPECT_THROW(gr::plan<Robot::Panda>(Eigen::VectorXd::Zero(3), arm, env), std::invalid_argument);
  EXPECT_THROW(gr::plan<Robot::Panda>(arm, Eigen::VectorXd::Zero(8), env), std::invalid_argument);
  const Eigen::VectorXd whole_body = Eigen::VectorXd::Zero(8);
  EXPECT_THROW(gr::plan<Robot::Stretch3>(Eigen::VectorXd::Zero(1), whole_body, env),
               std::invalid_argument);
  EXPECT_THROW(gr::plan<Robot::Stretch3>(whole_body, Eigen::VectorXd::Zero(5), env),
               std::invalid_argument);
  EXPECT_THROW(gr::plan(Robot::Ur5, Eigen::VectorXd::Zero(7), Eigen::VectorXd::Zero(6), env),
               std::invalid_argument);
}

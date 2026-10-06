// A Franka FR3 moves a box between two bays of a shelf, the C++ version of
// arm_ke.py.
//
// The program plans the move with the arm's kinetic-energy metric, measures the
// path, then plans the same move with the Euclidean metric on the joint angles and
// measures it too.
//
// Usage: arm_ke [--json out.json]

// [docs-start:load]
#include <cmath>
#include <cstdio>

#include <array>
#include <utility>
#include <vector>

#include <geodex/integration/vamp/registry.hpp>
#include <geodex/robots/planning.hpp>
// [docs-end:load]

#include "../../common/json_output.hpp"

// [docs-start:load]
namespace gr = geodex::robots;
namespace gv = geodex::integration::vamp;
namespace gp = geodex::planning;

/// An IKEA BILLY shelf in the arm's base frame, its seven boards (two sides, top,
/// base, two shelves, back) as center and size in meters.
const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> kShelf = {
    {{0.793492, 0.618718, 0.53}, {0.018, 0.28, 1.06}},
    {{0.011492, 0.618718, 0.53}, {0.018, 0.28, 1.06}},
    {{0.402492, 0.618718, 1.051}, {0.8, 0.28, 0.018}},
    {{0.402492, 0.618718, 0.04}, {0.764, 0.28, 0.08}},
    {{0.402492, 0.608718, 0.3977}, {0.764, 0.26, 0.018}},
    {{0.402492, 0.608718, 0.7243}, {0.764, 0.26, 0.018}},
    {{0.402492, 0.756218, 0.53}, {0.78, 0.005, 1.06}},
};

/// The held 0.04 x 0.24 x 0.16 m box as 112 spheres (x, y, z, radius) that
/// contain it and reach at most 1 cm outside it. The rows give the spheres with
/// x, y, z >= 0 in the box's frame, and their mirror images give the rest.
const std::vector<std::array<double, 4>> kOctant = {
    {0.012, 0.112, 0.072, 0.018}, {0.01, 0.0, 0.07, 0.02},
    {0.01, 0.052, 0.07, 0.02},    {0.01, 0.072, 0.07, 0.02},
    {0.01, 0.092, 0.07, 0.02},    {0.01, 0.11, 0.032, 0.02},
    {0.01, 0.11, 0.052, 0.02},    {0.008, 0.018, 0.068, 0.022},
    {0.008, 0.034, 0.068, 0.022}, {0.008, 0.108, 0.0, 0.022},
    {0.008, 0.108, 0.014, 0.022}, {0.0, 0.106, 0.066, 0.024},
    {0.0, 0.0, 0.014, 0.03},      {0.0, 0.0, 0.046, 0.03},
    {0.0, 0.026, 0.0, 0.03},      {0.0, 0.026, 0.036, 0.03},
    {0.0, 0.05, 0.046, 0.03},     {0.0, 0.052, 0.014, 0.03},
    {0.0, 0.08, 0.0, 0.03},       {0.0, 0.08, 0.026, 0.03},
    {0.0, 0.082, 0.046, 0.03},
};

/// The shelf as a collision scene, with the box attached to the gripper's TCP
/// frame. The gripper holds the box between its finger pads, its 40 mm side along
/// the x axis of the TCP frame and its center 0.055 m along the z axis.
gv::EnvHandle shelf_with_box() {
  auto builder = gv::make_scene_builder();
  for (const auto& [center, size] : kShelf) {
    gv::scene_add_box(builder, center, size, Eigen::Quaterniond::Identity());
  }
  auto env = gv::build_scene(builder);
  std::vector<std::array<double, 4>> box;
  for (const auto& [x, y, z, r] : kOctant) {
    for (double sx : {1.0, -1.0}) {
      for (double sy : {1.0, -1.0}) {
        for (double sz : {1.0, -1.0}) {
          if ((sx < 0 && x == 0.0) || (sy < 0 && y == 0.0) || (sz < 0 && z == 0.0)) {
            continue;
          }
          box.push_back({sx * x, sy * y, 0.055 + sz * z, r});
        }
      }
    }
  }
  gv::attach_spheres(env, box);
  return env;
}
// [docs-end:load]

// [docs-start:measure]
/// Joint-space and kinetic-energy length of a path with straight edges in joint
/// coordinates. The kinetic-energy length uses the midpoint rule.
std::pair<double, double> measure(const std::vector<Eigen::VectorXd>& path,
                                  int pieces = 16) {
  const gr::MassMatrix<gr::Robot::Fr3Gripper> mass;  // the arm's inertia M(q)
  double joint = 0.0, energy = 0.0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const Eigen::VectorXd d = path[i + 1] - path[i];
    joint += std::sqrt(d.dot(d));
    for (int k = 0; k < pieces; ++k) {
      const Eigen::VectorXd q = path[i] + ((k + 0.5) / pieces) * d;
      energy += std::sqrt(d.dot(mass(q) * d)) / pieces;
    }
  }
  return {joint, energy};
}
// [docs-end:measure]

geodex_examples::Json record(const gp::PlanResult<Eigen::VectorXd>& result) {
  using geodex_examples::Json;
  const auto [joint, energy] = measure(result.path);
  return Json::object({{"solved", result.solved},
                       {"cost", result.cost},
                       {"smoothed", result.smoothed},
                       {"path", Json::path(result.path)},
                       {"raw_path", Json::path(result.raw_path)},
                       {"joint_length", joint},
                       {"energy_length", energy}});
}

// [docs-start:load]
int main(int argc, char** argv) {
  constexpr auto R = gr::Robot::Fr3Gripper;
  const auto env = shelf_with_box();
  gr::RobotPlanOptions options;  // the kinetic-energy metric on the joints
  std::printf("%s joints: %d\n", gr::name(R).data(), gr::MassMatrix<R>::Nq);
  // [docs-end:load]

  // [docs-start:plan]
  Eigen::VectorXd start(7), goal(7);
  start << 0.1758, -0.1952, 0.4451, -2.1536, 1.9130, 2.0966, 1.0981;  // middle bay
  goal << 0.1591, -0.0746, 0.4710, -1.3579, 1.4042, 2.1810, 1.9137;    // top bay
  gp::PlanSettings settings;
  settings.iterations = 1500;
  settings.seed = 1;

  const auto result = gr::plan<R>(start, goal, env, settings, options);
  std::printf("solved=%d cost=%.3f waypoints=%zu\n", result.solved, result.cost,
              result.path.size());
  // [docs-end:plan]

  {
    // [docs-start:measure]
    const auto [joint, energy] = measure(result.path);
    std::printf("joint length=%.3f rad kinetic-energy length=%.3f\n", joint,
                energy);
    // [docs-end:measure]
  }

  // [docs-start:try-it]
  options.metric = gr::ArmMetric::Euclidean;
  const auto euclidean_result = gr::plan<R>(start, goal, env, settings, options);
  const auto [joint, energy] = measure(euclidean_result.path);
  std::printf("joint length=%.3f rad kinetic-energy length=%.3f\n", joint, energy);
  // [docs-end:try-it]

  using geodex_examples::Json;
  geodex_examples::write_json_arg(
      argc, argv,
      Json::object({{"kinetic_energy", record(result)},
                    {"euclidean", record(euclidean_result)}}));
  if (!result.solved || !euclidean_result.solved) return 1;
  // [docs-start:try-it]
  return 0;
}
// [docs-end:try-it]

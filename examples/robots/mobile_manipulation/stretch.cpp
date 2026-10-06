// The Stretch 3 and the Stretch 4 on one kitchen task, the C++ version of
// stretch.py.
//
// Each robot starts beside the kitchen table with its wrist tucked and ends on the
// far side of the island with its gripper over it. Base and arm move in one motion.
// The program plans the Stretch 3 on its differential-drive base, then the
// Stretch 4 on its holonomic base.
//
// Usage: stretch [--json out.json]

// [docs-start:load]
#include <cmath>
#include <cstdio>

#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

#include <geodex/robots/planning.hpp>
// [docs-end:load]

#include "../../common/json_output.hpp"

// [docs-start:load]
namespace gr = geodex::robots;
namespace gv = geodex::integration::vamp;
namespace gp = geodex::planning;

// The kitchen, a scene file in the scenes directory next to this program.
const std::string kScene =
    (std::filesystem::path(__FILE__).parent_path() / "scenes" / "kitchen.yaml")
        .string();
// [docs-end:load]

// [docs-start:sideways]
/// Share of the base's travel that is sideways in its own frame. Each edge moves
/// the base along one constant body twist (v_x, v_y, omega). The twist is the
/// SE(2) logarithm between the edge's waypoints.
double sideways_share(const std::vector<Eigen::VectorXd>& path) {
  const geodex::SE2<> se2;
  double sideways = 0.0, travel = 0.0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const Eigen::Vector3d twist = se2.log(path[i].head<3>(), path[i + 1].head<3>());
    sideways += std::abs(twist[1]);
    travel += std::sqrt(twist[0] * twist[0] + twist[1] * twist[1]);
  }
  return sideways / travel;
}
// [docs-end:sideways]

template <gr::Robot R>
geodex_examples::Json record(const gp::PlanResult<Eigen::VectorXd>& result) {
  using geodex_examples::Json;
  const bool differential = gr::base_drive<R> == gr::BaseDrive::Differential;
  return Json::object({{"solved", result.solved},
                       {"cost", result.cost},
                       {"drive", differential ? "differential_drive" : "holonomic"},
                       {"sideways_share", sideways_share(result.path)},
                       {"path", Json::path(result.path)},
                       {"raw_path", Json::path(result.raw_path)}});
}

// [docs-start:load]
int main(int argc, char** argv) {
  constexpr double pi = std::numbers::pi;
  const auto scene = gv::load_scene(kScene);
  gr::RobotPlanOptions options;  // each robot takes the base metric of its drive
  options.base_region = {Eigen::Vector2d(-3.0, -2.5), Eigen::Vector2d(3.0, 2.5)};
  constexpr auto R3 = gr::Robot::Stretch3;
  const bool differential = gr::base_drive<R3> == gr::BaseDrive::Differential;
  std::printf("%s drive: %s coordinates: %d\n", gr::name(R3).data(),
              differential ? "differential_drive" : "holonomic",
              gr::MassMatrix<R3>::Nq + 3);
  // [docs-end:load]

  // [docs-start:plan]
  // (x, y, theta, lift, arm extension, wrist yaw, wrist pitch, wrist roll)
  Eigen::VectorXd start(8), goal(8);
  start << 1.8, -0.2, pi / 2, 0.3, 0.0, 3.0, -0.5, 0.0;  // by the table
  goal << 0.0, 1.9, 0.0, 0.85, 0.4, 0.0, 0.0, 0.0;       // over the island
  gp::PlanSettings settings;
  settings.iterations = 1500;
  settings.seed = 1;
  settings.collision_check_resolution = 0.005;

  const auto result3 = gr::plan<R3>(start, goal, scene, settings, options);
  std::printf("solved=%d cost=%.3f waypoints=%zu\n", result3.solved, result3.cost,
              result3.path.size());
  // [docs-end:plan]

  // [docs-start:sideways]
  std::printf("sideways share=%.3f\n", sideways_share(result3.path));
  // [docs-end:sideways]

  // [docs-start:try-it]
  // The Stretch 4's arm points forward, and its goal faces the island.
  Eigen::VectorXd start4(8), goal4(8);
  start4 << 1.8, -0.2, pi / 2, 0.2, 0.0, 3.0, 0.0, 0.0;
  goal4 << 0.0, 1.85, -pi / 2, 0.8, 0.35, 0.0, 0.0, 0.0;

  const auto result4 =
      gr::plan<gr::Robot::Stretch4>(start4, goal4, scene, settings, options);
  std::printf("solved=%d cost=%.3f sideways share=%.3f\n", result4.solved,
              result4.cost, sideways_share(result4.path));
  // [docs-end:try-it]

  using geodex_examples::Json;
  geodex_examples::write_json_arg(
      argc, argv,
      Json::object({{"stretch3", record<gr::Robot::Stretch3>(result3)},
                    {"stretch4", record<gr::Robot::Stretch4>(result4)}}));
  if (!result3.solved || !result4.solved) return 1;
  // [docs-start:try-it]
  return 0;
}
// [docs-end:try-it]

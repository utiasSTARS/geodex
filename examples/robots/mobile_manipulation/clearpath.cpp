// A UR5e on a Husky and on a Ridgeback, the C++ version of clearpath.py.
//
// Each robot drives from the shelf of a workcell to a low table and reaches over it
// with the tool pointing down. Base and arm move in one motion on SE(2) x R^6. The
// program plans the Husky on its skid-steer base, measures how much its base
// slides, then plans the Ridgeback on its mecanum base.
//
// Usage: clearpath [--json out.json]

// [docs-start:load]
#include <cmath>
#include <cstdio>

#include <filesystem>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include <geodex/robots/planning.hpp>
// [docs-end:load]

#include "../../common/json_output.hpp"

// [docs-start:load]
namespace gr = geodex::robots;
namespace gv = geodex::integration::vamp;
namespace gp = geodex::planning;

// The workcell, a scene file in the scenes directory next to this program.
const std::string kScene =
    (std::filesystem::path(__FILE__).parent_path() / "scenes" / "workcell.yaml")
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
  constexpr auto Husky = gr::Robot::HuskyUr5e;
  const auto scene = gv::load_scene(kScene);
  gr::RobotPlanOptions options;  // each robot takes the base metric of its drive
  options.base_region = {Eigen::Vector2d(-3.0, -2.5), Eigen::Vector2d(3.0, 2.5)};
  const bool differential = gr::base_drive<Husky> == gr::BaseDrive::Differential;
  std::printf("%s drive: %s coordinates: %d\n", gr::name(Husky).data(),
              differential ? "differential_drive" : "holonomic",
              gr::MassMatrix<Husky>::Nq + 3);
  // [docs-end:load]

  // [docs-start:plan]
  // (x, y, theta, shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3)
  Eigen::VectorXd start(9), goal(9);
  start << -1.8, 1.3, pi / 2, 0.0, -2.3, 2.3, -1.57, -1.57, 0.0;
  goal << 1.6, 0.0, 0.0, 0.0, -0.6, 0.4, -1.4, -pi / 2, 0.0;
  gp::PlanSettings settings;
  settings.iterations = 1000;
  settings.seed = 1;
  settings.collision_check_resolution = 0.005;

  const auto husky_result = gr::plan<Husky>(start, goal, scene, settings, options);
  std::printf("solved=%d cost=%.3f\n", husky_result.solved, husky_result.cost);
  // [docs-end:plan]

  // [docs-start:sideways]
  std::printf("sideways share=%.3f\n", sideways_share(husky_result.path));
  // [docs-end:sideways]

  // [docs-start:try-it]
  const auto ridgeback_result =
      gr::plan<gr::Robot::RidgebackUr5e>(start, goal, scene, settings, options);
  std::printf("solved=%d cost=%.3f sideways share=%.3f\n", ridgeback_result.solved,
              ridgeback_result.cost, sideways_share(ridgeback_result.path));
  // [docs-end:try-it]

  using geodex_examples::Json;
  geodex_examples::write_json_arg(
      argc, argv,
      Json::object({{"husky_ur5e", record<Husky>(husky_result)},
                    {"ridgeback_ur5e",
                     record<gr::Robot::RidgebackUr5e>(ridgeback_result)}}));
  if (!husky_result.solved || !ridgeback_result.solved) return 1;
  // [docs-start:try-it]
  return 0;
}
// [docs-end:try-it]

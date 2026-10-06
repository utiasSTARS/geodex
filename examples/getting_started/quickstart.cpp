// Quickstart example, the C++ version of quickstart.py. A Franka Panda swings
// around a post.
//
// Usage: quickstart [--json out.json]

// [docs-start:load]
#include <cstdio>

#include <geodex/integration/vamp/registry.hpp>
#include <geodex/robots/planning.hpp>
// [docs-end:load]

#include "../common/json_output.hpp"

// [docs-start:load]
namespace gr = geodex::robots;
namespace gv = geodex::integration::vamp;
namespace gp = geodex::planning;

int main(int argc, char** argv) {
  constexpr auto R = gr::Robot::Panda;  // seven joints, the kinetic-energy metric
  std::printf("%s joints: %d\n", gr::name(R).data(), gr::MassMatrix<R>::Nq);
  // [docs-end:load]

  // [docs-start:scene]
  auto builder = gv::make_scene_builder();
  gv::scene_add_box(builder, {0.4, 0.0, 0.3}, {0.1, 0.1, 0.8},  // a post in front
                    Eigen::Quaterniond::Identity());
  const auto scene = gv::build_scene(builder);
  // [docs-end:scene]

  // [docs-start:plan]
  Eigen::VectorXd start(7), goal(7);
  start << -1.1, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
  goal << 1.1, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;

  gp::PlanSettings settings;
  settings.iterations = 1500;
  settings.seed = 1;
  const auto result = gr::plan<R>(start, goal, scene, settings);
  std::printf("solved=%d cost=%.4f waypoints=%zu\n", result.solved, result.cost,
              result.path.size());
  // [docs-end:plan]

  // [docs-start:result]
  std::printf("path: %zu x %ld, raw path: %zu x %ld\n", result.path.size(),
              result.path[0].size(), result.raw_path.size(),
              result.raw_path[0].size());
  // [docs-end:result]

  // [docs-start:try-it]
  gr::RobotPlanOptions options;
  options.metric = gr::ArmMetric::Euclidean;
  const auto other = gr::plan<R>(start, goal, scene, settings, options);
  for (const auto* r : {&result, &other}) {
    double turned = 0.0;
    for (std::size_t i = 0; i + 1 < r->path.size(); ++i) {
      turned += (r->path[i + 1] - r->path[i]).norm();
    }
    std::printf("%s: the joints turn %.3f rad in total\n",
                r == &result ? "kinetic energy" : "euclidean", turned);
  }
  // [docs-end:try-it]

  using geodex_examples::Json;
  geodex_examples::write_json_arg(
      argc, argv,
      Json::object(
          {{"solved", result.solved},
           {"cost", result.cost},
           {"raw_path", Json::path(result.raw_path)},
           {"path", Json::path(result.path)},
           {"euclidean", Json::object({{"solved", other.solved},
                                       {"cost", other.cost},
                                       {"path", Json::path(other.path)}})}}));
  // [docs-start:try-it]
  return 0;
}
// [docs-end:try-it]

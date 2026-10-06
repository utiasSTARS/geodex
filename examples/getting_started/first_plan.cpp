// A first plan. A differential-drive robot on SE(2) drives around a disc.
//
// The C++ version of first_plan.py. Usage: first_plan [--json out.json]

// [docs-start:first-plan]
#include <cmath>
#include <cstdio>

#include <geodex/geodex.hpp>
#include <geodex/planning/plan.hpp>
// [docs-end:first-plan]

#include "../common/json_output.hpp"

// [docs-start:first-plan]
int main(int argc, char** argv) {
  // The space holds poses (x, y, heading) in a 4 m x 4 m room. The weight 100 on
  // the squared sideways speed makes a meter of sliding as long as ten meters of
  // driving forward.
  geodex::SE2<> space{geodex::SE2LeftInvariantMetric{1.0, 100.0, 1.0},
                      geodex::SE2LeftExponentialMap{},
                      Eigen::Vector3d(0.0, 0.0, -M_PI),
                      Eigen::Vector3d(4.0, 4.0, M_PI)};

  auto is_valid = [](const Eigen::Vector3d& q) {  // outside a disc of radius 0.8 m
    return std::hypot(q[0] - 2.0, q[1] - 2.0) > 0.8;
  };

  const Eigen::Vector3d start(0.5, 2.0, 0.0), goal(3.5, 2.0, 0.0);
  geodex::planning::PlanSettings settings;
  settings.iterations = 2000;
  settings.seed = 1;
  auto result = geodex::planning::plan(space, start, goal, is_valid, settings);
  std::printf("%d %.3f %zu\n", result.solved, result.cost, result.path.size());
  // [docs-end:first-plan]

  using geodex_examples::Json;
  geodex_examples::write_json_arg(
      argc, argv,
      Json::object({{"solved", result.solved},
                    {"cost", result.cost},
                    {"path", Json::path(result.path)}}));
  // [docs-start:first-plan]
  return 0;
}
// [docs-end:first-plan]

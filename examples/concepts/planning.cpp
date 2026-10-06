// Examples of the Planning concept page, the C++ version of planning.py.
//
// The program makes two plans. The first passes between two caps on the sphere, and
// the second sets the fields of PlanSettings.
//
// Usage: planning [--json out.json]

// [docs-start:first-plan]
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include <geodex/geodex.hpp>
#include <geodex/planning/plan.hpp>
// [docs-end:first-plan]

#include "../common/json_output.hpp"

using geodex_examples::Json;

// [docs-start:first-plan]
namespace gp = geodex::planning;
// [docs-end:first-plan]

int main(int argc, char** argv) {
  std::vector<std::pair<std::string, Json>> out;

  // [docs-start:first-plan]
  geodex::Sphere<> sphere;

  // The point of the unit sphere at a longitude and latitude in degrees.
  auto point = [](double lon, double lat) {
    lon *= M_PI / 180.0;
    lat *= M_PI / 180.0;
    return Eigen::Vector3d(std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon),
                           std::sin(lat));
  };

  // Two caps are obstacles, each a center and an angular radius.
  const std::vector<std::pair<Eigen::Vector3d, double>> caps = {
      {point(-4, 6), 18.0 * (M_PI / 180.0)}, {point(14, 36), 12.0 * (M_PI / 180.0)}};
  auto is_valid = [&caps](const Eigen::Vector3d& q) {
    for (const auto& [center, radius] : caps) {
      if (q.dot(center) >= std::cos(radius)) return false;
    }
    return true;
  };

  const Eigen::Vector3d start = point(-40, 8), goal = point(40, 22);

  gp::PlanSettings settings;
  settings.iterations = 2000;
  settings.seed = 7;
  auto result = gp::plan(sphere, start, goal, is_valid, settings);

  std::printf("solved=%d cost=%.4f waypoints=%zu\n", result.solved, result.cost,
              result.path.size());
  // [docs-end:first-plan]

  // [docs-start:result]
  const auto& raw = result.raw_path;  // planner waypoints
  const auto& path = result.path;     // smoothed, checked and evenly spaced waypoints
  std::printf("raw waypoints=%zu smoothed=%d solve=%.1f ms smooth=%.1f ms\n",
              raw.size(), result.smoothed, result.time_ms, result.smooth_ms);
  // [docs-end:result]
  std::vector<Json> cap_list;
  for (const auto& [center, radius] : caps) {
    cap_list.push_back(Json::array({Json(center), Json(radius)}));
  }
  out.emplace_back("sphere", Json::object({{"solved", result.solved},
                                           {"cost", result.cost},
                                           {"raw_path", Json::path(raw)},
                                           {"path", Json::path(path)},
                                           {"smoothed", result.smoothed},
                                           {"caps", Json::array(cap_list)},
                                           {"raw_count", raw.size()}}));

  {
    // [docs-start:settings]
    gp::PlanSettings settings;
    settings.iterations = 2000;  // fixed budget, reproducible with a seed
    settings.seed = 7;           // seeds OMPL and the geodex samplers
    settings.time = 1.0;         // wall-clock budget, used when iterations is 0
    settings.planner = gp::planners::GreedyRRTstar{
        .range = 0.0, .greedy_ratio = 0.9, .rewire_factor = 1.1};
    settings.collision_check_resolution =
        0.01;  // spacing of the validity samples along an edge
    settings.interp =
        gp::InterpolationMode::BaseGeodesic;  // the curve of the planner's edges
    settings.goal_tolerance = 0.0;
    settings.smooth = true;  // run the smoother on the planner's path
    settings.smoothing.output_spacing = 0.0;
    auto result = gp::plan(sphere, start, goal, is_valid, settings);
    // [docs-end:settings]
    out.emplace_back("settings", Json::object({{"solved", result.solved},
                                               {"cost", result.cost},
                                               {"path", Json::path(result.path)}}));
  }

  geodex_examples::write_json_arg(argc, argv, Json::object(out));
  return 0;
}

// Three Clearpath bases cross a narrow office, the C++ version of bases.py.
//
// The program loads the office map as a distance grid, plans a Jackal from the
// lower aisle into a gap between two desks, measures how much of its travel is
// sideways, then plans the same query for every base. Run it from
// examples/robots/navigation, where the office map is.
//
// Usage: bases [--json out.json]

// [docs-start:map]
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <map>
#include <numbers>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <geodex/algorithm/precompute_matrix_lower_bound.hpp>
#include <geodex/collision/distance_grid.hpp>
#include <geodex/collision/footprint_grid_checker.hpp>
#include <geodex/collision/polygon_footprint.hpp>
#include <geodex/geodex.hpp>
#include <geodex/planning/plan.hpp>
// [docs-end:map]

#include "../../common/json_output.hpp"

// [docs-start:map]
namespace gc = geodex::collision;
namespace gp = geodex::planning;

/// DistanceGrid of the office map, the signed distance in meters to the nearest
/// occupied or unknown cell at the center of every 0.05 m cell.
gc::DistanceGrid office_grid() {
  gc::DistanceGrid grid;
  if (!grid.load("office_dist.txt")) std::exit(1);
  return grid;
}
// [docs-end:map]

// [docs-start:try-it]
struct Platform {
  std::string name;
  double length, width;  // footprint in meters, from Clearpath's specifications
  std::string drive;
  double planner_range;  // the planner's range under the base's metric
};
const std::vector<Platform> kPlatforms{
    {"jackal", 0.508, 0.430, "skid_steer", 6.5},
    {"dingo_d", 0.551, 0.517, "differential", 12.0},
    {"dingo_o", 0.686, 0.517, "holonomic", 6.5},
};

// Weights (w_x, w_y, w_theta) of the base metric for each drive.
const std::map<std::string, geodex::SE2LeftInvariantMetric> kDrives{
    {"differential", {1.0, 50.0, 1.0}},
    {"skid_steer", {1.0, 50.0, 2.0}},
    {"holonomic", {1.0, 1.0, 1.0}},
};
// [docs-end:try-it]

// [docs-start:plan]
/// Plan a rectangular base through the office with the given metric weights. The
/// planner adds tree edges of at most planner_range under the metric.
gp::PlanResult<Eigen::Vector3d> plan_base(
    const gc::DistanceGrid& grid, double length, double width,
    const geodex::SE2LeftInvariantMetric& weights, double planner_range,
    std::uint64_t seed = 1) {
  const double x_hi = (grid.width() - 1) * grid.resolution();
  const double y_hi = (grid.height() - 1) * grid.resolution();
  const geodex::SE2<> se2{weights, geodex::SE2LeftExponentialMap{},
                          Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                          Eigen::Vector3d(x_hi, y_hi, std::numbers::pi)};
  const auto footprint = gc::PolygonFootprint::rectangle(length / 2, width / 2, 6);
  const gc::FootprintGridChecker checker{&grid, footprint, 0.05};  // 5 cm margin
  // The clearance metric scales the base metric with kappa = 1.5 and beta = 3.
  const geodex::SDFConformalMetric metric{weights, checker, 1.5, 3.0};
  const geodex::ConfigurationSpace space{se2, metric};
  gp::PlanSettings settings;
  settings.iterations = 1000;
  settings.seed = seed;
  settings.planner = gp::planners::GreedyRRTstar{.range = planner_range};
  settings.collision_check_resolution = 0.05;
  settings.interp = gp::InterpolationMode::BaseGeodesic;
  // (x, y, heading) of the start and the goal, in meters and radians.
  constexpr double deg = std::numbers::pi / 180.0;
  const Eigen::Vector3d start(3.885, 1.235, -2.81 * deg);
  const Eigen::Vector3d goal(12.425, 3.675, -120.44 * deg);
  const auto is_valid = [&](const Eigen::Vector3d& q) {
    return checker.is_valid(q);
  };
  const auto heuristic =
      geodex::algorithm::precompute_matrix_lower_bound(se2).heuristic();
  return gp::plan(space, start, goal, is_valid, settings, heuristic);
}
// [docs-end:plan]

// [docs-start:sideways]
/// Share of the base's travel that is sideways in its own frame. Each edge moves
/// the base along one constant body twist (v_x, v_y, omega). The twist is the
/// SE(2) logarithm between the edge's waypoints.
double sideways_share(const std::vector<Eigen::Vector3d>& path) {
  const geodex::SE2<> se2;
  double sideways = 0.0, travel = 0.0;
  for (std::size_t i = 0; i + 1 < path.size(); ++i) {
    const Eigen::Vector3d twist = se2.log(path[i], path[i + 1]);
    sideways += std::abs(twist[1]);
    travel += std::sqrt(twist[0] * twist[0] + twist[1] * twist[1]);
  }
  return sideways / travel;
}
// [docs-end:sideways]

// [docs-start:plan]
int main(int argc, char** argv) {
  const gc::DistanceGrid grid = office_grid();
  const auto result =
      plan_base(grid, 0.508, 0.430, {1.0, 50.0, 2.0}, 6.5);  // a Jackal
  std::printf("solved=%d cost=%.3f waypoints=%zu\n", result.solved, result.cost,
              result.path.size());
  // [docs-end:plan]

  // [docs-start:sideways]
  std::printf("sideways share=%.3f\n", sideways_share(result.path));
  // [docs-end:sideways]

  using geodex_examples::Json;
  std::vector<std::pair<std::string, Json>> out;
  bool all_solved = true;
  // [docs-start:try-it]
  for (const auto& [name, length, width, drive, planner_range] : kPlatforms) {
    const auto result =
        plan_base(grid, length, width, kDrives.at(drive), planner_range);
    const double share = sideways_share(result.path);
    std::printf("%10s: solved=%d cost=%.2f sideways share=%.3f\n", name.c_str(),
                result.solved, result.cost, share);
    // [docs-end:try-it]
    all_solved = all_solved && result.solved;
    out.emplace_back(name,
                     Json::object({{"solved", result.solved},
                                   {"cost", result.cost},
                                   {"sideways_share", share},
                                   {"path", Json::path(result.path)},
                                   {"raw_path", Json::path(result.raw_path)},
                                   {"drive", drive},
                                   {"footprint", Json::array({length, width})}}));
    // [docs-start:try-it]
  }
  // [docs-end:try-it]

  geodex_examples::write_json_arg(argc, argv, Json::object(out));
  if (!all_solved) return 1;
  // [docs-start:try-it]
  return 0;
}
// [docs-end:try-it]

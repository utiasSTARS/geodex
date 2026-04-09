/// @file se2_willow_planning.cpp
/// @brief SE(2) planning on a real occupancy grid (Willow Garage) with distance transform.

#include <ompl/base/PlannerData.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/rrt/GreedyRRTstar.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

#include <cmath>
#include <fstream>
#include <functional>
#include <geodex/integration/ompl/geodex_optimization_objective.hpp>
#include <geodex/integration/ompl/geodex_state_space.hpp>
#include <geodex/manifold/se2.hpp>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// ---------------------------------------------------------------------------
// Distance grid (precomputed distance transform)
// ---------------------------------------------------------------------------

struct DistanceGrid {
  int width = 0, height = 0;
  double resolution = 0.05;
  std::vector<double> data;

  bool load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
      std::cerr << "Error: cannot open " << filename << "\n";
      return false;
    }

    in >> width >> height >> resolution;
    data.resize(static_cast<size_t>(width) * height);

    for (int r = 0; r < height; ++r) {
      for (int c = 0; c < width; ++c) {
        in >> data[static_cast<size_t>(r) * width + c];
      }
    }

    if (!in) {
      std::cerr << "Error: failed to read distance grid data\n";
      return false;
    }

    std::cout << "Loaded distance grid: " << width << "x" << height
              << ", resolution=" << resolution << " m/px"
              << ", world=" << width * resolution << "x" << height * resolution << " m\n";
    return true;
  }

  /// @brief Query the distance to nearest obstacle at world coordinates (x_m, y_m).
  double distance_at(double x_m, double y_m) const {
    int c = static_cast<int>(x_m / resolution);
    int r = static_cast<int>(y_m / resolution);

    // Clamp to bounds
    c = std::clamp(c, 0, width - 1);
    r = std::clamp(r, 0, height - 1);

    return data[static_cast<size_t>(r) * width + c];
  }
};

// ---------------------------------------------------------------------------
// Grid-based obstacle checker
// ---------------------------------------------------------------------------

template <typename StateType>
class GridObstacleChecker : public ob::StateValidityChecker {
  DistanceGrid grid_;
  double robot_radius_;

 public:
  GridObstacleChecker(const ob::SpaceInformationPtr& si, DistanceGrid grid, double robot_radius)
      : ob::StateValidityChecker(si), grid_(std::move(grid)), robot_radius_(robot_radius) {}

  bool isValid(const ob::State* state) const override {
    const auto* s = state->as<StateType>();
    double x = s->values[0], y = s->values[1];
    return grid_.distance_at(x, y) > robot_radius_;
  }
};

// ---------------------------------------------------------------------------
// Run result & runner
// ---------------------------------------------------------------------------

struct RunResult {
  std::string label;
  std::string metric_info;
  std::vector<std::array<double, 3>> path;
  std::vector<std::array<std::array<double, 3>, 2>> tree;
  bool solved = false;
};

using PlannerFactory = std::function<ob::PlannerPtr(const ob::SpaceInformationPtr&)>;

template <typename ManifoldT>
RunResult run_planner(ManifoldT manifold, const std::string& label, const std::string& metric_info,
                      const PlannerFactory& make_planner, double solve_time,
                      const ob::RealVectorBounds& bounds, const std::array<double, 3>& start_pose,
                      const std::array<double, 3>& goal_pose, const DistanceGrid& grid,
                      double robot_radius) {
  using StateSpace = geodex::ompl_integration::GeodexStateSpace<ManifoldT>;
  using StateType = geodex::ompl_integration::GeodexState<ManifoldT>;

  RunResult result;
  result.label = label;
  result.metric_info = metric_info;

  auto space = std::make_shared<StateSpace>(manifold, bounds);
  space->setCollisionResolution(grid.resolution);

  og::SimpleSetup ss(space);

  ss.setStateValidityChecker(
      std::make_shared<GridObstacleChecker<StateType>>(ss.getSpaceInformation(), grid, robot_radius));

  ob::ScopedState<StateSpace> start(space);
  start->values[0] = start_pose[0];
  start->values[1] = start_pose[1];
  start->values[2] = start_pose[2];

  ob::ScopedState<StateSpace> goal(space);
  goal->values[0] = goal_pose[0];
  goal->values[1] = goal_pose[1];
  goal->values[2] = goal_pose[2];

  ss.setStartAndGoalStates(start, goal);

  // Set optimization objective with admissible heuristic
  typename ManifoldT::Point goal_coords;
  goal_coords[0] = goal_pose[0];
  goal_coords[1] = goal_pose[1];
  goal_coords[2] = goal_pose[2];
  auto objective =
      std::make_shared<geodex::ompl_integration::GeodexOptimizationObjective<ManifoldT>>(
          ss.getSpaceInformation(), goal_coords);
  ss.setOptimizationObjective(objective);

  ss.setPlanner(make_planner(ss.getSpaceInformation()));

  ob::PlannerStatus status = ss.solve(solve_time);

  if (status) {
    result.solved = true;
    auto& path = ss.getSolutionPath();
    path.interpolate();

    for (const auto* state : path.getStates()) {
      const auto* s = state->as<StateType>();
      result.path.push_back({s->values[0], s->values[1], s->values[2]});
    }

    ob::PlannerData pdata(ss.getSpaceInformation());
    ss.getPlannerData(pdata);

    unsigned int nv = pdata.numVertices();
    for (unsigned int i = 0; i < nv; ++i) {
      std::vector<unsigned int> edges;
      pdata.getEdges(i, edges);
      const auto* vi = pdata.getVertex(i).getState()->as<StateType>();
      for (unsigned int j : edges) {
        const auto* vj = pdata.getVertex(j).getState()->as<StateType>();
        result.tree.push_back(
            {{{vi->values[0], vi->values[1], vi->values[2]},
              {vj->values[0], vj->values[1], vj->values[2]}}});
      }
    }

    std::cout << label << ": found solution (" << nv << " tree vertices, "
              << path.getStateCount() << " path waypoints)\n";
  } else {
    std::cerr << label << ": no solution found.\n";
  }

  return result;
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

void write_json(const std::string& filename, const std::vector<RunResult>& runs,
                const DistanceGrid& grid, const std::array<double, 3>& start_pose,
                const std::array<double, 3>& goal_pose) {
  std::ofstream out(filename);
  if (!out) {
    std::cerr << "Error: cannot open " << filename << "\n";
    return;
  }

  out << std::fixed << std::setprecision(8);
  out << "{\n";

  out << "  \"map\": { \"width\": " << grid.width << ", \"height\": " << grid.height
      << ", \"resolution\": " << grid.resolution << " },\n";

  out << "  \"start\": [" << start_pose[0] << ", " << start_pose[1] << ", " << start_pose[2]
      << "],\n";
  out << "  \"goal\": [" << goal_pose[0] << ", " << goal_pose[1] << ", " << goal_pose[2]
      << "],\n";

  out << "  \"runs\": [\n";
  for (size_t r = 0; r < runs.size(); ++r) {
    const auto& run = runs[r];
    out << "    {\n";
    out << "      \"label\": \"" << run.label << "\",\n";
    out << "      \"metric_info\": \"" << run.metric_info << "\",\n";

    out << "      \"path\": [\n";
    for (size_t i = 0; i < run.path.size(); ++i) {
      out << "        [" << run.path[i][0] << ", " << run.path[i][1] << ", " << run.path[i][2]
          << "]";
      if (i + 1 < run.path.size()) out << ",";
      out << "\n";
    }
    out << "      ],\n";

    out << "      \"tree\": [\n";
    for (size_t i = 0; i < run.tree.size(); ++i) {
      out << "        [[" << run.tree[i][0][0] << ", " << run.tree[i][0][1] << ", "
          << run.tree[i][0][2] << "], [" << run.tree[i][1][0] << ", " << run.tree[i][1][1] << ", "
          << run.tree[i][1][2] << "]]";
      if (i + 1 < run.tree.size()) out << ",";
      out << "\n";
    }
    out << "      ]\n";

    out << "    }";
    if (r + 1 < runs.size()) out << ",";
    out << "\n";
  }

  out << "  ]\n";
  out << "}\n";

  std::cout << "Wrote " << filename << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

struct PlannerInfo {
  std::string name;
  PlannerFactory factory;
};

std::vector<PlannerInfo> get_planners() {
  return {
      {"RRT*",
       [](const ob::SpaceInformationPtr& si) { return std::make_shared<og::RRTstar>(si); }},
      {"InformedRRT*",
       [](const ob::SpaceInformationPtr& si) { return std::make_shared<og::InformedRRTstar>(si); }},
      {"G-RRT*",
       [](const ob::SpaceInformationPtr& si) { return std::make_shared<og::GreedyRRTstar>(si); }},
      // {"BIT*",
      //  [](const ob::SpaceInformationPtr& si) { return std::make_shared<og::BITstar>(si); }},
  };
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <dist_map.txt> [output.json]\n";
    return 1;
  }

  std::string dist_map_file = argv[1];
  std::string output_file = argc > 2 ? argv[2] : "se2_willow.json";

  DistanceGrid grid;
  if (!grid.load(dist_map_file)) return 1;

  double robot_radius = 0.3;
  double world_w = grid.width * grid.resolution;
  double world_h = grid.height * grid.resolution;

  std::array<double, 3> start_pose = {8.0, 3.0, -std::numbers::pi / 2.0};
  std::array<double, 3> goal_pose = {10.0, 44.0, 0.0};

  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, 0.0);
  bounds.setHigh(0, world_w);
  bounds.setLow(1, 0.0);
  bounds.setHigh(1, world_h);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);

  std::cout << "World bounds: [0, " << world_w << "] x [0, " << world_h << "]\n";
  std::cout << "Start: (" << start_pose[0] << ", " << start_pose[1] << ", " << start_pose[2]
            << ")\n";
  std::cout << "Goal:  (" << goal_pose[0] << ", " << goal_pose[1] << ", " << goal_pose[2]
            << ")\n";

  auto planners = get_planners();
  std::vector<RunResult> runs;

  // Isotropic metric (omnidirectional)
  for (const auto& [pname, pfactory] : planners) {
    geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 0.5};
    geodex::SE2<> manifold{metric, geodex::SE2ExponentialMap{},
                           Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                           Eigen::Vector3d(world_w, world_h, std::numbers::pi)};
    runs.push_back(run_planner(manifold, "Isotropic (" + pname + ")", "w = (1, 1, 0.5)", pfactory,
                               5.0, bounds, start_pose, goal_pose, grid, robot_radius));
  }

  // Car-like metric (expensive lateral motion)
  for (const auto& [pname, pfactory] : planners) {
    geodex::SE2LeftInvariantMetric metric{1.0, 100.0, 0.5};
    geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap> manifold{
        metric, geodex::SE2ExponentialMap{},
        Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
        Eigen::Vector3d(world_w, world_h, std::numbers::pi)};
    runs.push_back(run_planner(manifold, "Car-like (" + pname + ")", "w = (1, 100, 0.5)", pfactory,
                               5.0, bounds, start_pose, goal_pose, grid, robot_radius));
  }

  write_json(output_file, runs, grid, start_pose, goal_pose);

  return 0;
}

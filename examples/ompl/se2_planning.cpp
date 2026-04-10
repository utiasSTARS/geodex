#include <ompl/base/PlannerData.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/rrt/GreedyRRTstar.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <geodex/algorithm/path_smoothing.hpp>
#include <geodex/integration/ompl/geodex_optimization_objective.hpp>
#include <geodex/integration/ompl/geodex_state_space.hpp>
#include <geodex/manifold/se2.hpp>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace ob = ompl::base;
namespace og = ompl::geometric;

struct Obstacle {
  double cx, cy, r;
};

/// Circular obstacle validity checker for SE(2) (checks position only).
template <typename StateType>
class SE2ObstacleChecker : public ob::StateValidityChecker {
  std::vector<Obstacle> obstacles_;

 public:
  SE2ObstacleChecker(const ob::SpaceInformationPtr& si, std::vector<Obstacle> obstacles)
      : ob::StateValidityChecker(si), obstacles_(std::move(obstacles)) {}

  bool isValid(const ob::State* state) const override {
    const auto* s = state->as<StateType>();
    double x = s->values[0], y = s->values[1];
    for (const auto& obs : obstacles_) {
      double dx = x - obs.cx, dy = y - obs.cy;
      if (dx * dx + dy * dy <= obs.r * obs.r) return false;
    }
    return true;
  }
};

struct RunResult {
  std::string label;
  std::string metric_info;
  std::vector<std::array<double, 3>> raw_path;
  std::vector<std::array<double, 3>> smoothed_path;
  std::vector<std::array<std::array<double, 3>, 2>> tree;
  bool solved = false;
  double planning_time_ms = 0.0;
  double smoothing_time_ms = 0.0;
  int vertices_removed = 0;
  int smooth_iterations = 0;
  int backward_segments = 0;
};

struct RunConfig {
  double x_lo = 0.0, x_hi = 10.0;
  double y_lo = 0.0, y_hi = 10.0;
  double start_x = 1.0, start_y = 1.0, start_theta = 0.0;
  double goal_x = 9.0, goal_y = 9.0, goal_theta = 0.0;
  std::vector<Obstacle> obstacles;
};

using PlannerFactory = std::function<ob::PlannerPtr(const ob::SpaceInformationPtr&)>;

/// Run a planner on a given SE2 manifold with post-planning pipeline.
template <typename ManifoldT>
RunResult run_planner(ManifoldT manifold, const std::string& label, const std::string& metric_info,
                      const PlannerFactory& make_planner, double solve_time, const RunConfig& cfg) {
  using StateSpace = geodex::ompl_integration::GeodexStateSpace<ManifoldT>;
  using StateType = geodex::ompl_integration::GeodexState<ManifoldT>;

  RunResult result;
  result.label = label;
  result.metric_info = metric_info;

  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, cfg.x_lo);
  bounds.setHigh(0, cfg.x_hi);
  bounds.setLow(1, cfg.y_lo);
  bounds.setHigh(1, cfg.y_hi);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);
  auto space = std::make_shared<StateSpace>(manifold, bounds);
  space->setCollisionResolution(0.1);

  og::SimpleSetup ss(space);

  ss.setStateValidityChecker(
      std::make_shared<SE2ObstacleChecker<StateType>>(ss.getSpaceInformation(), cfg.obstacles));

  ob::ScopedState<StateSpace> start(space);
  start->values[0] = cfg.start_x;
  start->values[1] = cfg.start_y;
  start->values[2] = cfg.start_theta;

  ob::ScopedState<StateSpace> goal(space);
  goal->values[0] = cfg.goal_x;
  goal->values[1] = cfg.goal_y;
  goal->values[2] = cfg.goal_theta;

  ss.setStartAndGoalStates(start, goal);

  typename ManifoldT::Point goal_coords;
  goal_coords[0] = cfg.goal_x;
  goal_coords[1] = cfg.goal_y;
  goal_coords[2] = cfg.goal_theta;
  auto objective =
      std::make_shared<geodex::ompl_integration::GeodexOptimizationObjective<ManifoldT>>(
          ss.getSpaceInformation(), goal_coords);
  ss.setOptimizationObjective(objective);

  ss.setPlanner(make_planner(ss.getSpaceInformation()));

  // --- Planning ---
  auto t0 = std::chrono::steady_clock::now();
  ob::PlannerStatus status = ss.solve(solve_time);
  auto t1 = std::chrono::steady_clock::now();
  result.planning_time_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (status) {
    result.solved = true;
    auto& path = ss.getSolutionPath();

    // Extract raw path.
    for (const auto* state : path.getStates()) {
      const auto* s = state->as<StateType>();
      result.raw_path.push_back({s->values[0], s->values[1], s->values[2]});
    }

    // --- Post-planning: metric-aware smoothing ---
    auto t2 = std::chrono::steady_clock::now();
    {
      std::vector<typename ManifoldT::Point> waypoints;
      for (const auto* state : path.getStates()) {
        const auto* s = state->as<StateType>();
        typename ManifoldT::Point p;
        p[0] = s->values[0];
        p[1] = s->values[1];
        p[2] = s->values[2];
        waypoints.push_back(p);
      }

      auto validity = [&](const typename ManifoldT::Point& q) {
        for (const auto& obs : cfg.obstacles) {
          double dx = q[0] - obs.cx, dy = q[1] - obs.cy;
          if (dx * dx + dy * dy <= obs.r * obs.r) return false;
        }
        return true;
      };

      geodex::PathSmoothingSettings smooth_settings;
      smooth_settings.max_shortcut_attempts = 200;
      smooth_settings.lbfgs_target_segments = 64;
      smooth_settings.lbfgs_max_iterations = 200;

      auto smooth_result = geodex::smooth_path(manifold, validity, waypoints, smooth_settings);
      result.vertices_removed = smooth_result.vertices_removed;
      result.smooth_iterations = smooth_result.smooth_iterations;

      // Forward/backward check.
      for (std::size_t k = 0; k + 1 < smooth_result.path.size(); ++k) {
        auto v = manifold.log(smooth_result.path[k], smooth_result.path[k + 1]);
        if (v[0] < -1e-10) ++result.backward_segments;
      }

      for (const auto& q : smooth_result.path) {
        result.smoothed_path.push_back({q[0], q[1], q[2]});
      }
    }
    auto t3 = std::chrono::steady_clock::now();
    result.smoothing_time_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Extract tree.
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

    std::cout << label << ": solved in " << result.planning_time_ms << " ms"
              << " (tree=" << nv << ", raw=" << result.raw_path.size()
              << ", smoothed=" << result.smoothed_path.size()
              << ", shortcut=-" << result.vertices_removed
              << ", lbfgs=" << result.smooth_iterations
              << ", bwd=" << result.backward_segments
              << ", smooth=" << result.smoothing_time_ms << " ms)\n";
  } else {
    std::cerr << label << ": no solution found.\n";
  }

  return result;
}

void write_json(const std::string& filename, const std::vector<RunResult>& runs,
                const RunConfig& cfg) {
  std::ofstream out(filename);
  if (!out) {
    std::cerr << "Error: cannot open " << filename << "\n";
    return;
  }

  out << std::fixed << std::setprecision(8);
  out << "{\n";

  // Obstacles array
  out << "  \"obstacles\": [\n";
  for (size_t i = 0; i < cfg.obstacles.size(); ++i) {
    const auto& obs = cfg.obstacles[i];
    out << "    { \"center\": [" << obs.cx << ", " << obs.cy << "], \"radius\": " << obs.r << " }";
    if (i + 1 < cfg.obstacles.size()) out << ",";
    out << "\n";
  }
  out << "  ],\n";

  out << "  \"start\": [" << cfg.start_x << ", " << cfg.start_y << ", " << cfg.start_theta
      << "],\n";
  out << "  \"goal\": [" << cfg.goal_x << ", " << cfg.goal_y << ", " << cfg.goal_theta << "],\n";
  out << "  \"runs\": [\n";

  for (size_t r = 0; r < runs.size(); ++r) {
    const auto& run = runs[r];
    out << "    {\n";
    out << "      \"label\": \"" << run.label << "\",\n";
    out << "      \"metric_info\": \"" << run.metric_info << "\",\n";
    out << "      \"planning_time_ms\": " << run.planning_time_ms << ",\n";
    out << "      \"smoothing_time_ms\": " << run.smoothing_time_ms << ",\n";
    out << "      \"vertices_removed\": " << run.vertices_removed << ",\n";
    out << "      \"smooth_iterations\": " << run.smooth_iterations << ",\n";
    out << "      \"backward_segments\": " << run.backward_segments << ",\n";

    out << "      \"raw_path\": [\n";
    for (size_t i = 0; i < run.raw_path.size(); ++i) {
      out << "        [" << run.raw_path[i][0] << ", " << run.raw_path[i][1] << ", "
          << run.raw_path[i][2] << "]";
      if (i + 1 < run.raw_path.size()) out << ",";
      out << "\n";
    }
    out << "      ],\n";

    out << "      \"smoothed_path\": [\n";
    for (size_t i = 0; i < run.smoothed_path.size(); ++i) {
      out << "        [" << run.smoothed_path[i][0] << ", " << run.smoothed_path[i][1] << ", "
          << run.smoothed_path[i][2] << "]";
      if (i + 1 < run.smoothed_path.size()) out << ",";
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
  std::string output_file = "se2_planning.json";
  if (argc > 1) output_file = argv[1];

  RunConfig cfg;
  cfg.x_lo = 0.0;
  cfg.x_hi = 10.0;
  cfg.y_lo = 0.0;
  cfg.y_hi = 10.0;
  cfg.start_x = 1.0;
  cfg.start_y = 1.0;
  cfg.start_theta = 0.0;
  cfg.goal_x = 9.0;
  cfg.goal_y = 9.0;
  cfg.goal_theta = 0.0;
  cfg.obstacles = {
      {3.0, 3.0, 1.2},
      {7.0, 3.0, 1.0},
      {5.0, 6.5, 1.3},
      {2.0, 7.0, 0.8},
      {8.0, 7.5, 0.9},
  };

  auto planners = get_planners();
  std::vector<RunResult> runs;

  // Isotropic metric (omnidirectional)
  for (const auto& [pname, pfactory] : planners) {
    geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 0.5};
    geodex::SE2<> manifold{metric};
    runs.push_back(
        run_planner(manifold, "Isotropic (" + pname + ")", "w = (1, 1, 0.5)", pfactory, 1.0, cfg));
  }

  // Car-like metric with turning radius control
  for (const auto& [pname, pfactory] : planners) {
    auto metric = geodex::SE2LeftInvariantMetric::car_like(0.7);  // r_eff ≈ 0.7
    geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap> manifold{metric};
    runs.push_back(run_planner(manifold, "Car-like (" + pname + ")",
                               "r_turn=0.7, w=(1,100,0.49)", pfactory, 1.0, cfg));
  }

  write_json(output_file, runs, cfg);

  return 0;
}

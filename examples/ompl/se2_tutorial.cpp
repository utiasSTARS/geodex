/// @file se2_tutorial.cpp
/// @brief SE(2) planning tutorial example: holonomic, differential-drive, and car-like
///        robots with footprint collision checking, clearance metrics, and path smoothing.
///
/// This consolidated example accompanies the "SE(2) Motion Planning" tutorial.
/// It supports five scenarios via the --scenario flag:
///   - holonomic:      circular robot, isotropic metric, grid SDF
///   - holo_clearance: circular robot, isotropic + SDFConformalMetric
///   - diff_drive:     rectangular robot, anisotropic metric, footprint checker
///   - diff_clearance: rectangular robot, anisotropic + SDFConformalMetric
///   - parking:        car-like robot, rectangle obstacles, parallel parking
///
/// Usage:
///   se2_tutorial <dist_map.txt> --scenario=holonomic -o output.json [--time=1.0] [--seed=42]
///   se2_tutorial dummy --scenario=parking -o parking.json

#include <cmath>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <ompl/base/PlannerData.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

#include "geodex/algorithm/path_smoothing.hpp"
#include "geodex/collision/collision.hpp"
#include "geodex/integration/ompl/geodex_optimization_objective.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/integration/ompl/validity_checker.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/metrics/clearance.hpp"

namespace ob = ompl::base;
namespace og = ompl::geometric;

using DistanceGrid = geodex::collision::DistanceGrid;
using GridSDF = geodex::collision::GridSDF;
using RectObstacle = geodex::collision::RectObstacle;
using RectSmoothSDF = geodex::collision::RectSmoothSDF;

// ---------------------------------------------------------------------------
// Run result
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Planner runner (generic over manifold and validity function)
// ---------------------------------------------------------------------------

using PlannerFactory = std::function<ob::PlannerPtr(const ob::SpaceInformationPtr&)>;

// Default StateSpace interpolation settings used during OMPL tree expansion.
// Scenarios can override this per-experiment by passing a custom value as the
// last argument to run_planner.
inline geodex::InterpolationSettings default_state_interp() {
  geodex::InterpolationSettings s;
  s.step_size = 1.0;
  s.convergence_tol = 1e-2;
  s.convergence_rel = 1e-2;
  s.max_steps = 20;
  s.force_log_direction = true;
  return s;
}

template <typename ManifoldT, typename ValidityFn>
RunResult run_planner(ManifoldT manifold, const std::string& label, const std::string& metric_info,
                      const PlannerFactory& make_planner, const double solve_time,
                      const ob::RealVectorBounds& bounds,
                      const std::array<double, 3>& start_pose,
                      const std::array<double, 3>& goal_pose, ValidityFn validity,
                      geodex::algorithm::PathSmoothingSettings smooth_settings = {},
                      const double range = 0.0, const double collision_resolution = 0.1,
                      geodex::InterpolationSettings state_space_interp = default_state_interp(),
                      const double rewire_factor = 0.5) {
  using StateSpace = geodex::integration::ompl::GeodexStateSpace<ManifoldT>;
  using StateType = geodex::integration::ompl::GeodexState<ManifoldT>;

  RunResult result;
  result.label = label;
  result.metric_info = metric_info;

  auto space = std::make_shared<StateSpace>(manifold, bounds);
  space->setCollisionResolution(collision_resolution);
  space->setInterpolationSettings(state_space_interp);

  og::SimpleSetup ss(space);
  ss.setStateValidityChecker(geodex::integration::ompl::make_validity_checker<ManifoldT>(
      ss.getSpaceInformation(), validity));

  ob::ScopedState<StateSpace> start(space);
  start->values[0] = start_pose[0];
  start->values[1] = start_pose[1];
  start->values[2] = start_pose[2];

  ob::ScopedState<StateSpace> goal(space);
  goal->values[0] = goal_pose[0];
  goal->values[1] = goal_pose[1];
  goal->values[2] = goal_pose[2];
  ss.setStartAndGoalStates(start, goal, 0.5);

  typename ManifoldT::Point goal_coords;
  goal_coords[0] = goal_pose[0];
  goal_coords[1] = goal_pose[1];
  goal_coords[2] = goal_pose[2];
  auto objective =
      std::make_shared<geodex::integration::ompl::GeodexOptimizationObjective<ManifoldT>>(
          ss.getSpaceInformation(), goal_coords);
  ss.setOptimizationObjective(objective);

  auto planner = make_planner(ss.getSpaceInformation());
  if (range > 0.0) {
    planner->params().setParam("range", std::to_string(range));
  }
  planner->params().setParam("rewire_factor", std::to_string(rewire_factor));
  ss.setPlanner(planner);

  // --- Planning ---
  auto t0 = std::chrono::steady_clock::now();
  ob::PlannerStatus status = ss.solve(solve_time);
  auto t1 = std::chrono::steady_clock::now();
  result.planning_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  const bool is_exact = (status == ob::PlannerStatus::EXACT_SOLUTION);
  const bool is_approx = (status == ob::PlannerStatus::APPROXIMATE_SOLUTION);
  if (is_exact || is_approx) {
    result.solved = is_exact;
    auto& path = ss.getSolutionPath();

    // Extract planner waypoints.
    std::vector<typename ManifoldT::Point> waypoints;
    for (const auto* state : path.getStates()) {
      const auto* s = state->as<StateType>();
      typename ManifoldT::Point p;
      p[0] = s->values[0];
      p[1] = s->values[1];
      p[2] = s->values[2];
      waypoints.push_back(p);
    }

    // Densify raw path via discrete_geodesic with log direction.
    geodex::InterpolationSettings dense_interp;
    dense_interp.step_size = 0.3;
    dense_interp.convergence_tol = 1e-3;
    dense_interp.max_steps = 50;
    dense_interp.force_log_direction = true;
    for (std::size_t i = 0; i < waypoints.size(); ++i) {
      result.raw_path.push_back({waypoints[i][0], waypoints[i][1], waypoints[i][2]});
      if (i + 1 < waypoints.size()) {
        auto geo =
            geodex::discrete_geodesic(manifold, waypoints[i], waypoints[i + 1], dense_interp);
        for (std::size_t k = 1; k + 1 < geo.path.size(); ++k) {
          result.raw_path.push_back({geo.path[k][0], geo.path[k][1], geo.path[k][2]});
        }
      }
    }

    // --- Post-planning: metric-aware smoothing ---
    auto t2 = std::chrono::steady_clock::now();
    {
      auto smooth_result =
          geodex::algorithm::smooth_path(manifold, validity, waypoints, smooth_settings);
      result.vertices_removed = smooth_result.vertices_removed;
      result.smooth_iterations = smooth_result.smooth_iterations;

      // Check for backward segments.
      for (std::size_t k = 0; k + 1 < smooth_result.path.size(); ++k) {
        auto v = manifold.log(smooth_result.path[k], smooth_result.path[k + 1]);
        if (v[0] < -1e-10) ++result.backward_segments;
      }

      // Densify smoothed path.
      for (std::size_t i = 0; i < smooth_result.path.size(); ++i) {
        result.smoothed_path.push_back(
            {smooth_result.path[i][0], smooth_result.path[i][1], smooth_result.path[i][2]});
        if (i + 1 < smooth_result.path.size()) {
          auto geo = geodex::discrete_geodesic(manifold, smooth_result.path[i],
                                               smooth_result.path[i + 1], dense_interp);
          for (std::size_t k = 1; k + 1 < geo.path.size(); ++k) {
            result.smoothed_path.push_back({geo.path[k][0], geo.path[k][1], geo.path[k][2]});
          }
        }
      }
    }
    auto t3 = std::chrono::steady_clock::now();
    result.smoothing_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Extract tree.
    ob::PlannerData pdata(ss.getSpaceInformation());
    ss.getPlannerData(pdata);
    const unsigned int nv = pdata.numVertices();
    if (nv < 20000) {
      for (unsigned int i = 0; i < nv; ++i) {
        std::vector<unsigned int> edges;
        pdata.getEdges(i, edges);
        const auto* vi = pdata.getVertex(i).getState()->as<StateType>();
        for (unsigned int j : edges) {
          const auto* vj = pdata.getVertex(j).getState()->as<StateType>();
          result.tree.push_back({{{vi->values[0], vi->values[1], vi->values[2]},
                                  {vj->values[0], vj->values[1], vj->values[2]}}});
        }
      }
    }

    std::cout << label << ": " << (is_exact ? "EXACT" : "APPROX") << " in "
              << result.planning_time_ms << " ms"
              << " (tree=" << nv << ", raw=" << result.raw_path.size()
              << ", smoothed=" << result.smoothed_path.size() << ", shortcut=-"
              << result.vertices_removed << ", lbfgs=" << result.smooth_iterations
              << ", bwd=" << result.backward_segments << ", smooth=" << result.smoothing_time_ms
              << " ms)\n";
  } else {
    std::cerr << label << ": FAILED (no solution found).\n";
  }

  return result;
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

struct TutorialOutput {
  std::string scenario;
  std::array<double, 3> start{}, goal{};

  // Robot description.
  std::string robot_type;     // "circle" or "rectangle"
  double robot_radius = 0.0;  // for circle
  double robot_hl = 0.0, robot_hw = 0.0;  // for rectangle

  // Grid-based environment (corridor scenarios).
  const DistanceGrid* grid = nullptr;
  std::string map_file;

  // Rectangle obstacles (parking scenario).
  std::vector<RectObstacle> rect_obstacles;

  // Run results.
  std::vector<RunResult> runs;

  // Optional conformal factor grid (for heatmap visualization).
  std::vector<double> conformal_values;
  int conformal_w = 0, conformal_h = 0;
  double conformal_res = 0.0;
};

void write_json(const std::string& filename, const TutorialOutput& out) {
  std::ofstream f(filename);
  if (!f) {
    std::cerr << "Error: cannot open " << filename << "\n";
    return;
  }

  f << std::fixed << std::setprecision(8);
  f << "{\n";
  f << "  \"scenario\": \"" << out.scenario << "\",\n";
  f << "  \"start\": [" << out.start[0] << ", " << out.start[1] << ", " << out.start[2] << "],\n";
  f << "  \"goal\": [" << out.goal[0] << ", " << out.goal[1] << ", " << out.goal[2] << "],\n";

  // Robot.
  f << "  \"robot\": { \"type\": \"" << out.robot_type << "\"";
  if (out.robot_type == "circle") {
    f << ", \"radius\": " << out.robot_radius;
  } else {
    f << ", \"half_length\": " << out.robot_hl << ", \"half_width\": " << out.robot_hw;
  }
  f << " },\n";

  // Map (grid-based environment).
  if (out.grid) {
    f << "  \"map\": { \"width\": " << out.grid->width() << ", \"height\": " << out.grid->height()
      << ", \"resolution\": " << out.grid->resolution() << ", \"file\": \"" << out.map_file
      << "\" },\n";
  }

  // Rectangle obstacles.
  if (!out.rect_obstacles.empty()) {
    f << "  \"rect_obstacles\": [\n";
    for (std::size_t i = 0; i < out.rect_obstacles.size(); ++i) {
      const auto& o = out.rect_obstacles[i];
      f << "    {\"center\": [" << o.cx << ", " << o.cy << "], \"theta\": " << o.theta
        << ", \"half_length\": " << o.half_length << ", \"half_width\": " << o.half_width << "}";
      if (i + 1 < out.rect_obstacles.size()) f << ",";
      f << "\n";
    }
    f << "  ],\n";
  }

  // Conformal factor grid.
  if (!out.conformal_values.empty()) {
    f << "  \"conformal_grid\": { \"width\": " << out.conformal_w
      << ", \"height\": " << out.conformal_h << ", \"resolution\": " << out.conformal_res
      << ", \"values\": [";
    for (std::size_t i = 0; i < out.conformal_values.size(); ++i) {
      if (i > 0) f << ", ";
      f << out.conformal_values[i];
    }
    f << "] },\n";
  }

  // Runs.
  f << "  \"runs\": [\n";
  for (std::size_t r = 0; r < out.runs.size(); ++r) {
    const auto& run = out.runs[r];
    f << "    {\n";
    f << "      \"label\": \"" << run.label << "\",\n";
    f << "      \"metric_info\": \"" << run.metric_info << "\",\n";
    f << "      \"solved\": " << (run.solved ? "true" : "false") << ",\n";
    f << "      \"planning_time_ms\": " << run.planning_time_ms << ",\n";
    f << "      \"smoothing_time_ms\": " << run.smoothing_time_ms << ",\n";
    f << "      \"vertices_removed\": " << run.vertices_removed << ",\n";
    f << "      \"smooth_iterations\": " << run.smooth_iterations << ",\n";
    f << "      \"backward_segments\": " << run.backward_segments << ",\n";

    f << "      \"raw_path\": [\n";
    for (std::size_t i = 0; i < run.raw_path.size(); ++i) {
      f << "        [" << run.raw_path[i][0] << ", " << run.raw_path[i][1] << ", "
        << run.raw_path[i][2] << "]";
      if (i + 1 < run.raw_path.size()) f << ",";
      f << "\n";
    }
    f << "      ],\n";

    f << "      \"smoothed_path\": [\n";
    for (std::size_t i = 0; i < run.smoothed_path.size(); ++i) {
      f << "        [" << run.smoothed_path[i][0] << ", " << run.smoothed_path[i][1] << ", "
        << run.smoothed_path[i][2] << "]";
      if (i + 1 < run.smoothed_path.size()) f << ",";
      f << "\n";
    }
    f << "      ],\n";

    f << "      \"tree\": [\n";
    for (std::size_t i = 0; i < run.tree.size(); ++i) {
      f << "        [[" << run.tree[i][0][0] << ", " << run.tree[i][0][1] << ", "
        << run.tree[i][0][2] << "], [" << run.tree[i][1][0] << ", " << run.tree[i][1][1] << ", "
        << run.tree[i][1][2] << "]]";
      if (i + 1 < run.tree.size()) f << ",";
      f << "\n";
    }
    f << "      ]\n";

    f << "    }";
    if (r + 1 < out.runs.size()) f << ",";
    f << "\n";
  }

  f << "  ]\n";
  f << "}\n";

  std::cout << "Wrote " << filename << "\n";
}

// ---------------------------------------------------------------------------
// Helper: sample conformal factor grid
// ---------------------------------------------------------------------------

template <typename SDFFn>
std::vector<double> sample_conformal_grid(const SDFFn& sdf, const double kappa, const double beta,
                                          const int width, const int height,
                                          const double resolution) {
  std::vector<double> values(static_cast<std::size_t>(width) * height);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const double x = (col + 0.5) * resolution;
      const double y = (row + 0.5) * resolution;
      const Eigen::Vector3d q{x, y, 0.0};
      const double d = sdf(q);
      values[static_cast<std::size_t>(row) * width + col] = 1.0 + kappa * std::exp(-beta * d);
    }
  }
  return values;
}

// ---------------------------------------------------------------------------
// Helper: shared bounds and planner factory
// ---------------------------------------------------------------------------

ob::RealVectorBounds make_bounds(const double world_w, const double world_h) {
  ob::RealVectorBounds bounds(3);
  bounds.setLow(0, 0.0);
  bounds.setHigh(0, world_w);
  bounds.setLow(1, 0.0);
  bounds.setHigh(1, world_h);
  bounds.setLow(2, -std::numbers::pi);
  bounds.setHigh(2, std::numbers::pi);
  return bounds;
}

PlannerFactory make_irrt() {
  return [](const ob::SpaceInformationPtr& si) -> ob::PlannerPtr {
    return std::make_shared<og::InformedRRTstar>(si);
  };
}

PlannerFactory make_rrt() {
  return [](const ob::SpaceInformationPtr& si) -> ob::PlannerPtr {
    return std::make_shared<og::RRTstar>(si);
  };
}

// ---------------------------------------------------------------------------
// Scenario: holonomic circular robot (baseline, no clearance)
// ---------------------------------------------------------------------------

TutorialOutput run_holonomic(const DistanceGrid& grid, const std::string& map_file,
                             const double solve_time) {
  constexpr double robot_radius = 0.3;
  // Safety margin beyond the robot radius. Pushes the tree (and the raw path)
  // away from walls so L-BFGS has room to round corners.
  constexpr double safety_margin = 0.10;
  const double world_w = grid.width() * grid.resolution();
  const double world_h = grid.height() * grid.resolution();

  TutorialOutput out;
  out.scenario = "holonomic";
  out.start = {2.0, 5.0, 0.0};
  out.goal = {12.0, 6.0, -std::numbers::pi / 2.0};
  out.robot_type = "circle";
  out.robot_radius = robot_radius;
  out.grid = &grid;
  out.map_file = map_file;

  auto validity = [&grid, robot_radius](const auto& q) {
    return grid.distance_at(q[0], q[1]) > robot_radius + safety_margin;
  };

  geodex::SE2LeftInvariantMetric metric{1.0, 1.0, 1.0};
  geodex::SE2<> manifold{metric, geodex::SE2ExponentialMap{},
                         Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                         Eigen::Vector3d(world_w, world_h, std::numbers::pi)};

  // Per-scenario StateSpace interp. Fine step_size gives OMPL metric-consistent
  // distances during tree growth so the raw path already tracks the geodesic.
  geodex::InterpolationSettings state_interp;
  state_interp.step_size = 0.2;
  state_interp.convergence_tol = 1e-3;
  state_interp.convergence_rel = 1e-3;
  state_interp.max_steps = 80;
  state_interp.force_log_direction = true;

  constexpr double planner_range = 4;
  constexpr double rewire_factor = 1.1;

  auto bounds = make_bounds(world_w, world_h);
  out.runs.push_back(run_planner(manifold, "Holonomic (isotropic)", "w=(1,1,1)", make_irrt(),
                                 solve_time, bounds, out.start, out.goal, validity, {},
                                 planner_range, grid.resolution(), state_interp, rewire_factor));
  return out;
}

// ---------------------------------------------------------------------------
// Scenario: holonomic + clearance metric
// ---------------------------------------------------------------------------

TutorialOutput run_holo_clearance(const DistanceGrid& grid, const std::string& map_file,
                                  const double solve_time) {
  constexpr double robot_radius = 0.3;
  constexpr double safety_margin = 0.10;
  constexpr double kappa = 1.5, beta = 1.5;
  const double world_w = grid.width() * grid.resolution();
  const double world_h = grid.height() * grid.resolution();

  TutorialOutput out;
  out.scenario = "holo_clearance";
  out.start = {2.0, 5.0, 0.0};
  out.goal = {12.0, 6.0, -std::numbers::pi / 2.0};
  out.robot_type = "circle";
  out.robot_radius = robot_radius;
  out.grid = &grid;
  out.map_file = map_file;

  auto validity = [&grid, robot_radius](const auto& q) {
    return grid.distance_at(q[0], q[1]) > robot_radius + safety_margin;
  };

  geodex::SE2LeftInvariantMetric base_metric{1.0, 1.0, 1.0};
  geodex::collision::InflatedSDF inflated_sdf{GridSDF{&grid}, robot_radius};
  geodex::SDFConformalMetric clearance_metric{base_metric, inflated_sdf, kappa, beta};

  geodex::SE2<> se2{base_metric, geodex::SE2ExponentialMap{},
                    Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                    Eigen::Vector3d(world_w, world_h, std::numbers::pi)};
  geodex::ConfigurationSpace cspace{se2, clearance_metric};

  // Per-scenario StateSpace interp. Clearance metric inflates distance near
  // walls so we need fine geodesic stepping to track the warped geometry.
  geodex::InterpolationSettings state_interp;
  state_interp.step_size = 0.2;
  state_interp.convergence_tol = 1e-3;
  state_interp.convergence_rel = 1e-3;
  state_interp.max_steps = 80;
  state_interp.force_log_direction = true;

  constexpr double planner_range = 3.0;
  constexpr double rewire_factor = 1.1;

  auto bounds = make_bounds(world_w, world_h);
  out.runs.push_back(run_planner(cspace, "Holonomic + clearance", "k=1.5 b=1.5", make_irrt(),
                                 solve_time, bounds, out.start, out.goal, validity, {},
                                 planner_range, grid.resolution(), state_interp, rewire_factor));

  // Sample conformal factor grid for heatmap visualization.
  out.conformal_w = grid.width();
  out.conformal_h = grid.height();
  out.conformal_res = grid.resolution();
  out.conformal_values =
      sample_conformal_grid(inflated_sdf, kappa, beta, grid.width(), grid.height(),
                            grid.resolution());

  return out;
}

// ---------------------------------------------------------------------------
// Scenario: differential-drive rectangular robot
// ---------------------------------------------------------------------------

TutorialOutput run_diff_drive(const DistanceGrid& grid, const std::string& map_file,
                              const double solve_time) {
  constexpr double robot_hl = 0.35, robot_hw = 0.25;
  constexpr double safety_margin = 0.10;
  const double world_w = grid.width() * grid.resolution();
  const double world_h = grid.height() * grid.resolution();

  TutorialOutput out;
  out.scenario = "diff_drive";
  out.start = {2.0, 5.0, 0.0};
  out.goal = {12.0, 6.0, -std::numbers::pi / 2.0};
  out.robot_type = "rectangle";
  out.robot_hl = robot_hl;
  out.robot_hw = robot_hw;
  out.grid = &grid;
  out.map_file = map_file;

  auto footprint = geodex::collision::PolygonFootprint::rectangle(robot_hl, robot_hw, 6);
  geodex::collision::FootprintGridChecker checker{&grid, footprint, safety_margin};
  auto validity = [&checker](const auto& q) { return checker.is_valid(q); };

  geodex::SE2LeftInvariantMetric metric{1.0, 10.0, 1.0};
  geodex::SE2<> manifold{metric, geodex::SE2ExponentialMap{},
                         Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                         Eigen::Vector3d(world_w, world_h, std::numbers::pi)};

  // Per-scenario StateSpace interp. Anisotropic metric (w_y = 10) means a
  // metric step of 0.3 is only ~0.095 m laterally, so finer stepping is needed
  // to track the true geodesic during tree growth.
  geodex::InterpolationSettings state_interp;
  state_interp.step_size = 0.3;
  state_interp.convergence_tol = 1e-3;
  state_interp.convergence_rel = 1e-3;
  state_interp.max_steps = 80;
  state_interp.force_log_direction = true;

  constexpr double planner_range = 4.0;
  constexpr double rewire_factor = 1.1;

  auto bounds = make_bounds(world_w, world_h);
  out.runs.push_back(run_planner(manifold, "Diff-drive (anisotropic)", "w=(1,10,1)", make_irrt(),
                                 solve_time, bounds, out.start, out.goal, validity, {},
                                 planner_range, grid.resolution(), state_interp, rewire_factor));
  return out;
}

// ---------------------------------------------------------------------------
// Scenario: differential-drive + clearance metric
// ---------------------------------------------------------------------------

TutorialOutput run_diff_clearance(const DistanceGrid& grid, const std::string& map_file,
                                  const double solve_time) {
  constexpr double robot_hl = 0.35, robot_hw = 0.25;
  constexpr double safety_margin = 0.05;
  constexpr double kappa = 1.5, beta = 1.5;
  const double world_w = grid.width() * grid.resolution();
  const double world_h = grid.height() * grid.resolution();

  TutorialOutput out;
  out.scenario = "diff_clearance";
  out.start = {2.0, 5.0, 0.0};
  out.goal = {12.0, 6.0, -std::numbers::pi / 2.0};
  out.robot_type = "rectangle";
  out.robot_hl = robot_hl;
  out.robot_hw = robot_hw;
  out.grid = &grid;
  out.map_file = map_file;

  auto footprint = geodex::collision::PolygonFootprint::rectangle(robot_hl, robot_hw, 6);
  geodex::collision::FootprintGridChecker checker{&grid, footprint, safety_margin};
  auto validity = [&checker](const auto& q) { return checker.is_valid(q); };

  // Use FootprintGridChecker as continuous SDF for the clearance metric.
  // operator() returns min grid distance across perimeter samples minus safety margin.
  geodex::SE2LeftInvariantMetric base_metric{1.0, 10.0, 1.0};
  geodex::SDFConformalMetric clearance_metric{base_metric, checker, kappa, beta};

  geodex::SE2<> se2{base_metric, geodex::SE2ExponentialMap{},
                    Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
                    Eigen::Vector3d(world_w, world_h, std::numbers::pi)};
  geodex::ConfigurationSpace cspace{se2, clearance_metric};

  // Per-scenario StateSpace interp.
  geodex::InterpolationSettings state_interp;
  state_interp.step_size = 0.3;
  state_interp.convergence_tol = 1e-3;
  state_interp.convergence_rel = 1e-3;
  state_interp.max_steps = 60;
  state_interp.force_log_direction = true;

  constexpr double planner_range = 5.0;
  constexpr double rewire_factor = 1.1;

  // RRT* (not InformedRRT*) — informed heuristic is too loose for
  // anisotropic+clearance on the Willow corridor.
  auto bounds = make_bounds(world_w, world_h);
  out.runs.push_back(run_planner(cspace, "Diff-drive + clearance", "w=(1,10,1) k=1.5 b=1.5",
                                 make_rrt(), solve_time, bounds, out.start, out.goal, validity,
                                 {}, planner_range, grid.resolution(), state_interp,
                                 rewire_factor));
  return out;
}

// ---------------------------------------------------------------------------
// Scenario: car-like parallel parking
// ---------------------------------------------------------------------------

TutorialOutput run_parking(const double solve_time) {
  constexpr double car_hl = 2.25, car_hw = 0.9;  // 4.5 x 1.8 m sedan
  constexpr double world_w = 30.0, world_h = 12.0;
  constexpr double turning_radius = 1.5, lateral_penalty = 20.0;
  constexpr double kappa = 8.0, beta = 3.0;

  std::vector<RectObstacle> obstacles = {
      {5.0, 1.35, 0.0, car_hl, car_hw},       // parked car 1
      {10.0, 1.35, 0.0, car_hl, car_hw},      // parked car 2
      {21.0, 1.35, 0.0, car_hl, car_hw},      // parked car 3
      {26.0, 1.35, 0.0, car_hl, car_hw},      // parked car 4
      {15.0, -0.05, 0.0, 15.0, 0.05},         // curb
      {15.0, 10.05, 0.0, 15.0, 0.05},         // sidewalk
  };

  TutorialOutput out;
  out.scenario = "parking";
  out.start = {25.0, 6.0, std::numbers::pi};
  out.goal = {15.5, 1.35, 0.0};
  out.robot_type = "rectangle";
  out.robot_hl = car_hl;
  out.robot_hw = car_hw;
  out.rect_obstacles = obstacles;

  auto bounds = make_bounds(world_w, world_h);

  // SAT-based polygon collision checking.
  auto validity = [&obstacles, car_hl, car_hw, &bounds](const auto& q) {
    const double x = q[0], y = q[1], theta = q[2];
    if (x - car_hl < bounds.low[0] || x + car_hl > bounds.high[0] ||
        y - car_hw < bounds.low[1] || y + car_hw > bounds.high[1])
      return false;
    RectObstacle ego{x, y, theta, car_hl, car_hw};
    for (const auto& obs : obstacles) {
      if (geodex::collision::rects_overlap(ego, obs)) return false;
    }
    return true;
  };

  auto base_metric = geodex::SE2LeftInvariantMetric::car_like(turning_radius, lateral_penalty);
  RectSmoothSDF sdf{obstacles, 20.0, car_hw};
  geodex::SDFConformalMetric clearance_metric{base_metric, sdf, kappa, beta};

  geodex::SE2<geodex::SE2LeftInvariantMetric, geodex::SE2ExponentialMap> se2{
      geodex::SE2LeftInvariantMetric::car_like(turning_radius, lateral_penalty),
      geodex::SE2ExponentialMap{}, Eigen::Vector3d(0.0, 0.0, -std::numbers::pi),
      Eigen::Vector3d(world_w, world_h, std::numbers::pi)};
  geodex::ConfigurationSpace cspace{se2, clearance_metric};

  geodex::algorithm::PathSmoothingSettings car_smooth;
  car_smooth.interp.step_size = 0.02;
  car_smooth.interp.max_steps = 2000;
  car_smooth.collision_resolution = 0.1;

  // Per-scenario StateSpace interp.
  geodex::InterpolationSettings state_interp;
  state_interp.step_size = 0.5;
  state_interp.convergence_tol = 1e-3;
  state_interp.convergence_rel = 1e-3;
  state_interp.max_steps = 80;
  state_interp.force_log_direction = true;

  out.runs.push_back(run_planner(cspace, "Car-like parallel parking", "r=1.5 lp=20 k=8 b=3",
                                 make_rrt(), solve_time, bounds, out.start, out.goal, validity,
                                 car_smooth, 3.0, 0.1, state_interp, 1.1));
  return out;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <dist_map.txt> --scenario=<name> [-o output.json] [--time=<s>] [--seed=<n>]\n"
              << "\nScenarios: holonomic, holo_clearance, diff_drive, diff_clearance, parking\n";
    return 1;
  }

  std::string dist_map_file = argv[1];
  std::string output_file;
  std::string scenario = "holonomic";
  double time_override = 0.0;
  int seed = -1;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.starts_with("--scenario=")) {
      scenario = arg.substr(11);
    } else if (arg.starts_with("-o")) {
      if (arg.size() > 2)
        output_file = arg.substr(2);
      else if (i + 1 < argc)
        output_file = argv[++i];
    } else if (arg.starts_with("--time=")) {
      time_override = std::stod(arg.substr(7));
    } else if (arg.starts_with("--seed=")) {
      seed = std::stoi(arg.substr(7));
    } else {
      output_file = arg;
    }
  }

  if (output_file.empty()) {
    output_file = "se2_tutorial_" + scenario + ".json";
  }

  // Set OMPL random seed for reproducibility.
  if (seed >= 0) {
    ompl::RNG::setSeed(static_cast<unsigned int>(seed));
  }

  // Default planning times per scenario.
  const double solve_time = (time_override > 0.0) ? time_override
                            : (scenario == "parking") ? 2.0
                                                      : 1.0;

  TutorialOutput result;

  if (scenario == "parking") {
    result = run_parking(solve_time);
  } else {
    // All non-parking scenarios need the distance grid.
    DistanceGrid grid;
    if (!grid.load(dist_map_file)) {
      std::cerr << "Error: could not load distance grid: " << dist_map_file << "\n";
      return 1;
    }

    std::cout << "Loaded grid: " << grid.width() << "x" << grid.height()
              << " (" << grid.width() * grid.resolution() << " x "
              << grid.height() * grid.resolution() << " m)\n";

    if (scenario == "holonomic") {
      result = run_holonomic(grid, dist_map_file, solve_time);
    } else if (scenario == "holo_clearance") {
      result = run_holo_clearance(grid, dist_map_file, solve_time);
    } else if (scenario == "diff_drive") {
      result = run_diff_drive(grid, dist_map_file, solve_time);
    } else if (scenario == "diff_clearance") {
      result = run_diff_clearance(grid, dist_map_file, solve_time);
    } else {
      std::cerr << "Unknown scenario: " << scenario << "\n";
      return 1;
    }
  }

  write_json(output_file, result);

  return 0;
}

#include <cmath>

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <ompl/base/PlannerData.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

#include "geodex/algorithm/path_smoothing.hpp"
#include "geodex/integration/ompl/geodex_optimization_objective.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/metrics/jacobi.hpp"
#include "geodex/metrics/kinetic_energy.hpp"

namespace ob = ompl::base;
namespace og = ompl::geometric;

struct RunResult {
  std::string label;
  std::string metric_info;
  std::vector<std::array<double, 2>> raw_path;
  std::vector<std::array<double, 2>> smoothed_path;
  std::vector<std::array<std::array<double, 2>, 2>> tree;
  bool solved = false;
};

struct RunConfig {
  double lo = -M_PI, hi = M_PI;
  double start_x = 0.0, start_y = 0.0;
  double goal_x = 0.0, goal_y = 0.0;
};

using PlannerFactory = std::function<ob::PlannerPtr(const ob::SpaceInformationPtr&)>;

/// Run RRT* on a given manifold.
template <typename ManifoldT>
RunResult run_planner(ManifoldT manifold, const std::string& label, const std::string& metric_info,
                      const PlannerFactory& make_planner, double solve_time, const RunConfig& cfg,
                      double range = 0.0) {
  using StateSpace = geodex::integration::ompl::GeodexStateSpace<ManifoldT>;
  using StateType = geodex::integration::ompl::GeodexState<ManifoldT>;

  RunResult result;
  result.label = label;
  result.metric_info = metric_info;

  ob::RealVectorBounds bounds(2);
  bounds.setLow(cfg.lo);
  bounds.setHigh(cfg.hi);
  auto space = std::make_shared<StateSpace>(manifold, bounds);

  og::SimpleSetup ss(space);

  // Always-valid checker (no obstacles)
  ss.setStateValidityChecker([](const ob::State*) { return true; });

  ob::ScopedState<StateSpace> start(space);
  start->values[0] = cfg.start_x;
  start->values[1] = cfg.start_y;

  ob::ScopedState<StateSpace> goal(space);
  goal->values[0] = cfg.goal_x;
  goal->values[1] = cfg.goal_y;

  ss.setStartAndGoalStates(start, goal);

  // Set optimization objective
  typename ManifoldT::Point goal_coords;
  goal_coords[0] = cfg.goal_x;
  goal_coords[1] = cfg.goal_y;
  auto objective =
      std::make_shared<geodex::integration::ompl::GeodexOptimizationObjective<ManifoldT>>(
          ss.getSpaceInformation(), goal_coords);
  ss.setOptimizationObjective(objective);

  auto planner = make_planner(ss.getSpaceInformation());
  if (range > 0.0) {
    if (auto* rrt = dynamic_cast<og::RRTstar*>(planner.get())) {
      rrt->setRange(range);
    }
  }
  ss.setPlanner(planner);

  ob::PlannerStatus status = ss.solve(solve_time);

  if (status) {
    result.solved = true;
    auto& path = ss.getSolutionPath();

    using Point = typename ManifoldT::Point;
    std::vector<Point> raw_points;
    raw_points.reserve(path.getStateCount());
    for (const auto* state : path.getStates()) {
      const auto* s = state->as<StateType>();
      Point p;
      p[0] = s->values[0];
      p[1] = s->values[1];
      raw_points.push_back(p);
      result.raw_path.push_back({s->values[0], s->values[1]});
    }

    if (raw_points.size() >= 3) {
      geodex::algorithm::PathSmoothingSettings settings;
      settings.max_shortcut_attempts = 0;
      auto smoothed = geodex::algorithm::smooth_path(
          manifold, [](const Point&) { return true; }, raw_points, settings);
      for (const auto& p : smoothed.path) {
        result.smoothed_path.push_back({p[0], p[1]});
      }
    } else {
      result.smoothed_path = result.raw_path;
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
        result.tree.push_back({{{vi->values[0], vi->values[1]}, {vj->values[0], vj->values[1]}}});
      }
    }

    std::cout << label << ": found solution (" << nv << " tree vertices, " << path.getStateCount()
              << " path waypoints)\n";
  } else {
    std::cerr << label << ": no solution found.\n";
  }

  return result;
}

/// 2-link planar arm mass matrix (matching tutorial parameters).
struct PlanarArmMassMatrix {
  double l1 = 1.0, l2 = 1.0, m1 = 1.0, m2 = 1.0;
  double lc1 = 0.5, lc2 = 0.5;
  double I1 = 1.0 / 12.0, I2 = 1.0 / 12.0;

  Eigen::Matrix2d operator()(const Eigen::Vector2d& q) const {
    double c2 = std::cos(q[1]);
    double h = l1 * lc2 * c2;

    Eigen::Matrix2d M;
    M(0, 0) = I1 + I2 + m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * h);
    M(0, 1) = I2 + m2 * (lc2 * lc2 + h);
    M(1, 0) = M(0, 1);
    M(1, 1) = I2 + m2 * lc2 * lc2;
    return M;
  }
};

void write_json(const std::string& filename, const std::vector<RunResult>& runs,
                const RunConfig& cfg, const PlanarArmMassMatrix& arm, double H) {
  std::ofstream out(filename);
  if (!out) {
    std::cerr << "Error: cannot open " << filename << "\n";
    return;
  }

  out << std::fixed << std::setprecision(8);
  out << "{\n";
  out << "  \"start\": [" << cfg.start_x << ", " << cfg.start_y << "],\n";
  out << "  \"goal\": [" << cfg.goal_x << ", " << cfg.goal_y << "],\n";
  out << "  \"arm\": {\n";
  out << "    \"l1\": " << arm.l1 << ", \"l2\": " << arm.l2 << ",\n";
  out << "    \"m1\": " << arm.m1 << ", \"m2\": " << arm.m2 << ",\n";
  out << "    \"lc1\": " << arm.lc1 << ", \"lc2\": " << arm.lc2 << ",\n";
  out << "    \"I1\": " << arm.I1 << ", \"I2\": " << arm.I2 << ",\n";
  out << "    \"g\": 9.81\n";
  out << "  },\n";
  out << "  \"H\": " << H << ",\n";
  out << "  \"runs\": [\n";

  for (size_t r = 0; r < runs.size(); ++r) {
    const auto& run = runs[r];
    out << "    {\n";
    out << "      \"label\": \"" << run.label << "\",\n";
    out << "      \"metric_info\": \"" << run.metric_info << "\",\n";

    out << "      \"raw_path\": [\n";
    for (size_t i = 0; i < run.raw_path.size(); ++i) {
      out << "        [" << run.raw_path[i][0] << ", " << run.raw_path[i][1] << "]";
      if (i + 1 < run.raw_path.size()) out << ",";
      out << "\n";
    }
    out << "      ],\n";

    out << "      \"smoothed_path\": [\n";
    for (size_t i = 0; i < run.smoothed_path.size(); ++i) {
      out << "        [" << run.smoothed_path[i][0] << ", " << run.smoothed_path[i][1] << "]";
      if (i + 1 < run.smoothed_path.size()) out << ",";
      out << "\n";
    }
    out << "      ],\n";

    out << "      \"tree\": [\n";
    for (size_t i = 0; i < run.tree.size(); ++i) {
      out << "        [[" << run.tree[i][0][0] << ", " << run.tree[i][0][1] << "], ["
          << run.tree[i][1][0] << ", " << run.tree[i][1][1] << "]]";
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

int main(int argc, char* argv[]) {
  std::string output_file = "minimum_energy_planning.json";
  if (argc > 1) output_file = argv[1];

  // Arm parameters (matching tutorial)
  PlanarArmMassMatrix mass_fn;

  // Gravitational potential
  auto potential = [&mass_fn](const Eigen::Vector2d& q) {
    constexpr double g = 9.81;
    return mass_fn.m1 * g * mass_fn.lc1 * std::sin(q[0]) +
           mass_fn.m2 * g * (mass_fn.l1 * std::sin(q[0]) + mass_fn.lc2 * std::sin(q[0] + q[1]));
  };

  // Jacobi metric energy level: H = 1.2 * Pmax
  constexpr double g = 9.81;
  const double pmax = g * (mass_fn.m1 * mass_fn.lc1 + mass_fn.m2 * (mass_fn.l1 + mass_fn.lc2));
  const double H = 1.2 * pmax;

  // Planning configuration: [-pi, pi] bounds
  RunConfig cfg;
  cfg.lo = -M_PI;
  cfg.hi = M_PI;
  cfg.start_x = -M_PI / 4.0;
  cfg.start_y = -M_PI / 4.0;
  cfg.goal_x = 3 * M_PI / 4.0;
  cfg.goal_y = 3 * M_PI / 4.0;

  auto rrt_star = [](const ob::SpaceInformationPtr& si) {
    return std::make_shared<og::RRTstar>(si);
  };

  constexpr double solve_time = 2.0;
  std::vector<RunResult> runs;

  // 1. Flat metric on R^2
  {
    geodex::Euclidean<2> manifold;
    runs.push_back(run_planner(manifold, "Flat", "M = I", rrt_star, solve_time, cfg));
  }

  // 2. Kinetic energy metric
  {
    geodex::KineticEnergyMetric ke_metric{mass_fn};
    geodex::ConfigurationSpace cspace{geodex::Euclidean<2>{}, ke_metric};

    // Compute range: ~0.3 coordinate units scaled by Riemannian factor at domain midpoint
    Eigen::Vector2d mid((cfg.start_x + cfg.goal_x) / 2, (cfg.start_y + cfg.goal_y) / 2);
    Eigen::Vector2d unit(1.0, 0.0);
    double range = 0.3 * cspace.norm(mid, unit);

    runs.push_back(
        run_planner(cspace, "Kinetic Energy", "M = M(q)", rrt_star, solve_time, cfg, range));
  }

  // 3. Jacobi metric
  {
    geodex::JacobiMetric jacobi_metric{mass_fn, potential, H};
    geodex::ConfigurationSpace cspace{geodex::Euclidean<2>{}, jacobi_metric};

    // Compute range: ~0.3 coordinate units scaled by Riemannian factor at domain midpoint
    Eigen::Vector2d mid((cfg.start_x + cfg.goal_x) / 2, (cfg.start_y + cfg.goal_y) / 2);
    Eigen::Vector2d unit(1.0, 0.0);
    double range = 0.3 * cspace.norm(mid, unit);

    runs.push_back(
        run_planner(cspace, "Jacobi", "J = 2(H-P)M(q)", rrt_star, solve_time, cfg, range));
  }

  write_json(output_file, runs, cfg, mass_fn, H);

  return 0;
}

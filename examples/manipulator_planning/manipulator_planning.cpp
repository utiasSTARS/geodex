// Manipulator motion planning under kinetic energy Riemannian metrics.
//
// Supported examples:
//   - Panda table-pick problem 2 (default)
//   - Fixed-base dual-arm PR2 problem from data/datasets/mbm/problems/
//
// Pipeline:
//   - geodex::robots::MassMatrix<Robot> drives a KineticEnergyMetric over a
//     Euclidean ConfigurationSpace (precompiled CRBA, no runtime URDF parse).
//   - VAMP load_scene + make_vamp_checker / make_vamp_motion_validator handle
//     SIMD collision checking (state validity + edge validity).
//   - robots::MassLowerBound<Robot> supplies the precompiled constant SPD Loewner
//     bound on M(q) (certified at build time) for an admissible MatrixLowerBound
//     heuristic — no runtime precompute.
//   - GreedyRRTstar runs in informed mode under GeodexOptimizationObjective.
//   - smooth_path refines the raw RRT* output before JSON export.

#include <cmath>
#include <cstdint>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <ompl/base/Cost.h>
#include <ompl/base/Planner.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorBounds.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/planners/rrt/GreedyRRTstar.h>
#include <ompl/util/RandomNumbers.h>
#include <yaml-cpp/yaml.h>

#include "geodex/geodex.hpp"
#include "geodex/integration/ompl/geodex_optimization_objective.hpp"
#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/integration/vamp/registry.hpp"
#include "geodex/robots/mass_lower_bound.hpp"

#ifndef GEODEX_ROBOT_DATA_DIR
#define GEODEX_ROBOT_DATA_DIR "."
#endif

#ifndef GEODEX_DATASET_DATA_DIR
#define GEODEX_DATASET_DATA_DIR "."
#endif

namespace ob = ompl::base;
namespace og = ompl::geometric;
namespace gio = geodex::integration::ompl;
namespace robots = geodex::robots;
namespace vamp_int = geodex::integration::vamp;
namespace gh = geodex::heuristics;

namespace {

constexpr double kRange = 1.2;
constexpr double kGreedyRatio = 0.9;
constexpr double kCollisionResolution = 0.05;
constexpr std::uint64_t kSeed = 42;

struct PandaConfig {
  static constexpr int kDim = 7;
  static constexpr robots::Robot kRobot = robots::Robot::Panda;
  static constexpr std::string_view kRobotName = "panda";
  static constexpr bool kUsesProblemFile = true;
  static constexpr double kPlanningTime = 0.1;
  static constexpr std::array<double, kDim> kStart{0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};
  static constexpr std::array<double, kDim> kGoal{-0.748, 0.823, -0.655, -1.160,
                                                  -2.897, 2.871, 1.017};
  static constexpr std::array<std::string_view, kDim> kJointNames{
      "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
      "panda_joint5", "panda_joint6", "panda_joint7"};

  static auto urdf_path() -> std::string {
    return std::string(GEODEX_ROBOT_DATA_DIR) + "/panda/urdf/panda.urdf";
  }

  static auto default_problem_path() -> std::string {
    return std::string(GEODEX_DATASET_DATA_DIR) +
           "/mbm/problems/panda/panda_arm/table_pick/problem0002.problem.yaml";
  }
};

struct Pr2Config {
  static constexpr int kDim = 14;
  static constexpr robots::Robot kRobot = robots::Robot::Pr2;
  static constexpr std::string_view kRobotName = "pr2";
  static constexpr bool kUsesProblemFile = true;
  static constexpr double kPlanningTime = 60.0;
  static constexpr std::array<std::string_view, kDim> kJointNames{
      "l_shoulder_pan_joint",   "l_shoulder_lift_joint", "l_upper_arm_roll_joint",
      "l_elbow_flex_joint",     "l_forearm_roll_joint",  "l_wrist_flex_joint",
      "l_wrist_roll_joint",     "r_shoulder_pan_joint",  "r_shoulder_lift_joint",
      "r_upper_arm_roll_joint", "r_elbow_flex_joint",    "r_forearm_roll_joint",
      "r_wrist_flex_joint",     "r_wrist_roll_joint"};

  static auto urdf_path() -> std::string {
    return std::string(GEODEX_ROBOT_DATA_DIR) + "/pr2/pr2.urdf";
  }

  static auto default_problem_path() -> std::string {
    return std::string(GEODEX_DATASET_DATA_DIR) +
           "/mbm/problems/pr2/dual_arm/tabletop_can_box/problem0001.problem.yaml";
  }
};

template <typename Config>
struct Types {
  using BaseManifold = geodex::Euclidean<Config::kDim>;
  using MassMatrixFn = robots::MassMatrix<Config::kRobot>;
  using Metric = geodex::KineticEnergyMetric<MassMatrixFn>;
  using Manifold = geodex::ConfigurationSpace<BaseManifold, Metric>;
  using Heuristic = gh::MatrixLowerBound<Config::kDim>;
  using StateSpace = gio::GeodexStateSpace<Manifold>;
  using StateType = gio::GeodexState<Manifold>;
  using Objective = gio::GeodexOptimizationObjective<Manifold, Heuristic>;
};

struct Cli {
  std::string robot;
  std::string problem_path;
  std::string out_path = "manipulator_planning.json";
  std::string scene_path;
  double planning_time = -1.0;
  bool dry_run = false;
};

template <typename Config>
struct PlanningInput {
  std::string id;
  std::string robot_name = std::string(Config::kRobotName);
  std::string planning_group;
  std::string problem_path;
  std::string scene_path;
  std::string urdf_path = Config::urdf_path();
  std::string ee_link;
  double planning_time = Config::kPlanningTime;
  std::array<std::string, Config::kDim> joint_names{};
  std::array<double, Config::kDim> start{};
  std::array<double, Config::kDim> goal{};
};

struct PathStats {
  double cost = 0.0;
  double energy = 0.0;
  std::size_t n_waypoints = 0;
  double time_ms = -1.0;
  int vertices_removed = -1;
  int smooth_iterations = -1;
  bool collision_free = true;
};

auto resolve_relative(const std::filesystem::path& base_file, const std::string& path)
    -> std::string {
  std::filesystem::path p(path);
  if (p.is_absolute()) return p.lexically_normal().string();
  return (base_file.parent_path() / p).lexically_normal().string();
}

auto require_string(const YAML::Node& node, const std::string& key) -> std::string {
  if (!node[key]) throw std::runtime_error("missing required YAML key: " + key);
  return node[key].as<std::string>();
}

auto read_problem_robot(const std::string& problem_path) -> std::string {
  const YAML::Node problem = YAML::LoadFile(problem_path);
  return require_string(problem, "robot");
}

auto read_joint_names(const YAML::Node& group, const std::string& context)
    -> std::vector<std::string> {
  if (!group["joints"] || !group["joints"].IsSequence()) {
    throw std::runtime_error(context + " must define a joints sequence");
  }
  std::vector<std::string> joints;
  joints.reserve(group["joints"].size());
  for (const auto& joint : group["joints"]) joints.push_back(joint.as<std::string>());
  return joints;
}

template <typename Config>
auto read_named_joint_vector(const YAML::Node& joints,
                             const std::array<std::string, Config::kDim>& order,
                             const std::string& context) -> std::array<double, Config::kDim> {
  if (!joints || !joints.IsMap()) {
    throw std::runtime_error(context + " must contain a joints map");
  }
  std::array<double, Config::kDim> q{};
  for (int i = 0; i < Config::kDim; ++i) {
    const auto& name = order[i];
    if (!joints[name]) throw std::runtime_error(context + " missing joint value: " + name);
    q[i] = joints[name].template as<double>();
  }
  return q;
}

template <typename Config>
auto config_joint_names() -> std::array<std::string, Config::kDim> {
  std::array<std::string, Config::kDim> names{};
  for (int i = 0; i < Config::kDim; ++i) names[i] = std::string(Config::kJointNames[i]);
  return names;
}

template <typename Config>
auto make_static_input() -> PlanningInput<Config> {
  PlanningInput<Config> input;
  input.scene_path = Config::scene_path();
  input.joint_names = config_joint_names<Config>();
  input.planning_group = "default";
  for (int i = 0; i < Config::kDim; ++i) {
    input.start[i] = Config::kStart[i];
    input.goal[i] = Config::kGoal[i];
  }
  return input;
}

template <typename Config>
auto load_problem_input(const std::string& problem_path) -> PlanningInput<Config> {
  const auto problem_file = std::filesystem::path(problem_path);
  const YAML::Node problem = YAML::LoadFile(problem_file.string());
  PlanningInput<Config> input;
  input.problem_path = problem_file.lexically_normal().string();
  input.id = problem["id"] ? problem["id"].as<std::string>() : input.problem_path;
  input.robot_name = require_string(problem, "robot");
  input.planning_group = require_string(problem, "planning_group");
  if (input.robot_name != Config::kRobotName) {
    throw std::runtime_error("problem robot '" + input.robot_name +
                             "' does not match selected robot '" + std::string(Config::kRobotName) +
                             "'");
  }

  input.scene_path = resolve_relative(problem_file, require_string(problem, "scene"));
  if (problem["allowed_planning_time"]) {
    input.planning_time = problem["allowed_planning_time"].as<double>();
  }

  const auto robot_yaml_path =
      std::filesystem::path(GEODEX_ROBOT_DATA_DIR) / input.robot_name / "robot.yaml";
  const YAML::Node robot = YAML::LoadFile(robot_yaml_path.string());
  if (robot["urdf"]) {
    input.urdf_path = resolve_relative(robot_yaml_path, robot["urdf"].as<std::string>());
  }

  const auto group = robot["planning_groups"][input.planning_group];
  if (!group) {
    throw std::runtime_error("robot metadata has no planning group: " + input.planning_group);
  }
  if (group["default_ee_link"]) input.ee_link = group["default_ee_link"].template as<std::string>();

  const auto joints = read_joint_names(group, "planning group '" + input.planning_group + "'");
  if (joints.size() != static_cast<std::size_t>(Config::kDim)) {
    throw std::runtime_error("planning group dimension does not match selected robot");
  }
  const auto expected = config_joint_names<Config>();
  for (int i = 0; i < Config::kDim; ++i) {
    input.joint_names[i] = joints[i];
    if (input.joint_names[i] != expected[i]) {
      throw std::runtime_error(
          "planning group joint order differs from generated robot order at index " +
          std::to_string(i) + ": got '" + input.joint_names[i] + "', expected '" + expected[i] +
          "'");
    }
  }

  input.start =
      read_named_joint_vector<Config>(problem["start"]["joints"], input.joint_names, "start");
  input.goal =
      read_named_joint_vector<Config>(problem["goal"]["joints"], input.joint_names, "goal");
  return input;
}

template <typename Config>
auto make_planning_input(const Cli& cli) -> PlanningInput<Config> {
  std::string problem_path = cli.problem_path;
  bool used_default_problem = false;
  if (problem_path.empty()) {
    if constexpr (Config::kUsesProblemFile) {
      problem_path = Config::default_problem_path();
      used_default_problem = true;
    }
  }

  PlanningInput<Config> input;
  if (problem_path.empty()) {
    if constexpr (Config::kUsesProblemFile) {
      throw std::runtime_error("no problem file provided for robot '" +
                               std::string(Config::kRobotName) + "'");
    } else {
      input = make_static_input<Config>();
    }
  } else {
    input = load_problem_input<Config>(problem_path);
    if (used_default_problem) input.planning_time = Config::kPlanningTime;
  }
  if (!cli.scene_path.empty()) input.scene_path = cli.scene_path;
  if (cli.planning_time > 0.0) input.planning_time = cli.planning_time;
  return input;
}

auto parse_cli(int argc, char* argv[]) -> Cli {
  Cli cli;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--robot" && i + 1 < argc) {
      cli.robot = argv[++i];
    } else if (arg == "--problem" && i + 1 < argc) {
      cli.problem_path = argv[++i];
    } else if (arg == "--scene" && i + 1 < argc) {
      cli.scene_path = argv[++i];
    } else if (arg == "--time" && i + 1 < argc) {
      cli.planning_time = std::stod(argv[++i]);
    } else if ((arg == "--out" || arg == "-o") && i + 1 < argc) {
      cli.out_path = argv[++i];
    } else if (arg == "--dry-run") {
      cli.dry_run = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: manipulator_planning [--robot panda|pr2] [--problem PATH] "
                   "[--scene PATH] [--time SECONDS] [--out PATH] [--dry-run]\n";
      std::exit(0);
    } else if (!arg.empty() && arg[0] != '-') {
      cli.out_path = arg;
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
  }
  return cli;
}

template <typename Config>
auto build_manifold() -> typename Types<Config>::Manifold {
  using T = Types<Config>;
  typename T::MassMatrixFn mm;
  const auto [lo, hi] = T::MassMatrixFn::joint_limits();
  static_assert(T::MassMatrixFn::nq() == Config::kDim, "Robot nq must match planning dimension");

  typename T::BaseManifold base;
  base.set_sampling_bounds(lo, hi);
  typename T::Metric metric{std::move(mm)};
  return typename T::Manifold{std::move(base), std::move(metric)};
}

// Loads the precompiled Loewner lower bound certified at build time
template <typename Config>
auto load_heuristic() -> typename Types<Config>::Heuristic {
  using LB = robots::MassLowerBound<Config::kRobot>;
  static_assert(LB::available,
                "no precompiled Loewner bound for this robot; run "
                "`cmake --build build --target regenerate_robot_bounds`");
  std::cerr << "[heuristic] using precompiled Loewner bound (certificate=" << LB::certificate
            << ", converged=" << std::boolalpha << LB::converged << ")\n";
  Eigen::Matrix<double, Config::kDim, Config::kDim> M_lower = LB::matrix();
  return typename Types<Config>::Heuristic{M_lower};
}

template <typename Config>
auto build_state_space(const typename Types<Config>::Manifold& manifold)
    -> std::shared_ptr<typename Types<Config>::StateSpace> {
  using T = Types<Config>;
  ob::RealVectorBounds bounds(Config::kDim);
  for (int i = 0; i < Config::kDim; ++i) {
    bounds.setLow(i, manifold.lo()[i]);
    bounds.setHigh(i, manifold.hi()[i]);
  }
  auto space = std::make_shared<typename T::StateSpace>(manifold, bounds);
  space->setInterpolationMode(gio::InterpolationMode::BaseGeodesic);
  space->setCollisionResolution(kCollisionResolution);
  return space;
}

template <typename Config>
auto build_si(const std::shared_ptr<typename Types<Config>::StateSpace>& space,
              const vamp_int::EnvHandle& env) -> ob::SpaceInformationPtr {
  using T = Types<Config>;
  auto si = std::make_shared<ob::SpaceInformation>(space);

  std::shared_ptr<vamp_int::CollisionChecker> state_checker(
      vamp_int::make_vamp_checker(std::string(Config::kRobotName), env).release());
  si->setStateValidityChecker([state_checker](const ob::State* state) {
    const auto* s = state->template as<typename T::StateType>();
    return state_checker->is_valid(s->values, Config::kDim);
  });

  std::shared_ptr<ob::MotionValidator> motion_validator(
      vamp_int::make_vamp_motion_validator(std::string(Config::kRobotName), si, env).release());
  si->setMotionValidator(motion_validator);

  si->setup();
  return si;
}

template <typename Config>
auto build_problem(const ob::SpaceInformationPtr& si,
                   const std::shared_ptr<typename Types<Config>::StateSpace>& space,
                   const typename Types<Config>::Heuristic& heuristic,
                   const PlanningInput<Config>& input)
    -> std::pair<ob::ProblemDefinitionPtr, std::shared_ptr<typename Types<Config>::Objective>> {
  using T = Types<Config>;
  auto pdef = std::make_shared<ob::ProblemDefinition>(si);

  ob::ScopedState<typename T::StateSpace> start(space);
  ob::ScopedState<typename T::StateSpace> goal(space);
  for (int i = 0; i < Config::kDim; ++i) {
    start->values[i] = input.start[i];
    goal->values[i] = input.goal[i];
  }
  pdef->setStartAndGoalStates(start, goal);

  typename T::Manifold::Point goal_coords;
  for (int i = 0; i < Config::kDim; ++i) goal_coords[i] = input.goal[i];

  auto objective = std::make_shared<typename T::Objective>(si, goal_coords, heuristic);
  objective->setGreedyBiasingRatio(kGreedyRatio);
  pdef->setOptimizationObjective(objective);

  return {pdef, objective};
}

template <typename Config>
auto build_planner(const ob::SpaceInformationPtr& si, const ob::ProblemDefinitionPtr& pdef)
    -> std::shared_ptr<og::GreedyRRTstar> {
  using StateType = typename Types<Config>::StateType;
  auto planner = std::make_shared<og::GreedyRRTstar>(si);
  planner->setRange(kRange);
  planner->setNNDistanceFunction([](const ob::State* a, const ob::State* b) {
    const auto* sa = a->as<StateType>();
    const auto* sb = b->as<StateType>();
    double d2 = 0.0;
    for (int i = 0; i < Config::kDim; ++i) {
      const double d = sa->values[i] - sb->values[i];
      d2 += d * d;
    }
    return std::sqrt(d2);
  });

  planner->setGreedyBiasingRatio(0.0);
  planner->setGreedyCostForTreePruning(true);
  planner->setProblemDefinition(pdef);
  planner->setup();
  return planner;
}

template <typename Manifold>
auto compute_path_stats(const Manifold& manifold, const std::vector<typename Manifold::Point>& path)
    -> PathStats {
  PathStats s;
  s.n_waypoints = path.size();
  if (path.size() < 2) return s;
  for (std::size_t i = 1; i < path.size(); ++i) {
    const double d = manifold.distance(path[i - 1], path[i]);
    s.cost += d;
    s.energy += d * d;
  }
  return s;
}

void print_stats_table(const PathStats& raw, const PathStats& smooth) {
  auto pct = [](double base, double x) {
    if (base <= 0.0) return 0.0;
    return 100.0 * (x - base) / base;
  };
  auto fmt_delta = [&](double base, double x) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << std::showpos << pct(base, x) << "%";
    return os.str();
  };
  std::cerr << std::fixed << std::setprecision(4);
  std::cerr << "\n=== Path stats ===\n";
  std::cerr << std::left << std::setw(16) << "" << std::setw(20) << "cost" << std::setw(20)
            << "energy" << std::setw(14) << "n_waypoints" << std::setw(10) << "time(ms)"
            << "\n";
  std::cerr << std::left << std::setw(16) << "raw planner:" << std::setw(20) << raw.cost
            << std::setw(20) << raw.energy << std::setw(14) << raw.n_waypoints << std::setw(10)
            << "-"
            << "\n";
  std::ostringstream cost_col, energy_col;
  cost_col << std::fixed << std::setprecision(4) << smooth.cost << " ("
           << fmt_delta(raw.cost, smooth.cost) << ")";
  energy_col << std::fixed << std::setprecision(4) << smooth.energy << " ("
             << fmt_delta(raw.energy, smooth.energy) << ")";
  std::cerr << std::left << std::setw(16) << "smooth_path:" << std::setw(20) << cost_col.str()
            << std::setw(20) << energy_col.str() << std::setw(14) << smooth.n_waypoints
            << std::setw(10) << smooth.time_ms << "\n\n";
}

template <int Dim>
void write_array(std::ofstream& out, const std::array<double, Dim>& q, const char* indent) {
  out << indent << "[";
  for (int j = 0; j < Dim; ++j) {
    out << q[j] << (j + 1 < Dim ? ", " : "");
  }
  out << "]";
}

template <int Dim>
void write_path_array(std::ofstream& out, const std::vector<std::array<double, Dim>>& path,
                      const char* key) {
  out << "  \"" << key << "\": [\n";
  for (std::size_t i = 0; i < path.size(); ++i) {
    write_array<Dim>(out, path[i], "    ");
    out << (i + 1 < path.size() ? "," : "") << "\n";
  }
  out << "  ],\n";
}

void write_stats_block(std::ofstream& out, const PathStats& s, const char* key, bool last = false) {
  out << "  \"" << key << "\": {"
      << "\"cost\": " << s.cost << ", \"energy\": " << s.energy
      << ", \"n_waypoints\": " << s.n_waypoints;
  if (s.time_ms >= 0.0) out << ", \"time_ms\": " << s.time_ms;
  if (s.vertices_removed >= 0) out << ", \"vertices_removed\": " << s.vertices_removed;
  if (s.smooth_iterations >= 0) out << ", \"smooth_iterations\": " << s.smooth_iterations;
  out << ", \"collision_free\": " << (s.collision_free ? "true" : "false");
  out << "}" << (last ? "" : ",") << "\n";
}

template <typename Config>
void write_json(const std::string& out_path, const PlanningInput<Config>& input, bool solved,
                double final_cost, const std::vector<std::array<double, Config::kDim>>& dense_path,
                const std::vector<std::array<double, Config::kDim>>& raw_path,
                const std::vector<std::array<double, Config::kDim>>& smooth_path_arr,
                const PathStats& raw_stats, const PathStats& smooth_stats,
                std::uint64_t total_motion_cost_calls) {
  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "error: could not open " << out_path << " for writing\n";
    return;
  }
  out << std::fixed << std::setprecision(8);
  out << "{\n";
  if (!input.id.empty()) out << "  \"problem_id\": \"" << input.id << "\",\n";
  if (!input.problem_path.empty()) out << "  \"problem\": \"" << input.problem_path << "\",\n";
  out << "  \"robot\": \"" << input.robot_name << "\",\n";
  if (!input.planning_group.empty()) {
    out << "  \"planning_group\": \"" << input.planning_group << "\",\n";
  }
  if (!input.ee_link.empty()) out << "  \"ee_link\": \"" << input.ee_link << "\",\n";
  out << "  \"urdf\": \"" << input.urdf_path << "\",\n";
  out << "  \"scene\": \"" << input.scene_path << "\",\n";
  out << "  \"joint_names\": [";
  for (int i = 0; i < Config::kDim; ++i) {
    out << "\"" << input.joint_names[i] << "\"" << (i + 1 < Config::kDim ? ", " : "");
  }
  out << "],\n";
  out << "  \"start\": [";
  for (int i = 0; i < Config::kDim; ++i) {
    out << input.start[i] << (i + 1 < Config::kDim ? ", " : "");
  }
  out << "],\n";
  out << "  \"goal\": [";
  for (int i = 0; i < Config::kDim; ++i) {
    out << input.goal[i] << (i + 1 < Config::kDim ? ", " : "");
  }
  out << "],\n";
  out << "  \"range\": " << kRange << ",\n";
  out << "  \"greedy_ratio\": " << kGreedyRatio << ",\n";
  out << "  \"planning_time_s\": " << input.planning_time << ",\n";
  out << "  \"seed\": " << kSeed << ",\n";
  out << "  \"solved\": " << (solved ? "true" : "false") << ",\n";
  out << "  \"final_cost\": " << final_cost << ",\n";

  write_path_array<Config::kDim>(out, dense_path, "path");
  write_path_array<Config::kDim>(out, raw_path, "raw_path");
  write_path_array<Config::kDim>(out, smooth_path_arr, "smooth_path");
  write_stats_block(out, raw_stats, "raw_stats");
  write_stats_block(out, smooth_stats, "smooth_stats");
  out << "  \"total_motion_cost_calls\": " << total_motion_cost_calls << "\n";
  out << "}\n";

  std::cout << "wrote " << out_path << "\n";
}

template <typename Config>
auto path_to_arrays(const std::vector<typename Types<Config>::Manifold::Point>& path)
    -> std::vector<std::array<double, Config::kDim>> {
  std::vector<std::array<double, Config::kDim>> out;
  out.reserve(path.size());
  for (const auto& q : path) {
    std::array<double, Config::kDim> arr{};
    for (int i = 0; i < Config::kDim; ++i) arr[i] = q[i];
    out.push_back(arr);
  }
  return out;
}

template <typename Config>
auto run_planning(const Cli& cli) -> int {
  using T = Types<Config>;
  const PlanningInput<Config> input = make_planning_input<Config>(cli);

  ompl::RNG::setSeed(kSeed);

  std::cerr << "[setup] robot=" << input.robot_name;
  if (!input.planning_group.empty()) std::cerr << " group=" << input.planning_group;
  std::cerr << "\n";
  if (!input.problem_path.empty()) std::cerr << "[setup] problem: " << input.problem_path << "\n";
  std::cerr << "[setup] scene: " << input.scene_path << "\n";
  if (cli.dry_run) {
    std::cerr << "[setup] urdf: " << input.urdf_path << "\n";
    std::cerr << "[setup] joints:";
    for (const auto& name : input.joint_names) std::cerr << " " << name;
    std::cerr << "\n[dry-run] problem parsed successfully; skipping planning\n";
    return 0;
  }
  std::cerr << "[setup] loading scene\n";
  auto env = vamp_int::load_scene(input.scene_path);

  std::cerr << "[setup] building kinetic-energy manifold (precompiled CRBA)\n";
  auto manifold = build_manifold<Config>();
  auto heuristic = load_heuristic<Config>();
  auto space = build_state_space<Config>(manifold);
  auto si = build_si<Config>(space, env);
  auto [pdef, objective] = build_problem<Config>(si, space, heuristic, input);
  auto planner = build_planner<Config>(si, pdef);

  std::cerr << "[plan] solving with G-RRT* (range=" << kRange << ", greedy_ratio=" << kGreedyRatio
            << ", time=" << input.planning_time << "s, seed=" << kSeed << ")\n";
  ob::PlannerStatus status =
      planner->solve(ob::timedPlannerTerminationCondition(input.planning_time));
  const bool solved = pdef->hasExactSolution();
  std::cerr << "[plan] status=" << status.asString() << " exact=" << solved << "\n";

  std::vector<std::array<double, Config::kDim>> dense_arr;
  std::vector<std::array<double, Config::kDim>> raw_arr;
  std::vector<std::array<double, Config::kDim>> smooth_arr;
  PathStats raw_stats, smooth_stats;
  double final_cost = std::numeric_limits<double>::infinity();

  if (solved) {
    auto path_ptr = std::dynamic_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
    if (path_ptr && path_ptr->getStateCount() >= 2) {
      using Point = typename T::Manifold::Point;
      std::vector<Point> raw_path;
      raw_path.reserve(path_ptr->getStateCount());
      for (const auto* state : path_ptr->getStates()) {
        raw_path.push_back(state->template as<typename T::StateType>()->asEigen());
      }
      raw_stats = compute_path_stats(manifold, raw_path);
      raw_arr = path_to_arrays<Config>(raw_path);

      auto smoothing_checker = vamp_int::make_vamp_checker(std::string(Config::kRobotName), env);
      auto validity_fn = [&smoothing_checker](const Point& q) {
        return smoothing_checker->is_valid(q.data(), Config::kDim);
      };

      geodex::algorithm::PathSmoothingSettings smooth_settings;
      const auto t_smooth_start = std::chrono::steady_clock::now();
      auto smooth_result =
          geodex::algorithm::smooth_path(manifold, validity_fn, raw_path, smooth_settings);
      const double smooth_dt_ms =
          1000.0 *
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_smooth_start).count();
      smooth_stats = compute_path_stats(manifold, smooth_result.path);
      smooth_stats.time_ms = smooth_dt_ms;
      smooth_stats.vertices_removed = smooth_result.vertices_removed;
      smooth_stats.smooth_iterations = smooth_result.smooth_iterations;
      smooth_stats.collision_free = smooth_result.collision_free;
      smooth_arr = path_to_arrays<Config>(smooth_result.path);

      print_stats_table(raw_stats, smooth_stats);

      og::PathGeometric dense_path(si);
      for (const auto& q : smooth_result.path) {
        ob::ScopedState<typename T::StateSpace> s(space);
        for (int i = 0; i < Config::kDim; ++i) s->values[i] = q[i];
        dense_path.append(s.get());
      }
      dense_path.interpolate(100);

      for (const auto* state : dense_path.getStates()) {
        const auto* s = state->template as<typename T::StateType>();
        std::array<double, Config::kDim> q{};
        for (int i = 0; i < Config::kDim; ++i) q[i] = s->values[i];
        dense_arr.push_back(q);
      }
      final_cost = smooth_stats.cost;
    }
  }

  write_json<Config>(cli.out_path, input, solved, final_cost, dense_arr, raw_arr, smooth_arr,
                     raw_stats, smooth_stats, objective->getMotionCostCallCount());
  return solved ? 0 : 1;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    Cli cli = parse_cli(argc, argv);
    if (cli.robot.empty()) {
      cli.robot = cli.problem_path.empty() ? "panda" : read_problem_robot(cli.problem_path);
    }
    if (cli.robot == "panda") return run_planning<PandaConfig>(cli);
    if (cli.robot == "pr2") return run_planning<Pr2Config>(cli);
    std::cerr << "error: unsupported robot '" << cli.robot << "' (expected 'panda' or 'pr2')\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
}

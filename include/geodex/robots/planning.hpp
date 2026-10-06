/// @file
/// @brief Planning for a built-in robot in a VAMP scene.
///
/// @details The C++ counterpart of Python's `geodex.plan(robot, start, goal, scene)`.
/// It builds the robot's configuration space, the VAMP point checker and motion
/// validator, and the precomputed admissible heuristic, then calls
/// `geodex::planning::plan`. Needs a build with `GEODEX_OMPL` and `GEODEX_VAMP`.
///
/// @code
/// namespace gr = geodex::robots;
/// const auto env = geodex::integration::vamp::load_scene("table.yaml");
/// const auto result = gr::plan<gr::Robot::Panda>(start, goal, env);
///
/// // A mobile robot plans on SE(2) x R^n, and the options pick only the metric.
/// gr::RobotPlanOptions options;
/// options.base = geodex::SE2LeftInvariantMetric::holonomic();
/// const auto whole_body = gr::plan<gr::Robot::Stretch3>(start8, goal8, env, {}, options);
/// @endcode

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/heuristics/product_lower_bound.hpp"
#include "geodex/integration/vamp/registry.hpp"
#include "geodex/integration/vamp/validity.hpp"
#include "geodex/manifold/product.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/metrics/se2_left_invariant.hpp"
#include "geodex/planning/plan.hpp"
#include "geodex/robots/joint_space.hpp"

namespace geodex::robots {

/// @brief Metric and base-region choices for `robots::plan`.
struct RobotPlanOptions {
  ArmMetric metric = ArmMetric::KineticEnergy;  ///< metric on the joints of the arm

  /// @brief Base metric of a mobile robot. Empty picks `SE2LeftInvariantMetric::holonomic()`
  /// or `SE2LeftInvariantMetric::differential_drive()` by the robot's `base_drive`.
  std::optional<SE2LeftInvariantMetric> base;

  /// @brief Lower and upper corner of the region a mobile base samples. Empty takes the
  /// start and goal positions grown by `base_margin` on every side.
  std::optional<std::pair<Eigen::Vector2d, Eigen::Vector2d>> base_region;

  double base_margin = 1.0;  ///< growth of the default base region, in meters
};

namespace detail {

/// @brief Copy the smoother's fields when both results have them.
template <typename Out, typename In>
void copy_smoothing(Out& out, const In& in) {
  if constexpr (requires { out.smoothed = in.smoothed; }) out.smoothed = in.smoothed;
  if constexpr (requires { out.smooth_ms = in.smooth_ms; }) out.smooth_ms = in.smooth_ms;
}

/// @brief Plan on @p space with the VAMP kernel of robot @p name bound to @p env.
template <typename SpaceT, typename HeuristicT>
planning::PlanResult<Eigen::VectorXd> plan_with_vamp(
    const SpaceT& space, const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
    const std::string& name, const integration::vamp::EnvHandle& env,
    const planning::PlanSettings& settings, const HeuristicT& heuristic) {
  using Point = typename SpaceT::Point;
  // The smoother checks each edge at configurations between which no sphere moves more than
  // `travel`, spaced by the edge's own sphere travel. The scene grows by the sphere motion of
  // the corner tolerance.
  if (!std::isfinite(settings.collision_check_resolution) ||
      settings.collision_check_resolution < 0.0) {
    throw std::invalid_argument("plan: collision_check_resolution must be finite and >= 0");
  }
  const double tolerance = settings.smoothing.corner_tolerance;
  if (!(std::isfinite(tolerance) && tolerance > 0.0)) {
    throw std::invalid_argument("plan: smoothing.corner_tolerance must be finite and > 0");
  }
  const double travel = settings.collision_check_resolution > 0.0
                            ? settings.collision_check_resolution
                            : integration::vamp::kDefaultMaxSphereStep;
  const double speed = integration::vamp::sphere_speed(name, env);
  const integration::vamp::EnvHandle padded = integration::vamp::pad_scene(env, speed * tolerance);
  // Keep the concrete validity type with its batched checks.
  const integration::vamp::VampValidity<Point> is_valid(
      integration::vamp::make_vamp_checker(name, padded));
  const auto motion_validator = [name, padded](const ompl::base::SpaceInformationPtr& si) {
    return std::shared_ptr<ompl::base::MotionValidator>(
        integration::vamp::make_vamp_motion_validator(name, si, padded));
  };
  const Point s = start;
  const Point g = goal;
  // The space's box holds the robot's joint limits and the base region. The plan treats it
  // as physical limits.
  planning::PlanSettings limited = settings;
  if (!limited.limits) limited.limits.emplace(space.lo(), space.hi());
  limited.collision_check_resolution = 0.0;
  limited.smoothing.collision_check_resolution = travel;
  limited.smoothing.edge_travel =
      [&space, sphere_travel = integration::vamp::make_sphere_travel(name, env)](
          const Eigen::Ref<const Eigen::VectorXd>& a, const Eigen::Ref<const Eigen::VectorXd>& b) {
        return sphere_travel(space.log(Point(a), Point(b)));
      };
  const auto r = planning::plan(space, s, g, is_valid, limited, heuristic, motion_validator);

  planning::PlanResult<Eigen::VectorXd> out;
  out.solved = r.solved;
  out.cost = r.cost;
  out.time_ms = r.time_ms;
  copy_smoothing(out, r);
  out.path.assign(r.path.begin(), r.path.end());
  out.raw_path.assign(r.raw_path.begin(), r.raw_path.end());
  // The structured binding fails to compile when PlanResult gains a member. Copy a new member
  // into `out`.
  [[maybe_unused]] const auto& [solved, smoothed, path, raw_path, cost, time_ms, smooth_ms,
                                first_solution_ms, first_solution_iterations, informed_samples,
                                focused_samples, uniform_samples] = r;
  out.first_solution_ms = r.first_solution_ms;
  out.first_solution_iterations = r.first_solution_iterations;
  out.informed_samples = r.informed_samples;
  out.focused_samples = r.focused_samples;
  out.uniform_samples = r.uniform_samples;
  return out;
}

}  // namespace detail

/// @brief Plan a collision-free path for robot @p R in the VAMP scene @p env.
///
/// A fixed-base robot plans on `joint_space<R>()`. A robot with a planar base plans on
/// `make_product(SE2<>(options.base, ...), joint_space<R>())` with configurations
/// `(x, y, theta, arm joints...)`. The informed planner uses the precomputed Loewner bound of
/// the chosen metric as its heuristic, `heuristics::product_lower_bound` for a planar
/// base.
///
/// The planner and the smoother check the scene grown by the sphere motion of the smoother's
/// corner tolerance, the planner's edges with `integration::vamp::make_vamp_motion_validator`.
/// The smoother checks its edges at steps along which no sphere center moves more than
/// `settings.collision_check_resolution` meters, and 0 uses
/// `integration::vamp::kDefaultMaxSphereStep` (5 mm). It spaces each edge by the edge's own
/// sphere travel (`integration::vamp::make_sphere_travel`), and the plan replaces
/// `settings.smoothing.collision_check_resolution` and `settings.smoothing.edge_travel`. A
/// sphere can come closer to an obstacle
/// between two checks than at the checks. With smoothing off, the path holds the planner's
/// states, checked at VAMP's own edge resolution. Self-collision and contact with an attached
/// body are checked at the samples only.
///
/// When `settings.smoothing.sharp_coordinates` is 0, a base whose metric makes sideways motion
/// cost more than driving gets `sharp_base_coordinates(metric)`. Its pose may then keep a
/// corner while the smoother rounds the arm's joints. A value of at least the configuration
/// size rounds every coordinate together.
///
/// @param start Start configuration.
/// @param goal Goal configuration.
/// @param env Scene from `integration::vamp::load_scene` or `build_scene`.
/// @param settings Planner settings forwarded to `planning::plan`.
/// @param options Metric and base-region choices.
/// @throw std::invalid_argument if @p start or @p goal has the wrong size.
template <Robot R>
planning::PlanResult<Eigen::VectorXd> plan(const Eigen::VectorXd& start,
                                           const Eigen::VectorXd& goal,
                                           const integration::vamp::EnvHandle& env,
                                           const planning::PlanSettings& settings = {},
                                           const RobotPlanOptions& options = {}) {
  const std::string robot_name(name(R));
  constexpr Eigen::Index dim = MassMatrix<R>::Nq + (has_planar_base<R> ? 3 : 0);
  if (start.size() != dim || goal.size() != dim) {
    throw std::invalid_argument("robots::plan: " + robot_name + " takes configurations of size " +
                                std::to_string(dim) + ", got start " +
                                std::to_string(start.size()) + " and goal " +
                                std::to_string(goal.size()));
  }
  if constexpr (has_planar_base<R>) {
    const SE2LeftInvariantMetric metric = options.base.value_or(
        base_drive<R> == BaseDrive::Differential ? SE2LeftInvariantMetric::differential_drive()
                                                 : SE2LeftInvariantMetric::holonomic());
    std::pair<Eigen::Vector2d, Eigen::Vector2d> region;
    if (options.base_region) {
      region = *options.base_region;
    } else {
      const Eigen::Vector2d a = start.head<2>(), b = goal.head<2>();
      region = {a.cwiseMin(b).array() - options.base_margin,
                a.cwiseMax(b).array() + options.base_margin};
    }
    const auto& [xy_lo, xy_hi] = region;
    const SE2<> base(metric, Eigen::Vector3d(xy_lo.x(), xy_lo.y(), -std::numbers::pi),
                     Eigen::Vector3d(xy_hi.x(), xy_hi.y(), std::numbers::pi));
    const auto heuristic =
        heuristics::product_lower_bound({{metric.coordinate_lower_bound(), base.periods()},
                                         {joint_lower_bound<R>(options.metric)}});
    planning::PlanSettings with_base = settings;
    if (with_base.smoothing.sharp_coordinates == 0) {
      with_base.smoothing.sharp_coordinates = sharp_base_coordinates(metric);
    }
    if (options.metric == ArmMetric::KineticEnergy) {
      const auto space = make_product(base, joint_space<R, ArmMetric::KineticEnergy>());
      return detail::plan_with_vamp(space, start, goal, robot_name, env, with_base, heuristic);
    }
    const auto space = make_product(base, joint_space<R, ArmMetric::Euclidean>());
    return detail::plan_with_vamp(space, start, goal, robot_name, env, with_base, heuristic);
  } else {
    const heuristics::MatrixLowerBound<Eigen::Dynamic> heuristic(
        joint_lower_bound<R>(options.metric));
    if (options.metric == ArmMetric::KineticEnergy) {
      return detail::plan_with_vamp(joint_space<R, ArmMetric::KineticEnergy>(), start, goal,
                                    robot_name, env, settings, heuristic);
    }
    return detail::plan_with_vamp(joint_space<R, ArmMetric::Euclidean>(), start, goal, robot_name,
                                  env, settings, heuristic);
  }
}

/// @brief Plan for the robot chosen at run time, e.g. from `robot_from_name`.
inline planning::PlanResult<Eigen::VectorXd> plan(Robot robot, const Eigen::VectorXd& start,
                                                  const Eigen::VectorXd& goal,
                                                  const integration::vamp::EnvHandle& env,
                                                  const planning::PlanSettings& settings = {},
                                                  const RobotPlanOptions& options = {}) {
  return visit(robot,
               [&]<Robot R>() { return plan<R>(start, goal, env, settings, options); });
}

}  // namespace geodex::robots

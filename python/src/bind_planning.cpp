/// @file bind_planning.cpp
/// @brief Python bindings for the geodex planning facade. Binds the `planners` submodule
/// (GreedyRRTstar), PlanSettings, PlanResult and plan(). The build compiles this file only with
/// OMPL.

#include <cstdint>

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <ompl/base/DiscreteMotionValidator.h>

#include "geodex/heuristics/heuristics.hpp"
#include "geodex/integration/ompl/directional_motion_validator.hpp"
#include "geodex/planning/plan.hpp"

#include "wrappers/dynamic_manifold.hpp"
#include "wrappers/extract_manifold.hpp"
#include "wrappers/native_collision.hpp"
#include "wrappers/native_plan.hpp"
#include "wrappers/native_validity.hpp"
#include "wrappers/py_callable.hpp"
#include "wrappers/py_config_space.hpp"
#include "wrappers/py_metrics.hpp"
#include "wrappers/py_se2.hpp"
#include "wrappers/py_so2.hpp"
#include "wrappers/py_torus.hpp"

#ifdef GEODEX_PYTHON_HAS_ROBOTS
#include "wrappers/py_robot_model.hpp"
#endif

#ifdef GEODEX_PYTHON_HAS_VAMP
#include "geodex/integration/vamp/registry.hpp"
#include "geodex/integration/vamp/validity.hpp"

#include "wrappers/py_robot_model.hpp"
#include "wrappers/py_scene.hpp"
#endif

namespace nb = nanobind;
namespace gp = geodex::planning;

namespace {

/// Interpolation strategy enum used by PlanSettings.
using InterpolationMode = geodex::integration::ompl::InterpolationMode;

/// Config for plan()'s motion_validator keyword.
struct DirectionalValidatorConfig {
  double max_reverse_length = 0.0;
};

/// Pack a path (sequence of points) into an (N, d) row matrix for numpy.
Eigen::MatrixXd path_to_matrix(const std::vector<Eigen::VectorXd>& path) {
  if (path.empty()) return Eigen::MatrixXd(0, 0);
  const auto n = static_cast<Eigen::Index>(path.size());
  const Eigen::Index d = path.front().size();
  Eigen::MatrixXd out(n, d);
  for (Eigen::Index i = 0; i < n; ++i) out.row(i) = path[static_cast<std::size_t>(i)].transpose();
  return out;
}

/// Map an interp string to the InterpolationMode enum.
InterpolationMode parse_interp(const std::string& name) {
  if (name == "auto") return InterpolationMode::Auto;
  if (name == "base_geodesic") return InterpolationMode::BaseGeodesic;
  if (name == "riemannian_geodesic") return InterpolationMode::RiemannianGeodesic;
  throw std::invalid_argument(
      "interp must be one of 'auto', 'base_geodesic', 'riemannian_geodesic'; got '" + name + "'.");
}

/// Whether a Python validity result counts as valid. Any truthy result counts, including a
/// numpy bool.
bool truthy(const nb::object& result) {
  const int t = PyObject_IsTrue(result.ptr());
  if (t < 0) throw nb::python_error();
  return t == 1;
}

/// The parts of a ConfigurationSpace over SE2 whose metric is a ClearanceMetric with an
/// SE2LeftInvariantMetric base and a bound geodex SDF. Any other space gives nothing.
std::optional<geodex::python::SE2ClearanceParts> se2_clearance_parts(nb::handle space) {
  namespace gpy = geodex::python;
  if (!nb::isinstance<gpy::PyConfigurationSpace>(space)) return std::nullopt;
  const auto& cs = nb::cast<const gpy::PyConfigurationSpace&>(space);
  if (!cs.se2_base() || !cs.metric_ref() || !cs.metric_ref()->get()) return std::nullopt;
  const nb::handle metric(cs.metric_ref()->get());
  if (!nb::isinstance<gpy::PyClearanceMetric>(metric)) return std::nullopt;
  const auto& clearance = nb::cast<const gpy::PyClearanceMetric&>(metric);
  const auto* function = clearance.impl().sdf().target<gpy::SdfFunction>();
  const auto sdf = function ? function->native() : std::nullopt;
  if (!clearance.se2_base() || !sdf) return std::nullopt;
  return gpy::SE2ClearanceParts{*cs.se2_base(), *clearance.se2_base(), *sdf, clearance.kappa(),
                                clearance.beta()};
}

/// The validity of a pose for the typed SE2 space. `native` is the C++ checker of `v`.
std::function<bool(const Eigen::Vector3d&)> pose_validity(
    const nb::object& v, const std::optional<geodex::python::NativeValidity>& native) {
  if (v.is_none()) return {};
  if (native) return [checker = *native](const Eigen::Vector3d& q) { return checker(q); };
  return [v](const Eigen::Vector3d& q) { return truthy(v(q)); };
}

/// Map an InterpolationMode enum back to its interp string.
std::string interp_name(InterpolationMode mode) {
  switch (mode) {
    case InterpolationMode::Auto:
      return "auto";
    case InterpolationMode::BaseGeodesic:
      return "base_geodesic";
    case InterpolationMode::RiemannianGeodesic:
      return "riemannian_geodesic";
  }
  return "auto";
}

}  // namespace

void bind_planning(nb::module_& m) {
  // --- log level ---
  nb::enum_<gp::LogLevel>(m, "LogLevel", "How much plan() lets its planners print.")
      .value("Debug", gp::LogLevel::Debug)
      .value("Info", gp::LogLevel::Info)
      .value("Warn", gp::LogLevel::Warn)
      .value("Error", gp::LogLevel::Error)
      .value("Off", gp::LogLevel::Off);
  m.def("set_log_level", &gp::set_log_level, nb::arg("level"),
        "Set how much plan() lets its planners print. The default, LogLevel.Warn, prints\n"
        "warnings and errors only. At LogLevel.Info, a plan that runs its planner prints its\n"
        "first-solution, refinement and smoothing times. The environment variable\n"
        "GEODEX_LOG_LEVEL (debug, info, warn, error or off, in any case) sets the starting\n"
        "level. OMPL's own level is restored after each plan.");
  m.def("log_level", &gp::log_level, "The level plan() runs its planners at.");

  // --- planners submodule ---
  auto planners = m.def_submodule("planners", "The planner of plan() and its parameters.");

  // --- planners.GreedyRRTstar ---
  nb::class_<gp::planners::GreedyRRTstar>(
      planners, "GreedyRRTstar",
      "Asymptotically optimal informed planner (G-RRT*) and its parameters.")
      .def(
          "__init__",
          [](gp::planners::GreedyRRTstar* self, double range, double greedy_ratio,
             double rewire_factor, bool greedy_cost_for_tree_pruning, unsigned int max_neighbors) {
            new (self) gp::planners::GreedyRRTstar{range, greedy_ratio, rewire_factor,
                                                   greedy_cost_for_tree_pruning, max_neighbors};
          },
          nb::arg("range") = 0.0, nb::arg("greedy_ratio") = 0.9, nb::arg("rewire_factor") = 1.1,
          nb::arg("greedy_cost_for_tree_pruning") = true, nb::arg("max_neighbors") = 0u,
          "Create GreedyRRTstar settings.\n\n"
          "Args:\n"
          "    range: Step size. 0 selects OMPL's automatic value.\n"
          "    greedy_ratio: Fraction of samples from the greedy ellipsoid.\n"
          "    rewire_factor: Rewiring radius scale.\n"
          "    greedy_cost_for_tree_pruning: Prune to the greedy set when greedy_ratio > 0.\n"
          "    max_neighbors: Cap on the k-nearest neighborhood. 0 leaves it unbounded.")
      .def_rw("range", &gp::planners::GreedyRRTstar::range,
              "Step size. 0 selects OMPL's automatic value.")
      .def_rw("greedy_ratio", &gp::planners::GreedyRRTstar::greedy_ratio,
              "Fraction of samples from the greedy ellipsoid.")
      .def_rw("rewire_factor", &gp::planners::GreedyRRTstar::rewire_factor,
              "Rewiring radius scale.")
      .def_rw("greedy_cost_for_tree_pruning",
              &gp::planners::GreedyRRTstar::greedy_cost_for_tree_pruning,
              "Prune to the greedy set when greedy_ratio > 0.")
      .def_rw("max_neighbors", &gp::planners::GreedyRRTstar::max_neighbors,
              "Cap on the k-nearest neighborhood. 0 leaves it unbounded.")
      .def("__repr__", [](const gp::planners::GreedyRRTstar& p) {
        return "GreedyRRTstar(range=" + std::to_string(p.range) +
               ", greedy_ratio=" + std::to_string(p.greedy_ratio) +
               ", rewire_factor=" + std::to_string(p.rewire_factor) +
               ", greedy_cost_for_tree_pruning=" +
               (p.greedy_cost_for_tree_pruning ? std::string{"True"} : std::string{"False"}) +
               ", max_neighbors=" + std::to_string(p.max_neighbors) + ")";
      });

  // --- PlanSettings ---
  nb::class_<gp::PlanSettings>(
      m, "PlanSettings",
      "Settings of a plan() call. The defaults solve most problems.\n\n"
      "interp selects the curve of the planner's edges by name, one of 'base_geodesic'\n"
      "(the default), 'auto' or 'riemannian_geodesic'.")
      .def(
          "__init__",
          [](gp::PlanSettings* self, double time, unsigned int iterations, double refine_time,
             std::optional<gp::planners::GreedyRRTstar> planner,
             double collision_check_resolution, const std::string& interp,
             double goal_tolerance, std::uint64_t seed, bool smooth, nb::object smoothing,
             std::optional<std::pair<Eigen::VectorXd, Eigen::VectorXd>> limits) {
            new (self) gp::PlanSettings{};
            self->time = time;
            self->iterations = iterations;
            self->refine_time = refine_time;
            if (planner) self->planner = *planner;
            self->collision_check_resolution = collision_check_resolution;
            self->interp = parse_interp(interp);
            self->goal_tolerance = goal_tolerance;
            self->seed = seed;
            self->smooth = smooth;
            if (!smoothing.is_none()) {
              self->smoothing = nb::cast<geodex::algorithm::PathSmoothingSettings>(smoothing);
            }
            self->limits = std::move(limits);
          },
          nb::arg("time") = 1.0, nb::arg("iterations") = 0u, nb::arg("refine_time") = 0.0,
          nb::arg("planner") = nb::none(), nb::arg("collision_check_resolution") = 0.0,
          nb::arg("interp") = "base_geodesic", nb::arg("goal_tolerance") = 0.0,
          nb::arg("seed") = std::uint64_t{0}, nb::arg("smooth") = true,
          nb::arg("smoothing") = nb::none(), nb::arg("limits") = nb::none(),
          "Create plan settings.\n\n"
          "Args:\n"
          "    time: Planning time budget in seconds.\n"
          "    iterations: Iteration budget. When > 0, the planner runs this many iterations\n"
          "        instead of running by time. A seeded plan with an iteration budget is\n"
          "        reproducible.\n"
          "    refine_time: Seconds of refinement after the first exact solution, within\n"
          "        time. 0 refines for the whole budget. plan() ignores it when iterations\n"
          "        is set.\n"
          "    planner: A planners.GreedyRRTstar with the planner's parameters. None selects\n"
          "        the defaults.\n"
          "    collision_check_resolution: Spacing of the edge checks of the planner and the\n"
          "        smoother, in coordinate distance. 0 keeps OMPL's default spacing for the\n"
          "        planner and gives the smoother the spacing in smoothing, or one hundredth\n"
          "        of the diagonal of the planning bounds when that is 0 too. plan() raises\n"
          "        ValueError when it is negative, not finite, or fine enough that an edge\n"
          "        across the bounds needs more than 1e7 checks. For a robot with a Scene, it\n"
          "        is instead the largest distance in meters that a robot sphere moves between\n"
          "        two checks of the smoother, 0.005 when 0 (see plan()).\n"
          "    interp: Curve of the planner's edges ('base_geodesic', 'auto',\n"
          "        'riemannian_geodesic'). 'base_geodesic', the default, uses the space's\n"
          "        geodesic. 'riemannian_geodesic' uses the metric's discrete geodesic. 'auto'\n"
          "        picks the base geodesic when the space's log is the Riemannian logarithm\n"
          "        of its metric or a motion validator is installed, and the discrete\n"
          "        geodesic otherwise.\n"
          "    goal_tolerance: Accepted distance to the goal.\n"
          "    seed: Seed of the plan. A nonzero seed gives the same plan in any state of the\n"
          "        space. 0, the default, takes a fresh seed from the space's own sampler and\n"
          "        advances it by one random_point(). Repeated plans are then independent,\n"
          "        and space.seed(s) repeats the same sequence of plans. The samples follow\n"
          "        the space's sampler kind and reproduce only under an iteration budget. On\n"
          "        a space whose distance breaks the triangle inequality, such as SE2 with\n"
          "        unequal weights, one seed can give different plans on Linux and macOS.\n"
          "    smooth: Run smooth_path on the planner's path.\n"
          "    smoothing: PathSmoothingSettings for smooth_path. None selects the defaults.\n"
          "        A nonzero collision_check_resolution overrides the one inside, except with\n"
          "        smoothing.edge_travel, whose resolution must then be positive.\n"
          "    limits: Physical limits of the coordinates as (lower, upper), such as joint\n"
          "        limits. The search stays inside them, and the smoother treats a point\n"
          "        outside them as invalid. None takes the limits that the space declares,\n"
          "        such as a robot's joint limits. Without declared limits, the search uses a\n"
          "        box around the region the space samples, and the smoother does not treat\n"
          "        this box as a limit. Limits that are not finite raise ValueError.")
      .def_rw("time", &gp::PlanSettings::time, "Planning time budget in seconds.")
      .def_rw("iterations", &gp::PlanSettings::iterations,
              "Iteration budget. When > 0, the planner runs this many iterations instead of "
              "running by time.")
      .def_rw("refine_time", &gp::PlanSettings::refine_time,
              "Seconds of refinement after the first exact solution. 0 uses the whole budget.")
      .def_rw("planner", &gp::PlanSettings::planner,
              "Parameters of the planner, a planners.GreedyRRTstar.")
      .def_rw("collision_check_resolution", &gp::PlanSettings::collision_check_resolution,
              "Edge-check spacing of the planner and the smoother. 0 keeps OMPL's default for "
              "the planner and gives the smoother the spacing in smoothing, or one hundredth "
              "of the bounds' diagonal.")
      .def_prop_rw(
          "interp", [](const gp::PlanSettings& s) { return interp_name(s.interp); },
          [](gp::PlanSettings& s, const std::string& v) { s.interp = parse_interp(v); },
          "Geodesic interpolation strategy ('auto', 'base_geodesic', 'riemannian_geodesic').")
      .def_rw("goal_tolerance", &gp::PlanSettings::goal_tolerance, "Accepted distance to the goal.")
      .def_rw("seed", &gp::PlanSettings::seed,
              "Seed of the plan. 0 takes a fresh seed from the space's own sampler.")
      .def_rw("smooth", &gp::PlanSettings::smooth, "Run smooth_path on the planner's path.")
      .def_rw("limits", &gp::PlanSettings::limits,
              "Physical (lower, upper) coordinate limits, or None.")
      .def_rw("smoothing", &gp::PlanSettings::smoothing,
              "PathSmoothingSettings forwarded to smooth_path.")
      .def("__repr__", [](const gp::PlanSettings& s) {
        return "PlanSettings(time=" + std::to_string(s.time) + ", planner=GreedyRRTstar" +
               ", interp=" + interp_name(s.interp) +
               ", goal_tolerance=" + std::to_string(s.goal_tolerance) +
               ", seed=" + std::to_string(s.seed) +
               ", smooth=" + (s.smooth ? std::string{"True"} : std::string{"False"}) + ")";
      });

  // --- DirectionalMotionValidator ---
  nb::class_<DirectionalValidatorConfig>(
      m, "DirectionalMotionValidator",
      "Forward-drivability constraint for plan() on SE(2)-like spaces.\n\n"
      "It rejects tree edges whose net body-forward motion is below -max_reverse_length,\n"
      "and planned paths drive forward. The manifold's log must return a body twist with\n"
      "the forward component at index 0. For SE2, that means retraction='exponential' with\n"
      "frame='body', and plan() refuses any other SE2. Pass it as\n"
      "plan(..., motion_validator=...).")
      .def(
          "__init__",
          [](DirectionalValidatorConfig* self, double max_reverse_length) {
            new (self) DirectionalValidatorConfig{max_reverse_length};
          },
          nb::arg("max_reverse_length") = 0.0,
          "Create the constraint.\n\n"
          "Args:\n"
          "    max_reverse_length: Reverse budget per edge in body-forward units. 0 forbids\n"
          "        any net reverse motion.")
      .def_rw("max_reverse_length", &DirectionalValidatorConfig::max_reverse_length,
              "Reverse budget per edge in body-forward units.")
      .def("__repr__", [](const DirectionalValidatorConfig& c) {
        return "DirectionalMotionValidator(max_reverse_length=" +
               std::to_string(c.max_reverse_length) + ")";
      });

  // --- PlanResult ---
  using PR = gp::PlanResult<Eigen::VectorXd>;
  nb::class_<PR>(m, "PlanResult", "Outcome of a plan() call.")
      .def_ro("solved", &PR::solved, "True when the planner found an exact solution.")
      .def_prop_ro(
          "path", [](const PR& r) { return path_to_matrix(r.path); }, nb::rv_policy::copy,
          "(N, d) float64 ndarray of the final path. When smoothed is True, it is the\n"
          "smoother's output, joined by the manifold's geodesic. Otherwise, it holds\n"
          "the planner's path, densified along the planner's own interpolation. The\n"
          "planner checked those edges on its own curve and spacing.")
      .def_prop_ro(
          "waypoints", [](const PR& r) { return r.path; }, nb::rv_policy::copy,
          "The final path as a list of np.ndarray.")
      .def_prop_ro(
          "raw_path", [](const PR& r) { return path_to_matrix(r.raw_path); }, nb::rv_policy::copy,
          "(N, d) float64 ndarray, the planner's waypoints before smoothing.")
      .def_ro("smoothed", &PR::smoothed, "True when path is the smoother's output.")
      .def_ro("smooth_ms", &PR::smooth_ms, "Wall-clock smoothing time in milliseconds.")
      .def_ro("cost", &PR::cost, "Geodesic length of the final path under the metric.")
      .def_ro("time_ms", &PR::time_ms, "Wall-clock solve time in milliseconds.")
      .def_ro("first_solution_ms", &PR::first_solution_ms,
              "Milliseconds from the start of the search to the first exact solution, -1 without "
              "one. The rest of time_ms refines it.")
      .def_ro("first_solution_iterations", &PR::first_solution_iterations,
              "Termination checks before the first exact solution, counted like "
              "PlanSettings.iterations, 0 without one.")
      .def_ro("informed_samples", &PR::informed_samples,
              "G-RRT* samples from an informed set, the greedy set included.")
      .def_ro("focused_samples", &PR::focused_samples,
              "G-RRT* samples from the greedy set.")
      .def_ro("uniform_samples", &PR::uniform_samples,
              "G-RRT* samples from the whole space, taken before the first solution or when "
              "the informed set is empty.")
      .def("__bool__", [](const PR& r) { return r.solved; })
      .def("__len__", [](const PR& r) { return r.path.size(); })
      .def("__repr__", [](const PR& r) {
        return "PlanResult(solved=" + (r.solved ? std::string{"True"} : std::string{"False"}) +
               ", cost=" + std::to_string(r.cost) +
               ", waypoints=" + std::to_string(r.path.size()) +
               ", time_ms=" + std::to_string(r.time_ms) + ")";
      });

  // --- plan ---
  m.def(
      "plan",
      [](nb::object space, const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
         nb::object collision, nb::object is_valid, nb::object settings, nb::object heuristic,
         nb::object motion_validator) -> PR {
        nb::object v = !is_valid.is_none() ? is_valid : collision;
        // The bound is_valid method of a C++ checker runs in C++.
        const std::optional<geodex::python::NativeValidity> native_valid =
            v.is_none() ? std::nullopt : geodex::python::NativeValidity::find(v);
        std::function<bool(const Eigen::VectorXd&)> vf;
#ifdef GEODEX_PYTHON_HAS_VAMP
        // Keep the scene's concrete validity type with its batched checks.
        std::optional<geodex::integration::vamp::VampValidity<Eigen::VectorXd>> scene_valid;
        std::optional<geodex::integration::vamp::SphereTravel> scene_travel;
#endif
        std::function<std::shared_ptr<ompl::base::MotionValidator>(
            const ompl::base::SpaceInformationPtr&)>
            mvf;
        // Smoother check spacing of a collision scene, the sphere travel between checks.
        std::optional<double> scene_resolution;
        if (!v.is_none()) {
          if (native_valid) {
            // A point that the bound method does not take goes through Python and raises its
            // TypeError.
            vf = [v, checker = *native_valid](const Eigen::VectorXd& q) {
              return checker.takes(q.size()) ? checker(q) : truthy(v(q));
            };
          } else if (PyCallable_Check(v.ptr())) {
            vf = [v](const Eigen::VectorXd& q) { return truthy(v(q)); };
          }
#ifdef GEODEX_PYTHON_HAS_VAMP
          else if (nb::isinstance<geodex::python::PyScene>(v) ||
                   nb::isinstance<geodex::integration::vamp::EnvHandle>(v)) {
            namespace gvamp = geodex::integration::vamp;
            if (!nb::isinstance<geodex::python::PyRobotModel>(space)) {
              throw std::invalid_argument(
                  "a collision scene requires a robot space (geodex.robots.*)");
            }
            const std::string name = nb::cast<const geodex::python::PyRobotModel&>(space).name();
            gvamp::EnvHandle env = nb::isinstance<geodex::python::PyScene>(v)
                                       ? nb::cast<const geodex::python::PyScene&>(v).env()
                                       : nb::cast<gvamp::EnvHandle>(v);
            // A robot's collision_check_resolution is the sphere-center travel between checks.
            // The planner and the smoother check the scene grown by the sphere travel of the
            // corner tolerance.
            const gp::PlanSettings given =
                settings.is_none() ? gp::PlanSettings{} : nb::cast<gp::PlanSettings>(settings);
            if (!std::isfinite(given.collision_check_resolution) ||
                given.collision_check_resolution < 0.0) {
              throw std::invalid_argument(
                  "plan: collision_check_resolution must be finite and >= 0");
            }
            const double tolerance = given.smoothing.corner_tolerance;
            if (!(std::isfinite(tolerance) && tolerance > 0.0)) {
              throw std::invalid_argument(
                  "plan: smoothing.corner_tolerance must be finite and > 0");
            }
            const double travel = given.collision_check_resolution > 0.0
                                      ? given.collision_check_resolution
                                      : gvamp::kDefaultMaxSphereStep;
            const double speed = gvamp::sphere_speed(name, env);
            const gvamp::EnvHandle padded = gvamp::pad_scene(env, speed * tolerance);
            scene_valid.emplace(gvamp::make_vamp_checker(name, padded));
            scene_travel = gvamp::make_sphere_travel(name, env);
            scene_resolution = travel;
            mvf = [name, padded](const ompl::base::SpaceInformationPtr& si) {
              return std::shared_ptr<ompl::base::MotionValidator>(
                  gvamp::make_vamp_motion_validator(name, si, padded).release());
            };
          }
#endif
          else {
            throw std::invalid_argument("collision must be a callable q -> bool, a Scene, or None");
          }
        }

        gp::PlanSettings s =
            settings.is_none() ? gp::PlanSettings{} : nb::cast<gp::PlanSettings>(settings);
        if (scene_resolution) {
          s.collision_check_resolution = 0.0;
          s.smoothing.collision_check_resolution = *scene_resolution;
        }
        // A base that turns at corners keeps them, and the smoother rounds the arm.
        if (nb::isinstance<geodex::python::PyRobotModel>(space) &&
            s.smoothing.sharp_coordinates == 0) {
          s.smoothing.sharp_coordinates =
              nb::cast<const geodex::python::PyRobotModel&>(space).sharp_coordinates();
        }

        namespace gh = geodex::heuristics;
        using Eig = gh::EigenvalueLowerBound<gh::Euclidean>;
        using MLB = gh::MatrixLowerBound<Eigen::Dynamic>;

        bool is_robot = false;
        geodex::python::DynamicManifold dm;
#ifdef GEODEX_PYTHON_HAS_ROBOTS
        if (nb::isinstance<geodex::python::PyRobotModel>(space)) {
          is_robot = true;
          dm = nb::cast<const geodex::python::PyRobotModel&>(space).to_dynamic_manifold();
        }
#endif
        if (!is_robot) dm = geodex::python::extract_dynamic_manifold(space);
#ifdef GEODEX_PYTHON_HAS_VAMP
        // The smoother spaces each edge of a scene plan by the edge's sphere travel.
        if (scene_travel) {
          s.smoothing.edge_travel = [sphere_travel = *scene_travel, dm](
                                        const Eigen::Ref<const Eigen::VectorXd>& a,
                                        const Eigen::Ref<const Eigen::VectorXd>& b) {
            return sphere_travel(dm.log(Eigen::VectorXd(a), Eigen::VectorXd(b)));
          };
        }
#endif

        std::optional<double> max_reverse_length;
        if (!motion_validator.is_none()) {
          if (!nb::isinstance<DirectionalValidatorConfig>(motion_validator)) {
            throw std::invalid_argument(
                "motion_validator must be a geodex.DirectionalMotionValidator");
          }
          if (nb::isinstance<geodex::python::PySE2>(space) &&
              !nb::cast<const geodex::python::PySE2&>(space).log_is_body_twist()) {
            throw std::invalid_argument(
                "DirectionalMotionValidator reads the forward speed from a body twist and "
                "needs SE2(retraction='exponential', frame='body')");
          }
          const auto cfg = nb::cast<DirectionalValidatorConfig>(motion_validator);
          max_reverse_length = cfg.max_reverse_length;
          const auto inner_factory = mvf;
          mvf = [dm, cfg, inner_factory](const ompl::base::SpaceInformationPtr& si) {
            ompl::base::MotionValidatorPtr inner =
                inner_factory ? inner_factory(si)
                              : std::make_shared<ompl::base::DiscreteMotionValidator>(si.get());
            return std::make_shared<geodex::integration::ompl::DirectionalMotionValidator<
                geodex::python::DynamicManifold>>(si.get(), dm, std::move(inner),
                                                  cfg.max_reverse_length);
          };
        }
        // The evenly spaced edges and, with a scene, every smoothed edge skip the motion
        // validator. Every path the smoother keeps drives forward within the budget on each
        // edge.
        if (max_reverse_length) {
          const double budget = *max_reverse_length;
          const auto previous = s.smoothing.path_predicate;
          s.smoothing.path_predicate = [dm, budget,
                                        previous](const std::vector<Eigen::VectorXd>& path) {
            for (std::size_t k = 1; k < path.size(); ++k) {
              if (!(dm.log(path[k - 1], path[k])[0] >= -budget)) return false;
            }
            return !previous || previous(path);
          };
        }

        // ConfigurationSpace(SE2, ClearanceMetric(SE2LeftInvariantMetric, <C++ SDF>)) plans on
        // its typed space, without std::function and Python. The plan equals the plan on
        // the type-erased space.
        const auto se2_clearance = se2_clearance_parts(space);

        auto run = [&](auto h) {
          if constexpr (std::is_constructible_v<geodex::python::PlanHeuristic, decltype(h)>) {
            if (se2_clearance && start.size() == 3 && goal.size() == 3) {
              return geodex::python::plan_se2_clearance(*se2_clearance, start, goal,
                                                        pose_validity(v, native_valid), s, h,
                                                        max_reverse_length);
            }
          }
#ifdef GEODEX_PYTHON_HAS_VAMP
          if (scene_valid) {
            return gp::plan<geodex::python::DynamicManifold, decltype(h)>(dm, start, goal,
                                                                          *scene_valid, s, h, mvf);
          }
#endif
          return gp::plan<geodex::python::DynamicManifold, decltype(h)>(dm, start, goal, vf, s, h,
                                                                        mvf);
        };

        // An explicit heuristic overrides the per-manifold default. The Euclidean chord
        // distance is admissible only when lambda_min(M) >= 1 everywhere. A custom-metric
        // ConfigurationSpace needs Zero or a lower-bound heuristic.
        if (!heuristic.is_none()) {
          if (nb::isinstance<gh::Zero>(heuristic)) return run(gh::Zero{});
          if (nb::isinstance<gh::Euclidean>(heuristic)) return run(gh::Euclidean{});
          if (nb::isinstance<Eig>(heuristic)) return run(nb::cast<Eig>(heuristic));
          if (nb::isinstance<MLB>(heuristic)) return run(nb::cast<MLB>(heuristic));
          throw std::invalid_argument(
              "heuristic must be None or a geodex.heuristics.* instance (Zero, Euclidean, "
              "EigenvalueLowerBound, MatrixLowerBound)");
        }

#ifdef GEODEX_PYTHON_HAS_ROBOTS
        // Robots default to their precomputed Loewner lower-bound heuristic, wrapped on a
        // mobile base's heading.
        if (is_robot) {
          return run(nb::cast<const geodex::python::PyRobotModel&>(space).heuristic());
        }
#endif
        // SE(2), SO(2) and the torus use their certified matrix lower bound, which wraps at
        // the angle cut. Their raw coordinate chords wrap, and SE(2) chords also measure a
        // rotating frame.
        if (nb::isinstance<geodex::python::PySE2>(space)) {
          return run(nb::cast<const geodex::python::PySE2&>(space).matrix_lower_bound());
        }
        if (nb::isinstance<geodex::python::PySO2>(space)) {
          return run(nb::cast<const geodex::python::PySO2&>(space).matrix_lower_bound());
        }
        if (nb::isinstance<geodex::python::PyTorus>(space)) {
          return run(nb::cast<const geodex::python::PyTorus&>(space).matrix_lower_bound());
        }
        return run(gh::Euclidean{});
      },
      nb::arg("space"), nb::arg("start"), nb::arg("goal"), nb::arg("collision") = nb::none(),
      nb::kw_only(), nb::arg("is_valid") = nb::none(), nb::arg("settings") = nb::none(),
      nb::arg("heuristic") = nb::none(), nb::arg("motion_validator") = nb::none(),
      "Plan a collision-free path from start to goal on a manifold.\n\n"
      "Args:\n"
      "    space: Any geodex manifold (Sphere, Euclidean, Torus, SE2, SO2, SO3, SE3,\n"
      "        ConfigurationSpace, or Product). A ConfigurationSpace over SE2 with a\n"
      "        ClearanceMetric of an SE2LeftInvariantMetric and a geodex.collision SDF plans\n"
      "        in C++ without calling into Python.\n"
      "    start: Start point (np.ndarray).\n"
      "    goal: Goal point (np.ndarray).\n"
      "    collision: Optional callable q -> bool that returns True when q is\n"
      "        collision-free, or a geodex.vamp Scene or EnvHandle for a robot space (SIMD\n"
      "        state and edge validity). Omit it for free-space planning. With a Scene, the\n"
      "        planner and the smoother check the obstacles grown by the sphere travel over\n"
      "        the corner tolerance. The smoother checks its edges at steps along which no\n"
      "        sphere moves more than settings.collision_check_resolution (0.005 m by default),\n"
      "        and it spaces each edge by the edge's own sphere travel. The plan replaces\n"
      "        settings.smoothing.collision_check_resolution and smoothing.edge_travel.\n"
      "        A sphere can come closer to an obstacle between two checks than at the checks.\n"
      "        Self-collision is tested only at the sampled states. The is_valid method of a\n"
      "        geodex.collision.FootprintGridChecker or a geodex.vamp.CollisionChecker runs\n"
      "        in C++ without calling into Python.\n"
      "    is_valid: Keyword-only alias for collision. It takes precedence when both are\n"
      "        given.\n"
      "    settings: Optional PlanSettings (defaults when omitted). For a robot whose\n"
      "        base metric makes sideways motion cost more than driving, a\n"
      "        settings.smoothing.sharp_coordinates of 0 becomes 3, and the base pose may keep\n"
      "        a corner while the smoother rounds the arm's joints. A value of at least the\n"
      "        number of coordinates rounds every coordinate together.\n"
      "    heuristic: Admissible heuristic of the informed planner, a geodex.heuristics\n"
      "        instance (Zero, Euclidean, EigenvalueLowerBound or MatrixLowerBound). None\n"
      "        selects the default. Robots use their precomputed Loewner lower bound, SE2, SO2\n"
      "        and Torus use a Loewner bound computed at setup and wrapped at the cut, and\n"
      "        other spaces use the Euclidean chord distance. Pass Zero for a custom-metric\n"
      "        ConfigurationSpace, where the Euclidean bound is inadmissible.\n"
      "    motion_validator: Optional geodex.DirectionalMotionValidator restricting tree\n"
      "        edges to forward drivability on SE(2)-like spaces.\n"
      "Returns:\n"
      "    PlanResult with the path, its cost and timings, and sampling counters.\n"
      "\n"
      "Examples:\n"
      "    Plan a Panda path in a table-pick scene, then on an abstract manifold::\n"
      "\n"
      "        import numpy as np\n"
      "        import geodex\n"
      "\n"
      "        robot = geodex.robots.Panda()\n"
      "        scene = geodex.load_scene(\"table_pick.scene.yaml\")\n"
      "        result = geodex.plan(robot, start, goal, collision=scene)\n"
      "        print(result.solved, result.path.shape, result.cost)\n"
      "\n"
      "        S2 = geodex.Sphere()\n"
      "        result = geodex.plan(S2, p0, p1, is_valid=lambda q: q[2] < 0.8)");
}

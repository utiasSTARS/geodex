/// @file bind_algorithms.cpp
/// @brief Python bindings for the geodex algorithms. Binds the discrete geodesic walk,
/// distance_midpoint, smooth_path and precompute_matrix_lower_bound with their settings
/// and result types.

#include <cmath>
#include <cstdint>

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "geodex/algorithm/distance.hpp"
#include "geodex/algorithm/interpolation.hpp"
#include "geodex/algorithm/path_smoothing.hpp"
#include "geodex/algorithm/precompute_matrix_lower_bound.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"

#include "wrappers/dynamic_manifold.hpp"
#include "wrappers/extract_manifold.hpp"
#include "wrappers/py_callable.hpp"
#include "wrappers/sizes.hpp"

namespace nb = nanobind;
using namespace geodex::python;

namespace {

/// Truth value of a Python call result. A callable may return any truthy object, including a
/// numpy bool.
bool truthy(const nb::object& result) {
  const int t = PyObject_IsTrue(result.ptr());
  if (t < 0) throw nb::python_error();
  return t == 1;
}

/// Extract a DynamicManifold from any known Python manifold type.
DynamicManifold extract_algo_manifold(nb::object obj) { return extract_dynamic_manifold(obj); }

/// Pack a path (sequence of points) into an (N, d) row matrix for numpy.
Eigen::MatrixXd path_to_matrix(const std::vector<Eigen::VectorXd>& path) {
  if (path.empty()) return Eigen::MatrixXd(0, 0);
  const auto n = static_cast<Eigen::Index>(path.size());
  const Eigen::Index d = path.front().size();
  Eigen::MatrixXd out(n, d);
  for (Eigen::Index i = 0; i < n; ++i) out.row(i) = path[static_cast<std::size_t>(i)].transpose();
  return out;
}

/// Euclidean RiemannianManifold with the metric of a Python callable `q -> M(q)` and box
/// bounds `lo`, `hi`. It provides the inner_matrix, lo() and hi() that
/// `precompute_matrix_lower_bound` needs. Python does not see this class.
class PrecomputeManifold {
 public:
  using Scalar = double;
  using Point = Eigen::VectorXd;
  using Tangent = Eigen::VectorXd;
  using MetricFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

  PrecomputeManifold(MetricFn fn, Eigen::VectorXd lo, Eigen::VectorXd hi)
      : fn_(std::move(fn)), lo_(std::move(lo)), hi_(std::move(hi)) {
    if (lo_.size() != hi_.size()) {
      throw std::invalid_argument("precompute_matrix_lower_bound: lo and hi must match in size.");
    }
    for (int i = 0; i < lo_.size(); ++i) {
      if (!(lo_[i] <= hi_[i])) {
        throw std::invalid_argument(
            "precompute_matrix_lower_bound: each lo[i] must be <= hi[i].");
      }
    }
  }

  int dim() const { return static_cast<int>(lo_.size()); }
  const Eigen::VectorXd& lo() const { return lo_; }
  const Eigen::VectorXd& hi() const { return hi_; }

  Eigen::VectorXd random_point() const { return 0.5 * (lo_ + hi_); }

  Eigen::VectorXd exp(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const { return p + v; }
  Eigen::VectorXd log(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const { return q - p; }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u,
               const Eigen::VectorXd& v) const {
    return u.dot(metric(p) * v);
  }
  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::sqrt(inner(p, v, v));
  }
  double distance(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    return norm(p, q - p);
  }
  Eigen::VectorXd geodesic(const Eigen::VectorXd& p, const Eigen::VectorXd& q, double t) const {
    return p + t * (q - p);
  }
  double injectivity_radius() const { return std::numeric_limits<double>::infinity(); }

  Eigen::MatrixXd inner_matrix(const Eigen::VectorXd& p, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const {
    return U.transpose() * metric(p) * V;
  }

 private:
  // The Python metric at p, checked to be dim() by dim().
  Eigen::MatrixXd metric(const Eigen::VectorXd& p) const {
    Eigen::MatrixXd M = fn_(p);
    require_shape(M, dim(), dim(), "precompute_matrix_lower_bound", "the metric");
    return M;
  }

  MetricFn fn_;
  Eigen::VectorXd lo_;
  Eigen::VectorXd hi_;
};

static_assert(geodex::RiemannianManifold<PrecomputeManifold>);
static_assert(geodex::HasBatchInnerMatrix<PrecomputeManifold>);

}  // namespace

void bind_algorithms(nb::module_& m) {
  using geodex::InterpolationResult;
  using geodex::InterpolationSettings;
  using geodex::InterpolationStatus;

  // --- InterpolationStatus ---
  nb::enum_<InterpolationStatus>(m, "InterpolationStatus",
                                 "Termination status of the discrete geodesic walk.")
      .value("Converged", InterpolationStatus::Converged,
             "The distance to the target fell below the convergence tolerance.")
      .value("MaxStepsReached", InterpolationStatus::MaxStepsReached,
             "The walk used its iteration budget before reaching the tolerance.")
      .value("GradientVanished", InterpolationStatus::GradientVanished,
             "The Riemannian gradient vanished at a point other than the target.")
      .value("CutLocus", InterpolationStatus::CutLocus,
             "log returned about zero for distinct points, for example antipodal ones.")
      .value("StepShrunkToZero", InterpolationStatus::StepShrunkToZero,
             "Distortion halvings drove the step size below min_step_size.")
      .value("DegenerateInput", InterpolationStatus::DegenerateInput,
             "Start and target are equal. The path has a single point.");

  // --- InterpolationSettings ---
  nb::class_<InterpolationSettings>(
      m, "InterpolationSettings",
      "Settings for the discrete geodesic walk.\n\n"
      "Each iteration takes a Riemannian step of length min(step_size, remaining_distance)\n"
      "in the descent direction. step_size also sets the path resolution. The iteration\n"
      "count and the path size scale as initial_distance / step_size.")
      .def(
          "__init__",
          [](InterpolationSettings* s, double step_size, double convergence_tol,
             double convergence_rel, int max_steps, double fd_epsilon, double distortion_ratio,
             double growth_factor, double min_step_size, double gradient_eps, double cut_locus_eps,
             bool force_log_direction, double fd_midpoint_guard_tau) {
            new (s) InterpolationSettings{step_size,
                                          convergence_tol,
                                          convergence_rel,
                                          max_steps,
                                          fd_epsilon,
                                          distortion_ratio,
                                          growth_factor,
                                          min_step_size,
                                          gradient_eps,
                                          cut_locus_eps,
                                          force_log_direction,
                                          fd_midpoint_guard_tau};
          },
          nb::arg("step_size") = 0.5, nb::arg("convergence_tol") = 1e-4,
          nb::arg("convergence_rel") = 1e-3, nb::arg("max_steps") = 100,
          nb::arg("fd_epsilon") = 0.0, nb::arg("distortion_ratio") = 1.5,
          nb::arg("growth_factor") = 1.5, nb::arg("min_step_size") = 1e-12,
          nb::arg("gradient_eps") = 1e-12, nb::arg("cut_locus_eps") = 1e-10,
          nb::arg("force_log_direction") = false, nb::arg("fd_midpoint_guard_tau") = 0.25,
          "Create interpolation settings.\n\n"
          "Args:\n"
          "    step_size: Largest Riemannian step per iteration and the path resolution.\n"
          "    convergence_tol: Absolute stop threshold on |log(current, target)|_R.\n"
          "    convergence_rel: Relative stop threshold (distance < rel * initial_distance).\n"
          "    max_steps: Maximum number of successful gradient-descent steps.\n"
          "    fd_epsilon: Central finite-difference step of the fallback gradient. 0 selects\n"
          "        it automatically.\n"
          "    distortion_ratio: Largest accepted ratio of the realized step length to the\n"
          "        intended step length. A larger step halves the step cap and retries.\n"
          "    growth_factor: Factor that regrows the step cap after a successful step.\n"
          "    min_step_size: Failure threshold after repeated distortion halvings.\n"
          "    gradient_eps: Gradient norm threshold of the GradientVanished status.\n"
          "    cut_locus_eps: |log|_R threshold that flags the CutLocus status.\n"
          "    force_log_direction: If True, always descend along -log(current, target) and\n"
          "        skip the finite-difference fallback. The path follows the base\n"
          "        retraction's geodesic instead of the metric's Riemannian geodesic.\n"
          "    fd_midpoint_guard_tau: Relative-error threshold of the midpoint distance\n"
          "        surrogate in the finite-difference gradient. Above it, the sample uses\n"
          "        |log|_R for that basis direction. 0 always uses |log|_R.")
      .def_rw("step_size", &InterpolationSettings::step_size,
              "Largest Riemannian step per iteration and the path resolution.")
      .def_rw("convergence_tol", &InterpolationSettings::convergence_tol,
              "Absolute stop threshold on |log(current, target)|_R.")
      .def_rw("convergence_rel", &InterpolationSettings::convergence_rel,
              "Relative stop threshold (distance < rel * initial_distance).")
      .def_rw("max_steps", &InterpolationSettings::max_steps,
              "Maximum number of successful gradient-descent steps.")
      .def_rw("fd_epsilon", &InterpolationSettings::fd_epsilon,
              "Central finite-difference step of the fallback gradient. 0 selects it "
              "automatically.")
      .def_rw("distortion_ratio", &InterpolationSettings::distortion_ratio,
              "Largest accepted ratio of the realized step length to the intended step "
              "length.")
      .def_rw("growth_factor", &InterpolationSettings::growth_factor,
              "Factor by which the step cap grows back after a successful iteration.")
      .def_rw("min_step_size", &InterpolationSettings::min_step_size,
              "Failure threshold after repeated distortion halvings.")
      .def_rw("gradient_eps", &InterpolationSettings::gradient_eps,
              "Gradient norm threshold of the GradientVanished status.")
      .def_rw("cut_locus_eps", &InterpolationSettings::cut_locus_eps,
              "|log|_R threshold that flags CutLocus.")
      .def_rw("force_log_direction", &InterpolationSettings::force_log_direction,
              "If True, always descend along -log(current, target) and skip the "
              "finite-difference fallback. The path follows the base retraction's geodesic "
              "instead of the metric's Riemannian geodesic.")
      .def_rw("fd_midpoint_guard_tau", &InterpolationSettings::fd_midpoint_guard_tau,
              "Relative-error threshold of the midpoint distance surrogate in the "
              "finite-difference gradient. Above it, the sample uses |log|_R.")
      .def("__repr__", [](const InterpolationSettings& s) {
        return "InterpolationSettings(step_size=" + std::to_string(s.step_size) +
               ", convergence_tol=" + std::to_string(s.convergence_tol) +
               ", max_steps=" + std::to_string(s.max_steps) + ")";
      });

  // --- InterpolationResult ---
  using PyResult = InterpolationResult<Eigen::VectorXd>;
  nb::class_<PyResult>(m, "InterpolationResult",
                       "Result of discrete_geodesic. It holds the path, the termination\n"
                       "status, the iteration count and the initial and final Riemannian\n"
                       "distances to the target.")
      .def_prop_ro(
          "path", [](const PyResult& r) { return path_to_matrix(r.path); }, nb::rv_policy::copy,
          "(N, d) float64 ndarray of the points from start toward target, start first.")
      .def_prop_ro(
          "waypoints", [](const PyResult& r) { return r.path; }, nb::rv_policy::copy,
          "The path as a list of np.ndarray.")
      .def_ro("status", &PyResult::status,
              "InterpolationStatus of the walk. Check it before using `path`.")
      .def_ro("iterations", &PyResult::iterations,
              "Number of successful gradient steps, excluding distortion retries.")
      .def_ro("distortion_halvings", &PyResult::distortion_halvings,
              "Number of times a failed progress check halved the step cap.")
      .def_ro("fd_midpoint_fallbacks", &PyResult::fd_midpoint_fallbacks,
              "Number of finite-difference samples that used |log|_R after the guard "
              "rejected the midpoint surrogate. A nonzero value flags a non-Riemannian "
              "retraction, a cut-locus crossing or a non-smooth metric near the point.")
      .def_ro("initial_distance", &PyResult::initial_distance,
              "Riemannian distance from start to target at entry.")
      .def_ro("final_distance", &PyResult::final_distance,
              "Riemannian distance from the final iterate to target at exit.")
      .def("__repr__", [](const PyResult& r) {
        return "InterpolationResult(status=" + std::string(geodex::to_string(r.status)) +
               ", iterations=" + std::to_string(r.iterations) +
               ", path_size=" + std::to_string(r.path.size()) +
               ", initial_distance=" + std::to_string(r.initial_distance) +
               ", final_distance=" + std::to_string(r.final_distance) + ")";
      });

  // --- distance_midpoint ---
  m.def(
      "distance_midpoint",
      [](nb::object manifold, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        auto dm = extract_algo_manifold(manifold);
        return geodex::distance_midpoint(dm, a, b);
      },
      nb::arg("manifold"), nb::arg("a"), nb::arg("b"),
      "Approximate geodesic distance between two points using the midpoint method.\n\n"
      "The third-order approximation is d(a,b) ≈ ||log_m(b) - log_m(a)||_m with the\n"
      "geodesic midpoint m = exp_a(0.5 * log_a(b)).\n\n"
      "Args:\n"
      "    manifold: Any geodex manifold (Sphere, Euclidean, Torus, SE2, ConfigurationSpace).\n"
      "    a: First point on the manifold.\n"
      "    b: Second point on the manifold.\n"
      "Returns:\n"
      "    Approximate geodesic distance (float).");

  // --- discrete_geodesic ---
  m.def(
      "discrete_geodesic",
      [](nb::object manifold, const Eigen::VectorXd& start, const Eigen::VectorXd& goal,
         const InterpolationSettings& settings) -> PyResult {
        auto dm = extract_algo_manifold(manifold);
        return geodex::discrete_geodesic(dm, start, goal, settings);
      },
      nb::arg("manifold"), nb::arg("start"), nb::arg("goal"),
      nb::arg("settings") = InterpolationSettings{},
      "Walk from start toward goal by Riemannian natural gradient descent.\n\n"
      "Each iteration first tries the Riemannian logarithm direction, using the identity\n"
      "grad((1/2) d^2) = -log inside the injectivity radius, and checks the progress.\n"
      "When the check fails, that step uses a central finite-difference natural gradient\n"
      "of the manifold's inner product. The iteration count and the path size scale as\n"
      "initial_distance / settings.step_size.\n\n"
      "Args:\n"
      "    manifold: Any geodex manifold (Sphere, Euclidean, Torus, SE2, ConfigurationSpace).\n"
      "    start: Starting point (np.ndarray).\n"
      "    goal: Target point (np.ndarray).\n"
      "    settings: InterpolationSettings (optional).\n"
      "Returns:\n"
      "    InterpolationResult with fields path, status, iterations, distortion_halvings,\n"
      "    fd_midpoint_fallbacks, initial_distance, final_distance.");

  // --- PathSmoothingSettings ---
  using PSS = geodex::algorithm::PathSmoothingSettings;
  using EdgeFn = std::function<bool(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using LengthFn = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using PathFn = std::function<bool(const std::vector<Eigen::VectorXd>&)>;
  // Adapt Python edge callables, which take arrays and may return any truthy object, to the
  // C++ hooks, which take Eigen::Ref and return bool. A settings object that Python owns
  // reports its hooks to the garbage collector, which frees a hook whose closure reaches the
  // settings. Copies of the settings hold their own untracked references.
  auto owner_of = [](const nb::handle self, const void* ptr) -> const void* {
    return nb::inst_state(self).second ? ptr : nullptr;
  };
  auto to_edge_predicate = [](const nb::object& fn, const void* owner) {
    geodex::algorithm::EdgePredicate out;
    if (!fn.is_none()) {
      out = [hook = PyCallable<bool>(nb::borrow<nb::callable>(fn), owner)](
                const Eigen::Ref<const Eigen::VectorXd>& a,
                const Eigen::Ref<const Eigen::VectorXd>& b) {
        nb::gil_scoped_acquire gil;
        return truthy(hook.call(Eigen::VectorXd(a), Eigen::VectorXd(b)));
      };
    }
    return out;
  };
  auto to_edge_length = [](const nb::object& fn, const void* owner) {
    geodex::algorithm::EdgeLength out;
    if (!fn.is_none()) {
      out = [hook = PyCallable<double>(nb::borrow<nb::callable>(fn), owner)](
                const Eigen::Ref<const Eigen::VectorXd>& a,
                const Eigen::Ref<const Eigen::VectorXd>& b) {
        nb::gil_scoped_acquire gil;
        return nb::cast<double>(hook.call(Eigen::VectorXd(a), Eigen::VectorXd(b)));
      };
    }
    return out;
  };
  auto to_path_predicate = [](const nb::object& fn, const void* owner) {
    PathFn out;
    if (!fn.is_none()) {
      out = [hook = PyCallable<bool>(nb::borrow<nb::callable>(fn), owner)](
                const std::vector<Eigen::VectorXd>& path) {
        nb::gil_scoped_acquire gil;
        return truthy(hook.call(path));
      };
    }
    return out;
  };
  auto from_edge_predicate =
      [](const geodex::algorithm::EdgePredicate& fn) -> std::optional<EdgeFn> {
    if (!fn) return std::nullopt;
    return EdgeFn([fn](const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return fn(a, b); });
  };
  nb::class_<PSS>(m, "PathSmoothingSettings",
                  "Settings for smooth_path. The defaults work in meters for SE(2) and in radians\n"
                  "for an arm.",
                  nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [to_edge_predicate, to_edge_length, to_path_predicate](
              PSS* self, double collision_check_resolution, double output_spacing,
              std::uint64_t seed, nb::object edge_provably_clear, nb::object edge_validator,
              nb::object path_predicate, bool round_corners, double corner_tolerance,
              double corner_max_angle, int sharp_coordinates, nb::object edge_travel) {
            new (self) PSS{};
            self->collision_check_resolution = collision_check_resolution;
            self->edge_travel = to_edge_length(edge_travel, self);
            self->output_spacing = output_spacing;
            self->seed = seed;
            self->edge_provably_clear = to_edge_predicate(edge_provably_clear, self);
            self->edge_validator = to_edge_predicate(edge_validator, self);
            self->path_predicate = to_path_predicate(path_predicate, self);
            self->round_corners = round_corners;
            self->corner_tolerance = corner_tolerance;
            self->corner_max_angle = corner_max_angle;
            self->sharp_coordinates = sharp_coordinates;
          },
          nb::arg("collision_check_resolution") = 0.0, nb::arg("output_spacing") = 0.0,
          nb::arg("seed") = std::uint64_t{42}, nb::arg("edge_provably_clear") = nb::none(),
          nb::arg("edge_validator") = nb::none(), nb::arg("path_predicate") = nb::none(),
          nb::arg("round_corners") = PSS{}.round_corners,
          nb::arg("corner_tolerance") = PSS{}.corner_tolerance,
          nb::arg("corner_max_angle") = PSS{}.corner_max_angle,
          nb::arg("sharp_coordinates") = PSS{}.sharp_coordinates,
          nb::arg("edge_travel") = nb::none(),
          "Create smoothing settings.\n\n"
          "Args:\n"
          "    collision_check_resolution: Largest spacing between validity samples along an\n"
          "        edge, measured as the coordinate norm of log(a, b), or in the units of\n"
          "        edge_travel when that is set. 0 uses one hundredth of the input path's\n"
          "        coordinate length and ignores edge_travel. smooth_path raises ValueError\n"
          "        when it is negative, not finite, or asks for more than 1e7 samples on one\n"
          "        edge.\n"
          "    output_spacing: Longest step between the returned waypoints, as the coordinate\n"
          "        norm of log. 0 leaves the step without a limit. The waypoints lie on the\n"
          "        smoothed path at equal steps between the ends and the corners without a\n"
          "        curve, at the longest step whose edges stay within corner_tolerance of it.\n"
          "    seed: Seed of the shortcut sampler. The result is a pure function of the input.\n"
          "    edge_provably_clear: Optional callable (a, b) -> bool. True proves the whole\n"
          "        geodesic from a to b valid and skips its samples. False proves nothing.\n"
          "    edge_validator: Optional callable (a, b) -> bool that decides every edge of\n"
          "        the smoothed path in place of the sampled test, for example a planner's\n"
          "        motion validator. The evenly spaced edges do not pass through it. Give an\n"
          "        asymmetric constraint also as path_predicate.\n"
          "    path_predicate: Optional callable (list of waypoints) -> bool. smooth_path\n"
          "        rejects a shortcut, a waypoint move or a rounded corner that fails it, and\n"
          "        returns the smoothed path's own waypoints when the evenly spaced path fails.\n"
          "    round_corners: Round the corners of the smoothed path into C2 curves. False\n"
          "        keeps the corners.\n"
          "    corner_tolerance: Largest distance between a rounding curve and the edges\n"
          "        between its samples, and between the smoothed path and the evenly spaced\n"
          "        edges, as the coordinate norm of log.\n"
          "    corner_max_angle: Largest turning angle of a rounded corner in radians, under\n"
          "        the metric. A sharper corner stays.\n"
          "    sharp_coordinates: Number of leading tangent coordinates whose path may keep a\n"
          "        corner, such as the pose of a differential-drive base before its arm's\n"
          "        joints. Where the curve through every coordinate stays smaller than its full\n"
          "        size, and at a corner sharper than corner_max_angle, smooth_path also tries a\n"
          "        curve that keeps the corner in these coordinates and rounds the others, and\n"
          "        the larger curve stays. exp and log must act on these coordinates apart from\n"
          "        the others, as on a product space. 0 rounds every coordinate together, and a\n"
          "        negative value raises ValueError.\n"
          "    edge_travel: Optional callable (a, b) -> float, a bound on how far the checked\n"
          "        geometry moves along the edge from a to b, in the units of a positive\n"
          "        collision_check_resolution. The checks then space the edge by it instead of\n"
          "        the coordinate norm of log(a, b). The bound must hold for every part of the\n"
          "        edge in proportion to its share, as a bound on the speed does. smooth_path\n"
          "        raises ValueError when it returns a negative or non-finite value.")
      .def_rw("collision_check_resolution", &PSS::collision_check_resolution,
              "Largest spacing between validity samples along an edge. 0 derives it from the "
              "input path.")
      .def_rw("output_spacing", &PSS::output_spacing,
              "Longest step between the evenly spaced waypoints. 0 leaves the step without a "
              "limit.")
      .def_rw("seed", &PSS::seed, "Seed of the shortcut sampler.")
      .def_rw("round_corners", &PSS::round_corners,
              "Round the corners of the smoothed path into C2 curves.")
      .def_rw("corner_tolerance", &PSS::corner_tolerance,
              "Largest distance between a rounding curve and the edges between its samples, "
              "and between the smoothed path and the evenly spaced edges, as the coordinate "
              "norm of log.")
      .def_rw("corner_max_angle", &PSS::corner_max_angle,
              "Largest turning angle of a rounded corner in radians, under the metric.")
      .def_rw("sharp_coordinates", &PSS::sharp_coordinates,
              "Number of leading tangent coordinates whose path may keep a corner while the "
              "others are rounded.")
      .def_prop_rw(
          "edge_provably_clear",
          [from_edge_predicate](const PSS& s) {
            return from_edge_predicate(s.edge_provably_clear);
          },
          [to_edge_predicate, owner_of](nb::pointer_and_handle<PSS> s, nb::object fn) {
            s.p->edge_provably_clear = to_edge_predicate(fn, owner_of(s.h, s.p));
          },
          nb::for_setter(nb::arg("fn").none()),
          "Optional sufficient test (a, b) -> bool for a whole edge, or None.")
      .def_prop_rw(
          "edge_validator",
          [from_edge_predicate](const PSS& s) { return from_edge_predicate(s.edge_validator); },
          [to_edge_predicate, owner_of](nb::pointer_and_handle<PSS> s, nb::object fn) {
            s.p->edge_validator = to_edge_predicate(fn, owner_of(s.h, s.p));
          },
          nb::for_setter(nb::arg("fn").none()),
          "Optional edge test (a, b) -> bool that replaces the sampled test, or None.")
      .def_prop_rw(
          "edge_travel",
          [](const PSS& s) -> std::optional<LengthFn> {
            if (!s.edge_travel) return std::nullopt;
            return LengthFn([fn = s.edge_travel](const Eigen::VectorXd& a,
                                                 const Eigen::VectorXd& b) { return fn(a, b); });
          },
          [to_edge_length, owner_of](nb::pointer_and_handle<PSS> s, nb::object fn) {
            s.p->edge_travel = to_edge_length(fn, owner_of(s.h, s.p));
          },
          nb::for_setter(nb::arg("fn").none()),
          "Optional bound (a, b) -> float on how far the checked geometry moves along an edge, "
          "or None.")
      .def_prop_rw(
          "path_predicate",
          [](const PSS& s) -> std::optional<PathFn> {
            if (!s.path_predicate) return std::nullopt;
            return s.path_predicate;
          },
          [to_path_predicate, owner_of](nb::pointer_and_handle<PSS> s, nb::object fn) {
            s.p->path_predicate = to_path_predicate(fn, owner_of(s.h, s.p));
          },
          nb::for_setter(nb::arg("fn").none()),
          "Optional predicate on the whole candidate path, or None.")

      .def("__repr__", [](const PSS& s) {
        return "PathSmoothingSettings(collision_check_resolution=" +
               std::to_string(s.collision_check_resolution) +
               ", output_spacing=" + std::to_string(s.output_spacing) +
               ", seed=" + std::to_string(s.seed) +
               ", round_corners=" + (s.round_corners ? std::string{"True"} : std::string{"False"}) + ")";
      });

  // --- PathSmoothingProfile ---
  using PSP = geodex::algorithm::PathSmoothingProfile;
  nb::class_<PSP>(m, "PathSmoothingProfile",
                  "Time and work counters of one smooth_path call.")
      .def_ro("total_ms", &PSP::total_ms, "Whole call, milliseconds.")
      .def_ro("shortcut_ms", &PSP::shortcut_ms, "Shortcut rounds, milliseconds.")
      .def_ro("descent_ms", &PSP::descent_ms, "Subdivision and local energy descent, milliseconds.")
      .def_ro("resample_ms", &PSP::resample_ms, "Even output spacing, milliseconds.")
      .def_ro("rounding_ms", &PSP::rounding_ms, "Corner rounding, milliseconds.")
      .def_ro("certify_ms", &PSP::certify_ms, "Final check, fallbacks included, milliseconds.")
      .def_ro("point_checks", &PSP::point_checks, "Configurations handed to the validity oracle.")
      .def_ro("batch_calls", &PSP::batch_calls, "Batched validity calls.")
      .def_ro("edge_checks", &PSP::edge_checks, "Edges tested.")
      .def_ro("edge_proofs", &PSP::edge_proofs, "Edges settled by edge_provably_clear.")
      .def_ro("shortcut_attempts", &PSP::shortcut_attempts, "Shortcuts tried.")
      .def_ro("shortcuts", &PSP::shortcuts, "Shortcuts accepted.")
      .def_ro("relax_visits", &PSP::relax_visits, "Waypoint visits of the energy descent.")
      .def_ro("relax_moves", &PSP::relax_moves, "Waypoint moves that passed the edge checks.")
      .def_ro("rounded_corners", &PSP::rounded_corners, "Corners replaced by a curve.")
      .def_ro("cusps", &PSP::cusps, "Corners that turn more than corner_max_angle.")
      .def_ro("kept_corners", &PSP::kept_corners,
              "Corners that stay after every curve size failed a check.")
      .def_ro("split_corners", &PSP::split_corners,
              "Rounded corners that keep the corner in the leading sharp_coordinates.")
      .def_ro("rounding_retries", &PSP::rounding_retries, "Curve sizes that failed a check.")
      .def_ro("input_waypoints", &PSP::input_waypoints, "Size of the input path.")
      .def_ro("output_waypoints", &PSP::output_waypoints, "Size of the returned path.")
      .def_ro("fallback", &PSP::fallback,
              "Returned stage. 0 is the smoothed path, 1 the optimized waypoints before "
              "resampling, 2 the first shortcut round and 3 the input.");

  // --- PathSmoothingResult ---
  using PSR = geodex::algorithm::PathSmoothingResult<Eigen::VectorXd>;
  nb::class_<PSR>(m, "PathSmoothingResult", "Result of smooth_path.")
      .def_prop_ro(
          "path", [](const PSR& r) { return path_to_matrix(r.path); }, nb::rv_policy::copy,
          "(N, d) float64 ndarray, the returned path.")
      .def_prop_ro(
          "waypoints", [](const PSR& r) { return r.path; }, nb::rv_policy::copy,
          "list[np.ndarray], the returned path as a Python list.")
      .def_ro("length", &PSR::length, "Metric length of the returned path.")
      .def_ro("collision_free", &PSR::collision_free,
              "True when every waypoint and every edge of the smoothed path passed the check. "
              "The returned waypoints lie on that path, and the edges between them stay within "
              "corner_tolerance of it.")
      .def_prop_ro(
          "first_invalid_index",
          [](const PSR& r) -> std::optional<std::size_t> {
            if (r.first_invalid_index == PSR::npos) return std::nullopt;
            return r.first_invalid_index;
          },
          "Index of the first waypoint that fails or starts a failing edge, None when the path "
          "passes.")
      .def_ro("profile", &PSR::profile, "Timing and work counters.")
      .def("__repr__", [](const PSR& r) {
        return "PathSmoothingResult(length=" + std::to_string(r.length) +
               ", waypoints=" + std::to_string(r.path.size()) + ", collision_free=" +
               (r.collision_free ? std::string{"True"} : std::string{"False"}) + ")";
      });

  // --- smooth_path ---
  m.def(
      "smooth_path",
      [](nb::object manifold_obj, nb::callable validity_fn,
         const std::vector<Eigen::VectorXd>& path, PSS settings) {
        const DynamicManifold manifold = extract_algo_manifold(manifold_obj);
        const auto valid = [&validity_fn](const Eigen::VectorXd& q) {
          return truthy(validity_fn(q));
        };
        return geodex::algorithm::smooth_path(manifold, valid, path, settings);
      },
      nb::arg("manifold"), nb::arg("validity_fn"), nb::arg("path"), nb::arg("settings") = PSS{},
      "Shorten and smooth a valid path under the manifold's metric.\n\n"
      "Alternates randomized shortcutting with local energy descent, rounds the corners\n"
      "into C2 curves when settings.round_corners is on, checks the result and spaces its\n"
      "waypoints evenly. collision_free is True exactly when every waypoint of the smoothed\n"
      "path passes validity_fn and every edge, interpolated along the manifold's geodesic,\n"
      "passes the edge test at settings.collision_check_resolution. Next to a rounding curve,\n"
      "a part of an edge of the shortened path passes at the samples of the whole edge, and an\n"
      "edge_validator tests the whole edge. The evenly spaced waypoints lie on that path, and\n"
      "the edges between them stay within corner_tolerance of it. When the smoothed path\n"
      "fails, smooth_path returns an earlier stage, down to the input itself.\n"
      "The check holds at the stated resolution only, and a path may touch an obstacle\n"
      "between two samples.\n\n"
      "Args:\n"
      "    manifold: Any geodex manifold.\n"
      "    validity_fn: Callable(q) -> bool, True when q is valid.\n"
      "    path: Input path as a list of waypoints, typically a planner's output.\n"
      "    settings: PathSmoothingSettings (optional).\n"
      "Returns:\n"
      "    PathSmoothingResult with path, waypoints, length, collision_free,\n"
      "    first_invalid_index and profile.");

  // --- PrecomputeMatrixLowerBoundSettings ---
  using PMLBS = geodex::algorithm::PrecomputeMatrixLowerBoundSettings;
  nb::class_<PMLBS>(
      m, "PrecomputeMatrixLowerBoundSettings",
      "Settings for `precompute_matrix_lower_bound`, which certifies a Loewner lower bound "
      "by constraint generation.")
      .def(nb::init<>(), "Create default precompute settings.")
      .def_rw("max_outer", &PMLBS::max_outer,
              "Maximum outer constraint-generation iterations.")
      .def_rw("tol", &PMLBS::tol, "Stop when lambda_min >= 1 - tol over the configuration space.")
      .def_rw("n_starts_per_iter", &PMLBS::n_starts_per_iter,
              "Multi-start seeds per outer iteration. 0 uses max(20, 10 * dim).")
      .def_rw("max_iters_per_start", &PMLBS::max_iters_per_start,
              "Max gradient-descent iterations per start.")
      .def_rw("grad_tol", &PMLBS::grad_tol,
              "Gradient-norm convergence for inner gradient descent.")
      .def_rw("fd_eps", &PMLBS::fd_eps,
              "Finite-difference step for the lambda_min gradient.")
      .def_rw("seed", &PMLBS::seed, "RNG seed for multi-start initial points.");

  // --- PrecomputeMatrixLowerBoundResult ---
  using PMLBR = geodex::algorithm::PrecomputeMatrixLowerBoundResult;
  nb::class_<PMLBR>(m, "PrecomputeMatrixLowerBoundResult",
                    "Result of `precompute_matrix_lower_bound`.")
      .def_ro("M_lower", &PMLBR::M_lower, "Certified Loewner lower bound on M(q).")
      .def_ro("lambda_min_certificate", &PMLBR::lambda_min_certificate,
              "Final worst-case lambda_min(L^-1 M(q) L^-T).")
      .def_ro("n_outer_iters", &PMLBR::n_outer_iters,
              "Outer constraint-generation iterations executed.")
      .def_ro("n_metric_evals", &PMLBR::n_metric_evals,
              "Total M(q) evaluations across the precompute.")
      .def_ro("converged", &PMLBR::converged,
              "True when lambda_min_certificate >= 1 - tol.")
      .def_ro("elapsed_ms", &PMLBR::elapsed_ms, "Wall-clock duration of the precompute (ms).")
      .def("__repr__", [](const PMLBR& r) {
        return "PrecomputeMatrixLowerBoundResult(lambda_min_certificate=" +
               std::to_string(r.lambda_min_certificate) +
               ", n_outer_iters=" + std::to_string(r.n_outer_iters) +
               ", n_metric_evals=" + std::to_string(r.n_metric_evals) +
               ", converged=" + (r.converged ? std::string{"True"} : std::string{"False"}) + ")";
      });

  // --- precompute_matrix_lower_bound ---
  m.def(
      "precompute_matrix_lower_bound",
      [](PrecomputeManifold::MetricFn metric_fn, const Eigen::VectorXd& lo,
         const Eigen::VectorXd& hi, PMLBS settings) {
        const PrecomputeManifold manifold(std::move(metric_fn), lo, hi);
        return geodex::algorithm::precompute_matrix_lower_bound(manifold, settings);
      },
      nb::arg("metric_fn"), nb::arg("lo"), nb::arg("hi"), nb::arg("settings") = PMLBS{},
      "Compute a constant SPD Loewner lower bound on M(q) by constraint generation.\n\n"
      "Args:\n"
      "    metric_fn: Callable(q) -> np.ndarray returning the SPD metric tensor M(q).\n"
      "    lo: Per-dimension lower bounds on the configuration space (np.ndarray, shape (d,)).\n"
      "    hi: Per-dimension upper bounds on the configuration space (np.ndarray, shape (d,)).\n"
      "    settings: PrecomputeMatrixLowerBoundSettings (optional).\n"
      "Returns:\n"
      "    PrecomputeMatrixLowerBoundResult with the certified bound and diagnostics.");
}

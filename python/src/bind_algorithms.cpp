/// @file bind_algorithms.cpp
/// @brief Python bindings for geodex algorithms: InterpolationSettings,
/// InterpolationStatus, InterpolationResult, distance_midpoint, discrete_geodesic,
/// PathSmoothingSettings, PathSmoothingResult, smooth_path,
/// SimplifyPathSettings, SimplifyPathResult, simplify_path,
/// PrecomputeMatrixLowerBoundSettings, PrecomputeMatrixLowerBoundResult,
/// precompute_matrix_lower_bound.

#include <cmath>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "geodex/algorithm/distance.hpp"
#include "geodex/algorithm/interpolation.hpp"
#include "geodex/algorithm/path_smoothing.hpp"
#include "geodex/algorithm/precompute_matrix_lower_bound.hpp"
#include "geodex/algorithm/simplify_path.hpp"
#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"

#include "wrappers/dynamic_manifold.hpp"
#include "wrappers/extract_manifold.hpp"

namespace nb = nanobind;
using namespace geodex::python;

namespace {

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

/// Lightweight RiemannianManifold wrapper exposing inner_matrix + lo/hi as
/// required by `precompute_matrix_lower_bound`. Backed by a Python callable
/// `q -> M(q)` plus per-dimension bounds. Not a Python-visible class; lives
/// only as the call-site bridge for `precompute_matrix_lower_bound` below.
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
    return u.dot(fn_(p) * v);
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
    return U.transpose() * fn_(p) * V;
  }

 private:
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
                                 "Termination status for the discrete geodesic walk.")
      .value("Converged", InterpolationStatus::Converged,
             "Distance to target fell below convergence tolerance.")
      .value("MaxStepsReached", InterpolationStatus::MaxStepsReached,
             "Iteration budget exhausted without reaching tolerance.")
      .value("GradientVanished", InterpolationStatus::GradientVanished,
             "Riemannian gradient norm is ~0 at a non-target point.")
      .value("CutLocus", InterpolationStatus::CutLocus,
             "log collapsed to ~0 while start and target are distinct (e.g. antipodal).")
      .value("StepShrunkToZero", InterpolationStatus::StepShrunkToZero,
             "Distortion halving drove the step size below min_step_size.")
      .value("DegenerateInput", InterpolationStatus::DegenerateInput,
             "start == target on entry; returned a single-point path.");

  // --- InterpolationSettings ---
  nb::class_<InterpolationSettings>(
      m, "InterpolationSettings",
      "Settings for the discrete geodesic walk.\n\n"
      "Walk semantics: each iteration takes a Riemannian step of length\n"
      "min(step_size, remaining_distance) in the descent direction. Iteration\n"
      "count and returned-path size scale as ~initial_distance / step_size,\n"
      "so step_size also serves as the effective path resolution.")
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
          "    step_size: Max Riemannian step per iteration (also effective path resolution).\n"
          "    convergence_tol: Absolute stop threshold on |log(current, target)|_R.\n"
          "    convergence_rel: Relative stop threshold (distance < rel * initial_distance).\n"
          "    max_steps: Maximum number of successful gradient-descent steps.\n"
          "    fd_epsilon: Central FD step for the fallback gradient; 0 means auto-select.\n"
          "    distortion_ratio: Progress-check tolerance; 1.5 requires at least 50% of the\n"
          "        intended step length in distance decrease before accepting a step.\n"
          "    growth_factor: After a successful step, regrow the step cap by this factor.\n"
          "    min_step_size: Failure threshold after repeated distortion halvings.\n"
          "    gradient_eps: Gradient norm threshold for GradientVanished status.\n"
          "    cut_locus_eps: |log|_R threshold that flags a cut-locus situation.\n"
          "    force_log_direction: If True, always use -log(current, target) as the descent\n"
          "        direction and skip the FD fallback. Produces smoother paths at the cost\n"
          "        of following the base retraction's geodesic instead of the true\n"
          "        Riemannian geodesic of the configured metric.\n"
          "    fd_midpoint_guard_tau: Relative-error threshold above which the midpoint\n"
          "        distance surrogate used inside the FD gradient is rejected and the sample\n"
          "        falls back to |log|_R for that basis direction. Set to 0 to force\n"
          "        via-log sampling every time.")
      .def_rw("step_size", &InterpolationSettings::step_size,
              "Max Riemannian step per iteration; also the effective path resolution.")
      .def_rw("convergence_tol", &InterpolationSettings::convergence_tol,
              "Absolute stop threshold on |log(current, target)|_R.")
      .def_rw("convergence_rel", &InterpolationSettings::convergence_rel,
              "Relative stop threshold (distance < rel * initial_distance).")
      .def_rw("max_steps", &InterpolationSettings::max_steps,
              "Maximum number of successful gradient-descent steps.")
      .def_rw("fd_epsilon", &InterpolationSettings::fd_epsilon,
              "Central FD step for the fallback gradient; 0 means auto-select.")
      .def_rw("distortion_ratio", &InterpolationSettings::distortion_ratio,
              "Progress-check tolerance.")
      .def_rw("growth_factor", &InterpolationSettings::growth_factor,
              "Factor by which the step cap grows back after a successful iteration.")
      .def_rw("min_step_size", &InterpolationSettings::min_step_size,
              "Failure threshold after repeated distortion halvings.")
      .def_rw("gradient_eps", &InterpolationSettings::gradient_eps,
              "Gradient Riemannian-norm threshold for GradientVanished status.")
      .def_rw("cut_locus_eps", &InterpolationSettings::cut_locus_eps,
              "|log|_R threshold that flags CutLocus.")
      .def_rw("force_log_direction", &InterpolationSettings::force_log_direction,
              "If True, always use -log(current, target) as the descent direction and skip "
              "the FD fallback. Produces smoother paths at the cost of following the base "
              "retraction's geodesic rather than the true Riemannian geodesic of the "
              "configured metric.")
      .def_rw("fd_midpoint_guard_tau", &InterpolationSettings::fd_midpoint_guard_tau,
              "Relative-error threshold above which the midpoint distance surrogate used "
              "inside the FD gradient is rejected and the sample falls back to |log|_R for "
              "that basis direction.")
      .def("__repr__", [](const InterpolationSettings& s) {
        return "InterpolationSettings(step_size=" + std::to_string(s.step_size) +
               ", convergence_tol=" + std::to_string(s.convergence_tol) +
               ", max_steps=" + std::to_string(s.max_steps) + ")";
      });

  // --- InterpolationResult ---
  using PyResult = InterpolationResult<Eigen::VectorXd>;
  nb::class_<PyResult>(m, "InterpolationResult",
                       "Output of discrete_geodesic.\n\n"
                       "Carries the discretised path, a termination status, iteration count,\n"
                       "and the initial/final Riemannian distances to target.")
      .def_prop_ro(
          "path", [](const PyResult& r) { return path_to_matrix(r.path); }, nb::rv_policy::copy,
          "(N, d) float64 ndarray — points traced from start toward target (starts with start).")
      .def_prop_ro("waypoints", [](const PyResult& r) { return r.path; }, nb::rv_policy::copy,
                   "list[np.ndarray] — the path as a Python list (back-compat with the old API).")
      .def_ro("status", &PyResult::status,
              "InterpolationStatus — termination reason. Always check before using `path`.")
      .def_ro("iterations", &PyResult::iterations,
              "Number of successful gradient steps taken (distortion retries do not count).")
      .def_ro("distortion_halvings", &PyResult::distortion_halvings,
              "Number of times the step cap was halved due to progress failure.")
      .def_ro("fd_midpoint_fallbacks", &PyResult::fd_midpoint_fallbacks,
              "Number of FD basis samples whose midpoint distance surrogate was rejected "
              "by the runtime guard and replaced with |log|_R. A nonzero value flags a "
              "non-Riemannian retraction, a cut-locus crossing, or a non-smooth metric "
              "feature within the FD neighbourhood.")
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
      "Computes a third-order approximation: d(a,b) ≈ ||log_m(b) - log_m(a)||_m\n"
      "where m = exp_a(0.5 * log_a(b)) is the geodesic midpoint.\n\n"
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
      "Walk from start toward goal via Riemannian natural gradient descent.\n\n"
      "Each iteration first tries the Riemannian logarithm direction (exploiting\n"
      "the identity grad((1/2) d^2) = -log at points inside the injectivity radius)\n"
      "and verifies via a progress check. When the check fails (non-Riemannian\n"
      "retraction or metric mismatch), the algorithm falls back for that step only\n"
      "to a central finite-difference natural gradient computed from the manifold's\n"
      "inner product.\n\n"
      "Walk semantics: iteration count and path size both scale as\n"
      "~initial_distance / settings.step_size; reduce step_size for higher resolution.\n\n"
      "Args:\n"
      "    manifold: Any geodex manifold (Sphere, Euclidean, Torus, SE2, ConfigurationSpace).\n"
      "    start: Starting point (np.ndarray).\n"
      "    goal: Target point (np.ndarray).\n"
      "    settings: InterpolationSettings (optional, uses defaults if omitted).\n"
      "Returns:\n"
      "    InterpolationResult with fields path, status, iterations, distortion_halvings,\n"
      "    fd_midpoint_fallbacks, initial_distance, final_distance.");

  // --- PathSmoothingSettings ---
  using PSS = geodex::algorithm::PathSmoothingSettings;
  nb::class_<PSS>(m, "PathSmoothingSettings", "Settings for metric-aware path smoothing.")
      .def(nb::init<>(), "Create default path smoothing settings.")
      .def_rw("max_shortcut_attempts", &PSS::max_shortcut_attempts)
      .def_rw("edge_collision_samples", &PSS::edge_collision_samples)
      .def_rw("collision_resolution", &PSS::collision_resolution)
      .def_rw("lbfgs_target_segments", &PSS::lbfgs_target_segments)
      .def_rw("lbfgs_max_iterations", &PSS::lbfgs_max_iterations)
      .def_rw("grad_tol", &PSS::grad_tol)
      .def_rw("energy_tol", &PSS::energy_tol)
      .def_rw("fd_epsilon", &PSS::fd_epsilon)
      .def_rw("lbfgs_memory", &PSS::lbfgs_memory)
      .def_rw("armijo_c", &PSS::armijo_c)
      .def_rw("max_displacement", &PSS::max_displacement)
      .def_rw("armijo_max_backtracks", &PSS::armijo_max_backtracks);

  // --- PathSmoothingResult ---
  using PSR = geodex::algorithm::PathSmoothingResult<Eigen::VectorXd>;
  nb::class_<PSR>(m, "PathSmoothingResult", "Result of path smoothing.")
      .def_prop_ro("path", [](const PSR& r) { return path_to_matrix(r.path); },
                   nb::rv_policy::copy, "(N, d) float64 ndarray — smoothed path.")
      .def_prop_ro("waypoints", [](const PSR& r) { return r.path; }, nb::rv_policy::copy,
                   "list[np.ndarray] — the path as a Python list (back-compat).")
      .def_ro("energy", &PSR::energy, "Discrete energy of the result.")
      .def_ro("distance", &PSR::distance, "Geodesic distance estimate.")
      .def_ro("vertices_removed", &PSR::vertices_removed, "Vertices removed in shortcutting.")
      .def_ro("smooth_iterations", &PSR::smooth_iterations, "L-BFGS iterations used.")
      .def_ro("collision_free", &PSR::collision_free, "Whether final path is collision-free.");

  // --- smooth_path ---
  using ValidityFn = std::function<bool(const Eigen::VectorXd&)>;
  m.def(
      "smooth_path",
      [](nb::object manifold_obj, ValidityFn validity_fn, const std::vector<Eigen::VectorXd>& path,
         PSS settings) {
        const DynamicManifold manifold = extract_algo_manifold(manifold_obj);
        return geodex::algorithm::smooth_path(manifold, validity_fn, path, settings);
      },
      nb::arg("manifold"), nb::arg("validity_fn"), nb::arg("path"), nb::arg("settings") = PSS{},
      "Smooth a path using metric-aware shortcutting and L-BFGS energy minimization.\n\n"
      "Args:\n"
      "    manifold: Any geodex manifold.\n"
      "    validity_fn: Callable(q) -> bool, returns True if collision-free.\n"
      "    path: List of waypoints (numpy arrays).\n"
      "    settings: PathSmoothingSettings (optional).");

  // --- SimplifyPathSettings ---
  using SPS = geodex::algorithm::SimplifyPathSettings;
  nb::class_<SPS>(
      m, "SimplifyPathSettings",
      "Settings for energy-aware shortcutting + collision-constrained L-BFGS smoothing.")
      .def(nb::init<>(), "Create default simplify-path settings.")
      .def_rw("max_shortcut_attempts", &SPS::max_shortcut_attempts,
              "Random shortcut attempts in phase 1.")
      .def_rw("edge_collision_samples", &SPS::edge_collision_samples,
              "Geodesic samples per edge for collision checks.")
      .def_rw("shortcut_seed", &SPS::shortcut_seed, "RNG seed for shortcut sampling.")
      .def_rw("smooth_target_segments", &SPS::smooth_target_segments,
              "Upsample resolution for the L-BFGS smoothing phase.")
      .def_rw("max_iter_per_level", &SPS::max_iter_per_level, "Max L-BFGS iterations per level.")
      .def_rw("grad_tol", &SPS::grad_tol, "Gradient infinity-norm convergence threshold.")
      .def_rw("energy_tol", &SPS::energy_tol, "Relative energy-change convergence threshold.")
      .def_rw("fd_epsilon", &SPS::fd_epsilon, "Finite-difference step for the gradient.")
      .def_rw("lbfgs_memory", &SPS::lbfgs_memory, "L-BFGS history size.")
      .def_rw("armijo_c", &SPS::armijo_c, "Armijo sufficient-decrease parameter.")
      .def_rw("max_displacement", &SPS::max_displacement,
              "Trust-region radius per waypoint (0 disables).")
      .def_rw("verbose", &SPS::verbose, "Print per-iteration info to stderr.");

  // --- SimplifyPathResult ---
  using SPR = geodex::algorithm::SimplifyPathResult<Eigen::VectorXd>;
  nb::class_<SPR>(m, "SimplifyPathResult", "Result of geodex.simplify_path.")
      .def_prop_ro("path", [](const SPR& r) { return path_to_matrix(r.path); },
                   nb::rv_policy::copy,
                   "(N, d) float64 ndarray — simplified path including endpoints.")
      .def_prop_ro("waypoints", [](const SPR& r) { return r.path; }, nb::rv_policy::copy,
                   "list[np.ndarray] — the path as a Python list (back-compat).")
      .def_ro("energy", &SPR::energy, "Discrete energy of the result.")
      .def_ro("distance", &SPR::distance, "Geodesic distance estimate sqrt(energy).")
      .def_ro("vertices_removed", &SPR::vertices_removed,
              "Vertices removed in the shortcutting phase.")
      .def_ro("smooth_iterations", &SPR::smooth_iterations,
              "L-BFGS iterations in the smoothing phase.")
      .def_ro("collision_free", &SPR::collision_free,
              "Whether the final path passed end-to-end collision validation.")
      .def("__repr__", [](const SPR& r) {
        return "SimplifyPathResult(distance=" + std::to_string(r.distance) +
               ", path_size=" + std::to_string(r.path.size()) +
               ", vertices_removed=" + std::to_string(r.vertices_removed) +
               ", smooth_iterations=" + std::to_string(r.smooth_iterations) +
               ", collision_free=" +
               (r.collision_free ? std::string{"True"} : std::string{"False"}) + ")";
      });

  // --- simplify_path ---
  m.def(
      "simplify_path",
      [](nb::object manifold_obj, ValidityFn validity_fn,
         const std::vector<Eigen::VectorXd>& initial_path, SPS settings) {
        const DynamicManifold manifold = extract_algo_manifold(manifold_obj);
        return geodex::algorithm::simplify_path(manifold, validity_fn, initial_path, settings);
      },
      nb::arg("manifold"), nb::arg("validity_fn"), nb::arg("initial_path"),
      nb::arg("settings") = SPS{},
      "Simplify a collision-free path via energy-aware shortcutting + collision-constrained "
      "L-BFGS smoothing.\n\n"
      "Args:\n"
      "    manifold: Any geodex manifold.\n"
      "    validity_fn: Callable(q) -> bool, returns True if collision-free.\n"
      "    initial_path: Collision-free path (>= 2 waypoints).\n"
      "    settings: SimplifyPathSettings (optional).");

  // --- PrecomputeMatrixLowerBoundSettings ---
  using PMLBS = geodex::algorithm::PrecomputeMatrixLowerBoundSettings;
  nb::class_<PMLBS>(
      m, "PrecomputeMatrixLowerBoundSettings",
      "Settings for `precompute_matrix_lower_bound` (Loewner-meet certificate via "
      "constraint generation).")
      .def(nb::init<>(), "Create default precompute settings.")
      .def_rw("max_outer", &PMLBS::max_outer,
              "Maximum outer constraint-generation iterations.")
      .def_rw("tol", &PMLBS::tol, "Stop when lambda_min >= 1 - tol over the configuration space.")
      .def_rw("n_starts_per_iter", &PMLBS::n_starts_per_iter,
              "Multi-start seeds per outer iter (0 = auto: max(20, 10 * dim)).")
      .def_rw("max_iters_per_start", &PMLBS::max_iters_per_start,
              "Max gradient-descent iterations per start.")
      .def_rw("grad_tol", &PMLBS::grad_tol,
              "Gradient-norm convergence for inner gradient descent.")
      .def_rw("fd_eps", &PMLBS::fd_eps,
              "Finite-difference step for the lambda_min gradient.")
      .def_rw("seed", &PMLBS::seed, "RNG seed for multi-start initial points.")
      .def_rw("verbose", &PMLBS::verbose, "Print outer-iteration diagnostics to stderr.");

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
      "Compute a constant SPD Loewner lower bound on M(q) via constraint generation.\n\n"
      "Args:\n"
      "    metric_fn: Callable(q) -> np.ndarray returning the SPD metric tensor M(q).\n"
      "    lo: Per-dimension lower bounds on the configuration space (np.ndarray, shape (d,)).\n"
      "    hi: Per-dimension upper bounds on the configuration space (np.ndarray, shape (d,)).\n"
      "    settings: PrecomputeMatrixLowerBoundSettings (optional).\n"
      "Returns:\n"
      "    PrecomputeMatrixLowerBoundResult with the certified bound and diagnostics.");
}

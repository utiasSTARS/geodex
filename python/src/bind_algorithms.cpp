/// @file bind_algorithms.cpp
/// @brief Python bindings for geodex algorithms: InterpolationSettings,
/// InterpolationStatus, InterpolationResult, distance_midpoint, discrete_geodesic,
/// EuclideanHeuristic.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <geodex/algorithm/distance.hpp>
#include <geodex/algorithm/heuristics.hpp>
#include <geodex/algorithm/interpolation.hpp>

#include "wrappers/dynamic_manifold.hpp"
#include "wrappers/py_config_space.hpp"
#include "wrappers/py_euclidean.hpp"
#include "wrappers/py_se2.hpp"
#include "wrappers/py_sphere.hpp"
#include "wrappers/py_torus.hpp"

namespace nb = nanobind;
using namespace geodex::python;

namespace {

/// Extract a DynamicManifold from any known Python manifold type.
DynamicManifold extract_algo_manifold(nb::object obj) {
  if (nb::isinstance<PySphere>(obj))
    return nb::cast<const PySphere&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyEuclidean>(obj))
    return nb::cast<const PyEuclidean&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyTorus>(obj))
    return nb::cast<const PyTorus&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySE2>(obj))
    return nb::cast<const PySE2&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyConfigurationSpace>(obj))
    return nb::cast<const PyConfigurationSpace&>(obj).to_dynamic_manifold();
  throw std::invalid_argument(
      "Unknown manifold type. Expected Sphere, Euclidean, Torus, SE2, or ConfigurationSpace.");
}

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
  nb::class_<InterpolationSettings>(m, "InterpolationSettings",
      "Settings for the discrete geodesic walk.\n\n"
      "Walk semantics: each iteration takes a Riemannian step of length\n"
      "min(step_size, remaining_distance) in the descent direction. Iteration\n"
      "count and returned-path size scale as ~initial_distance / step_size,\n"
      "so step_size also serves as the effective path resolution.")
      .def("__init__",
           [](InterpolationSettings* s, double step_size, double convergence_tol,
              double convergence_rel, int max_steps, double fd_epsilon,
              double distortion_ratio, double growth_factor, double min_step_size,
              double gradient_eps, double cut_locus_eps) {
             new (s) InterpolationSettings{step_size,       convergence_tol,  convergence_rel,
                                          max_steps,        fd_epsilon,       distortion_ratio,
                                          growth_factor,    min_step_size,    gradient_eps,
                                          cut_locus_eps};
           },
           nb::arg("step_size") = 0.5, nb::arg("convergence_tol") = 1e-4,
           nb::arg("convergence_rel") = 1e-3, nb::arg("max_steps") = 100,
           nb::arg("fd_epsilon") = 0.0, nb::arg("distortion_ratio") = 1.5,
           nb::arg("growth_factor") = 1.5, nb::arg("min_step_size") = 1e-12,
           nb::arg("gradient_eps") = 1e-12, nb::arg("cut_locus_eps") = 1e-10,
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
           "    cut_locus_eps: |log|_R threshold that flags a cut-locus situation.")
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
      .def_ro("path", &PyResult::path,
              "list[np.ndarray] — points traced from start toward target (always starts with start).")
      .def_ro("status", &PyResult::status,
              "InterpolationStatus — termination reason. Always check before using `path`.")
      .def_ro("iterations", &PyResult::iterations,
              "Number of successful gradient steps taken (distortion retries do not count).")
      .def_ro("distortion_halvings", &PyResult::distortion_halvings,
              "Number of times the step cap was halved due to progress failure.")
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
      "    initial_distance, final_distance.");

  // --- EuclideanHeuristic ---
  nb::class_<geodex::EuclideanHeuristic>(m, "EuclideanHeuristic",
      "Euclidean (L2) heuristic between coordinate vectors.\n\n"
      "Computes the chord distance ||a - b||_2. Admissible for any manifold where\n"
      "geodesic distance >= chord distance (e.g., convex subsets of Euclidean space).")
      .def(nb::init<>(), "Create a Euclidean heuristic.")
      .def(
          "__call__",
          [](const geodex::EuclideanHeuristic& h, const Eigen::VectorXd& a,
             const Eigen::VectorXd& b) { return h(a, b); },
          nb::arg("a"), nb::arg("b"), "Compute ||a - b||_2.");
}

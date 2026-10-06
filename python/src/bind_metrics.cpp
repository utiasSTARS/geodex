#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "wrappers/extract_metric.hpp"
#include "wrappers/native_collision.hpp"
#include "wrappers/py_callable.hpp"
#include "wrappers/py_metrics.hpp"

namespace nb = nanobind;
using namespace geodex::python;

namespace {

/// A composed metric calls its base metric object through a reference registered under the
/// composed metric. The garbage collector sees this edge.
DynamicMetric borrow_base_metric(nb::handle base, const void* owner) {
  return borrow_dynamic_metric(std::make_shared<const PyOwnedRef>(base, owner));
}

}  // namespace

void bind_metrics(nb::module_& m) {
  // --- KineticEnergyMetric ---
  nb::class_<PyKineticEnergyMetric>(
      m, "KineticEnergyMetric",
      "Kinetic-energy metric g(q) = M(q).\n\n"
      "The inner product at q is <u, v>_q = u^T M(q) v where M(q) is a\n"
      "symmetric positive-definite mass matrix returned by the callable.",
      nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](PyKineticEnergyMetric* self, nb::callable mass_matrix_fn) {
            new (self)
                PyKineticEnergyMetric(PyCallable<Eigen::MatrixXd>(std::move(mass_matrix_fn), self));
          },
          nb::arg("mass_matrix_fn"),
           "Create a kinetic-energy metric.\n\n"
           "Args:\n"
           "    mass_matrix_fn: Callable(q) -> np.ndarray returning the SPD mass matrix.")
      .def("inner", &PyKineticEnergyMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product <u, v>_p = u^T M(p) v.")
      .def("norm", &PyKineticEnergyMetric::norm, nb::arg("p"), nb::arg("v"),
           "Riemannian norm ||v||_p = sqrt(v^T M(p) v).")
      .def("__repr__", &PyKineticEnergyMetric::repr);

  // --- JacobiMetric ---
  nb::class_<PyJacobiMetric>(m, "JacobiMetric",
                             "Jacobi metric for minimum-time geodesics under a potential field.\n\n"
                             "The inner product at q is <u, v>_q = 2(H - P(q)) u^T M(q) v\n"
                             "where H is the total energy and P(q) is the potential energy.",
                             nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](PyJacobiMetric* self, nb::callable mass_matrix_fn, nb::callable potential_fn,
             double total_energy) {
            new (self) PyJacobiMetric(PyCallable<Eigen::MatrixXd>(std::move(mass_matrix_fn), self),
                                      PyCallable<double>(std::move(potential_fn), self),
                                      total_energy);
          },
          nb::arg("mass_matrix_fn"), nb::arg("potential_fn"), nb::arg("total_energy"),
           "Create a Jacobi metric.\n\n"
           "Args:\n"
           "    mass_matrix_fn: Callable(q) -> np.ndarray returning the SPD mass matrix.\n"
           "    potential_fn: Callable(q) -> float returning the potential energy.\n"
           "    total_energy: Total energy H (must satisfy H > P(q) everywhere).")
      .def("inner", &PyJacobiMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product 2(H - P(p)) u^T M(p) v.")
      .def("norm", &PyJacobiMetric::norm, nb::arg("p"), nb::arg("v"), "Riemannian norm.")
      .def("__repr__", &PyJacobiMetric::repr);

  // --- PullbackMetric ---
  nb::class_<PyPullbackMetric>(
      m, "PullbackMetric",
      "Pullback metric from task space to configuration space via the Jacobian.\n\n"
      "The inner product at q is <u, v>_q = u^T J(q)^T G(q) J(q) v + lambda * u^T v.",
      nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](PyPullbackMetric* self, nb::callable jacobian_fn, nb::callable task_metric_fn,
             double regularization) {
            new (self) PyPullbackMetric(PyCallable<Eigen::MatrixXd>(std::move(jacobian_fn), self),
                                        PyCallable<Eigen::MatrixXd>(std::move(task_metric_fn), self),
                                        regularization);
          },
          nb::arg("jacobian_fn"), nb::arg("task_metric_fn"), nb::arg("regularization") = 0.0,
           "Create a pullback metric.\n\n"
           "Args:\n"
           "    jacobian_fn: Callable(q) -> np.ndarray returning the Jacobian matrix.\n"
           "    task_metric_fn: Callable(q) -> np.ndarray returning the task-space SPD metric.\n"
           "    regularization: Regularization parameter lambda (default 0).")
      .def("inner", &PyPullbackMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product u^T J^T G J v + lambda * u^T v.")
      .def("norm", &PyPullbackMetric::norm, nb::arg("p"), nb::arg("v"), "Riemannian norm.")
      .def("__repr__", &PyPullbackMetric::repr);

  // --- ConstantSPDMetric ---
  nb::class_<PyConstantSPDMetric>(
      m, "ConstantSPDMetric",
      "Point-independent Riemannian metric defined by a constant SPD matrix.\n\n"
      "The inner product is <u, v> = u^T A v where A is a constant SPD matrix.")
      .def(nb::init<const Eigen::MatrixXd&>(), nb::arg("matrix"),
           "Create a constant SPD metric.\n\n"
           "Args:\n"
           "    matrix: Symmetric positive-definite weight matrix.")
      .def("inner", &PyConstantSPDMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product u^T A v.")
      .def("norm", &PyConstantSPDMetric::norm, nb::arg("p"), nb::arg("v"),
           "Riemannian norm sqrt(v^T A v).")
      .def("__repr__", &PyConstantSPDMetric::repr);

  // --- SE2LeftInvariantMetric ---
  nb::class_<PySE2LeftInvariantMetric>(
      m, "SE2LeftInvariantMetric",
      "Left-invariant metric on SE(2) with the constant diagonal inner product\n"
      "<u, v> = wx ux vx + wy uy vy + wtheta utheta vtheta on the (x, y, theta) tangent.\n\n"
      "High wy suppresses lateral sliding (differential-drive or car-like behavior).\n"
      "Pass it as the base metric of a ClearanceMetric for obstacle-aware SE(2) planning.")
      .def(nb::init<double, double, double>(), nb::arg("wx") = 1.0, nb::arg("wy") = 1.0,
           nb::arg("wtheta") = 1.0,
           "Create a left-invariant metric with weights (wx, wy, wtheta).")
      .def_static("car_like", &PySE2LeftInvariantMetric::car_like, nb::arg("turning_radius"),
                  nb::arg("lateral_penalty") = 100.0,
                  "Car-like weights with wtheta = turning_radius^2 and wy = lateral_penalty.\n"
                  "The geodesic turning radius is about sqrt(wtheta / wx).")
      .def_static("holonomic", &PySE2LeftInvariantMetric::holonomic, nb::arg("wtheta") = 1.0,
                  "Metric of a holonomic base, weights (1, 1, wtheta).")
      .def_static("differential_drive", &PySE2LeftInvariantMetric::differential_drive,
                  nb::arg("lateral_weight") = 100.0, nb::arg("wtheta") = 1.0,
                  "Metric of a differential-drive base, weights (1, lateral_weight, wtheta).\n"
                  "A large lateral weight makes sideways motion expensive.")
      .def("coordinate_lower_bound", &PySE2LeftInvariantMetric::coordinate_lower_bound,
           "Constant matrix below the metric on (x, y, theta) velocities at every heading,\n"
           "diag(min(wx, wy), min(wx, wy), wtheta). heuristics.MatrixLowerBound and\n"
           "heuristics.product_lower_bound take it with SE2.periods().")
      .def("inner", &PySE2LeftInvariantMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Left-invariant inner product.")
      .def("norm", &PySE2LeftInvariantMetric::norm, nb::arg("p"), nb::arg("v"),
           "Left-invariant norm.")
      .def_prop_ro("weights", &PySE2LeftInvariantMetric::weights, nb::rv_policy::copy,
                   "The diagonal weight vector (wx, wy, wtheta).")
      .def("__repr__", &PySE2LeftInvariantMetric::repr);

  // --- WeightedMetric ---
  nb::class_<PyWeightedMetric>(m, "WeightedMetric",
                               "Uniformly scaled metric wrapper.\n\n"
                               "The inner product is <u, v>_q = alpha * <u, v>^base_q.",
                               nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](PyWeightedMetric* self, nb::object base_metric, double alpha) {
            new (self) PyWeightedMetric(borrow_base_metric(base_metric, self), alpha);
          },
          nb::arg("base_metric"), nb::arg("alpha"),
          "Create a weighted metric.\n\n"
          "Args:\n"
          "    base_metric: Any geodex metric to scale.\n"
          "    alpha: Scaling factor (must be positive).")
      .def("inner", &PyWeightedMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Scaled Riemannian inner product alpha * <u, v>^base_p.")
      .def("norm", &PyWeightedMetric::norm, nb::arg("p"), nb::arg("v"), "Scaled Riemannian norm.")
      .def_prop_ro("alpha", &PyWeightedMetric::alpha, "The scaling factor.")
      .def("__repr__", &PyWeightedMetric::repr);

  // --- AffineCombinedMetric (dynamic-arity) ---
  nb::class_<PyAffineCombinedMetric>(
      m, "AffineCombinedMetric",
      "Positive linear combination of N Riemannian metric policies.\n\n"
      "Composes N metric policies g_1, ..., g_N with non-negative coefficients\n"
      "c_1, ..., c_N into the metric <u, v>_p = sum_k c_k <u, v>_p^{g_k}.\n"
      "Use it for composite metrics such as 'pullback + beta * kinetic-energy'. It needs\n"
      "at least one summand, non-negative coefficients and at least one positive one.",
      nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](PyAffineCombinedMetric* self, nb::list metrics, std::vector<double> coeffs) {
            std::vector<DynamicMetric> bases;
            bases.reserve(nb::len(metrics));
            for (auto handle : metrics) bases.push_back(borrow_base_metric(handle, self));
            new (self) PyAffineCombinedMetric(std::move(bases), std::move(coeffs));
          },
          nb::arg("metrics"), nb::arg("coeffs"),
          "Create an AffineCombinedMetric from a list of metrics and a list of\n"
          "non-negative coefficients (matching length, at least one > 0).")
      .def("inner", &PyAffineCombinedMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Combined inner product sum_k c_k <u, v>_p^{g_k}.")
      .def("norm", &PyAffineCombinedMetric::norm, nb::arg("p"), nb::arg("v"),
           "Combined Riemannian norm.")
      .def_prop_ro("coeffs", &PyAffineCombinedMetric::coeffs, "The coefficient list.")
      .def_prop_ro("size", &PyAffineCombinedMetric::size, "Number of summands.")
      .def("__repr__", &PyAffineCombinedMetric::repr);

  // --- ClearanceMetric ---
  nb::class_<PyClearanceMetric>(
      m, "ClearanceMetric",
      "SDF-based conformal metric that scales a base metric by obstacle proximity.\n\n"
      "The inner product is <u,v>_q = (1 + kappa * exp(-beta * sdf(q))) * <u,v>^base_q.",
      nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](PyClearanceMetric* self, nb::object base_metric, nb::callable sdf, double kappa,
             double beta) {
            std::optional<geodex::SE2LeftInvariantMetric> se2_base;
            if (nb::isinstance<PySE2LeftInvariantMetric>(base_metric)) {
              se2_base = nb::cast<const PySE2LeftInvariantMetric&>(base_metric).impl();
            }
            new (self) PyClearanceMetric(borrow_base_metric(base_metric, self),
                                         SdfFunction(std::move(sdf), self), kappa, beta,
                                         std::move(se2_base));
          },
          nb::arg("base_metric"), nb::arg("sdf"), nb::arg("kappa") = 5.0, nb::arg("beta") = 3.0,
          "Create an SDF-based conformal metric.\n\n"
          "Args:\n"
          "    base_metric: Any geodex metric to scale.\n"
          "    sdf: Callable(q) -> float returning signed distance (positive = free). A\n"
          "        geodex.collision SDF, such as a FootprintGridChecker or a GridSDF, runs\n"
          "        in C++ without calling into Python.\n"
          "    kappa: Strength of obstacle repulsion (default 5.0).\n"
          "    beta: Falloff rate (default 3.0).")
      .def("inner", &PyClearanceMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Conformally scaled inner product.")
      .def("norm", &PyClearanceMetric::norm, nb::arg("p"), nb::arg("v"), "Conformally scaled norm.")
      .def_prop_ro("kappa", &PyClearanceMetric::kappa, "Obstacle repulsion strength.")
      .def_prop_ro("beta", &PyClearanceMetric::beta, "Falloff rate.")
      .def("__repr__", &PyClearanceMetric::repr);
}

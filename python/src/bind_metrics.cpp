#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_metrics.hpp"

namespace nb = nanobind;
using namespace geodex::python;

namespace {

/// Extract a DynamicMetric from any known Python metric type.
DynamicMetric extract_dynamic_metric(nb::object obj) {
  if (nb::isinstance<PyKineticEnergyMetric>(obj))
    return nb::cast<const PyKineticEnergyMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyJacobiMetric>(obj))
    return nb::cast<const PyJacobiMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyPullbackMetric>(obj))
    return nb::cast<const PyPullbackMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyConstantSPDMetric>(obj))
    return nb::cast<const PyConstantSPDMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyWeightedMetric>(obj))
    return nb::cast<const PyWeightedMetric&>(obj).to_dynamic_metric();
  throw std::invalid_argument(
      "Unknown metric type. Expected KineticEnergyMetric, JacobiMetric, "
      "PullbackMetric, ConstantSPDMetric, or WeightedMetric.");
}

}  // namespace

void bind_metrics(nb::module_& m) {
  // --- KineticEnergyMetric ---
  nb::class_<PyKineticEnergyMetric>(m, "KineticEnergyMetric",
      "Kinetic energy metric g(q) = M(q).\n\n"
      "The inner product at q is <u, v>_q = u^T M(q) v where M(q) is a\n"
      "symmetric positive-definite mass matrix returned by the callable.")
      .def(nb::init<MassMatrixFn>(), nb::arg("mass_matrix_fn"),
           "Create a kinetic energy metric.\n\n"
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
      "where H is the total energy and P(q) is the potential energy.")
      .def(nb::init<MassMatrixFn, PotentialFn, double>(),
           nb::arg("mass_matrix_fn"), nb::arg("potential_fn"), nb::arg("total_energy"),
           "Create a Jacobi metric.\n\n"
           "Args:\n"
           "    mass_matrix_fn: Callable(q) -> np.ndarray returning the SPD mass matrix.\n"
           "    potential_fn: Callable(q) -> float returning the potential energy.\n"
           "    total_energy: Total energy H (must satisfy H > P(q) everywhere).")
      .def("inner", &PyJacobiMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product 2(H - P(p)) u^T M(p) v.")
      .def("norm", &PyJacobiMetric::norm, nb::arg("p"), nb::arg("v"),
           "Riemannian norm.")
      .def("__repr__", &PyJacobiMetric::repr);

  // --- PullbackMetric ---
  nb::class_<PyPullbackMetric>(m, "PullbackMetric",
      "Pullback metric from task space to configuration space via the Jacobian.\n\n"
      "The inner product at q is <u, v>_q = u^T J(q)^T G(q) J(q) v + lambda * u^T v.")
      .def(nb::init<JacobianFn, TaskMetricFn, double>(),
           nb::arg("jacobian_fn"), nb::arg("task_metric_fn"), nb::arg("regularization") = 0.0,
           "Create a pullback metric.\n\n"
           "Args:\n"
           "    jacobian_fn: Callable(q) -> np.ndarray returning the Jacobian matrix.\n"
           "    task_metric_fn: Callable(q) -> np.ndarray returning the task-space SPD metric.\n"
           "    regularization: Regularization parameter lambda (default 0).")
      .def("inner", &PyPullbackMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product u^T J^T G J v + lambda * u^T v.")
      .def("norm", &PyPullbackMetric::norm, nb::arg("p"), nb::arg("v"),
           "Riemannian norm.")
      .def("__repr__", &PyPullbackMetric::repr);

  // --- ConstantSPDMetric ---
  nb::class_<PyConstantSPDMetric>(m, "ConstantSPDMetric",
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

  // --- WeightedMetric ---
  nb::class_<PyWeightedMetric>(m, "WeightedMetric",
      "Uniformly scaled metric wrapper.\n\n"
      "The inner product is <u, v>_q = alpha * <u, v>^base_q.")
      .def("__init__",
           [](PyWeightedMetric* self, nb::object base_metric, double alpha) {
             new (self) PyWeightedMetric(extract_dynamic_metric(base_metric), alpha);
           },
           nb::arg("base_metric"), nb::arg("alpha"),
           "Create a weighted metric.\n\n"
           "Args:\n"
           "    base_metric: Any geodex metric to scale.\n"
           "    alpha: Scaling factor (must be positive).")
      .def("inner", &PyWeightedMetric::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Scaled Riemannian inner product alpha * <u, v>^base_p.")
      .def("norm", &PyWeightedMetric::norm, nb::arg("p"), nb::arg("v"),
           "Scaled Riemannian norm.")
      .def_prop_ro("alpha", &PyWeightedMetric::alpha, "The scaling factor.")
      .def("__repr__", &PyWeightedMetric::repr);
}

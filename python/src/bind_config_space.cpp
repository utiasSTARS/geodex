#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/dynamic_manifold.hpp"
#include "wrappers/py_config_space.hpp"
#include "wrappers/py_euclidean.hpp"
#include "wrappers/py_metrics.hpp"
#include "wrappers/py_se2.hpp"
#include "wrappers/py_sphere.hpp"
#include "wrappers/py_torus.hpp"

namespace nb = nanobind;
using namespace geodex::python;

namespace {

/// Extract a DynamicManifold from any known Python manifold type.
DynamicManifold extract_dynamic_manifold(nb::object obj) {
  if (nb::isinstance<PyTorus>(obj)) return nb::cast<const PyTorus&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyEuclidean>(obj))
    return nb::cast<const PyEuclidean&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySphere>(obj)) return nb::cast<const PySphere&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySE2>(obj)) return nb::cast<const PySE2&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyConfigurationSpace>(obj))
    return nb::cast<const PyConfigurationSpace&>(obj).to_dynamic_manifold();
  throw std::invalid_argument(
      "Unknown manifold type. Expected Sphere, Euclidean, Torus, SE2, or ConfigurationSpace.");
}

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

void bind_config_space(nb::module_& m) {
  nb::class_<PyConfigurationSpace>(
      m, "ConfigurationSpace",
      "A configuration space combining a base manifold's topology with a custom metric.\n\n"
      "Topology operations (exp, log, dim, random_point) come from the base manifold.\n"
      "Geometry operations (inner, norm, distance) come from the custom metric.")
      .def(
          "__init__",
          [](PyConfigurationSpace* self, nb::object base, nb::object metric) {
            auto dm = extract_dynamic_manifold(base);
            auto dmet = extract_dynamic_metric(metric);
            std::string base_name = nb::repr(base).c_str();
            std::string metric_name = nb::repr(metric).c_str();
            new (self) PyConfigurationSpace(std::move(dm), std::move(dmet), std::move(base_name),
                                            std::move(metric_name));
          },
          nb::arg("base_manifold"), nb::arg("metric"),
          "Create a configuration space.\n\n"
          "Args:\n"
          "    base_manifold: Base manifold (Sphere, Euclidean, Torus, SE2, etc.).\n"
          "    metric: Custom metric (KineticEnergyMetric, ConstantSPDMetric, etc.).")
      .def("dim", &PyConfigurationSpace::dim, "Return the intrinsic dimension.")
      .def("random_point", &PyConfigurationSpace::random_point,
           "Sample a random point from the base manifold.")
      .def("inner", &PyConfigurationSpace::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product from the custom metric.")
      .def("norm", &PyConfigurationSpace::norm, nb::arg("p"), nb::arg("v"),
           "Riemannian norm from the custom metric.")
      .def("exp", &PyConfigurationSpace::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map from the base manifold.")
      .def("log", &PyConfigurationSpace::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map from the base manifold.")
      .def("distance", &PyConfigurationSpace::distance, nb::arg("p"), nb::arg("q"),
           "Geodesic distance using the midpoint approximation with the custom metric.")
      .def("geodesic", &PyConfigurationSpace::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Geodesic interpolation at parameter t in [0, 1].")
      .def("__repr__", &PyConfigurationSpace::repr);
}

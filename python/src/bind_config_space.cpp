#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/dynamic_manifold.hpp"
#include "wrappers/extract_manifold.hpp"
#include "wrappers/extract_metric.hpp"
#include "wrappers/py_callable.hpp"
#include "wrappers/py_config_space.hpp"
#include "wrappers/py_se2.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_config_space(nb::module_& m) {
  nb::class_<PyConfigurationSpace>(
      m, "ConfigurationSpace",
      "A configuration space combining a base manifold's topology with a custom metric.\n\n"
      "Topology operations (exp, log, dim, random_point) come from the base manifold.\n"
      "Geometry operations (inner, norm, distance) come from the custom metric.",
      nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](PyConfigurationSpace* self, nb::object base, nb::object metric) {
            auto dm = extract_dynamic_manifold(base);
            // The space calls the metric object through a reference that the garbage
            // collector sees, and the collector frees a cycle through a metric callable.
            // Every manifold that the space hands out holds its own reference.
            auto own = std::make_shared<const PyOwnedRef>(metric, self);
            auto dmet = borrow_dynamic_metric(own);
            auto metric_for_copies = [own]() {
              if (!own->get()) {
                throw std::runtime_error("metric was released by the garbage collector");
              }
              return borrow_dynamic_metric(
                  std::make_shared<const PyOwnedRef>(nb::handle(own->get()), nullptr));
            };
            std::string base_name = nb::repr(base).c_str();
            std::string metric_name = nb::repr(metric).c_str();
            std::optional<PySE2> se2;
            if (nb::isinstance<PySE2>(base)) se2 = nb::cast<const PySE2&>(base);
            new (self) PyConfigurationSpace(std::move(dm), std::move(dmet), std::move(base_name),
                                            std::move(metric_name), std::move(metric_for_copies));
            self->set_sources(std::move(se2), own);
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

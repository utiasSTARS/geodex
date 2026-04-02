#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_euclidean.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_euclidean(nb::module_& m) {
  nb::class_<PyEuclidean>(m, "Euclidean",
      "Euclidean manifold R^n with the standard flat metric.\n\n"
      "Exp/log are trivial (addition/subtraction).")
      .def(nb::init<int>(), nb::arg("dim"),
           "Create a Euclidean space of the given dimension.")
      .def("dim", &PyEuclidean::dim, "Return the dimension.")
      .def("random_point", &PyEuclidean::random_point,
           "Sample a random point from a standard normal distribution.")
      .def("inner", &PyEuclidean::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Inner product <u, v> = u . v.")
      .def("norm", &PyEuclidean::norm, nb::arg("p"), nb::arg("v"),
           "Euclidean norm ||v||.")
      .def("exp", &PyEuclidean::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map: p + v.")
      .def("log", &PyEuclidean::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map: q - p.")
      .def("distance", &PyEuclidean::distance, nb::arg("p"), nb::arg("q"),
           "Euclidean distance ||p - q||.")
      .def("geodesic", &PyEuclidean::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Linear interpolation (1-t)*p + t*q.")
      .def("__repr__", &PyEuclidean::repr);
}

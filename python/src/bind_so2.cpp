#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_so2.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_so2(nb::module_& m) {
  nb::class_<PySO2>(m, "SO2",
                    "The special orthogonal group SO(2), the circle group S^1.\n\n"
                    "A configuration is a single angle theta in [-pi, pi) with wraparound.\n"
                    "Points and tangents are shape-(1,) arrays holding the angle and angular\n"
                    "velocity respectively. Uses the canonical (bi-invariant) metric with a\n"
                    "configurable weight.")
      .def(nb::init<double>(), nb::arg("weight") = 1.0,
           "Create an SO(2) manifold.\n\n"
           "Args:\n"
           "    weight: Positive rotational metric weight (norm scales as sqrt(weight)).")
      .def("dim", &PySO2::dim, "Return the intrinsic dimension (always 1).")
      .def("random_point", &PySO2::random_point,
           "Sample a random angle uniformly in [-pi, pi) as a shape-(1,) array.")
      .def("inner", &PySO2::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Canonical inner product <u, v>_p.")
      .def("norm", &PySO2::norm, nb::arg("p"), nb::arg("v"), "Canonical norm ||v||_p.")
      .def("exp", &PySO2::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map exp_p(v) = wrap(p + v).")
      .def("log", &PySO2::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map log_p(q) = wrap(q - p) (shortest arc).")
      .def("distance", &PySO2::distance, nb::arg("p"), nb::arg("q"), "Geodesic distance d(p, q).")
      .def("geodesic", &PySO2::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Geodesic interpolation at parameter t in [0, 1].")
      .def("__repr__", &PySO2::repr);
}

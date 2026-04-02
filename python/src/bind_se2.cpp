#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_se2.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_se2(nb::module_& m) {
  nb::class_<PySE2>(m, "SE2",
      "The special Euclidean group SE(2) = R^2 x SO(2).\n\n"
      "Poses are (x, y, theta) with theta in [-pi, pi).\n"
      "Uses a left-invariant metric with configurable weights.")
      .def(nb::init<double, double, double, const std::string&,
                    double, double, double, double>(),
           nb::arg("wx") = 1.0, nb::arg("wy") = 1.0, nb::arg("wtheta") = 1.0,
           nb::arg("retraction") = "exponential",
           nb::arg("x_lo") = 0.0, nb::arg("x_hi") = 10.0,
           nb::arg("y_lo") = 0.0, nb::arg("y_hi") = 10.0,
           "Create an SE(2) manifold.\n\n"
           "Args:\n"
           "    wx, wy, wtheta: Metric weights for (x, y, theta) components.\n"
           "    retraction: 'exponential' or 'euler'.\n"
           "    x_lo, x_hi, y_lo, y_hi: Workspace bounds for random sampling.")
      .def("dim", &PySE2::dim, "Return the intrinsic dimension (always 3).")
      .def("random_point", &PySE2::random_point,
           "Sample a random pose in the workspace bounds.")
      .def("inner", &PySE2::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Left-invariant inner product <u, v>_p.")
      .def("norm", &PySE2::norm, nb::arg("p"), nb::arg("v"),
           "Left-invariant norm ||v||_p.")
      .def("exp", &PySE2::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map (or retraction) exp_p(v).")
      .def("log", &PySE2::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map (or inverse retraction) log_p(q).")
      .def("distance", &PySE2::distance, nb::arg("p"), nb::arg("q"),
           "Geodesic distance d(p, q).")
      .def("geodesic", &PySE2::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Geodesic interpolation at parameter t in [0, 1].")
      .def("__repr__", &PySE2::repr);
}

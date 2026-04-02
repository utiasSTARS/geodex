#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_torus.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_torus(nb::module_& m) {
  nb::class_<PyTorus>(m, "Torus",
      "Flat torus T^n with periodic angle coordinates in [0, 2*pi)^n.\n\n"
      "Exp wraps to [0, 2*pi), log wraps differences to [-pi, pi).")
      .def(nb::init<int>(), nb::arg("dim"),
           "Create a flat torus of the given dimension.")
      .def("dim", &PyTorus::dim, "Return the dimension.")
      .def("random_point", &PyTorus::random_point,
           "Sample a uniformly random point in [0, 2*pi)^n.")
      .def("inner", &PyTorus::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Flat inner product <u, v> = u . v.")
      .def("norm", &PyTorus::norm, nb::arg("p"), nb::arg("v"),
           "Flat norm ||v||.")
      .def("exp", &PyTorus::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map: wrap(p + v) to [0, 2*pi)^n.")
      .def("log", &PyTorus::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map: shortest-path tangent in [-pi, pi)^n.")
      .def("distance", &PyTorus::distance, nb::arg("p"), nb::arg("q"),
           "Geodesic distance on the flat torus.")
      .def("geodesic", &PyTorus::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Geodesic interpolation at parameter t in [0, 1].")
      .def("__repr__", &PyTorus::repr);
}

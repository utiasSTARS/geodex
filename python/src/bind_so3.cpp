#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_so3.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_so3(nb::module_& m) {
  nb::class_<PySO3>(
      m, "SO3",
      "Special orthogonal group SO(3).\n\n"
      "Points are unit quaternions [x,y,z,w] (shape (4,)); tangents are body angular "
      "velocities omega (shape (3,)).\n"
      "frame='body' (left-invariant) or 'world' (right-invariant).")
      .def(nb::init<const std::string&, double>(), nb::arg("frame") = "body",
           nb::arg("weight") = 1.0,
           "Create an SO(3) manifold.\n\n"
           "Args:\n"
           "    frame: 'body' (left-invariant) or 'world' (right-invariant).\n"
           "    weight: Positive isotropic metric weight; norm scales as sqrt(weight).")
      .def("dim", &PySO3::dim, "Return the intrinsic dimension (always 3).")
      .def("random_point", &PySO3::random_point,
           "Sample a rotation uniformly (Haar measure) as a unit quaternion [x,y,z,w].")
      .def("inner", &PySO3::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product <u, v>_p of body angular velocities.")
      .def("norm", &PySO3::norm, nb::arg("p"), nb::arg("v"), "Riemannian norm ||v||_p.")
      .def("exp", &PySO3::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map (or retraction) exp_p(v).")
      .def("log", &PySO3::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map (or inverse retraction) log_p(q) (shortest arc).")
      .def("distance", &PySO3::distance, nb::arg("p"), nb::arg("q"),
           "Geodesic distance d(p, q) (rotation angle for the bi-invariant metric).")
      .def("geodesic", &PySO3::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Geodesic interpolation at parameter t in [0, 1] (quaternion SLERP).")
      .def("__repr__", &PySO3::repr);
}

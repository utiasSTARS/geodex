#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_se3.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_se3(nb::module_& m) {
  nb::class_<PySE3>(m, "SE3",
                    "The special Euclidean group SE(3) = R^3 x SO(3).\n\n"
                    "A pose is a 7-vector [tx, ty, tz, qx, qy, qz, qw]: a translation followed\n"
                    "by a scalar-last unit quaternion. A tangent is a twist [v; omega], shape (6,).\n"
                    "Geodesics are coupled screw motions. Uses a left-invariant metric with\n"
                    "configurable translation/rotation weights. frame='body' (left) or 'world' (right).")
      .def(nb::init<const std::string&, double, double, double, double, double, double, double,
                    double>(),
           nb::arg("frame") = "body", nb::arg("w_trans") = 1.0, nb::arg("w_rot") = 1.0,
           nb::arg("x_lo") = 0.0, nb::arg("x_hi") = 10.0, nb::arg("y_lo") = 0.0,
           nb::arg("y_hi") = 10.0, nb::arg("z_lo") = 0.0, nb::arg("z_hi") = 10.0,
           "Create an SE(3) manifold.\n\n"
           "Args:\n"
           "    frame: 'body' (left group exponential) or 'world' (right).\n"
           "    w_trans: Metric weight on each translational twist component.\n"
           "    w_rot: Metric weight on each rotational twist component.\n"
           "    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi: Translation sampling bounds.")
      .def("dim", &PySE3::dim, "Return the intrinsic dimension (always 6).")
      .def("random_point", &PySE3::random_point,
           "Sample a random pose: translation uniform in the box, rotation Haar-uniform on SO(3). "
           "Returns a 7-vector with a unit quaternion part.")
      .def("inner", &PySE3::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Invariant inner product <u, v>_p of two twists (6-vectors) at pose p (7-vector).")
      .def("norm", &PySE3::norm, nb::arg("p"), nb::arg("v"),
           "Invariant norm ||v||_p of a twist (6-vector) at pose p (7-vector).")
      .def("exp", &PySE3::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map (retraction) exp_p(v): a screw motion. p is (7,), v is (6,).")
      .def("log", &PySE3::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map (inverse retraction) log_p(q). Returns a twist (6,).")
      .def("distance", &PySE3::distance, nb::arg("p"), nb::arg("q"), "Geodesic distance d(p, q).")
      .def("geodesic", &PySE3::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Geodesic (screw) interpolation at parameter t in [0, 1]. Returns a pose (7,).")
      .def("__repr__", &PySE3::repr);
}

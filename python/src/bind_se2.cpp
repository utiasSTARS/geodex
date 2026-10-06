#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_se2.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_se2(nb::module_& m) {
  nb::class_<PySE2>(m, "SE2",
                    "The special Euclidean group SE(2) = R^2 x SO(2).\n\n"
                    "Poses are (x, y, theta) with theta in [-pi, pi).\n"
                    "Uses a left-invariant metric with configurable weights.")
      .def(nb::init<double, double, double, const std::string&, const std::string&, double, double,
                    double, double, const std::string&>(),
           nb::arg("wx") = 1.0, nb::arg("wy") = 1.0, nb::arg("wtheta") = 1.0,
           nb::arg("retraction") = "exponential", nb::arg("frame") = "body", nb::arg("x_lo") = 0.0,
           nb::arg("x_hi") = 10.0, nb::arg("y_lo") = 0.0, nb::arg("y_hi") = 10.0,
           nb::arg("sampler") = "scrambled",
           "Create an SE(2) manifold.\n\n"
           "Args:\n"
           "    wx, wy, wtheta: Metric weights for (x, y, theta) components.\n"
           "    retraction: 'exponential' or 'euler'. 'exponential' follows the screw\n"
           "        motion of a constant twist. 'euler' moves in a straight line while\n"
           "        turning at a constant rate, the geodesic of the metric when wx equals wy.\n"
           "    frame: 'body' (left-invariant) or 'world' (right-invariant). Applies to the\n"
           "        exponential retraction. The euler retraction ignores it.\n"
           "    x_lo, x_hi, y_lo, y_hi: Workspace bounds for random sampling.\n"
           "    sampler: 'scrambled' (default), 'halton', or 'random'.")
      .def("dim", &PySE2::dim, "Return the intrinsic dimension (always 3).")
      .def("random_point", &PySE2::random_point, "Sample a random pose in the workspace bounds.")
      .def("inner", &PySE2::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Left-invariant inner product <u, v>_p.")
      .def("norm", &PySE2::norm, nb::arg("p"), nb::arg("v"), "Left-invariant norm ||v||_p.")
      .def("exp", &PySE2::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map (or retraction) exp_p(v).")
      .def("log", &PySE2::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map (or inverse retraction) log_p(q).")
      .def("distance", &PySE2::distance, nb::arg("p"), nb::arg("q"),
           "Length of geodesic(p, q, .) under the metric. For the exponential retraction,\n"
           "it is the norm of the constant twist, which is at least the Riemannian distance.")
      .def("periods", &PySE2::periods,
           "Deck-group generators of the coordinate axes, (0, 0, 2*pi).\n\n"
           "Translation is unbounded and theta closes after a full turn. Pass this to\n"
           "heuristics.MatrixLowerBound to wrap the bound at the theta cut.")
      .def("coordinate_metric", &PySE2::coordinate_metric, nb::arg("q"),
           "The metric on coordinate velocities (xdot, ydot, thetadot).\n\n"
           "inner() and norm() measure body-frame velocities. This metric is their frame\n"
           "pullback J(theta)^T M J(theta), and the two agree only at theta = 0.")
      .def("matrix_lower_bound", &PySE2::matrix_lower_bound,
           "Certify a periods-aware Loewner lower bound for this metric.\n\n"
           "Runs constraint generation over the coordinate metric. The result is admissible\n"
           "for coordinate chords and wraps at theta. plan() uses it as the default\n"
           "heuristic on SE2.")
      .def("geodesic", &PySE2::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "The curve exp_p(t log_p(q)) at t in [0, 1]. For the exponential retraction,\n"
           "it is the screw motion of the constant twist, not a geodesic of the metric.")
      .def("seed", &PySE2::seed, nb::arg("seed"), "Reseed the sampler for reproducible sampling.")
      .def("set_sampler", &PySE2::set_sampler, nb::arg("sampler"),
           "Switch the sampler to 'scrambled', 'halton' or 'random'.")
      .def("__repr__", &PySE2::repr)
      .def_static("car_like", &PySE2::car_like, nb::arg("turning_radius"),
                  nb::arg("lateral_penalty") = 100.0, nb::arg("retraction") = "exponential",
                  nb::arg("frame") = "body", nb::arg("x_lo") = 0.0, nb::arg("x_hi") = 10.0,
                  nb::arg("y_lo") = 0.0, nb::arg("y_hi") = 10.0, nb::arg("sampler") = "scrambled",
                  "Create a car-like SE(2) manifold.\n\n"
                  "Args:\n"
                  "    turning_radius: Effective minimum turning radius.\n"
                  "    lateral_penalty: Weight suppressing sideslip (default 100).\n"
                  "    retraction: 'exponential' or 'euler'.\n"
                  "    frame: 'body' (left-invariant) or 'world' (right-invariant).\n"
                  "    x_lo, x_hi, y_lo, y_hi: Workspace bounds for random sampling.\n"
                  "    sampler: 'scrambled' (default), 'halton', or 'random'.");
}

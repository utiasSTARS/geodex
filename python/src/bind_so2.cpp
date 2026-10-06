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
      .def(nb::init<double, const std::string&>(), nb::arg("weight") = 1.0,
           nb::arg("sampler") = "scrambled",
           "Create an SO(2) manifold.\n\n"
           "Args:\n"
           "    weight: Positive rotational metric weight (norm scales as sqrt(weight)).\n"
           "    sampler: 'scrambled' (default), 'halton', or 'random'.")
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
      .def("periods", &PySO2::periods,
           "Period of the angle, (2*pi,). Pass this to heuristics.MatrixLowerBound to wrap\n"
           "the bound at the cut.")
      .def("coordinate_metric", &PySO2::coordinate_metric, nb::arg("q"),
           "The metric on the angle's velocity, a 1 by 1 matrix.")
      .def("matrix_lower_bound", &PySO2::matrix_lower_bound,
           "Certify a periods-aware Loewner lower bound for this metric. plan() uses it as\n"
           "the default heuristic on SO2.")
      .def("seed", &PySO2::seed, nb::arg("seed"), "Reseed the sampler for reproducible sampling.")
      .def("set_sampler", &PySO2::set_sampler, nb::arg("sampler"),
           "Switch the sampler to 'scrambled', 'halton' or 'random'.")
      .def("__repr__", &PySO2::repr);
}

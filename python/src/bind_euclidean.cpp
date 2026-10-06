#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_euclidean.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_euclidean(nb::module_& m) {
  nb::class_<PyEuclidean>(m, "Euclidean",
                          "Euclidean manifold R^n with the standard flat metric.\n\n"
                          "Exp/log are trivial (addition/subtraction).")
      .def(nb::init<int, const std::string&>(), nb::arg("dim"), nb::arg("sampler") = "scrambled",
           "Create a Euclidean space of the given dimension.\n\n"
           "Args:\n"
           "    dim: Dimension n.\n"
           "    sampler: 'scrambled' (default), 'halton', or 'random'.")
      .def("dim", &PyEuclidean::dim, "Return the dimension.")
      .def("random_point", &PyEuclidean::random_point,
           "Sample a random point uniformly in the sampling bounds.")
      .def("inner", &PyEuclidean::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Inner product <u, v> = u . v.")
      .def("norm", &PyEuclidean::norm, nb::arg("p"), nb::arg("v"), "Euclidean norm ||v||.")
      .def("exp", &PyEuclidean::exp, nb::arg("p"), nb::arg("v"), "Exponential map: p + v.")
      .def("log", &PyEuclidean::log, nb::arg("p"), nb::arg("q"), "Logarithmic map: q - p.")
      .def("distance", &PyEuclidean::distance, nb::arg("p"), nb::arg("q"),
           "Euclidean distance ||p - q||.")
      .def("geodesic", &PyEuclidean::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Linear interpolation (1-t)*p + t*q.")
      .def("seed", &PyEuclidean::seed, nb::arg("seed"),
           "Reseed the sampler for reproducible sampling.")
      .def("set_sampler", &PyEuclidean::set_sampler, nb::arg("sampler"),
           "Switch the sampler: 'scrambled', 'halton', or 'random'.")
      .def("set_sampling_bounds", &PyEuclidean::set_sampling_bounds, nb::arg("lo"), nb::arg("hi"),
           "Set the per-dimension sampling bounds (default [-1, 1]^n).\n\n"
           "Bounds affect random_point() and the search domain that plan() derives; the\n"
           "exp/log/metric operations are unchanged. A ConfigurationSpace built on this\n"
           "manifold inherits these bounds.")
      .def("lo", &PyEuclidean::lo, "Lower per-dimension sampling bound.")
      .def("hi", &PyEuclidean::hi, "Upper per-dimension sampling bound.")
      .def("__repr__", &PyEuclidean::repr);
}

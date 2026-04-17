#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/py_sphere.hpp"
#include "wrappers/py_sphere_n.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_sphere(nb::module_& m) {
  nb::class_<PySphere>(m, "Sphere",
                       "The 2-sphere S^2 with interchangeable retraction policy.\n\n"
                       "Points are unit vectors in R^3. Tangent vectors lie in the\n"
                       "orthogonal complement of the base point.")
      .def(nb::init<const std::string&>(), nb::arg("retraction") = "exponential",
           "Create a Sphere with the round metric.\n\n"
           "Args:\n"
           "    retraction: 'exponential' (true exp/log) or 'projection' (first-order).")
      .def("dim", &PySphere::dim, "Return the intrinsic dimension (always 2).")
      .def("random_point", &PySphere::random_point, "Sample a uniformly random point on S^2.")
      .def("project", &PySphere::project, nb::arg("p"), nb::arg("v"),
           "Project an ambient vector onto the tangent space at p.")
      .def("inner", &PySphere::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product <u, v>_p.")
      .def("norm", &PySphere::norm, nb::arg("p"), nb::arg("v"), "Riemannian norm ||v||_p.")
      .def("exp", &PySphere::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map (or retraction) exp_p(v).")
      .def("log", &PySphere::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map (or inverse retraction) log_p(q).")
      .def("distance", &PySphere::distance, nb::arg("p"), nb::arg("q"),
           "Geodesic distance d(p, q).")
      .def("geodesic", &PySphere::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Geodesic interpolation at parameter t in [0, 1].")
      .def("__repr__", &PySphere::repr);

  nb::class_<PySphereN>(m, "SphereN",
                        "The n-sphere S^n with interchangeable retraction policy.\n\n"
                        "Points are unit vectors in R^(n+1). The dimension n is set\n"
                        "at construction time.")
      .def(nb::init<int, const std::string&>(), nb::arg("dim"),
           nb::arg("retraction") = "exponential",
           "Create an n-sphere with the round metric.\n\n"
           "Args:\n"
           "    dim: Intrinsic dimension n of S^n.\n"
           "    retraction: 'exponential' (true exp/log) or 'projection' (first-order).")
      .def("dim", &PySphereN::dim, "Return the intrinsic dimension n.")
      .def("random_point", &PySphereN::random_point, "Sample a uniformly random point on S^n.")
      .def("project", &PySphereN::project, nb::arg("p"), nb::arg("v"),
           "Project an ambient vector onto the tangent space at p.")
      .def("inner", &PySphereN::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Riemannian inner product <u, v>_p.")
      .def("norm", &PySphereN::norm, nb::arg("p"), nb::arg("v"), "Riemannian norm ||v||_p.")
      .def("exp", &PySphereN::exp, nb::arg("p"), nb::arg("v"),
           "Exponential map (or retraction) exp_p(v).")
      .def("log", &PySphereN::log, nb::arg("p"), nb::arg("q"),
           "Logarithmic map (or inverse retraction) log_p(q).")
      .def("distance", &PySphereN::distance, nb::arg("p"), nb::arg("q"),
           "Geodesic distance d(p, q).")
      .def("geodesic", &PySphereN::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Geodesic interpolation at parameter t in [0, 1].")
      .def("__repr__", &PySphereN::repr);
}

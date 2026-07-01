#include <stdexcept>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "wrappers/extract_manifold.hpp"
#include "wrappers/py_product.hpp"

namespace nb = nanobind;
using namespace geodex::python;

void bind_product(nb::module_& m) {
  nb::class_<PyProduct>(
      m, "Product",
      "Riemannian product manifold M1 x M2 x ... x Mk.\n\n"
      "Compose a list of manifolds into their metric product. Points and tangents\n"
      "are the block-concatenation of the sub-manifold points/tangents; exp, log,\n"
      "and geodesic act block-wise and distance is sqrt(sum of squared block\n"
      "distances). Example: geodex.Product([geodex.Euclidean(3), geodex.SE2()]) is\n"
      "an R^3 x SE(2) mobile-manipulator configuration space.")
      .def(
          "__init__",
          [](PyProduct* self, nb::list manifolds) {
            std::vector<DynamicManifold> blocks;
            blocks.reserve(nb::len(manifolds));
            for (nb::handle h : manifolds)
              blocks.push_back(extract_dynamic_manifold(nb::borrow<nb::object>(h)));
            if (blocks.empty())
              throw std::invalid_argument("Product requires at least one manifold.");
            new (self) PyProduct(std::move(blocks));
          },
          nb::arg("manifolds"),
          "Create a product manifold from a list of manifolds.\n\n"
          "Args:\n"
          "    manifolds: A non-empty list of geodex manifolds (Euclidean, SE2, SO3, ...).")
      .def("dim", &PyProduct::dim, "Total intrinsic dimension (sum of block dims).")
      .def("random_point", &PyProduct::random_point, "Sample a random product point.")
      .def("inner", &PyProduct::inner, nb::arg("p"), nb::arg("u"), nb::arg("v"),
           "Product inner product (sum of block inner products).")
      .def("norm", &PyProduct::norm, nb::arg("p"), nb::arg("v"), "Product norm.")
      .def("exp", &PyProduct::exp, nb::arg("p"), nb::arg("v"), "Block-wise exponential map.")
      .def("log", &PyProduct::log, nb::arg("p"), nb::arg("q"), "Block-wise logarithmic map.")
      .def("distance", &PyProduct::distance, nb::arg("p"), nb::arg("q"),
           "Product geodesic distance sqrt(sum of squared block distances).")
      .def("geodesic", &PyProduct::geodesic, nb::arg("p"), nb::arg("q"), nb::arg("t"),
           "Block-wise geodesic interpolation at t in [0, 1].")
      .def("__repr__", &PyProduct::repr);
}

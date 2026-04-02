/// @file bind_algorithms.cpp
/// @brief Python bindings for geodex algorithms: distance_midpoint.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <geodex/algorithm/distance.hpp>

#include "wrappers/dynamic_manifold.hpp"
#include "wrappers/py_config_space.hpp"
#include "wrappers/py_euclidean.hpp"
#include "wrappers/py_se2.hpp"
#include "wrappers/py_sphere.hpp"
#include "wrappers/py_torus.hpp"

namespace nb = nanobind;
using namespace geodex::python;

namespace {

/// Extract a DynamicManifold from any known Python manifold type.
/// All returned DynamicManifolds have project() set (identity for flat manifolds).
DynamicManifold extract_algo_manifold(nb::object obj) {
  if (nb::isinstance<PySphere>(obj))
    return nb::cast<const PySphere&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyEuclidean>(obj))
    return nb::cast<const PyEuclidean&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyTorus>(obj))
    return nb::cast<const PyTorus&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySE2>(obj))
    return nb::cast<const PySE2&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyConfigurationSpace>(obj))
    return nb::cast<const PyConfigurationSpace&>(obj).to_dynamic_manifold();
  throw std::invalid_argument(
      "Unknown manifold type. Expected Sphere, Euclidean, Torus, SE2, or ConfigurationSpace.");
}

}  // namespace

void bind_algorithms(nb::module_& m) {
  // --- distance_midpoint ---
  m.def(
      "distance_midpoint",
      [](nb::object manifold, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        auto dm = extract_algo_manifold(manifold);
        return geodex::distance_midpoint(dm, a, b);
      },
      nb::arg("manifold"), nb::arg("a"), nb::arg("b"),
      "Approximate geodesic distance between two points using the midpoint method.\n\n"
      "Computes a third-order approximation: d(a,b) ≈ ||log_m(b) - log_m(a)||_m\n"
      "where m = exp_a(0.5 * log_a(b)) is the geodesic midpoint.\n\n"
      "Args:\n"
      "    manifold: Any geodex manifold (Sphere, Euclidean, Torus, SE2, ConfigurationSpace).\n"
      "    a: First point on the manifold.\n"
      "    b: Second point on the manifold.\n"
      "Returns:\n"
      "    Approximate geodesic distance (float).");
}

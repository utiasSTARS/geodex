/// @file extract_manifold.hpp
/// @brief Convert any known Python manifold object into a type-erased DynamicManifold.

#pragma once

#include <stdexcept>

#include <nanobind/nanobind.h>

#include "dynamic_manifold.hpp"
#include "py_config_space.hpp"
#include "py_euclidean.hpp"
#include "py_product.hpp"
#include "py_se2.hpp"
#include "py_se3.hpp"
#include "py_so2.hpp"
#include "py_so3.hpp"
#include "py_sphere.hpp"
#include "py_sphere_n.hpp"
#include "py_torus.hpp"

namespace geodex::python {

/// @brief Extract a DynamicManifold from any known Python manifold type.
/// @throws std::invalid_argument if `obj` is not a recognised manifold.
inline DynamicManifold extract_dynamic_manifold(nanobind::object obj) {
  namespace nb = nanobind;
  if (nb::isinstance<PySphere>(obj)) return nb::cast<const PySphere&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySphereN>(obj)) return nb::cast<const PySphereN&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyEuclidean>(obj))
    return nb::cast<const PyEuclidean&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyTorus>(obj)) return nb::cast<const PyTorus&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySE2>(obj)) return nb::cast<const PySE2&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySO2>(obj)) return nb::cast<const PySO2&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySO3>(obj)) return nb::cast<const PySO3&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PySE3>(obj)) return nb::cast<const PySE3&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyConfigurationSpace>(obj))
    return nb::cast<const PyConfigurationSpace&>(obj).to_dynamic_manifold();
  if (nb::isinstance<PyProduct>(obj)) return nb::cast<const PyProduct&>(obj).to_dynamic_manifold();
  throw std::invalid_argument(
      "Unknown manifold type. Expected Sphere, SphereN, Euclidean, Torus, SE2, "
      "SO2, SO3, SE3, ConfigurationSpace, or Product.");
}

}  // namespace geodex::python

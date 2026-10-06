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

#ifdef GEODEX_PYTHON_HAS_ROBOTS
#include "py_robot_model.hpp"
#endif

namespace geodex::python {

namespace detail {

inline DynamicManifold extract_unsized(nanobind::object obj) {
  namespace nb = nanobind;
#ifdef GEODEX_PYTHON_HAS_ROBOTS
  if (nb::isinstance<PyRobotModel>(obj))
    return nb::cast<const PyRobotModel&>(obj).to_dynamic_manifold();
#endif
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

}  // namespace detail

/// @brief Extract a DynamicManifold from any known Python manifold type. It checks
/// the sizes of the points and tangent vectors it is called with.
/// @throws std::invalid_argument if `obj` is not a recognized manifold.
inline DynamicManifold extract_dynamic_manifold(nanobind::object obj) {
  DynamicManifold dm = detail::extract_unsized(std::move(obj));
  dm.probe_sizes();
  return dm;
}

}  // namespace geodex::python

/// @file bind_vamp.cpp
/// @brief Python bindings for the geodex::integration::vamp submodule.
///
/// @details Exposes opaque scene loading + per-robot collision checking under
/// the `geodex.vamp` Python submodule. The OMPL `MotionValidator` factory is
/// intentionally not bound — geodex does not (yet) expose OMPL types to
/// Python; users who need batch-edge motion validation work with the C++
/// surface directly.

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "geodex/integration/vamp/registry.hpp"

namespace nb = nanobind;
namespace gvamp = geodex::integration::vamp;

void bind_vamp(nb::module_& m) {
  auto v = m.def_submodule("vamp", "VAMP-accelerated SIMD collision checking.");

  // --- EnvHandle (opaque) ---
  nb::class_<gvamp::EnvHandle>(
      v, "EnvHandle",
      "Opaque handle to a VAMP scene environment. Construct via `load_scene`; "
      "copy freely.");

  // --- CollisionChecker ---
  nb::class_<gvamp::CollisionChecker>(
      v, "CollisionChecker",
      "Per-robot point-validity collision checker.\n\n"
      "Instances are produced by `make_vamp_checker`.")
      .def(
          "is_valid",
          [](const gvamp::CollisionChecker& self, const Eigen::VectorXd& q) -> bool {
            return self.is_valid(q.data(), static_cast<int>(q.size()));
          },
          nb::arg("q"), "Check whether the configuration q is collision-free.");

  v.def(
      "load_scene",
      [](const std::string& yaml_path) { return gvamp::load_scene(yaml_path); },
      nb::arg("yaml_path"),
      "Load an MBM-style scene YAML into an opaque VAMP environment handle.\n\n"
      "Supports primitive collision objects (boxes, cylinders, spheres) and mesh\n"
      "objects (axis-aligned bounding-box approximation).");

  v.def(
      "make_vamp_checker",
      [](const std::string& robot_name, gvamp::EnvHandle env) {
        return gvamp::make_vamp_checker(robot_name, std::move(env));
      },
      nb::arg("robot_name"), nb::arg("env"),
      "Build a per-robot CollisionChecker bound to `env`.\n\n"
      "Supported robot names: 'baxter', 'fetch', 'panda', 'pr2', 'ur5'.");

  v.def(
      "registered_robots",
      []() { return gvamp::registered_robots(); },
      "Names of robots compiled into the geodex_vamp archive (sorted).");
}

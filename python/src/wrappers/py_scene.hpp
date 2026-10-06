/// @file py_scene.hpp
/// @brief Python wrapper for the in-memory VAMP collision-scene builder.

#pragma once

#include <array>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "geodex/integration/vamp/registry.hpp"

namespace geodex::python {

/// @brief Programmatic collision-scene builder exposed to Python as geodex.Scene.
///
/// Accumulates primitive obstacles (boxes, spheres, cylinders) and finalizes
/// them into an opaque VAMP environment handle. Orientation arguments use the
/// [qx, qy, qz, qw] convention. Box sizes are full extents.
class PyScene {
 public:
  PyScene() : builder_(geodex::integration::vamp::make_scene_builder()) {}

  /// Add an oriented box. size holds the full extents, and orientation is [qx, qy, qz, qw].
  void add_box(const std::array<double, 3>& position, const std::array<double, 3>& size,
               const std::array<double, 4>& orientation) {
    geodex::integration::vamp::scene_add_box(builder_, vec3(position), vec3(size),
                                             quaternion(orientation));
  }

  /// Add a sphere centered at center.
  void add_sphere(const std::array<double, 3>& center, double radius) {
    geodex::integration::vamp::scene_add_sphere(builder_, vec3(center), radius);
  }

  /// Add an oriented cylinder. Its local z axis is the cylinder axis.
  void add_cylinder(const std::array<double, 3>& position, double radius, double height,
                    const std::array<double, 4>& orientation) {
    geodex::integration::vamp::scene_add_cylinder(builder_, vec3(position), radius, height,
                                                  quaternion(orientation));
  }

  /// Finalize the scene into a sorted VAMP environment handle for the C++ planner.
  geodex::integration::vamp::EnvHandle env() const {
    return geodex::integration::vamp::build_scene(builder_);
  }

 private:
  static Eigen::Vector3d vec3(const std::array<double, 3>& a) {
    return Eigen::Vector3d(a[0], a[1], a[2]);
  }

  static Eigen::Quaterniond quaternion(const std::array<double, 4>& q) {
    return Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized();
  }

  geodex::integration::vamp::SceneBuilder builder_;
};

}  // namespace geodex::python

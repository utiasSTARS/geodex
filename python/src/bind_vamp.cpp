/// @file bind_vamp.cpp
/// @brief Python bindings for the geodex::integration::vamp submodule.
///
/// @details Exposes opaque scene loading and per-robot collision checking under the
/// `geodex.vamp` submodule. It does not bind the OMPL `MotionValidator` factory. geodex does
/// not expose OMPL types to Python, and batch-edge motion validation is available in C++.

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
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
      "Opaque handle to a VAMP scene environment. Create it with `load_scene`. Copying "
      "it is safe.");

  // --- CollisionChecker ---
  nb::class_<gvamp::CollisionChecker>(
      v, "CollisionChecker",
      "Per-robot point-validity collision checker.\n\n"
      "`make_vamp_checker` creates instances.")
      .def(
          "is_valid",
          [](const gvamp::CollisionChecker& self, const Eigen::VectorXd& q) -> bool {
            return self.is_valid(q.data(), static_cast<int>(q.size()));
          },
          nb::arg("q"), "Check whether the configuration q is inside the joint box and "
          "collision-free.")
      .def(
          "all_valid",
          [](const gvamp::CollisionChecker& self,
             const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& qs)
              -> bool {
            return self.all_valid(qs.data(), static_cast<int>(qs.cols()),
                                  static_cast<int>(qs.rows()));
          },
          nb::arg("qs"),
          "Check whether every row of the (N, dof) array qs is valid, in SIMD batches.")
      .def("batch_width", &gvamp::CollisionChecker::batch_width,
           "Number of configurations checked in one SIMD pass.");

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
      "robot_name is one of registered_robots(). Mobile manipulators take the\n"
      "whole-body configuration (x, y, theta, arm joints...).");

  v.def(
      "pad_scene",
      [](const gvamp::EnvHandle& env, double padding) { return gvamp::pad_scene(env, padding); },
      nb::arg("env"), nb::arg("padding"),
      "A copy of env with every obstacle grown by padding meters on every side.\n\n"
      "The copy keeps the attached spheres. Raises ValueError when padding is negative or\n"
      "not finite.");

  v.def(
      "sphere_speed",
      [](const std::string& robot_name, const gvamp::EnvHandle& env) {
        return gvamp::sphere_speed(robot_name, env);
      },
      nb::arg("robot_name"), nb::arg("env"),
      "Bound on how far any sphere center of the robot, and of the spheres attached in env,\n"
      "moves per unit coordinate norm of a motion, in meters.");

  v.def(
      "attach_spheres",
      [](gvamp::EnvHandle& env, const std::vector<std::array<double, 4>>& spheres) {
        gvamp::attach_spheres(env, spheres);
      },
      nb::arg("env"), nb::arg("spheres"),
      "Attach a rigid sphere set to the robot's end-effector frame.\n\n"
      "Spheres are [x, y, z, radius] in the end-effector frame and move with it. They\n"
      "are checked against the environment and the robot's own spheres. An empty list\n"
      "detaches. A checker copies the attachment when it is built. Attach or detach\n"
      "before building the checker. A checker built earlier does not see the change.");

  v.def(
      "robot_dimension", [](const std::string& robot_name) {
        return gvamp::robot_dimension(robot_name);
      },
      nb::arg("robot_name"), "Configuration dimension of a registered robot's VAMP model.");

  v.def(
      "robot_joint_names",
      [](const std::string& robot_name) { return gvamp::robot_joint_names(robot_name); },
      nb::arg("robot_name"),
      "Joint names of a registered robot's VAMP model, in configuration order.\n\n"
      "Configurations for the robot's checker and validators list these joints in this\n"
      "order. Mobile manipulators start with base_x_joint, base_y_joint and\n"
      "base_theta_joint. The names match data/robots/<robot>/robot.yaml.");

  v.def(
      "robot_spheres",
      [](const std::string& robot_name, const Eigen::VectorXd& q) {
        const auto spheres = gvamp::robot_spheres(robot_name, q.data(), static_cast<int>(q.size()));
        Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> out(
            static_cast<Eigen::Index>(spheres.size()), 4);
        for (std::size_t i = 0; i < spheres.size(); ++i) {
          for (int k = 0; k < 4; ++k) out(static_cast<Eigen::Index>(i), k) = spheres[i][k];
        }
        return out;
      },
      nb::arg("robot_name"), nb::arg("q"),
      "Collision spheres of a registered robot's VAMP model at q, one row\n"
      "[x, y, z, radius] per sphere in the world frame.");

  v.def(
      "robot_end_effector",
      [](const std::string& robot_name) { return gvamp::robot_end_effector(robot_name); },
      nb::arg("robot_name"),
      "Name of the link that holds attached spheres in a registered robot's VAMP\n"
      "model. attach_spheres poses spheres in its frame.");

  v.def(
      "registered_robots",
      []() { return gvamp::registered_robots(); },
      "Names of robots compiled into the geodex_vamp archive (sorted).");
}

/// @file scene_loader.hpp
/// @brief Header-only implementation of the scene-loader body (internal).
///
/// Pulls in VAMP collision types. Only translation units inside the @c geodex_vamp
/// static archive, which carries the matching SIMD compile options, include it.
/// Consumer translation units reach this code through the public @c load_scene
/// declaration in @c registry.hpp and do not include this header.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vamp/collision/environment.hh>
#include <vamp/collision/shapes.hh>

#include <yaml-cpp/yaml.h>

#include "geodex/integration/vamp/registry.hpp"
#include "vamp_env.hpp"

namespace geodex::integration::vamp::detail {

inline auto compose_pose(const Eigen::Vector3d& obj_t,
                         const Eigen::Matrix3d& obj_R,
                         const YAML::Node& pose)
    -> std::pair<Eigen::Vector3d, Eigen::Matrix3d> {
  const auto& pos = pose["position"];
  const auto& ori = pose["orientation"];
  Eigen::Vector3d local_t(pos[0].as<double>(), pos[1].as<double>(),
                          pos[2].as<double>());
  Eigen::Quaterniond local_q(ori[3].as<double>(), ori[0].as<double>(),
                             ori[1].as<double>(), ori[2].as<double>());
  local_q.normalize();
  return {obj_R * local_t + obj_t, obj_R * local_q.toRotationMatrix()};
}

// Typed primitive builders shared by the YAML loader and the SceneBuilder. Each shape
// takes its min_distance from VAMP's own constructor or compute_min_distance(), the
// distance from the world origin to the nearest point of the shape, which orders
// env.sort().
inline void add_box_shape(::vamp::collision::Environment<float>& env,
                          const Eigen::Vector3d& center,
                          const Eigen::Vector3d& half_extents,
                          const Eigen::Matrix3d& R) {
  ::vamp::collision::Cuboid<float> cuboid(
      center.x(), center.y(), center.z(),
      R(0, 0), R(1, 0), R(2, 0),
      R(0, 1), R(1, 1), R(2, 1),
      R(0, 2), R(1, 2), R(2, 2),
      half_extents.x(), half_extents.y(), half_extents.z());
  cuboid.min_distance = cuboid.compute_min_distance();
  env.cuboids.push_back(cuboid);
}

/// @brief Add a cylinder as the capsule on the same axis and radius.
///
/// VAMP's validity check reads `capsules` and `z_aligned_capsules` and does not read
/// `cylinders`. The capsule contains the cylinder and extends @p radius beyond each
/// flat cap, and the check stays conservative.
inline void add_cylinder_shape(::vamp::collision::Environment<float>& env,
                               const Eigen::Vector3d& center, double radius,
                               double height, const Eigen::Matrix3d& R) {
  const Eigen::Vector3d axis = R.col(2);
  const Eigen::Vector3d p1 = center - axis * (height / 2.0);
  const Eigen::Vector3d vec = axis * height;
  ::vamp::collision::Capsule<float> capsule(
      static_cast<float>(p1.x()), static_cast<float>(p1.y()), static_cast<float>(p1.z()),
      static_cast<float>(vec.x()), static_cast<float>(vec.y()), static_cast<float>(vec.z()),
      static_cast<float>(radius), static_cast<float>(1.0 / vec.squaredNorm()));
  // Compute min_distance from the closest point on the axis segment. VAMP's
  // compute_min_distance divides by the origin's distance to the axis and gives NaN when
  // the axis passes through the origin. The check then skips the whole list on some
  // platforms.
  const double s = std::clamp(-p1.dot(vec) / vec.squaredNorm(), 0.0, 1.0);
  capsule.min_distance = static_cast<float>(std::max(0.0, (p1 + s * vec).norm() - radius));
  if (capsule.xv == 0.0F && capsule.yv == 0.0F) {
    env.z_aligned_capsules.push_back(capsule);
  } else {
    env.capsules.push_back(capsule);
  }
}

inline void add_sphere_shape(::vamp::collision::Environment<float>& env,
                             const Eigen::Vector3d& center, double radius) {
  env.spheres.emplace_back(static_cast<float>(center.x()), static_cast<float>(center.y()),
                           static_cast<float>(center.z()), static_cast<float>(radius));
}

inline void add_primitive(::vamp::collision::Environment<float>& env,
                          const std::string& type, const YAML::Node& dims,
                          const Eigen::Vector3d& t, const Eigen::Matrix3d& R) {
  if (type == "box") {
    const double hx = dims[0].as<double>() / 2.0;
    const double hy = dims[1].as<double>() / 2.0;
    const double hz = dims[2].as<double>() / 2.0;
    add_box_shape(env, t, Eigen::Vector3d(hx, hy, hz), R);
  } else if (type == "cylinder") {
    const double height = dims[0].as<double>();
    const double radius = dims[1].as<double>();
    add_cylinder_shape(env, t, radius, height, R);
  } else if (type == "sphere") {
    const double radius = dims[0].as<double>();
    add_sphere_shape(env, t, radius);
  } else {
    throw std::invalid_argument("load_scene: unsupported primitive type '" + type +
                                "'; the scene takes box, cylinder and sphere");
  }
}

inline void add_mesh_aabb(::vamp::collision::Environment<float>& env,
                          const YAML::Node& mesh, const YAML::Node& pose) {
  const auto& vertices = mesh["vertices"];
  const auto& pos = pose["position"];
  const auto& ori = pose["orientation"];
  Eigen::Vector3d t(pos[0].as<double>(), pos[1].as<double>(),
                    pos[2].as<double>());
  Eigen::Quaterniond q(ori[3].as<double>(), ori[0].as<double>(),
                       ori[1].as<double>(), ori[2].as<double>());
  q.normalize();
  Eigen::Matrix3d R = q.toRotationMatrix();
  Eigen::Vector3d vmin(1e10, 1e10, 1e10);
  Eigen::Vector3d vmax(-1e10, -1e10, -1e10);
  for (const auto& v : vertices) {
    Eigen::Vector3d vl(v[0].as<double>(), v[1].as<double>(),
                       v[2].as<double>());
    Eigen::Vector3d vw = R * vl + t;
    vmin = vmin.cwiseMin(vw);
    vmax = vmax.cwiseMax(vw);
  }
  Eigen::Vector3d center = (vmin + vmax) / 2.0;
  Eigen::Vector3d half = (vmax - vmin) / 2.0;
  ::vamp::collision::Cuboid<float> cuboid(
      center.x(), center.y(), center.z(),
      1, 0, 0, 0, 1, 0, 0, 0, 1,
      half.x(), half.y(), half.z());
  cuboid.min_distance = cuboid.compute_min_distance();
  env.cuboids.push_back(cuboid);
}

inline auto load_scene_impl(const std::string& yaml_path) -> EnvHandle {
  ::vamp::collision::Environment<float> env;

  YAML::Node config = YAML::LoadFile(yaml_path);
  if (!config["world"] || !config["world"]["collision_objects"]) {
    auto p = std::make_shared<VampEnvT>(env);
    return EnvHandle{std::static_pointer_cast<void>(p)};
  }

  for (const auto& obj : config["world"]["collision_objects"]) {
    Eigen::Vector3d obj_t = Eigen::Vector3d::Zero();
    Eigen::Matrix3d obj_R = Eigen::Matrix3d::Identity();
    if (obj["pose"]) {
      const auto& opos = obj["pose"]["position"];
      const auto& oori = obj["pose"]["orientation"];
      obj_t = Eigen::Vector3d(opos[0].as<double>(), opos[1].as<double>(),
                              opos[2].as<double>());
      Eigen::Quaterniond oq(oori[3].as<double>(), oori[0].as<double>(),
                            oori[1].as<double>(), oori[2].as<double>());
      oq.normalize();
      obj_R = oq.toRotationMatrix();
    }

    if (obj["primitives"] && obj["primitive_poses"]) {
      const auto& primitives = obj["primitives"];
      const auto& poses = obj["primitive_poses"];
      for (std::size_t i = 0; i < primitives.size(); ++i) {
        const auto& prim = primitives[i];
        const std::string type = prim["type"].as<std::string>();
        const auto [t, R] = compose_pose(obj_t, obj_R, poses[i]);
        add_primitive(env, type, prim["dimensions"], t, R);
      }
    }

    if (obj["meshes"] && obj["mesh_poses"]) {
      const auto& meshes = obj["meshes"];
      const auto& poses = obj["mesh_poses"];
      for (std::size_t i = 0; i < meshes.size(); ++i) {
        add_mesh_aabb(env, meshes[i], poses[i]);
      }
    }
  }

  env.sort();

  auto p = std::make_shared<VampEnvT>(env);
  return EnvHandle{std::static_pointer_cast<void>(p)};
}

}  // namespace geodex::integration::vamp::detail

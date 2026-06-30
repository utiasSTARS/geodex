/// @file scene_loader.hpp
/// @brief Internal: header-only implementation of the scene-loader body.
///
/// Pulls in VAMP collision types; included only by @c vamp_impl.cpp inside
/// the @c geodex_vamp static archive (which has the matching SIMD compile
/// options applied to that single source file). Consumer translation units
/// reach this code via the public @c load_scene declaration in
/// @c registry.hpp; they never include this header.

#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
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

inline void add_primitive(::vamp::collision::Environment<float>& env,
                          const std::string& type, const YAML::Node& dims,
                          const Eigen::Vector3d& t, const Eigen::Matrix3d& R,
                          int& count) {
  if (type == "box") {
    const double hx = dims[0].as<double>() / 2.0;
    const double hy = dims[1].as<double>() / 2.0;
    const double hz = dims[2].as<double>() / 2.0;
    ::vamp::collision::Cuboid<float> cuboid(
        t.x(), t.y(), t.z(),
        R(0, 0), R(1, 0), R(2, 0),
        R(0, 1), R(1, 1), R(2, 1),
        R(0, 2), R(1, 2), R(2, 2),
        hx, hy, hz);
    cuboid.min_distance = cuboid.compute_min_distance();
    env.cuboids.push_back(cuboid);
    ++count;
  } else if (type == "cylinder") {
    const double height = dims[0].as<double>();
    const double radius = dims[1].as<double>();
    const Eigen::Vector3d axis = R.col(2);
    const Eigen::Vector3d p1 = t - axis * (height / 2.0);
    const Eigen::Vector3d vec = axis * height;
    ::vamp::collision::Cylinder<float> cyl;
    cyl.x1 = p1.x(); cyl.y1 = p1.y(); cyl.z1 = p1.z();
    cyl.xv = vec.x(); cyl.yv = vec.y(); cyl.zv = vec.z();
    cyl.r = radius;
    cyl.rdv = 1.0 / vec.squaredNorm();
    cyl.min_distance = std::sqrt(t.x() * t.x() + t.y() * t.y() + t.z() * t.z());
    env.cylinders.push_back(cyl);
    ++count;
  } else if (type == "sphere") {
    const double radius = dims[0].as<double>();
    ::vamp::collision::Sphere<float> sph;
    sph.x = t.x(); sph.y = t.y(); sph.z = t.z();
    sph.r = radius;
    sph.min_distance = std::sqrt(t.x() * t.x() + t.y() * t.y() + t.z() * t.z());
    env.spheres.push_back(sph);
    ++count;
  }
}

inline void add_mesh_aabb(::vamp::collision::Environment<float>& env,
                          const YAML::Node& mesh, const YAML::Node& pose,
                          int& count) {
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
  ++count;
}

inline auto load_scene_impl(const std::string& yaml_path) -> EnvHandle {
  ::vamp::collision::Environment<float> env;

  YAML::Node config = YAML::LoadFile(yaml_path);
  if (!config["world"] || !config["world"]["collision_objects"]) {
    auto p = std::make_shared<VampEnvT>(env);
    return EnvHandle{std::static_pointer_cast<void>(p)};
  }

  int count = 0;
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
        add_primitive(env, type, prim["dimensions"], t, R, count);
      }
    }

    if (obj["meshes"] && obj["mesh_poses"]) {
      const auto& meshes = obj["meshes"];
      const auto& poses = obj["mesh_poses"];
      for (std::size_t i = 0; i < meshes.size(); ++i) {
        add_mesh_aabb(env, meshes[i], poses[i], count);
      }
    }
  }

  env.sort();
  std::cerr << "geodex::integration::vamp: loaded " << count << " obstacles ("
            << env.spheres.size() << " spheres, "
            << env.cuboids.size() << " cuboids, "
            << env.cylinders.size() << " cylinders)\n";

  auto p = std::make_shared<VampEnvT>(env);
  return EnvHandle{std::static_pointer_cast<void>(p)};
}

}  // namespace geodex::integration::vamp::detail

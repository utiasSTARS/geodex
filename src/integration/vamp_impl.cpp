/// @file vamp_impl.cpp
/// @brief Scene building and the robot registry of the VAMP integration.
///
/// The @c geodex_vamp static archive compiles this file and one translation unit per
/// robot kernel with AVX2/FMA (x86_64) or NEON by default (aarch64) as PRIVATE flags.
/// Consumer translation units do not get SIMD options. The header-only public API in
/// @c registry.hpp resolves to the out-of-line definitions below and in the kernels.

#include <algorithm>
#include <cmath>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "geodex/integration/vamp/detail/robot_impl.hpp"
#include "geodex/integration/vamp/detail/scene_loader.hpp"
#include "geodex/integration/vamp/registry.hpp"
#include "vamp/robot_entry.hpp"

namespace geodex::integration::vamp {

namespace detail {
// One declaration per kernel translation unit.
#define GEODEX_VAMP_ROBOT(name) auto robot_entry_##name() -> RobotEntry;
#include "geodex_vamp_robots.inc"
#undef GEODEX_VAMP_ROBOT
}  // namespace detail

auto load_scene(const std::string& yaml_path) -> EnvHandle {
  return detail::load_scene_impl(yaml_path);
}

auto make_scene_builder() -> SceneBuilder {
  return SceneBuilder{std::make_shared<::vamp::collision::Environment<float>>()};
}

void scene_add_box(SceneBuilder& s, const Eigen::Vector3d& center, const Eigen::Vector3d& size,
                   const Eigen::Matrix3d& rotation) {
  auto& env = *std::static_pointer_cast<::vamp::collision::Environment<float>>(s.impl);
  detail::add_box_shape(env, center, 0.5 * size, rotation);
}

void scene_add_sphere(SceneBuilder& s, const Eigen::Vector3d& center, double radius) {
  auto& env = *std::static_pointer_cast<::vamp::collision::Environment<float>>(s.impl);
  detail::add_sphere_shape(env, center, radius);
}

void scene_add_cylinder(SceneBuilder& s, const Eigen::Vector3d& center, double radius,
                        double height, const Eigen::Matrix3d& rotation) {
  auto& env = *std::static_pointer_cast<::vamp::collision::Environment<float>>(s.impl);
  detail::add_cylinder_shape(env, center, radius, height, rotation);
}

auto build_scene(const SceneBuilder& s) -> EnvHandle {
  // Copy the accumulated scalar environment, sort, then convert to SIMD layout.
  ::vamp::collision::Environment<float> env =
      *std::static_pointer_cast<::vamp::collision::Environment<float>>(s.impl);
  env.sort();
  auto p = std::make_shared<detail::VampEnvT>(env);
  return EnvHandle{std::static_pointer_cast<void>(p)};
}

namespace {

using detail::RobotEntry;

/// Compiled robots, sorted by name.
auto robots() -> const std::vector<RobotEntry>& {
  static const std::vector<RobotEntry> table = [] {
    std::vector<RobotEntry> t{
#define GEODEX_VAMP_ROBOT(name) detail::robot_entry_##name(),
#include "geodex_vamp_robots.inc"
#undef GEODEX_VAMP_ROBOT
    };
    std::sort(t.begin(), t.end(),
              [](const RobotEntry& a, const RobotEntry& b) { return a.name < b.name; });
    return t;
  }();
  return table;
}

auto find_robot(const std::string& robot_name) -> const RobotEntry& {
  const auto& table = robots();
  const auto it = std::find_if(table.begin(), table.end(),
                               [&](const RobotEntry& e) { return e.name == robot_name; });
  if (it == table.end()) {
    throw std::runtime_error("geodex::integration::vamp: no robot registered as '" +
                             robot_name + "'");
  }
  return *it;
}

}  // namespace

auto make_vamp_checker(const std::string& robot_name, EnvHandle env)
    -> std::unique_ptr<CollisionChecker> {
  return find_robot(robot_name).checker(std::move(env));
}

auto make_vamp_motion_validator(const std::string& robot_name,
                                const ompl::base::SpaceInformationPtr& si, EnvHandle env)
    -> std::unique_ptr<ompl::base::MotionValidator> {
  return find_robot(robot_name).motion_validator(si, std::move(env));
}

void attach_spheres(EnvHandle& env, const std::vector<std::array<double, 4>>& spheres) {
  auto& e = detail::env_cast(env);
  if (spheres.empty()) {
    e.attachments.reset();
    return;
  }
  // Express the spheres in the end-effector frame. Every check derives the attachment pose
  // from forward kinematics.
  ::vamp::collision::Attachment<float> attachment(
      Eigen::Transform<float, 3, Eigen::Isometry>::Identity());
  for (const auto& s : spheres) {
    attachment.spheres.emplace_back(static_cast<float>(s[0]), static_cast<float>(s[1]),
                                    static_cast<float>(s[2]), static_cast<float>(s[3]));
  }
  e.attachments = ::vamp::collision::Attachment<::vamp::FloatVector<::vamp::FloatVectorWidth>>(
      attachment);
}

auto pad_scene(const EnvHandle& env, double padding) -> EnvHandle {
  if (!(padding >= 0.0) || !std::isfinite(padding)) {
    throw std::invalid_argument("pad_scene: padding must be finite and >= 0");
  }
  auto p = std::make_shared<detail::VampEnvT>(
      detail::padded_environment(detail::env_cast(env), static_cast<float>(padding)));
  return EnvHandle{std::static_pointer_cast<void>(p)};
}

auto sphere_speed(const std::string& robot_name, const EnvHandle& env) -> double {
  return find_robot(robot_name).sphere_travel(env).speed();
}

auto make_sphere_travel(const std::string& robot_name, const EnvHandle& env) -> SphereTravel {
  const auto& entry = find_robot(robot_name);
  return [bound = entry.sphere_travel(env), dim = entry.dimension, robot_name](
             const Eigen::Ref<const Eigen::VectorXd>& tangent) {
    if (tangent.size() != dim) {
      throw std::invalid_argument("sphere travel of '" + robot_name + "': tangent size " +
                                  std::to_string(tangent.size()) + " is not the dimension " +
                                  std::to_string(dim));
    }
    return bound(tangent.data());
  };
}

auto make_vamp_certified_motion_validator(const std::string& robot_name,
                                          const ompl::base::SpaceInformationPtr& si,
                                          EnvHandle env, double max_sphere_step)
    -> std::unique_ptr<ompl::base::MotionValidator> {
  return find_robot(robot_name).certified_motion_validator(si, std::move(env), max_sphere_step);
}

auto robot_spheres(const std::string& robot_name, const double* q, int dim)
    -> std::vector<std::array<double, 4>> {
  const auto& entry = find_robot(robot_name);
  if (dim != entry.dimension) {
    throw std::invalid_argument("geodex::integration::vamp: robot_spheres expects dimension " +
                                std::to_string(entry.dimension) + ", got " +
                                std::to_string(dim));
  }
  return entry.spheres(q);
}

auto robot_dimension(const std::string& robot_name) -> int {
  return find_robot(robot_name).dimension;
}

auto robot_joint_names(const std::string& robot_name) -> std::vector<std::string> {
  const auto& names = find_robot(robot_name).joint_names;
  return {names.begin(), names.end()};
}

auto robot_end_effector(const std::string& robot_name) -> std::string {
  return std::string(find_robot(robot_name).end_effector);
}

auto registered_robots() -> std::vector<std::string> {
  std::vector<std::string> names;
  names.reserve(robots().size());
  for (const auto& e : robots()) names.emplace_back(e.name);
  return names;
}

}  // namespace geodex::integration::vamp

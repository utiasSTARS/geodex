/// @file robot_entry.hpp
/// @brief One compiled robot kernel in the VAMP registry (internal).
///
/// Each kernel compiles in a translation unit of its own, generated from
/// @c robot.cpp.in, that defines @c robot_entry_<name>(). @c vamp_impl.cpp collects the
/// entries listed in the generated @c geodex_vamp_robots.inc. This header does not use
/// VAMP or SIMD types.

#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>
#include <string_view>

#include "geodex/integration/vamp/detail/sweep_model.hpp"
#include "geodex/integration/vamp/registry.hpp"

namespace geodex::integration::vamp::detail {

/// @brief Builds a collision checker bound to an environment.
using CheckerFactory = std::unique_ptr<CollisionChecker> (*)(EnvHandle);

/// @brief Builds an OMPL motion validator bound to an environment.
using ValidatorFactory = std::unique_ptr<ompl::base::MotionValidator> (*)(
    const ompl::base::SpaceInformationPtr&, EnvHandle);

/// @brief Builds the certified motion validator for a sphere step.
using CertifiedFactory = std::unique_ptr<ompl::base::MotionValidator> (*)(
    const ompl::base::SpaceInformationPtr&, EnvHandle, double);

/// @brief Returns the sphere centers and radii at a configuration.
using SpheresFn = std::vector<std::array<double, 4>> (*)(const double*);

/// @brief Returns the sphere travel bound with the spheres attached in an environment.
using TravelFn = TravelBound (*)(const EnvHandle&);

/// @brief Name, joint layout and factories of one robot kernel.
struct RobotEntry {
  std::string_view name;
  int dimension;
  std::span<const std::string_view> joint_names;  ///< configuration order
  std::string_view end_effector;                  ///< frame of attached spheres
  CheckerFactory checker;
  ValidatorFactory motion_validator;
  CertifiedFactory certified_motion_validator;
  SpheresFn spheres;
  TravelFn sphere_travel;
};

}  // namespace geodex::integration::vamp::detail

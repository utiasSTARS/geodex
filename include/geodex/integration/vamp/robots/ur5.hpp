/// @file ur5.hpp
/// @brief Internal: factory functions for the UR5 6-DOF VAMP collision
///        checker and motion validator.
///
/// Pulls in VAMP's UR5 model + SIMD intrinsics; included only by the
/// @c vamp_impl.cpp source file in the @c geodex_vamp static archive.

#pragma once

#include <memory>
#include <utility>

#include <vamp/robots/ur5.hh>

#include "geodex/integration/vamp/registry.hpp"
#include "geodex/integration/vamp/detail/robot_impl.hpp"

namespace geodex::integration::vamp::detail {

inline auto make_ur5_checker(EnvHandle env)
    -> std::unique_ptr<CollisionChecker> {
  return std::make_unique<VampCollisionCheckerImpl<::vamp::robots::UR5>>(
      std::move(env));
}

inline auto make_ur5_motion_validator(
    const ompl::base::SpaceInformationPtr& si, EnvHandle env)
    -> std::unique_ptr<ompl::base::MotionValidator> {
  return std::make_unique<VampMotionValidatorImpl<::vamp::robots::UR5>>(
      si, std::move(env));
}

}  // namespace geodex::integration::vamp::detail

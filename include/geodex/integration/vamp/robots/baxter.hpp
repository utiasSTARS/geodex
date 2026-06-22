/// @file baxter.hpp
/// @brief Internal: factory functions for the Baxter 14-DOF VAMP collision
///        checker and motion validator.
///
/// Pulls in VAMP's Baxter model + SIMD intrinsics; included only by the
/// @c vamp_impl.cpp source file in the @c geodex_vamp static archive.

#pragma once

#include <memory>
#include <utility>

#include <vamp/robots/baxter.hh>

#include "geodex/integration/vamp/registry.hpp"
#include "geodex/integration/vamp/detail/robot_impl.hpp"

namespace geodex::integration::vamp::detail {

inline auto make_baxter_checker(EnvHandle env)
    -> std::unique_ptr<CollisionChecker> {
  return std::make_unique<VampCollisionCheckerImpl<::vamp::robots::Baxter>>(
      std::move(env));
}

inline auto make_baxter_motion_validator(
    const ompl::base::SpaceInformationPtr& si, EnvHandle env)
    -> std::unique_ptr<ompl::base::MotionValidator> {
  return std::make_unique<VampMotionValidatorImpl<::vamp::robots::Baxter>>(
      si, std::move(env));
}

}  // namespace geodex::integration::vamp::detail

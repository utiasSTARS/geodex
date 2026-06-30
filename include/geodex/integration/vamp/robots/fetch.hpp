/// @file fetch.hpp
/// @brief Internal: factory functions for the Fetch 8-DOF VAMP collision
///        checker and motion validator.
///
/// Pulls in VAMP's Fetch model + SIMD intrinsics; included only by the
/// @c vamp_impl.cpp source file in the @c geodex_vamp static archive.

#pragma once

#include <memory>
#include <utility>

#include <vamp/robots/fetch.hh>

#include "geodex/integration/vamp/registry.hpp"
#include "geodex/integration/vamp/detail/robot_impl.hpp"

namespace geodex::integration::vamp::detail {

inline auto make_fetch_checker(EnvHandle env)
    -> std::unique_ptr<CollisionChecker> {
  return std::make_unique<VampCollisionCheckerImpl<::vamp::robots::Fetch>>(
      std::move(env));
}

inline auto make_fetch_motion_validator(
    const ompl::base::SpaceInformationPtr& si, EnvHandle env)
    -> std::unique_ptr<ompl::base::MotionValidator> {
  return std::make_unique<VampMotionValidatorImpl<::vamp::robots::Fetch>>(
      si, std::move(env));
}

}  // namespace geodex::integration::vamp::detail

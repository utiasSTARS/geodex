/// @file pr2.hpp
/// @brief Internal: factory functions for the PR2 fixed-base dual-arm VAMP
///        collision checker and motion validator.
///
/// Pulls in the generated PR2 model + SIMD intrinsics; included only by the
/// @c vamp_impl.cpp source file in the @c geodex_vamp static archive.

#pragma once

#include <memory>
#include <utility>

#include "geodex/integration/vamp/detail/robot_impl.hpp"
#include "geodex/integration/vamp/registry.hpp"
#include "geodex/integration/vamp/robots/generated/pr2.hh"

namespace geodex::integration::vamp::detail {

inline auto make_pr2_checker(EnvHandle env) -> std::unique_ptr<CollisionChecker> {
  return std::make_unique<VampCollisionCheckerImpl<::vamp::robots::Pr2>>(std::move(env));
}

inline auto make_pr2_motion_validator(const ompl::base::SpaceInformationPtr& si, EnvHandle env)
    -> std::unique_ptr<ompl::base::MotionValidator> {
  return std::make_unique<VampMotionValidatorImpl<::vamp::robots::Pr2>>(si, std::move(env));
}

}  // namespace geodex::integration::vamp::detail

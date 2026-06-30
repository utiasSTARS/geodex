/// @file robot_impl.hpp
/// @brief Internal: per-robot collision-checker and motion-validator templates.
///
/// Each per-robot header instantiates these against its @c vamp::robots::X
/// struct and exposes the result through a small @c make_X_checker /
/// @c make_X_motion_validator factory pair. Pulls in VAMP SIMD intrinsics;
/// included only by @c vamp_impl.cpp inside the @c geodex_vamp static
/// archive, which has SIMD compile options applied PRIVATEly so they never
/// propagate to consumer translation units.

#pragma once

#include <cstddef>
#include <utility>

#include <vamp/vector.hh>

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>

#include "geodex/integration/vamp/registry.hpp"
#include "check_motion.hpp"
#include "vamp_env.hpp"

namespace geodex::integration::vamp::detail {

template <typename VampRobot>
class VampCollisionCheckerImpl : public CollisionChecker {
 public:
  explicit VampCollisionCheckerImpl(EnvHandle env) : env_(std::move(env)) {}

  auto is_valid(const double* values, int dim) const -> bool override {
    if (dim != static_cast<int>(VampRobot::dimension)) return false;
    auto& env = env_cast(env_);
    typename VampRobot::template ConfigurationBlock<::vamp::FloatVectorWidth> block;
    for (std::size_t i = 0; i < VampRobot::dimension; ++i) {
      block[i] = ::vamp::FloatVector<::vamp::FloatVectorWidth>(
          static_cast<float>(values[i]));
    }
    return VampRobot::template fkcc<::vamp::FloatVectorWidth>(env, block);
  }

 private:
  EnvHandle env_;
};

template <typename VampRobot>
class VampMotionValidatorImpl : public ompl::base::MotionValidator {
 public:
  VampMotionValidatorImpl(const ompl::base::SpaceInformationPtr& si, EnvHandle env)
      : ompl::base::MotionValidator(si), env_(std::move(env)) {}

  auto checkMotion(const ompl::base::State* s1,
                   const ompl::base::State* s2) const -> bool override {
    return check_motion_impl<VampRobot>(s1, s2, env_);
  }

  auto checkMotion(const ompl::base::State* s1, const ompl::base::State* s2,
                   std::pair<ompl::base::State*, double>& /*lastValid*/) const
      -> bool override {
    return checkMotion(s1, s2);
  }

 private:
  EnvHandle env_;
};

}  // namespace geodex::integration::vamp::detail

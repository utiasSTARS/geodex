/// @file check_motion.hpp
/// @brief Internal: SIMD batch motion validator template shared by all robots.
///
/// Pulls in VAMP SIMD intrinsics; included only by @c vamp_impl.cpp inside
/// the @c geodex_vamp static archive, which has SIMD compile options applied
/// PRIVATEly so they never propagate to consumer translation units.

#pragma once

#include <cstddef>
#include <utility>

#include <vamp/planning/validate.hh>
#include <vamp/vector.hh>

#include <ompl/base/State.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include "geodex/integration/vamp/registry.hpp"
#include "vamp_env.hpp"

namespace geodex::integration::vamp::detail {

template <typename VampRobot>
auto check_motion_impl(const ompl::base::State* s1,
                       const ompl::base::State* s2,
                       const EnvHandle& env_handle) -> bool {
  auto& env = env_cast(env_handle);
  constexpr std::size_t dim = VampRobot::dimension;
  constexpr std::size_t rake = ::vamp::FloatVectorWidth;
  constexpr std::size_t resolution = VampRobot::resolution;

  const auto* rv1 = s1->as<ompl::base::RealVectorStateSpace::StateType>();
  const auto* rv2 = s2->as<ompl::base::RealVectorStateSpace::StateType>();

  alignas(32) typename VampRobot::ConfigurationBuffer buf1{}, buf2{};
  for (std::size_t i = 0; i < dim; ++i) {
    buf1[i] = static_cast<float>(rv1->values[i]);
    buf2[i] = static_cast<float>(rv2->values[i]);
  }
  typename VampRobot::Configuration q1(buf1.data());
  typename VampRobot::Configuration q2(buf2.data());

  return ::vamp::planning::validate_motion<VampRobot, rake, resolution>(q1, q2, env);
}

}  // namespace geodex::integration::vamp::detail

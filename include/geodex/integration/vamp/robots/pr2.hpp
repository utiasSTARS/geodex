/// @file pr2.hpp
/// @brief VAMP model of the PR2, both arms, 14 joints.
///
/// This internal header pulls in the kernel and its SIMD intrinsics. Only the robot's own
/// translation unit in the @c geodex_vamp static archive includes it and builds the registry
/// entry from `pr2_model` and `generated::pr2_sweep`.

#pragma once

#include "geodex/integration/vamp/robots/generated/pr2.hh"

#include "geodex/integration/vamp/robots/generated/pr2_sweep.hh"

namespace geodex::integration::vamp::detail {

/// @brief VAMP model of this robot.
using pr2_model = ::vamp::robots::Pr2;

}  // namespace geodex::integration::vamp::detail

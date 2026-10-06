/// @file stretch3.hpp
/// @brief VAMP model of the Hello Robot Stretch 3, whole body (x, y, theta, lift, arm, wrist yaw,
/// pitch, roll).
///
/// This internal header pulls in the kernel and its SIMD intrinsics. Only the robot's own
/// translation unit in the @c geodex_vamp static archive includes it and builds the registry
/// entry from `stretch3_model` and `generated::stretch3_sweep`.

#pragma once

#include "geodex/integration/vamp/robots/generated/stretch3.hh"

#include "geodex/integration/vamp/robots/generated/stretch3_sweep.hh"

namespace geodex::integration::vamp::detail {

/// @brief VAMP model of this robot.
using stretch3_model = ::vamp::robots::Stretch3;

}  // namespace geodex::integration::vamp::detail

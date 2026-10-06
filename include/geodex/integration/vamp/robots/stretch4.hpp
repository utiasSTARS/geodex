/// @file stretch4.hpp
/// @brief VAMP model of the Hello Robot Stretch 4, whole body (x, y, theta, lift, arm, wrist yaw,
/// pitch, roll).
///
/// This internal header pulls in the kernel and its SIMD intrinsics. Only the robot's own
/// translation unit in the @c geodex_vamp static archive includes it and builds the registry
/// entry from `stretch4_model` and `generated::stretch4_sweep`.

#pragma once

#include "geodex/integration/vamp/robots/generated/stretch4.hh"

#include "geodex/integration/vamp/robots/generated/stretch4_sweep.hh"

namespace geodex::integration::vamp::detail {

/// @brief VAMP model of this robot.
using stretch4_model = ::vamp::robots::Stretch4;

}  // namespace geodex::integration::vamp::detail

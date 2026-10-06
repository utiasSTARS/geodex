/// @file husky_ur5e.hpp
/// @brief VAMP model of the Clearpath Husky with a UR5e, whole body (x, y, theta, six arm joints).
///
/// This internal header pulls in the kernel and its SIMD intrinsics. Only the robot's own
/// translation unit in the @c geodex_vamp static archive includes it and builds the registry
/// entry from `husky_ur5e_model` and `generated::husky_ur5e_sweep`.

#pragma once

#include "geodex/integration/vamp/robots/generated/husky_ur5e.hh"

#include "geodex/integration/vamp/robots/generated/husky_ur5e_sweep.hh"

namespace geodex::integration::vamp::detail {

/// @brief VAMP model of this robot.
using husky_ur5e_model = ::vamp::robots::HuskyUR5e;

}  // namespace geodex::integration::vamp::detail

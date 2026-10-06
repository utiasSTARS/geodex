/// @file ur5.hpp
/// @brief VAMP model of the Universal Robots UR5 with a Robotiq gripper, 6 joints. It uses VAMP's
/// own kernel.
///
/// This internal header pulls in the kernel and its SIMD intrinsics. Only the robot's own
/// translation unit in the @c geodex_vamp static archive includes it and builds the registry
/// entry from `ur5_model` and `generated::ur5_sweep`.

#pragma once

#include <vamp/robots/ur5.hh>

#include "geodex/integration/vamp/robots/generated/ur5_sweep.hh"

namespace geodex::integration::vamp::detail {

/// @brief VAMP model of this robot.
using ur5_model = ::vamp::robots::UR5;

}  // namespace geodex::integration::vamp::detail

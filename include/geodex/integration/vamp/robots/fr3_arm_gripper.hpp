/// @file fr3_arm_gripper.hpp
/// @brief VAMP model of the Franka FR3 with a Robotiq 2F-85 on the flange, 7 joints.
///
/// This internal header pulls in the kernel and its SIMD intrinsics. Only the robot's own
/// translation unit in the @c geodex_vamp static archive includes it and builds the registry
/// entry from `fr3_arm_gripper_model` and `generated::fr3_arm_gripper_sweep`.

#pragma once

#include "geodex/integration/vamp/robots/generated/fr3_arm_gripper.hh"

#include "geodex/integration/vamp/robots/generated/fr3_arm_gripper_sweep.hh"

namespace geodex::integration::vamp::detail {

/// @brief VAMP model of this robot.
using fr3_arm_gripper_model = ::vamp::robots::FR3ArmGripper;

}  // namespace geodex::integration::vamp::detail

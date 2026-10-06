/// @file panda.hpp
/// @brief VAMP model of the Franka Emika Panda, 7 joints. It uses VAMP's own kernel.
///
/// This internal header pulls in the kernel and its SIMD intrinsics. Only the robot's own
/// translation unit in the @c geodex_vamp static archive includes it and builds the registry
/// entry from `panda_model` and `generated::panda_sweep`.

#pragma once

#include <vamp/robots/panda.hh>

#include "geodex/integration/vamp/robots/generated/panda_sweep.hh"

namespace geodex::integration::vamp::detail {

/// @brief VAMP model of this robot.
using panda_model = ::vamp::robots::Panda;

}  // namespace geodex::integration::vamp::detail

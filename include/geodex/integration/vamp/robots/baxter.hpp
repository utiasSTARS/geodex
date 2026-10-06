/// @file baxter.hpp
/// @brief VAMP model of the Rethink Baxter, both arms, 14 joints. It uses VAMP's own kernel.
///
/// This internal header pulls in the kernel and its SIMD intrinsics. Only the robot's own
/// translation unit in the @c geodex_vamp static archive includes it and builds the registry
/// entry from `baxter_model` and `generated::baxter_sweep`.

#pragma once

#include <vamp/robots/baxter.hh>

#include "geodex/integration/vamp/robots/generated/baxter_sweep.hh"

namespace geodex::integration::vamp::detail {

/// @brief VAMP model of this robot.
using baxter_model = ::vamp::robots::Baxter;

}  // namespace geodex::integration::vamp::detail

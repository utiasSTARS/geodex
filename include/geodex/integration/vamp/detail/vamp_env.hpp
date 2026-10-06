/// @file vamp_env.hpp
/// @brief VAMP environment alias and opaque-handle accessor (internal).
///
/// Not part of the public API. Pulls in VAMP SIMD types. Only translation units
/// inside the @c geodex_vamp static archive include it. The archive's SIMD compile
/// options are PRIVATE and do not reach consumer translation units.

#pragma once

#include <vamp/collision/environment.hh>
#include <vamp/vector.hh>

#include "geodex/integration/vamp/registry.hpp"

namespace geodex::integration::vamp::detail {

using VampEnvT = ::vamp::collision::Environment<
    ::vamp::FloatVector<::vamp::FloatVectorWidth>>;

inline auto env_cast(const EnvHandle& h) -> VampEnvT& {
  return *static_cast<VampEnvT*>(h.impl.get());
}

}  // namespace geodex::integration::vamp::detail

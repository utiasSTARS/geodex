/// @file vamp_env.hpp
/// @brief Internal: VAMP environment alias + opaque-handle accessor.
///
/// Not part of the public API. Pulls in VAMP SIMD types; included only by
/// @c vamp_impl.cpp inside the @c geodex_vamp static archive, which has
/// SIMD compile options applied PRIVATEly so they never propagate to
/// consumer translation units.

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

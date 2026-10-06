/// @file sampler_kind.hpp
/// @brief Map a sampler-kind name to a runtime DynamicSampler for the bindings.

#pragma once

#include <stdexcept>
#include <string>

#include "geodex/core/sampler.hpp"

namespace geodex::python {

/// @brief Build a DynamicSampler from a kind name.
/// @param kind One of "scrambled", "halton", or "random".
inline DynamicSampler make_sampler(const std::string& kind) {
  if (kind == "scrambled") return DynamicSampler{ScrambledHaltonSampler{}};
  if (kind == "halton") return DynamicSampler{HaltonSampler{}};
  if (kind == "random") return DynamicSampler{PseudoRandomSampler{}};
  throw std::invalid_argument("Unknown sampler: '" + kind +
                              "'. Options: 'scrambled', 'halton', 'random'");
}

}  // namespace geodex::python

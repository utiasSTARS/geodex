/// @file random.hpp
/// @brief Random values from a standard engine that are the same with every
/// standard library.
///
/// @details The engines in `<random>` produce fully specified sequences, and the
/// distributions (`std::uniform_int_distribution`, `std::uniform_real_distribution`,
/// `std::normal_distribution`) map them in implementation-defined ways. libstdc++ and
/// libc++ give different values from one seed. geodex samples through these functions,
/// and a seeded result reproduces on any machine.

#pragma once

#include <cstdint>

#include "geodex/utils/normal.hpp"

namespace geodex::utils {

/// @brief 64 random bits from `g`, a single call of a 64-bit engine such as
/// `std::mt19937_64` or two calls of a 32-bit one such as `std::mt19937`.
template <typename G>
std::uint64_t random_bits64(G& g) {
  static_assert(G::min() == 0, "the engine must start at 0");
  if constexpr (G::max() == 0xFFFFFFFFFFFFFFFFULL) {
    return static_cast<std::uint64_t>(g());
  } else {
    static_assert(G::max() == 0xFFFFFFFFULL, "the engine must give 32 or 64 bits per call");
    const auto hi = static_cast<std::uint64_t>(g());
    const auto lo = static_cast<std::uint64_t>(g());
    return (hi << 32) | lo;
  }
}

/// @brief A uniform double in [0, 1) with 53 random bits.
template <typename G>
double uniform01(G& g) {
  return static_cast<double>(random_bits64(g) >> 11) * 0x1.0p-53;
}

/// @brief A uniform double in [lo, hi).
template <typename G>
double uniform_real(G& g, const double lo, const double hi) {
  return lo + (hi - lo) * uniform01(g);
}

/// @brief A uniform index in [0, n), unbiased by rejection. `n` must be positive.
template <typename G>
std::uint64_t uniform_index(G& g, const std::uint64_t n) {
  // Values below 2^64 mod n would make the low residues more likely.
  const std::uint64_t threshold = (0 - n) % n;
  for (;;) {
    const std::uint64_t x = random_bits64(g);
    if (x >= threshold) return x % n;
  }
}

/// @brief A normal variate with the given mean and standard deviation, the inverse
/// normal CDF of a uniform sample in (0, 1).
template <typename G>
double normal(G& g, const double mean, const double stddev) {
  const double u = (static_cast<double>(random_bits64(g) >> 11) + 0.5) * 0x1.0p-53;
  return mean + stddev * normal_quantile(u);
}

}  // namespace geodex::utils

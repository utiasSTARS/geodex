/// @file sampler.hpp
/// @brief Sampler concepts and implementations for uniform sampling in \f$[0,1]^d\f$.
///
/// @details Manifolds compose a sampler with a `box_to_point` map to produce
/// points on the manifold. This split lets the same sampler back any manifold
/// while keeping Halton (deterministic low-discrepancy) sampling meaningful:
/// Halton is only well-defined on a fixed-cardinality coordinate space, so
/// the manifold owns the mapping from the box to its native representation.

#pragma once

#include <Eigen/Core>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>

namespace geodex {

namespace detail {

/// @brief First 30 primes — upper bound on Halton sampling dimensions.
inline constexpr std::array<int, 30> halton_primes = {
    2,  3,  5,  7,  11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113};

/// @brief Van der Corput sequence — the 1-D building block of Halton sampling.
/// @param index 1-based index into the sequence.
/// @param base Prime base for the digit expansion.
/// @return The value of the van der Corput sequence at `index` in base `base`.
inline double van_der_corput(const std::uint64_t index, const int base) {
  double result = 0.0;
  double f = 1.0 / static_cast<double>(base);
  std::uint64_t i = index;
  const std::uint64_t b = static_cast<std::uint64_t>(base);
  while (i > 0) {
    result += static_cast<double>(i % b) * f;
    i /= b;
    f /= static_cast<double>(base);
  }
  return result;
}

}  // namespace detail

/// @brief Concept: a type that fills a length-`d` vector with uniform samples in \f$[0, 1)\f$.
template <typename S>
concept Sampler = requires(S& s, const int d, Eigen::Ref<Eigen::VectorXd> out) {
  { s.sample_box(d, out) } -> std::same_as<void>;
};

/// @brief Concept: a `Sampler` that also supports explicit reseeding.
template <typename S>
concept SeedableSampler = Sampler<S> && requires(S& s, std::uint64_t seed) {
  { s.seed(seed) } -> std::same_as<void>;
};

/// @brief Pseudo-random sampler wrapping `std::mt19937`.
///
/// @details Default construction uses a shared `thread_local` Mersenne-Twister
/// generator, preserving the pre-refactor per-manifold random_point() semantics
/// and zero-cost instantiation. Passing an explicit seed creates an owned
/// generator with a reproducible sequence — useful for tests and benchmarks.
class StochasticSampler {
 public:
  /// @brief Default: share a thread-local generator.
  StochasticSampler() = default;

  /// @brief Seeded: own a generator for reproducible sequences.
  explicit StochasticSampler(std::uint64_t seed) : gen_(seed), owned_(true) {}

  /// @brief Reseed the sampler; transitions to an owned generator.
  void seed(std::uint64_t s) {
    gen_.seed(s);
    owned_ = true;
  }

  /// @brief Fill `out[0..d-1]` with uniform values in \f$[0, 1)\f$.
  void sample_box(const int d, Eigen::Ref<Eigen::VectorXd> out) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    auto& g = owned_ ? gen_ : thread_local_gen();
    for (int i = 0; i < d; ++i) {
      out[i] = dist(g);
    }
  }

 private:
  static std::mt19937& thread_local_gen() {
    thread_local std::mt19937 g{std::random_device{}()};
    return g;
  }

  std::mt19937 gen_{};
  bool owned_ = false;
};

/// @brief Halton low-discrepancy sampler (deterministic quasi-random).
///
/// @details Produces a fully reproducible quasi-random sequence by advancing
/// an internal 1-based index. Coordinates are computed per dimension via
/// `detail::van_der_corput` with the first `d` primes. Maximum supported
/// dimension is `detail::halton_primes.size()` (30).
class HaltonSampler {
 public:
  /// @brief Default: start the sequence at index 1.
  HaltonSampler() = default;

  /// @brief Start the sequence at an explicit index.
  explicit HaltonSampler(std::uint64_t start_index) : index_(start_index) {}

  /// @brief Reset the sequence index.
  void seed(std::uint64_t s) { index_ = s; }

  /// @brief Fill `out[0..d-1]` with the next Halton sample.
  void sample_box(const int d, Eigen::Ref<Eigen::VectorXd> out) {
    ++index_;  // 1-based
    for (int i = 0; i < d; ++i) {
      out[i] = detail::van_der_corput(index_,
                                      detail::halton_primes[static_cast<std::size_t>(i)]);
    }
  }

 private:
  std::uint64_t index_ = 0;
};

// Self-verification: our concrete samplers model the concepts.
static_assert(Sampler<StochasticSampler>);
static_assert(Sampler<HaltonSampler>);
static_assert(SeedableSampler<StochasticSampler>);
static_assert(SeedableSampler<HaltonSampler>);

}  // namespace geodex

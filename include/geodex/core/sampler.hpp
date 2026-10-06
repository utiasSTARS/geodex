/// @file sampler.hpp
/// @brief Sampler concepts and implementations for uniform sampling in \f$[0,1)^d\f$.
///
/// @details A sampler fills a unit-cube vector and does not know the geometry. Each
/// manifold's `from_unit_cube` map turns that vector into a point, and one sampler
/// backs every manifold. A low-discrepancy (Halton) sequence stays well distributed
/// only when one sequence spans the whole coordinate space.

#pragma once

#include <cstddef>
#include <cstdint>

#include <array>
#include <concepts>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "geodex/utils/random.hpp"

namespace geodex {

namespace detail {

/// @brief Number of van der Corput bases, the upper bound on Halton dimensions.
inline constexpr int kHaltonMaxDim = 1024;

/// @brief The first `kHaltonMaxDim` primes, sieved at compile time. Base `i` is
/// used for coordinate `i` of the Halton sequence.
inline constexpr std::array<int, kHaltonMaxDim> halton_primes = [] {
  std::array<int, kHaltonMaxDim> primes{};
  int count = 0;
  for (int n = 2; count < kHaltonMaxDim; ++n) {
    bool is_prime = true;
    for (int i = 0; i < count; ++i) {
      if (primes[i] * primes[i] > n) break;
      if (n % primes[i] == 0) {
        is_prime = false;
        break;
      }
    }
    if (is_prime) primes[count++] = n;
  }
  return primes;
}();

/// @brief Van der Corput radical inverse, the 1-D building block of the Halton
/// sequence (van der Corput 1935; Halton 1960).
/// @param index 1-based index into the sequence.
/// @param base Prime base for the digit expansion.
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

/// @brief Scrambled radical inverse using one digit permutation per base
/// (Braaten and Weller 1979). `perm` is a bijection on \f$\{0,\dots,base-1\}\f$.
///
/// @details Permutes only the significant digits of `index`. The implicit leading
/// zeros stay unscrambled and keep the result in \f$[0,1)\f$.
/// @param index 1-based index into the sequence.
/// @param base Prime base for the digit expansion.
/// @param perm Digit permutation applied to each base-`base` digit.
inline double scrambled_van_der_corput(const std::uint64_t index, const int base,
                                       const std::vector<int>& perm) {
  double result = 0.0;
  double f = 1.0 / static_cast<double>(base);
  std::uint64_t i = index;
  const std::uint64_t b = static_cast<std::uint64_t>(base);
  while (i > 0) {
    result += static_cast<double>(perm[static_cast<std::size_t>(i % b)]) * f;
    i /= b;
    f /= static_cast<double>(base);
  }
  return result;
}

/// @brief Throw unless a unit-cube vector holds at least the `n` coordinates a
/// `from_unit_cube` map reads.
inline void require_unit_cube_size(const Eigen::Index size, const int n) {
  if (size < n) {
    throw std::invalid_argument("from_unit_cube: needs " + std::to_string(n) +
                                " unit-cube coordinates, got " + std::to_string(size));
  }
}

/// @brief Thread-local seed source of the default-constructed low-discrepancy
/// samplers. Reseed it via `geodex::set_default_seed` to make default
/// sampling reproducible.
inline std::mt19937_64& seed_source() {
  thread_local std::mt19937_64 g{std::random_device{}()};
  return g;
}

}  // namespace detail

/// @brief A type that fills `out[0..n-1]` with uniform samples in \f$[0, 1)\f$.
template <typename S>
concept Sampler = requires(S& s, const int n, Eigen::Ref<Eigen::VectorXd> out) {
  { s.sample(n, out) } -> std::same_as<void>;
};

/// @brief A `Sampler` that also supports explicit reseeding.
///
/// @details `seed(s)` puts the sampler in a state determined by `s` alone, and two
/// samplers of one kind seeded alike give the same sequence. `PseudoRandomSampler`
/// reseeds its generator, `ScrambledHaltonSampler` samples a new scramble and start, and
/// the deterministic `HaltonSampler` moves its sequence index to `s`.
template <typename S>
concept SeedableSampler = Sampler<S> && requires(S& s, std::uint64_t seed) {
  { s.seed(seed) } -> std::same_as<void>;
};

/// @brief Pseudo-random sampler wrapping `std::mt19937`.
///
/// @details Default construction shares a `thread_local` generator and costs nothing to
/// set up. An explicitly seeded sampler owns a generator for a reproducible sequence.
class PseudoRandomSampler {
 public:
  /// @brief Share a thread-local generator by default.
  PseudoRandomSampler() = default;

  /// @brief Own a generator for a reproducible sequence.
  explicit PseudoRandomSampler(std::uint64_t seed) : gen_(seed), owned_(true) {}

  /// @brief Reseed and switch to an owned generator.
  void seed(std::uint64_t s) {
    gen_.seed(s);
    owned_ = true;
  }

  /// @brief Fill `out[0..n-1]` with uniform values in \f$[0, 1)\f$.
  void sample(const int n, Eigen::Ref<Eigen::VectorXd> out) {
    auto& g = owned_ ? gen_ : thread_local_gen();
    for (int i = 0; i < n; ++i) {
      out[i] = utils::uniform01(g);
    }
  }

  /// @brief Reseed the shared thread-local generator used by default samplers.
  static void reseed_thread_local(std::uint64_t s) { thread_local_gen().seed(s); }

 private:
  static std::mt19937& thread_local_gen() {
    thread_local std::mt19937 g{std::random_device{}()};
    return g;
  }

  std::mt19937 gen_{};
  bool owned_ = false;
};

/// @brief Halton low-discrepancy sampler (Halton 1960), deterministic quasi-random.
///
/// @details Advances a 1-based index and computes each coordinate from the first
/// `n` primes via van der Corput. Maximum dimension is
/// `detail::halton_primes.size()` (`detail::kHaltonMaxDim`).
class HaltonSampler {
 public:
  /// @brief Start the sequence at index 1 by default.
  HaltonSampler() = default;

  /// @brief Start the sequence at an explicit index.
  explicit HaltonSampler(std::uint64_t start_index) : index_(start_index) {}

  /// @brief Reset the sequence index.
  void seed(std::uint64_t s) { index_ = s; }

  /// @brief Fill `out[0..n-1]` with the next Halton sample.
  void sample(const int n, Eigen::Ref<Eigen::VectorXd> out) {
    if (n > static_cast<int>(detail::halton_primes.size())) {
      throw std::invalid_argument("Halton dimension exceeds the prime table");
    }
    ++index_;
    for (int i = 0; i < n; ++i) {
      out[i] = detail::van_der_corput(index_, detail::halton_primes[static_cast<std::size_t>(i)]);
    }
  }

 private:
  std::uint64_t index_ = 0;
};

/// @brief Scrambled Halton sampler, randomized low-discrepancy (Braaten and
/// Weller 1979; Owen 2017).
///
/// @details Samples a random digit permutation per prime base and a random start. Each
/// seed gives a low-discrepancy realization, independent seeds give independent
/// replicates, and the scramble removes the correlation between dimensions of the raw
/// Halton sequence. Permutations are built lazily per base. Maximum dimension is
/// `detail::kHaltonMaxDim`.
class ScrambledHaltonSampler {
 public:
  /// @brief Random scramble and start from the default seed source.
  ScrambledHaltonSampler() { init(detail::seed_source()()); }

  /// @brief Reproducible scramble and start from a seed.
  explicit ScrambledHaltonSampler(std::uint64_t seed) { init(seed); }

  /// @brief Reseed with a fresh scramble and start.
  void seed(std::uint64_t s) { init(s); }

  /// @brief Fill `out[0..n-1]` with the next scrambled Halton sample.
  void sample(const int n, Eigen::Ref<Eigen::VectorXd> out) {
    if (n > static_cast<int>(detail::halton_primes.size())) {
      throw std::invalid_argument("Halton dimension exceeds the prime table");
    }
    ensure_perms(n);
    ++index_;
    for (int i = 0; i < n; ++i) {
      const auto d = static_cast<std::size_t>(i);
      out[i] = detail::scrambled_van_der_corput(index_, detail::halton_primes[d], perms_[d]);
    }
  }

 private:
  void init(std::uint64_t s) {
    seed_ = s;
    perms_.clear();
    std::mt19937_64 g(s);
    index_ = g() & 0xFFFFFFFFULL;  // random start
  }

  /// @brief Build permutations up to dimension `n`. Each base's permutation is
  /// seeded from the base index and does not depend on the build order.
  void ensure_perms(int n) {
    while (static_cast<int>(perms_.size()) < n) {
      const std::size_t k = perms_.size();
      const std::size_t b = static_cast<std::size_t>(detail::halton_primes[k]);
      std::vector<int> p(b);
      std::iota(p.begin(), p.end(), 0);
      // Portable Fisher-Yates seeded per base. The scramble is deterministic and
      // identical across standard libraries.
      std::mt19937_64 g(seed_ ^ (0x9E3779B97F4A7C15ULL * (k + 1)));
      for (std::size_t i = b - 1; i >= 1; --i) {
        const std::size_t j = static_cast<std::size_t>(g() % (i + 1));
        const int tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
      }
      perms_.push_back(std::move(p));
    }
  }

  std::uint64_t seed_ = 0;
  std::uint64_t index_ = 0;
  std::vector<std::vector<int>> perms_;
};

/// @brief Type-erased runtime sampler handle with value semantics.
///
/// @details Wraps any `Sampler` and lets the Python bindings and the OMPL integration
/// choose the sampler kind and seed at runtime. A copy deep-copies the wrapped sampler
/// into an independent stream, as the concrete samplers do.
class DynamicSampler {
  struct Concept {
    virtual ~Concept() = default;
    virtual void sample(int n, Eigen::Ref<Eigen::VectorXd> out) = 0;
    virtual void seed(std::uint64_t s) = 0;
    virtual std::unique_ptr<Concept> clone() const = 0;
  };

  template <typename S>
  struct Model final : Concept {
    S wrapped;
    explicit Model(S s) : wrapped(std::move(s)) {}
    void sample(int n, Eigen::Ref<Eigen::VectorXd> out) override { wrapped.sample(n, out); }
    void seed(std::uint64_t s) override {
      if constexpr (SeedableSampler<S>) wrapped.seed(s);
    }
    std::unique_ptr<Concept> clone() const override { return std::make_unique<Model>(*this); }
  };

 public:
  /// @brief Wrap a scrambled Halton sampler by default.
  DynamicSampler() : DynamicSampler(ScrambledHaltonSampler{}) {}

  /// @brief Wrap any sampler.
  template <typename S>
    requires Sampler<std::decay_t<S>> && (!std::is_same_v<std::decay_t<S>, DynamicSampler>)
  DynamicSampler(S sampler)
      : impl_(std::make_unique<Model<std::decay_t<S>>>(std::move(sampler))) {}

  /// @brief Copy by deep-cloning the wrapped sampler into an independent stream.
  DynamicSampler(const DynamicSampler& other) : impl_(other.impl_->clone()) {}

  /// @brief Assign a deep clone of the wrapped sampler.
  DynamicSampler& operator=(const DynamicSampler& other) {
    impl_ = other.impl_->clone();
    return *this;
  }

  /// @brief Take over the wrapped sampler, leaving `other` empty.
  DynamicSampler(DynamicSampler&&) noexcept = default;

  /// @brief Take over the wrapped sampler, leaving `other` empty.
  DynamicSampler& operator=(DynamicSampler&&) noexcept = default;

  /// @brief Fill `out[0..n-1]` via the wrapped sampler.
  void sample(const int n, Eigen::Ref<Eigen::VectorXd> out) { impl_->sample(n, out); }

  /// @brief Reseed the wrapped sampler, or do nothing if it is not seedable.
  void seed(std::uint64_t s) { impl_->seed(s); }

 private:
  std::unique_ptr<Concept> impl_;
};

/// @brief Reseed the thread-local generators of the default-constructed samplers,
/// making subsequent default sampling reproducible.
inline void set_default_seed(std::uint64_t s) {
  detail::seed_source().seed(s);
  PseudoRandomSampler::reseed_thread_local(s);
}

// Check that the concrete samplers model the concepts.
static_assert(Sampler<PseudoRandomSampler>);
static_assert(Sampler<HaltonSampler>);
static_assert(Sampler<ScrambledHaltonSampler>);
static_assert(Sampler<DynamicSampler>);
static_assert(SeedableSampler<PseudoRandomSampler>);
static_assert(SeedableSampler<HaltonSampler>);
static_assert(SeedableSampler<ScrambledHaltonSampler>);
static_assert(SeedableSampler<DynamicSampler>);

}  // namespace geodex

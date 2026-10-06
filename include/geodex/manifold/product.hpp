/// @file product.hpp
/// @brief Riemannian product manifold \f$ \mathcal{M}_1 \times \cdots \times \mathcal{M}_N \f$.

#pragma once

#include <cmath>

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"
#include "geodex/core/sampler.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/manifold/sphere.hpp"

namespace geodex {

namespace detail {

/// @brief True when a manifold provides a `project(p, v)` method mapping an
/// ambient vector to the tangent space at `p`.
template <typename M>
concept ProductBlockHasProject =
    requires(const M m, const typename M::Point p, const typename M::Tangent v) {
      { m.project(p, v) } -> std::same_as<typename M::Tangent>;
    };

/// @brief True when a block has box sampling bounds `lo()` / `hi()` on its coordinates.
template <typename M>
concept ProductBlockBounded = requires(const M m) {
  { m.lo() } -> std::convertible_to<Eigen::VectorXd>;
  { m.hi() } -> std::convertible_to<Eigen::VectorXd>;
};

/// @brief True when a block provides the batched inner product `inner_matrix(p, U, V)`.
template <typename M>
concept ProductBlockInnerMatrix =
    requires(const M m, const typename M::Point p, const Eigen::MatrixXd A) {
      { m.inner_matrix(p, A, A) } -> std::convertible_to<Eigen::MatrixXd>;
    };

}  // namespace detail

// ---------------------------------------------------------------------------
// ProductManifold
// ---------------------------------------------------------------------------

/// @brief The Riemannian product of \f$ N \f$ sub-manifolds \f$ \mathcal{M}_1 \times \cdots \times
/// \mathcal{M}_N \f$.
///
/// @details The product metric is the direct sum
/// \f$ g = g_1 \oplus \cdots \oplus g_N \f$, and every operation decouples across
/// blocks. The exp and log maps and geodesics act block-wise, the inner product is a
/// block sum, and \f$ d = \sqrt{\sum_i d_i(p_i, q_i)^2} \f$. A typical use composes a
/// mobile base with a manipulator, \f$ \mathrm{SE}(2) \times \mathbb{R}^n \f$.
///
/// Points and tangents stack the ambient representations of the blocks in contiguous
/// segments. The intrinsic `dim()`, the sum of the block dimensions, can be smaller
/// than the stored tangent length. For `Sphere<> x Euclidean(2)`, `dim() == 4` and the
/// stored vectors have length 5.
///
/// random_point uses one joint sampler over all blocks, set with set_sampler
/// (scrambled Halton by default). Place higher-resolution blocks first. They get the
/// lower Halton primes.
///
/// @tparam Ms The sub-manifold types, each satisfying `RiemannianManifold`.
template <typename... Ms>
class ProductManifold {
 public:
  using Scalar = double;             ///< Scalar type.
  using Point = Eigen::VectorXd;     ///< Stacked ambient point representation.
  using Tangent = Eigen::VectorXd;   ///< Stacked ambient tangent representation.
  using SamplerType = DynamicSampler;  ///< joint sampler over all blocks

  /// @brief Number of sub-manifold blocks.
  static constexpr std::size_t N = sizeof...(Ms);

  /// @brief Construct from one instance of each sub-manifold.
  ///
  /// @details Caches, once at construction, each block's ambient point size
  /// (`random_point().size()`), its ambient tangent size (`log(p, p).size()`),
  /// its intrinsic dimension (`dim()`), and the corresponding prefix offsets.
  /// @param ms The sub-manifold instances, in order.
  explicit ProductManifold(Ms... ms) : blocks_(std::move(ms)...) {
    int poff = 0, toff = 0, coff = 0, d = 0;
    for_each_index([&]<std::size_t I>() {
      const auto& blk = std::get<I>(blocks_);
      const auto rp = blk.random_point();
      const int ps = static_cast<int>(rp.size());
      // Ambient tangent size, from log(p, p), the correctly sized zero tangent.
      const int ts = static_cast<int>(blk.log(rp, rp).size());
      const int cs = blk.unit_cube_dim();
      point_size_[I] = ps;
      point_off_[I] = poff;
      poff += ps;
      tan_size_[I] = ts;
      tan_off_[I] = toff;
      toff += ts;
      cube_size_[I] = cs;
      cube_off_[I] = coff;
      coff += cs;
      d += blk.dim();
    });
    total_point_ = poff;
    total_tan_ = toff;
    total_cube_ = coff;
    dim_ = d;
  }

  /// @brief Intrinsic dimension, the sum of the block dimensions.
  int dim() const { return dim_; }

  /// @brief Runtime check whether `log` is the Riemannian logarithm of the product
  /// metric.
  ///
  /// @details True exactly when every block's `log` is the Riemannian logarithm
  /// of its own metric (the product log is the direct sum of the block logs).
  bool has_riemannian_log_runtime() const {
    bool all = true;
    for_each_index(
        [&]<std::size_t I>() { all = all && geodex::is_riemannian_log(std::get<I>(blocks_)); });
    return all;
  }

  /// @brief Number of unit-cube coordinates that from_unit_cube consumes, the sum
  /// over the blocks.
  int unit_cube_dim() const { return total_cube_; }

  /// @brief Map one joint unit-cube vector to a stacked point by slicing it into
  /// per-block segments. One low-discrepancy sequence spans all blocks.
  /// @param u Unit-cube coordinates of length unit_cube_dim().
  Point from_unit_cube(Eigen::Ref<const Eigen::VectorXd> u) const {
    detail::require_unit_cube_size(u.size(), total_cube_);
    Point out(total_point_);
    for_each_index([&]<std::size_t I>() {
      out.segment(point_off_[I], point_size_[I]) =
          std::get<I>(blocks_).from_unit_cube(u.segment(cube_off_[I], cube_size_[I]));
    });
    return out;
  }

  /// @brief Sample a random point from one joint sampler over all blocks.
  Point random_point() const {
    cube_buf_.resize(total_cube_);
    sampler_.sample(total_cube_, cube_buf_);
    return from_unit_cube(cube_buf_);
  }

  /// @brief Set the sampler that random_point uses over the joint cube.
  void set_sampler(DynamicSampler s) { sampler_ = std::move(s); }

  /// @brief The joint sampler behind random_point(). Planning samples through copies of it.
  const DynamicSampler& sampler() const { return sampler_; }

  /// @brief Reseed the joint sampler.
  void seed(std::uint64_t s) { sampler_.seed(s); }

  /// @brief Exponential map \f$ \exp_p(v) \f$, applied block-wise.
  /// @param p Base point (stacked).
  /// @param v Tangent vector (stacked).
  /// @return The resulting point (stacked).
  Point exp(const Point& p, const Tangent& v) const {
    Point out(total_point_);
    for_each_index([&]<std::size_t I>() {
      using M = block_t<I>;
      const auto bp = slice<typename M::Point>(p, point_off_[I], point_size_[I]);
      const auto bv = slice<typename M::Tangent>(v, tan_off_[I], tan_size_[I]);
      out.segment(point_off_[I], point_size_[I]) = std::get<I>(blocks_).exp(bp, bv);
    });
    return out;
  }

  /// @brief Logarithmic map \f$ \log_p(q) \f$, applied block-wise.
  /// @param p Base point (stacked).
  /// @param q Target point (stacked).
  /// @return The tangent vector from \f$ p \f$ to \f$ q \f$ (stacked).
  Tangent log(const Point& p, const Point& q) const {
    Tangent out(total_tan_);
    for_each_index([&]<std::size_t I>() {
      using M = block_t<I>;
      const auto bp = slice<typename M::Point>(p, point_off_[I], point_size_[I]);
      const auto bq = slice<typename M::Point>(q, point_off_[I], point_size_[I]);
      out.segment(tan_off_[I], tan_size_[I]) = std::get<I>(blocks_).log(bp, bq);
    });
    return out;
  }

  /// @brief Riemannian inner product, the sum of the block inner products.
  /// @param p Base point (stacked).
  /// @param u First tangent vector (stacked).
  /// @param v Second tangent vector (stacked).
  Scalar inner(const Point& p, const Tangent& u, const Tangent& v) const {
    Scalar s = 0.0;
    for_each_index([&]<std::size_t I>() {
      using M = block_t<I>;
      const auto bp = slice<typename M::Point>(p, point_off_[I], point_size_[I]);
      const auto bu = slice<typename M::Tangent>(u, tan_off_[I], tan_size_[I]);
      const auto bv = slice<typename M::Tangent>(v, tan_off_[I], tan_size_[I]);
      s += std::get<I>(blocks_).inner(bp, bu, bv);
    });
    return s;
  }

  /// @brief Riemannian norm \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$.
  Scalar norm(const Point& p, const Tangent& v) const { return std::sqrt(inner(p, v, v)); }

  /// @brief Exact product-metric geodesic distance
  /// \f$ d(p, q) = \sqrt{\sum_i d_i(p_i, q_i)^2} \f$.
  /// @param p First point (stacked).
  /// @param q Second point (stacked).
  Scalar distance(const Point& p, const Point& q) const {
    Scalar s2 = 0.0;
    for_each_index([&]<std::size_t I>() {
      using M = block_t<I>;
      const auto bp = slice<typename M::Point>(p, point_off_[I], point_size_[I]);
      const auto bq = slice<typename M::Point>(q, point_off_[I], point_size_[I]);
      const Scalar d = std::get<I>(blocks_).distance(bp, bq);
      s2 += d * d;
    });
    return std::sqrt(s2);
  }

  /// @brief Geodesic interpolation, applied block-wise.
  /// @param p Start point (stacked).
  /// @param q End point (stacked).
  /// @param t Interpolation parameter in \f$ [0, 1] \f$.
  /// @return The interpolated point (stacked).
  Point geodesic(const Point& p, const Point& q, Scalar t) const {
    Point out(total_point_);
    for_each_index([&]<std::size_t I>() {
      using M = block_t<I>;
      const auto bp = slice<typename M::Point>(p, point_off_[I], point_size_[I]);
      const auto bq = slice<typename M::Point>(q, point_off_[I], point_size_[I]);
      out.segment(point_off_[I], point_size_[I]) = std::get<I>(blocks_).geodesic(bp, bq, t);
    });
    return out;
  }

  /// @brief Project an ambient vector onto the tangent space at \f$ p \f$,
  /// block-wise.
  ///
  /// @details Blocks that expose a `project` method (e.g. `Sphere`) have it
  /// applied to their tangent segment. Blocks whose tangent space is the whole
  /// ambient space (`Euclidean`, `SE2`, `Torus`) use the identity.
  /// @param p Base point (stacked).
  /// @param v Ambient vector to project (stacked).
  Tangent project(const Point& p, const Tangent& v) const {
    Tangent out(total_tan_);
    for_each_index([&]<std::size_t I>() {
      using M = block_t<I>;
      const auto bv = slice<typename M::Tangent>(v, tan_off_[I], tan_size_[I]);
      if constexpr (detail::ProductBlockHasProject<M>) {
        const auto bp = slice<typename M::Point>(p, point_off_[I], point_size_[I]);
        out.segment(tan_off_[I], tan_size_[I]) = std::get<I>(blocks_).project(bp, bv);
      } else {
        out.segment(tan_off_[I], tan_size_[I]) = bv;
      }
    });
    return out;
  }

  /// @name Coordinate facts of a bounded product
  /// Available when every block has sampling bounds `lo()` / `hi()`, as for SE(2),
  /// Euclidean and configuration-space blocks. Such a product certifies its own periodic
  /// Loewner bound through `algorithm::precompute_matrix_lower_bound`.
  /// @{

  /// @brief Lower sampling bounds, stacked from the blocks.
  Eigen::VectorXd lo() const
    requires(detail::ProductBlockBounded<Ms> && ...)
  {
    return stack_points([](const auto& blk) { return Eigen::VectorXd(blk.lo()); });
  }

  /// @brief Upper sampling bounds, stacked from the blocks.
  Eigen::VectorXd hi() const
    requires(detail::ProductBlockBounded<Ms> && ...)
  {
    return stack_points([](const auto& blk) { return Eigen::VectorXd(blk.hi()); });
  }

  /// @brief Per-coordinate periods, stacked from the blocks, 0 on aperiodic coordinates.
  Eigen::VectorXd periods() const
    requires(detail::ProductBlockBounded<Ms> && ...)
  {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(total_point_);
    for_each_index([&]<std::size_t I>() {
      if constexpr (HasPeriods<block_t<I>>) {
        out.segment(point_off_[I], point_size_[I]) = std::get<I>(blocks_).periods();
      }
    });
    return out;
  }

  /// @brief Batched inner product \f$ U^\top G(p) V \f$ in the blocks' tangent frames.
  Eigen::MatrixXd inner_matrix(const Point& p, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const
    requires(detail::ProductBlockInnerMatrix<Ms> && ...)
  {
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(total_tan_, total_tan_);
    for_each_index([&]<std::size_t I>() {
      using M = block_t<I>;
      const auto bp = slice<typename M::Point>(p, point_off_[I], point_size_[I]);
      const Eigen::MatrixXd E = Eigen::MatrixXd::Identity(tan_size_[I], tan_size_[I]);
      G.block(tan_off_[I], tan_off_[I], tan_size_[I], tan_size_[I]) =
          std::get<I>(blocks_).inner_matrix(bp, E, E);
    });
    return U.transpose() * G * V;
  }

  /// @brief The metric on coordinate velocities, block diagonal.
  ///
  /// @details Blocks with a frame (SE(2)) contribute their `coordinate_metric`. Flat
  /// bounded blocks measure their coordinates directly and contribute their inner
  /// product on the coordinate basis.
  Eigen::MatrixXd coordinate_metric(const Point& p) const
    requires((detail::ProductBlockBounded<Ms> && detail::ProductBlockInnerMatrix<Ms>) && ...)
  {
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(total_point_, total_point_);
    for_each_index([&]<std::size_t I>() {
      using M = block_t<I>;
      const auto bp = slice<typename M::Point>(p, point_off_[I], point_size_[I]);
      const auto& blk = std::get<I>(blocks_);
      if constexpr (HasCoordinateMetric<M>) {
        G.block(point_off_[I], point_off_[I], point_size_[I], point_size_[I]) =
            blk.coordinate_metric(bp);
      } else {
        const Eigen::MatrixXd E = Eigen::MatrixXd::Identity(point_size_[I], point_size_[I]);
        G.block(point_off_[I], point_off_[I], point_size_[I], point_size_[I]) =
            blk.inner_matrix(bp, E, E);
      }
    });
    return 0.5 * (G + G.transpose());
  }

  /// @}

  /// @brief Access the tuple of sub-manifold blocks (const).
  const std::tuple<Ms...>& blocks() const { return blocks_; }

 private:
  /// @brief The tuple element type of block `I`.
  template <std::size_t I>
  using block_t = std::tuple_element_t<I, std::tuple<Ms...>>;

  /// @brief Invoke `f.operator()<I>()` for each block index `I = 0 .. N-1`.
  template <typename F>
  static void for_each_index(F&& f) {
    [&f]<std::size_t... I>(std::index_sequence<I...>) {
      (f.template operator()<I>(), ...);
    }(std::make_index_sequence<N>{});
  }

  /// @brief Stack one per-block vector, given by @p f, into a full point-sized vector.
  template <typename F>
  Eigen::VectorXd stack_points(F&& f) const {
    Eigen::VectorXd out(total_point_);
    for_each_index([&]<std::size_t I>() {
      out.segment(point_off_[I], point_size_[I]) = f(std::get<I>(blocks_));
    });
    return out;
  }

  /// @brief Slice a length-`size` segment of `big` starting at `off` and return
  /// it as the (possibly fixed-size) Eigen vector type `Vec`.
  ///
  /// @details Resizes a dynamic target (`VectorXd`), default-constructs a fixed-size
  /// target (`Vector3d`) at its compile-time size, and assigns the same-length segment
  /// to either.
  template <typename Vec>
  static Vec slice(const Eigen::VectorXd& big, int off, int size) {
    Vec out;
    if constexpr (Vec::SizeAtCompileTime == Eigen::Dynamic) {
      out.resize(size);
    }
    out = big.segment(off, size);
    return out;
  }

  std::tuple<Ms...> blocks_;         ///< The sub-manifold instances.
  std::array<int, N> point_size_{};  ///< Ambient point size of each block.
  std::array<int, N> point_off_{};   ///< Prefix offset of each block's point segment.
  std::array<int, N> tan_size_{};    ///< Ambient tangent size of each block.
  std::array<int, N> tan_off_{};     ///< Prefix offset of each block's tangent segment.
  std::array<int, N> cube_size_{};   ///< Unit-cube size of each block.
  std::array<int, N> cube_off_{};    ///< Prefix offset of each block's cube segment.
  int total_point_ = 0;              ///< Total stacked point length.
  int total_tan_ = 0;                ///< Total stacked tangent length.
  int total_cube_ = 0;               ///< Total unit-cube length.
  int dim_ = 0;                      ///< Intrinsic dimension (sum of block dims).
  mutable DynamicSampler sampler_;  ///< Joint sampler over all blocks (default scrambled Halton).
  mutable Eigen::VectorXd cube_buf_;                       ///< Preallocated joint unit-cube buffer.
};

/// @brief Deduce `Ms...` and build a `ProductManifold`.
/// @param ms The sub-manifold instances, in order.
/// @return `ProductManifold<Ms...>` holding the given blocks.
template <typename... Ms>
auto make_product(Ms... ms) {
  return ProductManifold<Ms...>(std::move(ms)...);
}

// Verify the composed types satisfy RiemannianManifold.
static_assert(RiemannianManifold<ProductManifold<Euclidean<Eigen::Dynamic>, SE2<>>>);
static_assert(RiemannianManifold<ProductManifold<Sphere<>, Euclidean<Eigen::Dynamic>>>);

}  // namespace geodex

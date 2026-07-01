/// @file product.hpp
/// @brief Riemannian product manifold \f$ \mathcal{M}_1 \times \cdots \times \mathcal{M}_N \f$.

#pragma once

#include <cmath>

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/manifold/sphere.hpp"

namespace geodex {

namespace detail {

/// @brief True when a manifold provides a `project(p, v)` method mapping an
/// ambient vector to the tangent space at `p`.
///
/// @details Named distinctly from `geodex::detail::HasProject` (defined in
/// `algorithm/interpolation.hpp`) so both headers can be included in the same
/// translation unit without a concept re-definition. `ProductManifold::project`
/// dispatches on this per block: blocks that provide `project` (e.g. `Sphere`)
/// have it applied to their tangent segment; blocks that do not (their tangent
/// space is the whole ambient space) use the identity.
template <typename M>
concept ProductBlockHasProject =
    requires(const M m, const typename M::Point p, const typename M::Tangent v) {
      { m.project(p, v) } -> std::same_as<typename M::Tangent>;
    };

}  // namespace detail

// ---------------------------------------------------------------------------
// ProductManifold
// ---------------------------------------------------------------------------

/// @brief The Riemannian product of \f$ N \f$ sub-manifolds \f$ \mathcal{M}_1 \times \cdots \times
/// \mathcal{M}_N \f$.
///
/// @details The product metric is the direct sum
/// \f$ g = g_1 \oplus \cdots \oplus g_N \f$, so every operation decouples across
/// blocks: the exponential/logarithmic maps and geodesics act block-wise, the
/// inner product is a block sum, and the geodesic distance is
/// \f$ d = \sqrt{\sum_i d_i(p_i, q_i)^2} \f$. All of these are closed-form
/// whenever the blocks are — no numerical integration is required.
///
/// A typical use is composing a mobile base with a manipulator, e.g.
/// \f$ \mathrm{SE}(2) \times \mathbb{R}^n \f$, or a base with an orientation on a
/// sphere.
///
/// **Point and tangent layout.** Points are stacked ambient representations:
/// block \f$ i \f$ occupies a contiguous segment of a single `Eigen::VectorXd`
/// whose length is that block's ambient point size (`M_i::Point::size()`).
/// Tangent vectors are likewise stacked in each block's *ambient* tangent
/// representation (`M_i::Tangent::size()`, which equals the point size for every
/// manifold in this library). This mirrors how `Sphere` itself stores a
/// 2-dimensional tangent as a size-3 ambient vector orthogonal to the base
/// point: the ambient representation is what the block's `exp`/`log`/`inner`/
/// `project` consume, and it is the representation `discrete_geodesic`'s
/// finite-difference path expects (it builds a tangent basis with `p.size()`
/// rows and `dim()` columns).
///
/// Consequently `dim()` — the intrinsic dimension, the sum of the block
/// dimensions — may be strictly smaller than the stored tangent length. For
/// `Sphere<> x Euclidean(2)`, `dim() == 2 + 2 == 4` while the stored point and
/// tangent vectors both have length `3 + 2 == 5`. Point offsets and tangent
/// offsets are cached separately (per the two-representation design) even though
/// they coincide for every block here; keeping them apart makes the slicing
/// robust should a block ever use a tangent representation smaller than its
/// point.
///
/// @tparam Ms The sub-manifold types, each satisfying `RiemannianManifold`.
template <typename... Ms>
class ProductManifold {
 public:
  using Scalar = double;             ///< Scalar type.
  using Point = Eigen::VectorXd;     ///< Stacked ambient point representation.
  using Tangent = Eigen::VectorXd;   ///< Stacked ambient tangent representation.

  /// @brief Number of sub-manifold blocks.
  static constexpr std::size_t N = sizeof...(Ms);

  /// @brief Construct from one instance of each sub-manifold.
  ///
  /// @details Caches, once at construction, each block's ambient point size
  /// (`random_point().size()`), its ambient tangent size (`log(p, p).size()`),
  /// its intrinsic dimension (`dim()`), and the corresponding prefix offsets.
  /// @param ms The sub-manifold instances, in order.
  explicit ProductManifold(Ms... ms) : blocks_(std::move(ms)...) {
    int poff = 0, toff = 0, d = 0;
    for_each_index([&]<std::size_t I>() {
      const auto& blk = std::get<I>(blocks_);
      const auto rp = blk.random_point();
      const int ps = static_cast<int>(rp.size());
      // Ambient tangent size: log(p, p) is the (correctly sized) zero tangent.
      const int ts = static_cast<int>(blk.log(rp, rp).size());
      point_size_[I] = ps;
      point_off_[I] = poff;
      poff += ps;
      tan_size_[I] = ts;
      tan_off_[I] = toff;
      toff += ts;
      d += blk.dim();
    });
    total_point_ = poff;
    total_tan_ = toff;
    dim_ = d;
  }

  /// @brief Intrinsic dimension: the sum of the block dimensions.
  int dim() const { return dim_; }

  /// @brief Runtime query: is `log` the Riemannian logarithm of the product
  /// metric?
  ///
  /// @details True exactly when every block's `log` is the Riemannian logarithm
  /// of its own metric (the product log is the direct sum of the block logs).
  /// When true, `discrete_geodesic` takes the fast log-direction natural
  /// gradient; otherwise it falls back to finite differences, which is always
  /// safe.
  bool has_riemannian_log_runtime() const {
    bool all = true;
    for_each_index(
        [&]<std::size_t I>() { all = all && geodex::is_riemannian_log(std::get<I>(blocks_)); });
    return all;
  }

  /// @brief Sample a random point: the concatenation of each block's
  /// `random_point()`.
  Point random_point() const {
    Point out(total_point_);
    for_each_index([&]<std::size_t I>() {
      out.segment(point_off_[I], point_size_[I]) = std::get<I>(blocks_).random_point();
    });
    return out;
  }

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

  /// @brief Riemannian inner product: the sum of the block inner products.
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
  /// applied to their tangent segment; blocks whose tangent space is the whole
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

  /// @brief Slice a length-`size` segment of `big` starting at `off` and return
  /// it as the (possibly fixed-size) Eigen vector type `Vec`.
  ///
  /// @details Handles both fixed-size (`Vector3d`) and dynamic (`VectorXd`)
  /// targets: dynamic targets are resized, fixed targets are default-constructed
  /// at their compile-time size; the same-length segment is then assigned in
  /// both cases.
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
  int total_point_ = 0;              ///< Total stacked point length.
  int total_tan_ = 0;                ///< Total stacked tangent length.
  int dim_ = 0;                      ///< Intrinsic dimension (sum of block dims).
};

/// @brief Convenience factory: deduce `Ms...` and build a `ProductManifold`.
/// @param ms The sub-manifold instances, in order.
/// @return `ProductManifold<Ms...>` holding the given blocks.
template <typename... Ms>
auto make_product(Ms... ms) {
  return ProductManifold<Ms...>(std::move(ms)...);
}

// Verify the composed types satisfy RiemannianManifold, including the
// point-size != dim case (Sphere contributes ambient size 3 but dim 2).
static_assert(RiemannianManifold<ProductManifold<Euclidean<Eigen::Dynamic>, SE2<>>>);
static_assert(RiemannianManifold<ProductManifold<Sphere<>, Euclidean<Eigen::Dynamic>>>);

}  // namespace geodex

/// @file concepts.hpp
/// @brief Core C++20 concepts defining the manifold interface hierarchy.

#pragma once

#include <concepts>
#include <type_traits>

namespace geodex {

/// @brief A smooth manifold with point/tangent types and basic operations.
///
/// @details A type satisfying `Manifold` must provide:
/// - `Scalar`, `Point`, and `Tangent` type aliases
/// - `dim()` returning the intrinsic dimension
/// - `random_point()` returning a uniformly sampled point
template <typename M>
concept Manifold = requires(const M m) {
  typename M::Scalar;
  typename M::Point;
  typename M::Tangent;
  { m.dim() } -> std::convertible_to<int>;
  { m.random_point() } -> std::same_as<typename M::Point>;
};

/// @brief A Riemannian manifold with metric, distance, and geodesic operations.
///
/// @details Extends `Manifold` with:
/// - `inner(p, u, v)` — Riemannian inner product \f$ \langle u, v \rangle_p \f$
/// - `norm(p, v)` — Riemannian norm \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$
/// - `distance(p, q)` — geodesic distance \f$ d(p, q) \f$
/// - `geodesic(p, q, t)` — geodesic interpolation at parameter \f$ t \in [0, 1] \f$
/// - `exp(p, v)` — exponential map (or retraction) \f$ \exp_p(v) \f$
/// - `log(p, q)` — logarithmic map (or inverse retraction) \f$ \log_p(q) \f$
template <typename M>
concept RiemannianManifold =
    Manifold<M> &&
    requires(const M m, const typename M::Point p, const typename M::Point q,
             const typename M::Tangent u, const typename M::Tangent v, const typename M::Scalar t) {
      { m.inner(p, u, v) } -> std::convertible_to<typename M::Scalar>;
      { m.norm(p, v) } -> std::convertible_to<typename M::Scalar>;
      { m.distance(p, q) } -> std::convertible_to<typename M::Scalar>;
      { m.geodesic(p, q, t) } -> std::same_as<typename M::Point>;
      { m.exp(p, v) } -> std::same_as<typename M::Point>;
      { m.log(p, q) } -> std::same_as<typename M::Tangent>;
    };

/// @brief A manifold that provides its injectivity radius.
///
/// @details The injectivity radius \f$ \mathrm{inj}(\mathcal{M}) \f$ is the largest
/// radius for which the exponential map is a diffeomorphism. For the round sphere
/// this is \f$ \pi \f$; for Euclidean space it is \f$ \infty \f$.
template <typename M>
concept HasInjectivityRadius = Manifold<M> && requires(const M m) {
  { m.injectivity_radius() } -> std::convertible_to<typename M::Scalar>;
};

}  // namespace geodex

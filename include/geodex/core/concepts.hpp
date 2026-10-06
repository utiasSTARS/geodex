/// @file concepts.hpp
/// @brief Core C++20 concepts defining the manifold interface hierarchy.

#pragma once

#include <concepts>
#include <type_traits>

#include <Eigen/Core>

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
/// @details Extends `Manifold` with
/// - `inner(p, u, v)`, the Riemannian inner product \f$ \langle u, v \rangle_p \f$
/// - `norm(p, v)`, the Riemannian norm \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$
/// - `distance(p, q)`, the geodesic distance \f$ d(p, q) \f$
/// - `geodesic(p, q, t)`, geodesic interpolation at parameter \f$ t \in [0, 1] \f$
/// - `exp(p, v)`, the exponential map (or retraction) \f$ \exp_p(v) \f$
/// - `log(p, q)`, the logarithmic map (or inverse retraction) \f$ \log_p(q) \f$
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
/// radius for which the exponential map is a diffeomorphism. It is \f$ \pi \f$ for the
/// round sphere and \f$ \infty \f$ for Euclidean space.
template <typename M>
concept HasInjectivityRadius = Manifold<M> && requires(const M m) {
  { m.injectivity_radius() } -> std::convertible_to<typename M::Scalar>;
};

/// @brief A manifold that provides its metric tensor on coordinate velocities.
///
/// @details Returns the frame pullback \f$ G(q) = J(q)^\top M(q)\, J(q) \f$, where
/// `inner_matrix` measures tangent vectors in the manifold's own frame (the body frame
/// for a Lie group). A flat periodic manifold (SO(2), the torus) defines it as its Gram
/// matrix. Other flat manifolds do not have a frame to convert and leave it undefined.
template <typename M>
concept HasCoordinateMetric = Manifold<M> && requires(const M m, const typename M::Point p) {
  { m.coordinate_metric(p) } -> std::convertible_to<Eigen::MatrixXd>;
};

/// @brief A manifold whose coordinates are periodic on some axes.
///
/// @details `periods()` gives the deck-group generator per coordinate, 0 where the
/// axis is aperiodic. Curved manifolds do not have a flat quotient and do not define it.
template <typename M>
concept HasPeriods = Manifold<M> && requires(const M m) {
  { m.periods() } -> std::convertible_to<Eigen::VectorXd>;
};

/// @brief A manifold that exposes the sampler behind `random_point()`.
///
/// @details The OMPL integration copies it, with its kind, configuration and state, into
/// every planning sampler.
template <typename M>
concept HasSampler = Manifold<M> && requires(const M m) {
  typename M::SamplerType;
  { m.sampler() } -> std::convertible_to<typename M::SamplerType>;
};

}  // namespace geodex

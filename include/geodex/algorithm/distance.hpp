/// @file distance.hpp
/// @brief Midpoint distance approximation algorithm.

#pragma once

#include "geodex/core/concepts.hpp"

namespace geodex {

/// @brief Approximate geodesic distance using the midpoint method.
///
/// @details Computes the distance between two points by evaluating the difference of
/// log maps at their geodesic midpoint:
/// \f[
///   m = \exp_a\!\bigl(\tfrac{1}{2}\,\log_a(b)\bigr), \qquad
///   d_{\mathrm{mid}} = \bigl\|\log_m(b) - \log_m(a)\bigr\|_m
/// \f]
///
/// This method provides a third-order approximation to the exact geodesic distance
/// on general Riemannian manifolds. It is computationally cheaper than numerical
/// geodesic integration while maintaining high accuracy for moderately curved spaces.
/// For manifolds with exact exponential and logarithmic maps, this formula yields
/// the exact geodesic distance.
///
/// @see Phone Thiha Kyaw, Jonathan Kelly. "Geometry-Aware Sampling-Based Motion
///   Planning on Riemannian Manifolds." Proceedings of the 17th World Symposium on the
///   Algorithmic Foundations of Robotics (WAFR), 2026. arXiv:2602.00992.
///
/// @tparam M A type satisfying `RiemannianManifold`.
/// @param m The manifold instance.
/// @param a First point on the manifold.
/// @param b Second point on the manifold.
/// @return The approximate geodesic distance between \p a and \p b.
template <RiemannianManifold M>
auto distance_midpoint(const M& m, const typename M::Point& a, const typename M::Point& b) ->
    typename M::Scalar {
  auto v_ab = m.log(a, b);
  auto midpoint = m.exp(a, 0.5 * v_ab);
  auto v_ma = m.log(midpoint, a);
  auto v_mb = m.log(midpoint, b);
  auto v_diff = v_mb - v_ma;
  auto d = m.norm(midpoint, v_diff);
  return d;
}

}  // namespace geodex

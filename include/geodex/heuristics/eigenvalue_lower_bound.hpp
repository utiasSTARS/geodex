/// @file eigenvalue_lower_bound.hpp
/// @brief Eigenvalue-lower-bound heuristic for configuration-dependent metrics.

#pragma once

#include <cmath>
#include <utility>

#include "geodex/heuristics/euclidean.hpp"

namespace geodex::heuristics {

/// @brief Eigenvalue lower-bound heuristic.
///
/// @details For a Riemannian metric \f$ M(q) \f$, the geodesic distance
/// satisfies
/// \f[
///   d_M(a,b) \ge \sqrt{\lambda_{\min}}\; d_{\mathrm{base}}(a,b),
/// \f]
/// where \f$ \lambda_{\min} \f$ is the global minimum eigenvalue of
/// \f$ M(q) \f$ over the configuration space \f$ \mathcal{Q} \f$.
///
/// Admissibility follows from Cauchy--Schwarz: for any curve \f$ \gamma \f$,
/// \f[
///   L_M(\gamma) = \int \sqrt{\dot\gamma^\top M(\gamma)\,\dot\gamma}\,dt
///               \ge \sqrt{\lambda_{\min}} \int \|\dot\gamma\|\,dt
///               = \sqrt{\lambda_{\min}}\, L_{\mathrm{base}}(\gamma).
/// \f]
///
/// @tparam BaseHeuristicT Heuristic for the base manifold distance (default: Euclidean).
template <typename BaseHeuristicT = Euclidean>
class EigenvalueLowerBound {
 public:
  /// @brief Construct with the minimum eigenvalue.
  /// @param lambda_min Minimum eigenvalue of \f$ M(q) \f$ over \f$ \mathcal{Q} \f$.
  /// @param base_heuristic Heuristic for the base manifold distance.
  explicit EigenvalueLowerBound(double lambda_min,
                                BaseHeuristicT base_heuristic = BaseHeuristicT{})
      : sqrt_lambda_min_(std::sqrt(lambda_min)),
        base_heuristic_(std::move(base_heuristic)) {}

  /// @brief Compute \f$ \sqrt{\lambda_{\min}}\, h_{\mathrm{base}}(a, b) \f$.
  /// @param a First point.
  /// @param b Second point.
  /// @return Admissible lower bound on geodesic distance.
  template <typename PointA, typename PointB>
  auto operator()(const PointA& a, const PointB& b) const -> double {
    return sqrt_lambda_min_ * base_heuristic_(a, b);
  }

  /// @brief Cached \f$ \sqrt{\lambda_{\min}} \f$.
  auto sqrt_lambda_min() const -> double { return sqrt_lambda_min_; }

  /// @brief Access the base heuristic.
  auto base_heuristic() const -> const BaseHeuristicT& { return base_heuristic_; }

 private:
  double sqrt_lambda_min_;
  BaseHeuristicT base_heuristic_;
};

}  // namespace geodex::heuristics

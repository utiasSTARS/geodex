/// @file affine_combined.hpp
/// @brief Variadic positive-affine combination of Riemannian metric policies.

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <utility>

#include <Eigen/Core>

#include "geodex/core/metric.hpp"

namespace geodex {

/// @brief Positive linear combination of Riemannian metric policies.
///
/// @details Composes \f$ N \f$ metric policies \f$ g_1, \dots, g_N \f$ with
/// non-negative coefficients \f$ c_1, \dots, c_N \f$ into the metric
/// \f[
///   \langle u, v \rangle_p \;=\; \sum_{k=1}^{N} c_k \, \langle u, v \rangle_p^{g_k}.
/// \f]
/// The set of symmetric positive-definite matrices is a convex cone, so any
/// non-negative combination of SPD inner products is itself SPD as long as at
/// least one coefficient is positive and paired with a strictly positive-definite
/// summand. This underpins composite metrics such as
/// "pullback + \f$ \beta \f$ · kinetic-energy".
///
/// The class makes no `Manifold` claims of its own — it is a metric *policy*
/// like `IdentityMetric`, `ConstantSPDMetric`, or `KineticEnergyMetric`. Plug
/// it into a `ConfigurationSpace` (or any consumer of `inner` / `norm`) the
/// same way as the underlying policies.
///
/// @tparam Ms Metric policy types. Each must expose `inner(p, u, v)` and
///   `norm(p, v)`. The optional `inner_matrix(p, U, V)` overload is enabled
///   only when every summand provides one (`MetricHasInnerMatrix`).
template <typename... Ms>
class AffineCombinedMetric {
 public:
  /// @brief Number of metrics in the combination.
  static constexpr std::size_t arity = sizeof...(Ms);
  static_assert(arity > 0, "AffineCombinedMetric requires at least one metric");

  /// @brief Construct from a coefficient array and a parameter pack of metrics.
  /// @param coeffs Per-summand coefficients; each must be \f$\ge 0\f$.
  /// @param metrics The metric policy instances.
  AffineCombinedMetric(std::array<double, arity> coeffs, Ms... metrics)
      : metrics_(std::forward<Ms>(metrics)...), coeffs_(coeffs) {
    if constexpr (arity > 0) {
      bool any_positive = false;
      for (std::size_t k = 0; k < arity; ++k) {
        assert(coeffs_[k] >= 0.0 && "AffineCombinedMetric: coefficients must be non-negative");
        if (coeffs_[k] > 0.0) any_positive = true;
      }
      assert(any_positive && "AffineCombinedMetric: at least one coefficient must be > 0");
      (void)any_positive;
    }
  }

  /// @brief Inner product \f$ \sum_k c_k \langle u, v \rangle_p^{g_k} \f$.
  template <typename Point, typename Tangent>
  double inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return inner_impl(p, u, v, std::make_index_sequence<arity>{});
  }

  /// @brief Induced norm \f$ \|v\|_p = \sqrt{\langle v, v \rangle_p} \f$.
  template <typename Point, typename Tangent>
  double norm(const Point& p, const Tangent& v) const {
    return riemannian_norm(*this, p, v);
  }

  /// @brief Batched inner product \f$ \sum_k c_k\, U^\top M_k(p)\, V \f$.
  ///
  /// @details Enabled only when every summand provides `inner_matrix`. Returns
  /// the coefficient-weighted sum of per-summand tensors in a single call.
  template <typename Point>
    requires (MetricHasInnerMatrix<Ms, Point> && ...)
  Eigen::MatrixXd inner_matrix(const Point& p, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const {
    return inner_matrix_impl(p, U, V, std::make_index_sequence<arity>{});
  }

  /// @brief Access the coefficient array.
  const std::array<double, arity>& coeffs() const { return coeffs_; }

  /// @brief Access the I-th metric (compile-time index).
  template <std::size_t I>
  const auto& metric() const {
    static_assert(I < arity, "AffineCombinedMetric::metric<I>: index out of range");
    return std::get<I>(metrics_);
  }

 private:
  template <typename Point, typename Tangent, std::size_t... Is>
  double inner_impl(const Point& p, const Tangent& u, const Tangent& v,
                    std::index_sequence<Is...>) const {
    return ((coeffs_[Is] * std::get<Is>(metrics_).inner(p, u, v)) + ...);
  }

  template <typename Point, std::size_t... Is>
  Eigen::MatrixXd inner_matrix_impl(const Point& p, const Eigen::MatrixXd& U,
                                    const Eigen::MatrixXd& V, std::index_sequence<Is...>) const {
    Eigen::MatrixXd acc =
        coeffs_[0] * std::get<0>(metrics_).inner_matrix(p, U, V);
    ((Is == 0 ? void()
              : void(acc.noalias() += coeffs_[Is] * std::get<Is>(metrics_).inner_matrix(p, U, V))),
     ...);
    return acc;
  }

  std::tuple<Ms...> metrics_;
  std::array<double, arity> coeffs_;
};

/// @brief Deduction guide enabling `AffineCombinedMetric({0.5, 0.5}, m1, m2)`.
template <typename... Ms>
AffineCombinedMetric(std::array<double, sizeof...(Ms)>, Ms...) -> AffineCombinedMetric<Ms...>;

}  // namespace geodex

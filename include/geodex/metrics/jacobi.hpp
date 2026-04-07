/// @file jacobi.hpp
/// @brief Jacobi metric — a configuration-dependent scaling of the kinetic energy metric.

#pragma once

#include <Eigen/Core>
#include <utility>

#include <geodex/core/metric.hpp>
#include <geodex/metrics/kinetic_energy.hpp>
#include <geodex/metrics/weighted.hpp>

namespace geodex {

namespace detail {

/// @brief Functor capturing the Jacobi scaling \f$\alpha(q) = 2(H - P(q))\f$.
///
/// @details Used as the `AlphaT` parameter of `WeightedMetric` to turn a
/// kinetic-energy metric into a Jacobi metric. We use a named functor struct
/// (rather than a lambda) so that `JacobiMetric` has a nameable type.
template <typename PotentialFn>
struct JacobiAlphaFunctor {
  PotentialFn potential_fn_;
  double total_energy_;

  template <typename Point>
  double operator()(const Point& q) const {
    return 2.0 * (total_energy_ - potential_fn_(q));
  }
};

}  // namespace detail

/// @brief Jacobi metric conformally scaling a kinetic energy metric by the
/// available kinetic energy \f$ H - P(q) \f$.
///
/// @details The inner product at configuration \f$ q \f$ is:
/// \f$ \langle u, v \rangle_q = 2\,(H - P(q))\, u^\top M(q) \, v \f$
/// where \f$ H \f$ is the total energy, \f$ P(q) \f$ is the potential, and
/// \f$ M(q) \f$ is the mass matrix. Geodesics of this metric are the natural
/// motions of the mechanical system (Maupertuis' principle).
///
/// Implementation: this is a thin composition of `KineticEnergyMetric` (the
/// mass matrix) and `WeightedMetric` (the configuration-dependent scaling).
/// The `inner`, `inner_matrix`, and `norm` methods forward to the composed
/// metric — no duplicated mass-matrix or potential-evaluation logic.
///
/// @tparam MassMatrixFn Callable returning the SPD mass matrix at \f$ q \f$.
/// @tparam PotentialFn Callable returning the scalar potential \f$ P(q) \f$.
template <typename MassMatrixFn, typename PotentialFn>
struct JacobiMetric {
  using KEMetric = KineticEnergyMetric<MassMatrixFn>;
  using AlphaFn = detail::JacobiAlphaFunctor<PotentialFn>;
  using InnerMetric = WeightedMetric<KEMetric, AlphaFn>;

  InnerMetric inner_metric_;

  /// @brief Construct a Jacobi metric.
  /// @param mass_fn Callable returning the SPD mass matrix.
  /// @param pot_fn Callable returning the potential energy.
  /// @param H Total energy (must satisfy \f$ H > P(q) \f$ everywhere on the path).
  JacobiMetric(MassMatrixFn mass_fn, PotentialFn pot_fn, double H)
      : inner_metric_(KEMetric{std::move(mass_fn)},
                      AlphaFn{std::move(pot_fn), H}) {}

  /// @brief Compute the inner product \f$ 2(H - P(q))\, u^\top M(q) \, v \f$.
  template <typename Point, typename Tangent>
  double inner(const Point& q, const Tangent& u, const Tangent& v) const {
    return inner_metric_.inner(q, u, v);
  }

  /// @brief Compute the norm \f$ \|v\|_q \f$.
  template <typename Point, typename Tangent>
  double norm(const Point& q, const Tangent& v) const {
    return riemannian_norm(*this, q, v);
  }

  /// @brief Batched inner product: \f$U^\top \bigl(2(H - P(q)) M(q)\bigr) V\f$
  /// computed with a single evaluation of \f$M(q)\f$ and \f$P(q)\f$ through
  /// the wrapped `WeightedMetric`.
  template <typename Point>
  Eigen::MatrixXd inner_matrix(const Point& q, const Eigen::MatrixXd& U,
                                const Eigen::MatrixXd& V) const {
    return inner_metric_.inner_matrix(q, U, V);
  }

  /// @brief Return the total energy \f$ H \f$ set at construction.
  double total_energy() const { return inner_metric_.alpha_.total_energy_; }
};

/// @brief Factory function for `JacobiMetric` — convenience wrapper that lets
/// users write `make_jacobi_metric(mass_fn, pot_fn, H)` without naming the
/// template parameters.
template <typename MassMatrixFn, typename PotentialFn>
auto make_jacobi_metric(MassMatrixFn mass_fn, PotentialFn pot_fn, double H) {
  return JacobiMetric<MassMatrixFn, PotentialFn>{std::move(mass_fn), std::move(pot_fn), H};
}

}  // namespace geodex

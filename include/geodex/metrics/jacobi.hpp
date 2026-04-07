/// @file jacobi.hpp
/// @brief Jacobi metric for minimum-time geodesics under a potential field.

#pragma once

#include <Eigen/Core>

#include <geodex/core/metric.hpp>

namespace geodex {

/// @brief Jacobi metric conformally scaling a kinetic energy metric by
/// the available kinetic energy \f$ H - P(q) \f$.
///
/// @details The inner product at configuration \f$ q \f$ is:
/// \f$ \langle u, v \rangle_q = 2\,(H - P(q))\, u^\top M(q) \, v \f$
/// where \f$ H \f$ is the total energy (conserved), \f$ P(q) \f$ is the
/// potential energy, and \f$ M(q) \f$ is the mass matrix. Geodesics of
/// this metric are natural motions of the mechanical system (Maupertuis'
/// principle).
///
/// @tparam MassMatrixFn Callable returning the SPD mass matrix at \f$ q \f$.
/// @tparam PotentialFn Callable returning the scalar potential \f$ P(q) \f$.
template <typename MassMatrixFn, typename PotentialFn>
struct JacobiMetric {
  MassMatrixFn mass_matrix_fn_;  ///< Callable returning M(q).
  PotentialFn potential_fn_;     ///< Callable returning P(q).
  double total_energy_;          ///< Total energy \f$ H \f$.

  /// @brief Construct a Jacobi metric.
  /// @param mass_fn Callable returning the SPD mass matrix.
  /// @param pot_fn Callable returning the potential energy.
  /// @param H Total energy (must satisfy \f$ H > P(q) \f$ everywhere on the path).
  JacobiMetric(MassMatrixFn mass_fn, PotentialFn pot_fn, double H)
      : mass_matrix_fn_(std::move(mass_fn)),
        potential_fn_(std::move(pot_fn)),
        total_energy_(H) {}

  /// @brief Compute the inner product \f$ 2(H - P(q))\, u^\top M(q) \, v \f$.
  /// @param q Configuration point.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  template <typename Point, typename Tangent>
  double inner(const Point& q, const Tangent& u, const Tangent& v) const {
    double kinetic_factor = 2.0 * (total_energy_ - potential_fn_(q));
    return kinetic_factor * u.dot(mass_matrix_fn_(q) * v);
  }

  /// @brief Compute the norm \f$ \|v\|_q \f$.
  /// @param q Configuration point.
  /// @param v Tangent vector.
  /// @return The norm value.
  template <typename Point, typename Tangent>
  double norm(const Point& q, const Tangent& v) const {
    return riemannian_norm(*this, q, v);
  }

  /// @brief Batched inner product: \f$U^\top \bigl(2(H - P(q)) M(q)\bigr) V\f$
  /// computed with a single evaluation of \f$M(q)\f$ and \f$P(q)\f$.
  template <typename Point>
  Eigen::MatrixXd inner_matrix(const Point& q, const Eigen::MatrixXd& U,
                                const Eigen::MatrixXd& V) const {
    const double kinetic_factor = 2.0 * (total_energy_ - potential_fn_(q));
    return kinetic_factor * (U.transpose() * mass_matrix_fn_(q) * V);
  }
};

}  // namespace geodex

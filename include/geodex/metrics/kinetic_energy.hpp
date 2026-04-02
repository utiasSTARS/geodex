/// @file kinetic_energy.hpp
/// @brief Configuration-dependent kinetic energy metric \f$ g(q) = M(q) \f$.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace geodex {

/// @brief Kinetic energy metric where the inner product is defined by a
/// configuration-dependent mass matrix.
///
/// @details The inner product at configuration \f$ q \f$ is:
/// \f$ \langle u, v \rangle_q = u^\top M(q) \, v \f$
/// where \f$ M(q) \f$ is a symmetric positive-definite mass matrix returned
/// by the user-provided callable.
///
/// @tparam MassMatrixFn A callable type with signature
///   `auto operator()(const Point& q) -> Eigen::MatrixXd` (or fixed-size matrix).
template <typename MassMatrixFn>
struct KineticEnergyMetric {
  MassMatrixFn mass_matrix_fn_;  ///< Callable returning M(q).

  /// @brief Construct with a mass matrix function.
  /// @param fn Callable returning the SPD mass matrix at a given configuration.
  explicit KineticEnergyMetric(MassMatrixFn fn) : mass_matrix_fn_(std::move(fn)) {}

  /// @brief Compute the inner product \f$ \langle u, v \rangle_q = u^\top M(q) \, v \f$.
  /// @param q Configuration point.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  template <typename Point, typename Tangent>
  double inner(const Point& q, const Tangent& u, const Tangent& v) const {
    return u.dot(mass_matrix_fn_(q) * v);
  }

  /// @brief Compute the norm \f$ \|v\|_q = \sqrt{v^\top M(q) \, v} \f$.
  /// @param q Configuration point.
  /// @param v Tangent vector.
  /// @return The norm value.
  template <typename Point, typename Tangent>
  double norm(const Point& q, const Tangent& v) const {
    return std::sqrt(inner(q, v, v));
  }

  /// @brief Return the injectivity radius \f$ \infty \f$ (assumes flat topology).
  double injectivity_radius() const { return std::numeric_limits<double>::infinity(); }
};

}  // namespace geodex

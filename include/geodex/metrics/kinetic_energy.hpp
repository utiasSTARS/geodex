/// @file kinetic_energy.hpp
/// @brief Configuration-dependent kinetic energy metric \f$ g(q) = M(q) \f$.

#pragma once

#include <Eigen/Core>
#include <geodex/core/metric.hpp>
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
class KineticEnergyMetric {
 public:
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
    return riemannian_norm(*this, q, v);
  }

  /// @brief Batched inner product: \f$U^\top M(q)\, V\f$ computed with a single
  /// call to the mass-matrix function.
  ///
  /// @details This is the performance-critical path for `natural_gradient_fd`
  /// when the mass matrix is expensive to compute (e.g., forward kinematics for
  /// a manipulator): instead of calling `mass_matrix_fn_(q)` for every scalar
  /// \f$G_{ij} = \langle e_i, e_j\rangle_q\f$, we call it once and form the
  /// entire \f$d\times d\f$ tensor in a single matmul.
  template <typename Point>
  Eigen::MatrixXd inner_matrix(const Point& q, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const {
    return U.transpose() * mass_matrix_fn_(q) * V;
  }

  /// @brief Return the injectivity radius \f$ \infty \f$ (assumes flat topology).
  double injectivity_radius() const { return std::numeric_limits<double>::infinity(); }

 private:
  MassMatrixFn mass_matrix_fn_;  ///< Callable returning M(q).
};

}  // namespace geodex

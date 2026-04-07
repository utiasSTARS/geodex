/// @file constant_spd.hpp
/// @brief Point-independent Riemannian metric defined by a constant SPD matrix.

#pragma once

#include <Eigen/Core>

#include <geodex/core/metric.hpp>

namespace geodex {

/// @brief Point-independent Riemannian metric defined by a constant SPD matrix.
///
/// @details The inner product is \f$ \langle u, v \rangle_p = u^\top A \, v \f$
/// where \f$ A \f$ is a symmetric positive-definite matrix. The metric does not
/// depend on the base point \f$ p \f$.
///
/// This is a manifold-agnostic metric: the same class works for spheres,
/// Euclidean spaces, tori, or any manifold whose tangent vectors are
/// Eigen column vectors.
///
/// @tparam Dim Compile-time dimension, or `Eigen::Dynamic`.
template <int Dim = Eigen::Dynamic>
struct ConstantSPDMetric {
  Eigen::Matrix<double, Dim, Dim> A_;  ///< The SPD weight matrix.

  /// @brief Construct with a given SPD weight matrix.
  /// @param A Symmetric positive-definite matrix defining the metric.
  explicit ConstantSPDMetric(const Eigen::Matrix<double, Dim, Dim>& A) : A_(A) {}

  /// @brief Compute the inner product \f$ \langle u, v \rangle = u^\top A \, v \f$.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  double inner(const Eigen::Vector<double, Dim>& /*p*/, const Eigen::Vector<double, Dim>& u,
               const Eigen::Vector<double, Dim>& v) const {
    return u.dot(A_ * v);
  }

  /// @brief Compute the norm \f$ \|v\| = \sqrt{v^\top A \, v} \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The norm value.
  double norm(const Eigen::Vector<double, Dim>& p, const Eigen::Vector<double, Dim>& v) const {
    return riemannian_norm(*this, p, v);
  }

  /// @brief Batched inner product: \f$U^\top A\, V\f$ in a single matrix multiply.
  ///
  /// @details Provides the `HasBatchInnerMatrix` fast path for algorithms that
  /// evaluate a tangent-metric tensor in a basis (e.g., `natural_gradient_fd`).
  /// For `ConstantSPDMetric` this is a simple linear-algebra shortcut; the
  /// bigger win is for point-dependent metrics like `KineticEnergyMetric`
  /// where the expensive mass matrix is evaluated once instead of \f$d^2\f$
  /// times.
  Eigen::MatrixXd inner_matrix(const Eigen::Vector<double, Dim>& /*p*/,
                                const Eigen::MatrixXd& U, const Eigen::MatrixXd& V) const {
    return U.transpose() * A_ * V;
  }
};

}  // namespace geodex

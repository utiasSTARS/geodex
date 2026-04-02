/// @file constant_spd.hpp
/// @brief Point-independent Riemannian metric defined by a constant SPD matrix.

#pragma once

#include <Eigen/Core>
#include <cmath>

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
    return std::sqrt(inner(p, v, v));
  }
};

}  // namespace geodex

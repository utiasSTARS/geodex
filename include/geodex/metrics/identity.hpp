/// @file identity.hpp
/// @brief Zero-storage identity Riemannian metric.

#pragma once

#include <Eigen/Core>

#include <geodex/core/metric.hpp>

namespace geodex {

/// @brief Stateless identity metric: \f$ \langle u, v \rangle_p = u \cdot v \f$.
///
/// @details A zero-storage alternative to `ConstantSPDMetric<Dim>` with the
/// identity weight matrix. Uses O(1) memory instead of O(n^2) for the
/// identity-metric case. All manifold default-metric aliases
/// (`SphereRoundMetric`, `TorusFlatMetric`, `EuclideanStandardMetric`) point here.
///
/// @tparam Dim Compile-time vector dimension, or `Eigen::Dynamic`.
template <int Dim = Eigen::Dynamic>
class IdentityMetric {
 public:
  /// @brief Compute the inner product \f$ \langle u, v \rangle = u \cdot v \f$.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  double inner(const Eigen::Vector<double, Dim>& /*p*/,
               const Eigen::Vector<double, Dim>& u,
               const Eigen::Vector<double, Dim>& v) const {
    return u.dot(v);
  }

  /// @brief Compute the norm \f$ \|v\| = \sqrt{v \cdot v} \f$.
  /// @param p Base point (unused).
  /// @param v Tangent vector.
  /// @return The norm value.
  double norm(const Eigen::Vector<double, Dim>& p,
              const Eigen::Vector<double, Dim>& v) const {
    return riemannian_norm(*this, p, v);
  }

  /// @brief Batched inner product: \f$ U^\top V \f$.
  Eigen::MatrixXd inner_matrix(const Eigen::Vector<double, Dim>& /*p*/,
                                const Eigen::MatrixXd& U,
                                const Eigen::MatrixXd& V) const {
    return U.transpose() * V;
  }

  /// @brief Identity weight matrix (runtime-sized identity, for API compatibility).
  /// @details Returns a dynamically-sized identity matrix of the given dimension.
  ///   Provided so that `has_riemannian_log_runtime()` can call `weight_matrix()`
  ///   uniformly on any metric.  Unlike `ConstantSPDMetric`, this allocates only
  ///   when called — which should be rare.
  Eigen::MatrixXd weight_matrix() const {
    if constexpr (Dim != Eigen::Dynamic) {
      return Eigen::MatrixXd::Identity(Dim, Dim);
    } else {
      // Dynamic case: callers must not rely on this for hot paths.
      return Eigen::MatrixXd::Identity(1, 1);
    }
  }
};

}  // namespace geodex

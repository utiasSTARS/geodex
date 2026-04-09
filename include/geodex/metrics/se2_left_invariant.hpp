/// @file se2_left_invariant.hpp
/// @brief Left-invariant metric on SE(2) — thin wrapper over ConstantSPDMetric<3>.

#pragma once

#include <Eigen/Core>

#include <geodex/core/metric.hpp>
#include <geodex/metrics/constant_spd.hpp>

namespace geodex {

/// @brief Left-invariant metric on SE(2).
///
/// @details The inner product is constant (left-invariant):
/// \f$ \langle u, v \rangle = w_x u_x v_x + w_y u_y v_y + w_\theta u_\theta v_\theta \f$.
/// The weights \f$ (w_x, w_y, w_\theta) \f$ allow anisotropic cost, e.g. penalizing
/// lateral motion for car-like robots.
///
/// Implementation: this is `ConstantSPDMetric<3>` with `A = diag(w_x, w_y, w_\theta)`.
/// The `weights_` field is preserved alongside the base metric so that
/// `SE2::has_riemannian_log_runtime()` can quickly check unit weights without
/// inspecting the full SPD matrix.
class SE2LeftInvariantMetric {
 public:
  /// @brief Construct with unit weights (isotropic).
  SE2LeftInvariantMetric() : SE2LeftInvariantMetric(1.0, 1.0, 1.0) {}

  /// @brief Construct with explicit weights.
  /// @param wx Translational weight in x.
  /// @param wy Translational weight in y.
  /// @param wtheta Rotational weight.
  SE2LeftInvariantMetric(double wx, double wy, double wtheta)
      : weights_(wx, wy, wtheta),
        base_(Eigen::Vector3d(wx, wy, wtheta).asDiagonal().toDenseMatrix()) {}

  /// @brief Access the diagonal weight vector \f$ (w_x, w_y, w_\theta) \f$.
  const Eigen::Vector3d& weights() const { return weights_; }

  /// @brief Compute the inner product via the wrapped ConstantSPDMetric.
  /// @param p Base point (unused for a constant metric).
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  double inner(const Eigen::Vector3d& p, const Eigen::Vector3d& u,
               const Eigen::Vector3d& v) const {
    return base_.inner(p, u, v);
  }

  /// @brief Batched inner product via the wrapped ConstantSPDMetric.
  /// @param p Base point.
  /// @param U Matrix whose columns are tangent vectors.
  /// @param V Matrix whose columns are tangent vectors.
  /// @return \f$ U^\top A \, V \f$.
  Eigen::MatrixXd inner_matrix(const Eigen::Vector3d& p, const Eigen::MatrixXd& U,
                                const Eigen::MatrixXd& V) const {
    return base_.inner_matrix(p, U, V);
  }

  /// @brief Compute the norm \f$ \|v\| = \sqrt{\langle v, v \rangle} \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The norm value.
  double norm(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return riemannian_norm(*this, p, v);
  }

 private:
  Eigen::Vector3d weights_;   ///< Diagonal weight vector \f$ (w_x, w_y, w_\theta) \f$.
  ConstantSPDMetric<3> base_; ///< Wrapped SPD metric with `A = diag(weights_)`.
};

}  // namespace geodex

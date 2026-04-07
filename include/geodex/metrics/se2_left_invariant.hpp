/// @file se2_left_invariant.hpp
/// @brief Left-invariant metric on SE(2).

#pragma once

#include <Eigen/Core>
#include <cmath>

namespace geodex {

/// @brief Left-invariant metric on SE(2).
///
/// @details The inner product is constant (left-invariant):
/// \f$ \langle u, v \rangle = w_x u_x v_x + w_y u_y v_y + w_\theta u_\theta v_\theta \f$.
/// The weights \f$ (w_x, w_y, w_\theta) \f$ allow anisotropic cost, e.g. penalizing
/// lateral motion for car-like robots.
struct SE2LeftInvariantMetric {
  Eigen::Vector3d weights_;  ///< Diagonal weight vector \f$ (w_x, w_y, w_\theta) \f$.

  /// @brief Construct with unit weights (isotropic).
  SE2LeftInvariantMetric() : weights_(1.0, 1.0, 1.0) {}

  /// @brief Construct with explicit weights.
  /// @param wx Translational weight in x.
  /// @param wy Translational weight in y.
  /// @param wtheta Rotational weight.
  SE2LeftInvariantMetric(double wx, double wy, double wtheta) : weights_(wx, wy, wtheta) {}

  /// @brief Compute the inner product \f$ \langle u, v \rangle = \sum_i w_i u_i v_i \f$.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  double inner(const Eigen::Vector3d& /*p*/, const Eigen::Vector3d& u,
               const Eigen::Vector3d& v) const {
    return weights_[0] * u[0] * v[0] + weights_[1] * u[1] * v[1] + weights_[2] * u[2] * v[2];
  }

  /// @brief Batched inner product: \f$U^\top \operatorname{diag}(w)\, V\f$.
  Eigen::MatrixXd inner_matrix(const Eigen::Vector3d& /*p*/, const Eigen::MatrixXd& U,
                                const Eigen::MatrixXd& V) const {
    return U.transpose() * weights_.asDiagonal() * V;
  }

  /// @brief Compute the norm \f$ \|v\| = \sqrt{\langle v, v \rangle} \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The norm value.
  double norm(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::sqrt(inner(p, v, v));
  }
};

}  // namespace geodex

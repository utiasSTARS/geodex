/// @file se3_invariant.hpp
/// @brief Invariant metric on SE(3) — thin wrapper over ConstantSPDMetric<6>.

#pragma once

#include <Eigen/Core>

#include "geodex/core/metric.hpp"
#include "geodex/metrics/constant_spd.hpp"

namespace geodex {

/// @brief Invariant metric on SE(3) (left- or right-invariant, depending on the
/// retraction it is paired with).
///
/// @details The inner product is constant on the Lie algebra \f$ \mathfrak{se}(3) \f$:
/// \f$ \langle u, v \rangle = u^\top \mathrm{diag}(w)\, v \f$, where a twist is
/// ordered \f$ \xi = [v;\,\omega] \f$ and the weight vector is
/// \f$ w = (w_{v_x}, w_{v_y}, w_{v_z},\; w_{\omega_x}, w_{\omega_y}, w_{\omega_z}) \f$.
/// The two-scalar constructor sets \f$ w = (w_t, w_t, w_t, w_r, w_r, w_r) \f$ so
/// translation and rotation can be weighted independently — SE(3) admits no
/// bi-invariant metric, so any choice of weights is a modeling decision (see
/// `manifold/se3.hpp`).
///
/// Implementation: this is `ConstantSPDMetric<6>` with `A = diag(w)`. The
/// `weights_` field is kept alongside the base metric so that
/// `SE3::has_riemannian_log_runtime()` can cheaply detect unit weights without
/// inspecting the full SPD matrix.
///
/// Because the metric is left-invariant it does not depend on the base point;
/// the acted-on quantity is the 6-vector twist, so both the point and tangent
/// types are `Eigen::Matrix<double,6,1>` (the "p" argument is unused/constant).
class SE3InvariantMetric {
 public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;  ///< Twist / metric-argument type.

  /// @brief Construct the isotropic identity metric (unit weights).
  SE3InvariantMetric() : SE3InvariantMetric(Vector6d::Ones()) {}

  /// @brief Construct with separate translational and rotational weights.
  /// @param w_trans Weight applied to each of the 3 translational twist components.
  /// @param w_rot Weight applied to each of the 3 rotational twist components.
  SE3InvariantMetric(double w_trans, double w_rot)
      : SE3InvariantMetric(
            (Vector6d() << w_trans, w_trans, w_trans, w_rot, w_rot, w_rot).finished()) {}

  /// @brief Construct with an explicit diagonal weight vector on the twist.
  /// @param diag Six positive weights ordered translation-first, rotation-last.
  explicit SE3InvariantMetric(const Vector6d& diag)
      : weights_(diag), base_(diag.asDiagonal().toDenseMatrix()) {}

  /// @brief Access the diagonal weight vector \f$ (w_{v_x}, \dots, w_{\omega_z}) \f$.
  const Vector6d& weights() const { return weights_; }

  /// @brief Compute the inner product via the wrapped ConstantSPDMetric.
  /// @param p Base twist (unused for a constant metric).
  /// @param u First tangent (twist) vector.
  /// @param v Second tangent (twist) vector.
  /// @return The inner product value.
  double inner(const Vector6d& p, const Vector6d& u, const Vector6d& v) const {
    return base_.inner(p, u, v);
  }

  /// @brief Batched inner product \f$ U^\top A\, V \f$ via the wrapped ConstantSPDMetric.
  /// @param p Base twist (unused).
  /// @param U Matrix whose columns are tangent (twist) vectors.
  /// @param V Matrix whose columns are tangent (twist) vectors.
  /// @return \f$ U^\top A\, V \f$.
  Eigen::MatrixXd inner_matrix(const Vector6d& p, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const {
    return base_.inner_matrix(p, U, V);
  }

  /// @brief Compute the norm \f$ \|v\| = \sqrt{\langle v, v \rangle} \f$.
  /// @param p Base twist.
  /// @param v Tangent (twist) vector.
  /// @return The norm value.
  double norm(const Vector6d& p, const Vector6d& v) const { return riemannian_norm(*this, p, v); }

 private:
  Vector6d weights_;           ///< Diagonal weight vector on the twist \f$ [v;\,\omega] \f$.
  ConstantSPDMetric<6> base_;  ///< Wrapped SPD metric with `A = diag(weights_)`.
};

}  // namespace geodex

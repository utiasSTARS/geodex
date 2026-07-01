/// @file so3_canonical.hpp
/// @brief Canonical metric on SO(3) — thin wrapper over ConstantSPDMetric<3>.

#pragma once

#include <Eigen/Core>

#include "geodex/core/metric.hpp"
#include "geodex/metrics/constant_spd.hpp"

namespace geodex {

/// @brief Canonical left-invariant metric on SO(3).
///
/// @details The inner product is evaluated on body angular velocities
/// \f$ \omega \in \mathfrak{so}(3) \cong \mathbb{R}^3 \f$ and is constant
/// (left-invariant):
/// \f$ \langle u, v \rangle = w_x u_x v_x + w_y u_y v_y + w_z u_z v_z \f$.
///
/// The default unit weights \f$ (1, 1, 1) \f$ give the round, **bi-invariant**
/// metric on SO(3), for which the Lie-group `log` is the Riemannian logarithm
/// and geodesics are constant-speed rotations (quaternion SLERP). Any isotropic
/// scaling \f$ w \cdot I \f$ (all three weights equal) is likewise bi-invariant:
/// it rescales lengths uniformly without changing the geodesics. Anisotropic
/// weights break bi-invariance and yield a genuinely left-invariant (not
/// bi-invariant) metric.
///
/// Implementation: this is `ConstantSPDMetric<3>` with
/// \f$ A = \mathrm{diag}(w_x, w_y, w_z) \f$. The `weights_` field is preserved
/// alongside the base metric so that `SO3::has_riemannian_log_runtime()` can
/// quickly test isotropy without inspecting the wrapped SPD matrix. Points and
/// tangents are 3-vectors, mirroring `SE2LeftInvariantMetric`.
class SO3CanonicalMetric {
 public:
  using Point = Eigen::Vector3d;    ///< Body angular velocity base (unused; constant metric).
  using Tangent = Eigen::Vector3d;  ///< Body angular velocity \f$ \omega \f$.

  /// @brief Construct with unit weights (round bi-invariant metric).
  SO3CanonicalMetric() : SO3CanonicalMetric(Eigen::Vector3d(1.0, 1.0, 1.0)) {}

  /// @brief Construct an isotropic metric \f$ w \cdot I \f$ (bi-invariant).
  /// @param w Positive rotational weight; the norm scales as \f$ \sqrt{w} \f$.
  explicit SO3CanonicalMetric(double w) : SO3CanonicalMetric(Eigen::Vector3d(w, w, w)) {}

  /// @brief Construct an anisotropic diagonal metric
  /// \f$ \mathrm{diag}(w_x, w_y, w_z) \f$ (left-invariant, not bi-invariant).
  /// @param diag Positive diagonal weights \f$ (w_x, w_y, w_z) \f$.
  explicit SO3CanonicalMetric(const Eigen::Vector3d& diag)
      : weights_(diag), base_(diag.asDiagonal().toDenseMatrix()) {}

  /// @brief Access the diagonal weight vector \f$ (w_x, w_y, w_z) \f$.
  const Eigen::Vector3d& weights() const { return weights_; }

  /// @brief Compute the inner product via the wrapped ConstantSPDMetric.
  /// @param p Base point (unused for a constant metric).
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  double inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return base_.inner(p, u, v);
  }

  /// @brief Batched inner product via the wrapped ConstantSPDMetric.
  /// @param p Base point.
  /// @param U Matrix whose columns are tangent vectors.
  /// @param V Matrix whose columns are tangent vectors.
  /// @return \f$ U^\top A \, V \f$.
  Eigen::MatrixXd inner_matrix(const Point& p, const Eigen::MatrixXd& U,
                               const Eigen::MatrixXd& V) const {
    return base_.inner_matrix(p, U, V);
  }

  /// @brief Compute the norm \f$ \|v\| = \sqrt{\langle v, v \rangle} \f$.
  /// @param p Base point.
  /// @param v Tangent vector.
  /// @return The norm value.
  double norm(const Point& p, const Tangent& v) const { return riemannian_norm(*this, p, v); }

 private:
  Eigen::Vector3d weights_;    ///< Diagonal weight vector \f$ (w_x, w_y, w_z) \f$.
  ConstantSPDMetric<3> base_;  ///< Wrapped SPD metric with `A = diag(weights_)`.
};

}  // namespace geodex

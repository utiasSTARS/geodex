/// @file integration/pinocchio/pullback.hpp
/// @brief URDF-driven pullback metric factory built on Pinocchio primitives.
///
/// @details `make_pullback_metric` composes an end-effector pullback metric
/// \f$ J(q)^\top W J(q) \f$ from a URDF, delegating Jacobian evaluation to
/// `integration::pinocchio` primitives and reusing master's `PullbackMetric`
/// and `AffineCombinedMetric` policies.
///
/// Regularization is opt-in via a tag parameter:
/// - no tag: returns an unregularized `PullbackMetric`.
/// - `IsotropicRegularization{lambda}`: returns
///   \f$ \text{AffineCombinedMetric}(\{1, \lambda\}, \text{pullback}, I) \f$.
/// - `KineticEnergyRegularization{beta}`: returns
///   \f$ \text{AffineCombinedMetric}(\{1, \beta\}, \text{pullback}, M_{\text{CRBA}}) \f$.
///
/// Each overload returns a distinct concrete type; users deduce via `auto`.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "geodex/metrics/affine_combined.hpp"
#include "geodex/metrics/identity.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/metrics/pullback.hpp"

#include "geodex/integration/pinocchio/jacobian.hpp"
#include "geodex/integration/pinocchio/mass_matrix.hpp"

namespace geodex::integration::pinocchio {

/// @brief End-effector selection and task-space shaping for
///        `make_pullback_metric`.
struct PullbackOptions {
  /// End-effector frame names, in stacking order. An empty vector — or a
  /// single-element vector holding an empty string — triggers auto-detect of
  /// the last BODY frame attached to the final movable joint (the common
  /// single-arm case).
  std::vector<std::string> ee_frames{};

  /// Per-axis task-space weights in `LOCAL_WORLD_ALIGNED` coordinates,
  /// ordered `(x, y, z, r, p, y)`. The per-EE task metric is
  /// \f$ \text{diag}(w_x, w_y, w_z, w_r, w_p, w_y) \f$ and the same weights
  /// are applied to every end-effector when `ee_frames.size() > 1`. Zero an
  /// entry to drop that axis from the pullback (e.g. `{1, 1, 1, 0, 0, 0}` for
  /// a position-only pullback).
  std::array<double, 6> task_weights = {1.0, 1.0, 1.0, 0.1, 0.1, 0.1};
};

/// @brief Tag requesting isotropic regularization: adds \f$\lambda I\f$ to the
///        pullback metric.
struct IsotropicRegularization {
  double lambda;
};

/// @brief Tag requesting kinetic-energy regularization: adds
///        \f$\beta M_{\text{CRBA}}(q)\f$ to the pullback metric.
struct KineticEnergyRegularization {
  double beta;
};

namespace detail {

/// @brief Point-independent task-metric callable used by the pullback factory.
///
/// @details Captures a precomputed SPD weight matrix \f$W\f$ and returns it by
/// const reference for every configuration \f$q\f$ — matches the callable
/// contract `q \mapsto W(q)` expected by `PullbackMetric::task_metric_fn_`.
class ConstantTaskMetric {
 public:
  explicit ConstantTaskMetric(Eigen::MatrixXd W) : W_(std::move(W)) {}

  template <typename Q>
  auto operator()(const Q& /*q*/) const -> const Eigen::MatrixXd& {
    return W_;
  }

 private:
  Eigen::MatrixXd W_;
};

inline auto build_jacobian(const std::string& urdf_path, const PullbackOptions& opts)
    -> FrameJacobianImpl {
  return FrameJacobianImpl(urdf_path, opts.ee_frames, /*position_only=*/false);
}

inline auto task_dim(const FrameJacobianImpl& jac) -> int {
  return 6 * static_cast<int>(jac.frame_ids().size());
}

inline auto build_task_weight(int task_dim, const std::array<double, 6>& per_axis)
    -> Eigen::MatrixXd {
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(task_dim, task_dim);
  const int k = task_dim / 6;
  for (int f = 0; f < k; ++f) {
    const int base = f * 6;
    for (int a = 0; a < 6; ++a) {
      W(base + a, base + a) = per_axis[a];
    }
  }
  return W;
}

}  // namespace detail

/// @brief Build an unregularized pullback metric from a URDF.
/// @return `PullbackMetric` whose Jacobian is a Pinocchio frame-Jacobian
///         callable and whose task metric is a constant SPD matrix shaped
///         by `PullbackOptions::task_weights`.
inline auto make_pullback_metric(const std::string& urdf_path,
                                 const PullbackOptions& opts) {
  auto jac = detail::build_jacobian(urdf_path, opts);
  const int task_d = detail::task_dim(jac);
  detail::ConstantTaskMetric task_metric{detail::build_task_weight(task_d, opts.task_weights)};
  return PullbackMetric(std::move(jac), std::move(task_metric));
}

/// @brief Build a pullback metric with isotropic regularization
///        \f$ + \lambda I \f$.
inline auto make_pullback_metric(const std::string& urdf_path,
                                 const PullbackOptions& opts,
                                 IsotropicRegularization reg) {
  if (!(reg.lambda > 0.0)) {
    throw std::invalid_argument(
        "IsotropicRegularization requires lambda > 0; use the unregularized overload instead.");
  }
  auto base = make_pullback_metric(urdf_path, opts);
  IdentityMetric<Eigen::Dynamic> ident{};
  return AffineCombinedMetric(std::array<double, 2>{1.0, reg.lambda}, std::move(base),
                              std::move(ident));
}

/// @brief Build a pullback metric with kinetic-energy regularization
///        \f$ + \beta M_{\text{CRBA}}(q) \f$.
inline auto make_pullback_metric(const std::string& urdf_path,
                                 const PullbackOptions& opts,
                                 KineticEnergyRegularization reg) {
  if (!(reg.beta > 0.0)) {
    throw std::invalid_argument(
        "KineticEnergyRegularization requires beta > 0; use the unregularized overload instead.");
  }
  auto base = make_pullback_metric(urdf_path, opts);
  KineticEnergyMetric ke{MassMatrix(urdf_path)};
  return AffineCombinedMetric(std::array<double, 2>{1.0, reg.beta}, std::move(base),
                              std::move(ke));
}

}  // namespace geodex::integration::pinocchio

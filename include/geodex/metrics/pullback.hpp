/// @file pullback.hpp
/// @brief Pullback metric from task space to configuration space.

#pragma once

#include <Eigen/Core>
#include <cmath>

namespace geodex {

/// @brief Pullback metric that maps a task-space metric to configuration space
/// via the Jacobian.
///
/// @details The inner product at configuration \f$ q \f$ is:
/// \f$ \langle u, v \rangle_q = u^\top J(q)^\top G_X(q) \, J(q) \, v + \lambda \, u^\top v \f$
/// where \f$ J(q) \f$ is the task-space Jacobian, \f$ G_X(q) \f$ is the task-space
/// metric (SPD), and \f$ \lambda \f$ is an optional regularization parameter ensuring
/// positive-definiteness even at singularities.
///
/// @tparam JacobianFn Callable returning the Jacobian matrix at \f$ q \f$.
/// @tparam TaskMetricFn Callable returning the task-space SPD metric at \f$ q \f$.
template <typename JacobianFn, typename TaskMetricFn>
struct PullbackMetric {
  JacobianFn jacobian_fn_;       ///< Callable returning J(q).
  TaskMetricFn task_metric_fn_;  ///< Callable returning G_X(q).
  double lambda_;                ///< Regularization parameter.

  /// @brief Construct a pullback metric.
  /// @param jac_fn Callable returning the Jacobian.
  /// @param task_fn Callable returning the task-space metric.
  /// @param lambda Regularization (default 0).
  PullbackMetric(JacobianFn jac_fn, TaskMetricFn task_fn, double lambda = 0.0)
      : jacobian_fn_(std::move(jac_fn)),
        task_metric_fn_(std::move(task_fn)),
        lambda_(lambda) {}

  /// @brief Compute the inner product.
  /// @param q Configuration point.
  /// @param u First tangent vector.
  /// @param v Second tangent vector.
  /// @return The inner product value.
  template <typename Point, typename Tangent>
  double inner(const Point& q, const Tangent& u, const Tangent& v) const {
    auto J = jacobian_fn_(q);
    auto G = task_metric_fn_(q);
    double val = u.dot(J.transpose() * G * J * v);
    if (lambda_ > 0.0) {
      val += lambda_ * u.dot(v);
    }
    return val;
  }

  /// @brief Compute the norm.
  /// @param q Configuration point.
  /// @param v Tangent vector.
  /// @return The norm value.
  template <typename Point, typename Tangent>
  double norm(const Point& q, const Tangent& v) const {
    return std::sqrt(inner(q, v, v));
  }

  /// @brief Batched inner product: \f$U^\top (J^\top G J + \lambda I)\, V\f$ with
  /// a single pair of Jacobian / task-metric evaluations.
  template <typename Point>
  Eigen::MatrixXd inner_matrix(const Point& q, const Eigen::MatrixXd& U,
                                const Eigen::MatrixXd& V) const {
    auto J = jacobian_fn_(q);
    auto G = task_metric_fn_(q);
    Eigen::MatrixXd result = (J * U).transpose() * G * (J * V);
    if (lambda_ > 0.0) {
      result.noalias() += lambda_ * (U.transpose() * V);
    }
    return result;
  }
};

/// @brief Task-space metric that always returns the identity matrix.
///
/// @details Used as the default task metric for `make_pullback_euclidean_metric`.
struct IdentityTaskMetric {
  int task_dim_;  ///< Dimension of the task space.

  /// @brief Construct with the task-space dimension.
  /// @param task_dim Dimension of the task space.
  explicit IdentityTaskMetric(int task_dim) : task_dim_(task_dim) {}

  /// @brief Return the identity matrix.
  template <typename Point>
  Eigen::MatrixXd operator()(const Point& /*q*/) const {
    return Eigen::MatrixXd::Identity(task_dim_, task_dim_);
  }
};

/// @brief Create a pullback metric with Euclidean task-space metric (\f$ G_X = I \f$).
///
/// @details Computes \f$ u^\top J^\top J \, v + \lambda \, u^\top v \f$.
///
/// @tparam JacobianFn Callable returning the Jacobian matrix at \f$ q \f$.
/// @param jac_fn Callable returning the Jacobian.
/// @param task_dim Dimension of the task space.
/// @param lambda Regularization parameter (default 0).
/// @return A PullbackMetric with identity task-space metric.
template <typename JacobianFn>
auto make_pullback_euclidean_metric(JacobianFn jac_fn, int task_dim, double lambda = 0.0) {
  return PullbackMetric<JacobianFn, IdentityTaskMetric>{
      std::move(jac_fn), IdentityTaskMetric{task_dim}, lambda};
}

}  // namespace geodex

/// @file py_metrics.hpp
/// @brief Python wrappers for geodex metric types using std::function instantiations.

#pragma once

#include <cmath>

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>

#include "geodex/metrics/clearance.hpp"
#include "geodex/metrics/constant_spd.hpp"
#include "geodex/metrics/jacobi.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/metrics/pullback.hpp"
#include "geodex/metrics/se2_left_invariant.hpp"
#include "geodex/metrics/weighted.hpp"
#include "geodex/utils/ordered_sum.hpp"

#include "dynamic_manifold.hpp"
#include "sizes.hpp"

namespace geodex::python {

// Type aliases for std::function-based metric instantiations.
using MassMatrixFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
using PotentialFn = std::function<double(const Eigen::VectorXd&)>;
using JacobianFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
using TaskMetricFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

// --- KineticEnergyMetric ---

/// Kinetic energy metric u^T M(q) v of a mass matrix callable. Every call checks the sizes
/// of the vectors and of the returned matrix and raises ValueError on a mismatch.
class PyKineticEnergyMetric {
 public:
  explicit PyKineticEnergyMetric(MassMatrixFn fn) : mass_fn_(std::move(fn)) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    require_size(v, u.size(), "KineticEnergyMetric.inner", "v");
    const Eigen::MatrixXd M = mass_fn_(p);
    require_shape(M, u.size(), u.size(), "KineticEnergyMetric.inner", "the mass matrix");
    return utils::ordered_quadratic_form(u, M, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::sqrt(inner(p, v, v));
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<const PyKineticEnergyMetric>(*this);
    return DynamicMetric{[shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
                         [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
                           return shared->norm(p, v);
                         }};
  }

  std::string repr() const { return "KineticEnergyMetric()"; }

 private:
  MassMatrixFn mass_fn_;
};

// --- JacobiMetric ---

/// Jacobi metric 2 (H - P(q)) u^T M(q) v, the value of the C++ `JacobiMetric`,
/// with the sizes checked as in `PyKineticEnergyMetric`.
class PyJacobiMetric {
 public:
  PyJacobiMetric(MassMatrixFn mass_fn, PotentialFn pot_fn, double H)
      : mass_fn_(std::move(mass_fn)), potential_fn_(std::move(pot_fn)), total_energy_(H) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    require_size(v, u.size(), "JacobiMetric.inner", "v");
    const Eigen::MatrixXd M = mass_fn_(p);
    require_shape(M, u.size(), u.size(), "JacobiMetric.inner", "the mass matrix");
    return 2.0 * (total_energy_ - potential_fn_(p)) * utils::ordered_quadratic_form(u, M, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::sqrt(inner(p, v, v));
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<const PyJacobiMetric>(*this);
    return DynamicMetric{[shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
                         [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
                           return shared->norm(p, v);
                         }};
  }

  std::string repr() const { return "JacobiMetric(H=" + std::to_string(total_energy_) + ")"; }

 private:
  MassMatrixFn mass_fn_;
  PotentialFn potential_fn_;
  double total_energy_;
};

// --- PullbackMetric ---

/// Pullback metric u^T J^T G J v + lambda u^T v, the value of the C++
/// `PullbackMetric`, with the sizes of J and G checked on every call.
class PyPullbackMetric {
 public:
  PyPullbackMetric(JacobianFn jac_fn, TaskMetricFn task_fn, double lambda = 0.0)
      : jacobian_fn_(std::move(jac_fn)), task_metric_fn_(std::move(task_fn)), lambda_(lambda) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    require_size(v, u.size(), "PullbackMetric.inner", "v");
    const Eigen::MatrixXd J = jacobian_fn_(p);
    require_shape(J, J.rows(), u.size(), "PullbackMetric.inner", "the Jacobian");
    const Eigen::MatrixXd G = task_metric_fn_(p);
    require_shape(G, J.rows(), J.rows(), "PullbackMetric.inner", "the task metric");
    double val = u.dot(J.transpose() * G * J * v);
    if (lambda_ > 0.0) val += lambda_ * u.dot(v);
    return val;
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::sqrt(inner(p, v, v));
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<const PyPullbackMetric>(*this);
    return DynamicMetric{[shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
                         [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
                           return shared->norm(p, v);
                         }};
  }

  std::string repr() const { return "PullbackMetric(lambda=" + std::to_string(lambda_) + ")"; }

 private:
  JacobianFn jacobian_fn_;
  TaskMetricFn task_metric_fn_;
  double lambda_;
};

// --- ConstantSPDMetric ---

class PyConstantSPDMetric {
 public:
  using Impl = ConstantSPDMetric<Eigen::Dynamic>;

  explicit PyConstantSPDMetric(const Eigen::MatrixXd& A) : impl_(checked_square(A)) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    const Eigen::Index n = impl_.weight_matrix().rows();
    require_size(u, n, "ConstantSPDMetric.inner", "u");
    require_size(v, n, "ConstantSPDMetric.inner", "v");
    return impl_.inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(v, impl_.weight_matrix().rows(), "ConstantSPDMetric.norm", "v");
    return impl_.norm(p, v);
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<const PyConstantSPDMetric>(*this);
    return DynamicMetric{[shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
                         [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
                           return shared->norm(p, v);
                         }};
  }

  std::string repr() const {
    return "ConstantSPDMetric(dim=" + std::to_string(impl_.weight_matrix().rows()) + ")";
  }

 private:
  static const Eigen::MatrixXd& checked_square(const Eigen::MatrixXd& A) {
    require_shape(A, A.rows(), A.rows(), "ConstantSPDMetric", "A");
    return A;
  }

  Impl impl_;
};

// --- SE2LeftInvariantMetric ---

/// Left-invariant SE(2) metric with the constant diagonal inner product
/// diag(wx, wy, wtheta) on the (x, y, theta) tangent. It can serve as the base of a
/// ClearanceMetric, like the C++ composition
/// `SDFConformalMetric{SE2LeftInvariantMetric{...}, sdf}`.
class PySE2LeftInvariantMetric {
 public:
  using Impl = SE2LeftInvariantMetric;

  PySE2LeftInvariantMetric(double wx, double wy, double wtheta) : impl_(wx, wy, wtheta) {}

  static PySE2LeftInvariantMetric car_like(double turning_radius, double lateral_penalty) {
    const auto m = SE2LeftInvariantMetric::car_like(turning_radius, lateral_penalty);
    return PySE2LeftInvariantMetric(m.weights()[0], m.weights()[1], m.weights()[2]);
  }

  static PySE2LeftInvariantMetric holonomic(const double wtheta) {
    const auto m = SE2LeftInvariantMetric::holonomic(wtheta);
    return PySE2LeftInvariantMetric(m.weights()[0], m.weights()[1], m.weights()[2]);
  }

  static PySE2LeftInvariantMetric differential_drive(const double lateral_weight,
                                                     const double wtheta) {
    const auto m = SE2LeftInvariantMetric::differential_drive(lateral_weight, wtheta);
    return PySE2LeftInvariantMetric(m.weights()[0], m.weights()[1], m.weights()[2]);
  }

  Eigen::Matrix3d coordinate_lower_bound() const { return impl_.coordinate_lower_bound(); }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    require_size(p, 3, "SE2LeftInvariantMetric.inner", "p");
    require_size(u, 3, "SE2LeftInvariantMetric.inner", "u");
    require_size(v, 3, "SE2LeftInvariantMetric.inner", "v");
    return impl_.inner(Eigen::Vector3d(p), Eigen::Vector3d(u), Eigen::Vector3d(v));
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, 3, "SE2LeftInvariantMetric.norm", "p");
    require_size(v, 3, "SE2LeftInvariantMetric.norm", "v");
    return impl_.norm(Eigen::Vector3d(p), Eigen::Vector3d(v));
  }

  Eigen::Vector3d weights() const { return impl_.weights(); }

  /// @brief The wrapped C++ metric.
  const Impl& impl() const { return impl_; }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<const PySE2LeftInvariantMetric>(*this);
    return DynamicMetric{[shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
                         [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
                           return shared->norm(p, v);
                         }};
  }

  std::string repr() const {
    const auto w = impl_.weights();
    return "SE2LeftInvariantMetric(wx=" + std::to_string(w[0]) + ", wy=" + std::to_string(w[1]) +
           ", wtheta=" + std::to_string(w[2]) + ")";
  }

 private:
  Impl impl_;
};

// --- WeightedMetric ---

/// WeightedMetric wraps a type-erased DynamicMetric and uniformly scales it.
class PyWeightedMetric {
 public:
  PyWeightedMetric(DynamicMetric base, double alpha) : base_(std::move(base)), alpha_(alpha) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    return alpha_ * base_.inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::sqrt(inner(p, v, v));
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared_base = std::make_shared<DynamicMetric>(base_);
    double a = alpha_;
    return DynamicMetric{
        [shared_base, a](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                         const Eigen::VectorXd& v) { return a * shared_base->inner(p, u, v); },
        [shared_base, a](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
          return std::sqrt(a * shared_base->inner(p, v, v));
        }};
  }

  double alpha() const { return alpha_; }

  std::string repr() const { return "WeightedMetric(alpha=" + std::to_string(alpha_) + ")"; }

 private:
  DynamicMetric base_;
  double alpha_;
};

// --- AffineCombinedMetric (dynamic-arity) ---

/// Dynamic-arity positive linear combination of metric policies. It has the semantics of
/// the variadic C++ `geodex::AffineCombinedMetric<Ms...>` and dispatches at runtime over a
/// vector of type-erased `DynamicMetric` summands. Python cannot express the variadic form
/// without a binding per arity.
class PyAffineCombinedMetric {
 public:
  PyAffineCombinedMetric(std::vector<DynamicMetric> bases, std::vector<double> coeffs)
      : bases_(std::move(bases)), coeffs_(std::move(coeffs)) {
    if (bases_.size() != coeffs_.size()) {
      throw std::invalid_argument(
          "AffineCombinedMetric: metrics and coeffs must have matching length.");
    }
    if (bases_.empty()) {
      throw std::invalid_argument("AffineCombinedMetric: requires at least one summand.");
    }
    bool any_positive = false;
    for (const double c : coeffs_) {
      if (c < 0.0) {
        throw std::invalid_argument(
            "AffineCombinedMetric: coefficients must be non-negative.");
      }
      if (c > 0.0) any_positive = true;
    }
    if (!any_positive) {
      throw std::invalid_argument(
          "AffineCombinedMetric: at least one coefficient must be > 0.");
    }
  }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    double sum = 0.0;
    for (std::size_t k = 0; k < bases_.size(); ++k) {
      sum += coeffs_[k] * bases_[k].inner(p, u, v);
    }
    return sum;
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::sqrt(inner(p, v, v));
  }

  std::size_t size() const { return bases_.size(); }
  const std::vector<double>& coeffs() const { return coeffs_; }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<PyAffineCombinedMetric>(*this);
    return DynamicMetric{[shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
                         [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
                           return shared->norm(p, v);
                         }};
  }

  std::string repr() const {
    return "AffineCombinedMetric(arity=" + std::to_string(bases_.size()) + ")";
  }

 private:
  std::vector<DynamicMetric> bases_;
  std::vector<double> coeffs_;
};

// --- ClearanceMetric ---

using SDFFn = std::function<double(const Eigen::VectorXd&)>;

/// ClearanceMetric wraps SDFConformalMetric with a DynamicMetric base and a callable SDF.
class PyClearanceMetric {
 public:
  using Impl = SDFConformalMetric<DynamicMetric, SDFFn>;

  /// @param se2_base The base metric when it is an SE2LeftInvariantMetric. A
  ///        ConfigurationSpace over SE2 then plans with the typed metric.
  PyClearanceMetric(DynamicMetric base, SDFFn sdf, const double kappa = 5.0,
                    const double beta = 3.0,
                    std::optional<SE2LeftInvariantMetric> se2_base = std::nullopt)
      : impl_(std::move(base), std::move(sdf), kappa, beta), se2_base_(std::move(se2_base)) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    return impl_.inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const { return impl_.norm(p, v); }

  double kappa() const { return impl_.kappa(); }
  double beta() const { return impl_.beta(); }

  /// @brief The base metric when it is an SE2LeftInvariantMetric.
  const std::optional<SE2LeftInvariantMetric>& se2_base() const { return se2_base_; }

  /// @brief The wrapped C++ metric.
  const Impl& impl() const { return impl_; }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<Impl>(impl_);
    return DynamicMetric{[shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                  const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
                         [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
                           return shared->norm(p, v);
                         }};
  }

  std::string repr() const {
    return "ClearanceMetric(kappa=" + std::to_string(impl_.kappa()) +
           ", beta=" + std::to_string(impl_.beta()) + ")";
  }

 private:
  Impl impl_;
  std::optional<SE2LeftInvariantMetric> se2_base_;
};

}  // namespace geodex::python

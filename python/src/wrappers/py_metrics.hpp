/// @file py_metrics.hpp
/// @brief Python wrappers for geodex metric types using std::function instantiations.

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <functional>
#include <geodex/metrics/constant_spd.hpp>
#include <geodex/metrics/jacobi.hpp>
#include <geodex/metrics/kinetic_energy.hpp>
#include <geodex/metrics/pullback.hpp>
#include <geodex/metrics/weighted.hpp>
#include <memory>
#include <string>

#include "dynamic_manifold.hpp"

namespace geodex::python {

// Type aliases for std::function-based metric instantiations.
using MassMatrixFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
using PotentialFn = std::function<double(const Eigen::VectorXd&)>;
using JacobianFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;
using TaskMetricFn = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

// --- KineticEnergyMetric ---

class PyKineticEnergyMetric {
 public:
  using Impl = KineticEnergyMetric<MassMatrixFn>;

  explicit PyKineticEnergyMetric(MassMatrixFn fn) : impl_(std::move(fn)) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u,
               const Eigen::VectorXd& v) const {
    return impl_.inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return impl_.norm(p, v);
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<Impl>(impl_);
    return DynamicMetric{
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
          return shared->norm(p, v);
        }};
  }

  std::string repr() const { return "KineticEnergyMetric()"; }

 private:
  Impl impl_;
};

// --- JacobiMetric ---

class PyJacobiMetric {
 public:
  using Impl = JacobiMetric<MassMatrixFn, PotentialFn>;

  PyJacobiMetric(MassMatrixFn mass_fn, PotentialFn pot_fn, double H)
      : impl_(std::move(mass_fn), std::move(pot_fn), H) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u,
               const Eigen::VectorXd& v) const {
    return impl_.inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return impl_.norm(p, v);
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<Impl>(impl_);
    return DynamicMetric{
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
          return shared->norm(p, v);
        }};
  }

  std::string repr() const { return "JacobiMetric(H=" + std::to_string(impl_.total_energy()) + ")"; }

 private:
  Impl impl_;
};

// --- PullbackMetric ---

class PyPullbackMetric {
 public:
  using Impl = PullbackMetric<JacobianFn, TaskMetricFn>;

  PyPullbackMetric(JacobianFn jac_fn, TaskMetricFn task_fn, double lambda = 0.0)
      : impl_(std::move(jac_fn), std::move(task_fn), lambda) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u,
               const Eigen::VectorXd& v) const {
    return impl_.inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return impl_.norm(p, v);
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<Impl>(impl_);
    return DynamicMetric{
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
          return shared->norm(p, v);
        }};
  }

  std::string repr() const {
    return "PullbackMetric(lambda=" + std::to_string(impl_.lambda()) + ")";
  }

 private:
  Impl impl_;
};

// --- ConstantSPDMetric ---

class PyConstantSPDMetric {
 public:
  using Impl = ConstantSPDMetric<Eigen::Dynamic>;

  explicit PyConstantSPDMetric(const Eigen::MatrixXd& A) : impl_(A) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u,
               const Eigen::VectorXd& v) const {
    return impl_.inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return impl_.norm(p, v);
  }

  DynamicMetric to_dynamic_metric() const {
    auto shared = std::make_shared<Impl>(impl_);
    return DynamicMetric{
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) { return shared->inner(p, u, v); },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
          return shared->norm(p, v);
        }};
  }

  std::string repr() const {
    return "ConstantSPDMetric(dim=" + std::to_string(impl_.weight_matrix().rows()) + ")";
  }

 private:
  Impl impl_;
};

// --- WeightedMetric ---

/// WeightedMetric wraps a type-erased DynamicMetric and uniformly scales it.
class PyWeightedMetric {
 public:
  PyWeightedMetric(DynamicMetric base, double alpha)
      : base_(std::move(base)), alpha_(alpha) {}

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u,
               const Eigen::VectorXd& v) const {
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
                         const Eigen::VectorXd& v) {
          return a * shared_base->inner(p, u, v);
        },
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

}  // namespace geodex::python

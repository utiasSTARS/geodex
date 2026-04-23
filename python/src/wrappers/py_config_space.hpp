/// @file py_config_space.hpp
/// @brief Python wrapper for ConfigurationSpace combining base topology + custom metric.

#pragma once

#include <string>

#include "dynamic_manifold.hpp"

namespace geodex::python {

/// @brief Python wrapper for ConfigurationSpace combining base topology + custom metric.
///
/// Topology operations (exp, log, dim, random_point) come from the base manifold.
/// Geometry operations (inner, norm) come from the custom metric.
/// Distance and geodesic are derived from the composed operations.
class PyConfigurationSpace {
 public:
  PyConfigurationSpace(DynamicManifold dm, DynamicMetric dmet, std::string base_name,
                       std::string metric_name)
      : base_name_(std::move(base_name)), metric_name_(std::move(metric_name)) {
    // Compose: topology from base, geometry from metric.
    impl_ = DynamicManifold{
        [dm]() { return dm.dim(); },
        [dm]() { return dm.random_point(); },
        [dm](const Eigen::VectorXd& p, const Eigen::VectorXd& v) { return dm.exp(p, v); },
        [dm](const Eigen::VectorXd& p, const Eigen::VectorXd& q) { return dm.log(p, q); },
        [dmet](const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) {
          return dmet.inner(p, u, v);
        },
        [dmet](const Eigen::VectorXd& p, const Eigen::VectorXd& v) { return dmet.norm(p, v); },
        dm.has_project()
            ? DynamicManifold::ProjectFn{[dm](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
                return dm.project(p, v);
              }}
            : nullptr};
  }

  int dim() const { return impl_.dim(); }
  Eigen::VectorXd random_point() const { return impl_.random_point(); }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    return impl_.inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const { return impl_.norm(p, v); }

  Eigen::VectorXd exp(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return impl_.exp(p, v);
  }

  Eigen::VectorXd log(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    return impl_.log(p, q);
  }

  double distance(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    return impl_.distance(p, q);
  }

  Eigen::VectorXd geodesic(const Eigen::VectorXd& p, const Eigen::VectorXd& q, double t) const {
    return impl_.geodesic(p, q, t);
  }

  DynamicManifold to_dynamic_manifold() const { return impl_; }

  std::string repr() const {
    return "ConfigurationSpace(base=" + base_name_ + ", metric=" + metric_name_ + ")";
  }

 private:
  DynamicManifold impl_;
  std::string base_name_;
  std::string metric_name_;
};

}  // namespace geodex::python

/// @file py_config_space.hpp
/// @brief Python wrapper for ConfigurationSpace combining a base topology with a custom metric.

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "dynamic_manifold.hpp"
#include "py_se2.hpp"

namespace geodex::python {

class PyOwnedRef;

/// @brief Python wrapper for ConfigurationSpace combining a base topology with a custom metric.
///
/// Topology operations (exp, log, dim, random_point) come from the base manifold.
/// Geometry operations (inner, norm) come from the custom metric.
/// Distance and geodesic come from the composed operations.
class PyConfigurationSpace {
 public:
  /// @brief Compose a base manifold with a metric.
  /// @param dm The base manifold, supplying topology and sampling.
  /// @param dmet The metric this space evaluates with.
  /// @param base_name Name of the base for repr.
  /// @param metric_name Name of the metric for repr.
  /// @param metric_for_copies Builds the metric of every manifold that
  ///        to_dynamic_manifold() hands out. When empty, those manifolds share `dmet`.
  PyConfigurationSpace(DynamicManifold dm, DynamicMetric dmet, std::string base_name,
                       std::string metric_name,
                       std::function<DynamicMetric()> metric_for_copies = nullptr)
      : base_(std::move(dm)),
        metric_for_copies_(std::move(metric_for_copies)),
        impl_(compose(base_, dmet)),
        base_name_(std::move(base_name)),
        metric_name_(std::move(metric_name)) {}

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

  DynamicManifold to_dynamic_manifold() const {
    return metric_for_copies_ ? compose(base_, metric_for_copies_()) : impl_;
  }

  /// @brief Record the Python objects the space was built from. plan() reads them to plan
  /// on a typed copy of the space.
  /// @param se2 The base when it is an SE2.
  /// @param metric The reference that the space holds to its metric object.
  void set_sources(std::optional<PySE2> se2, std::shared_ptr<const PyOwnedRef> metric) {
    se2_ = std::move(se2);
    metric_ref_ = std::move(metric);
  }

  /// @brief The base when it is an SE2.
  const std::optional<PySE2>& se2_base() const { return se2_; }

  /// @brief The reference to the metric object, or null when none was recorded.
  const std::shared_ptr<const PyOwnedRef>& metric_ref() const { return metric_ref_; }

  std::string repr() const {
    return "ConfigurationSpace(base=" + base_name_ + ", metric=" + metric_name_ + ")";
  }

 private:
  /// @brief Topology from the base and geometry from the metric. It forwards the base's
  /// sampling interface (unit_cube_dim, from_unit_cube, sampler) and any explicit bounds.
  /// plan() derives its search domain and samplers from them.
  static DynamicManifold compose(const DynamicManifold& dm, const DynamicMetric& dmet) {
    DynamicManifold out{
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
            : nullptr,
        dm.has_from_unit_cube()
            ? DynamicManifold::UnitCubeDimFn{[dm]() { return dm.unit_cube_dim(); }}
            : nullptr,
        dm.has_from_unit_cube()
            ? DynamicManifold::FromUnitCubeFn{[dm](Eigen::Ref<const Eigen::VectorXd> u) {
                return dm.from_unit_cube(u);
              }}
            : nullptr};
    if (dm.has_bounds()) {
      auto [lo, hi] = dm.bounds();
      out.set_bounds(std::move(lo), std::move(hi));
    }
    out.set_sampler_fn([dm]() { return dm.sampler(); });
    out.set_sizes(dm.point_size(), dm.tangent_size());
    return out;
  }

  DynamicManifold base_;
  std::function<DynamicMetric()> metric_for_copies_;
  DynamicManifold impl_;
  std::string base_name_;
  std::string metric_name_;
  std::optional<PySE2> se2_;
  std::shared_ptr<const PyOwnedRef> metric_ref_;
};

}  // namespace geodex::python

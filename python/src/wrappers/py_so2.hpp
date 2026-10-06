/// @file py_so2.hpp
/// @brief Python wrapper for geodex::SO2 (the circle group).

#pragma once

#include <memory>
#include <string>

#include <Eigen/Core>

#include "geodex/algorithm/precompute_matrix_lower_bound.hpp"
#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/manifold/so2.hpp"

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"
#include "sizes.hpp"

namespace geodex::python {

/// @brief Python-facing wrapper for the abelian manifold SO(2).
class PySO2 {
 public:
  /// @brief The wrapped C++ manifold with the canonical metric and the true exp and log.
  using Impl = SO2<SO2CanonicalMetric, SO2ExponentialMap, DynamicSampler>;

  /// @brief Fixed-size 1-vector used internally by the C++ manifold.
  using M1 = Eigen::Matrix<double, 1, 1>;

  /// @brief Construct SO(2) with a scalar rotational weight.
  /// @param weight Positive metric weight. The norm scales as sqrt(weight).
  PySO2(double weight = 1.0, const std::string& sampler = "scrambled")
      : impl_(std::make_shared<Impl>(SO2CanonicalMetric{weight})), weight_(weight) {
    if (sampler != "scrambled") impl_->set_sampler(make_sampler(sampler));
  }

  int dim() const { return impl_->dim(); }

  Eigen::VectorXd random_point() const { return impl_->random_point(); }

  /// @brief Reseed the sampler for reproducible sampling.
  void seed(std::uint64_t s) { impl_->seed(s); }

  /// @brief Switch the sampler kind at runtime.
  void set_sampler(const std::string& kind) { impl_->set_sampler(make_sampler(kind)); }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    require_size(p, 1, "SO2.inner", "p");
    require_size(u, 1, "SO2.inner", "u");
    require_size(v, 1, "SO2.inner", "v");
    M1 p1(p), u1(u), v1(v);
    return impl_->inner(p1, u1, v1);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, 1, "SO2.norm", "p");
    require_size(v, 1, "SO2.norm", "v");
    M1 p1(p), v1(v);
    return impl_->norm(p1, v1);
  }

  Eigen::VectorXd exp(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, 1, "SO2.exp", "p");
    require_size(v, 1, "SO2.exp", "v");
    M1 p1(p), v1(v);
    return impl_->exp(p1, v1);
  }

  Eigen::VectorXd log(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    require_size(p, 1, "SO2.log", "p");
    require_size(q, 1, "SO2.log", "q");
    M1 p1(p), q1(q);
    return impl_->log(p1, q1);
  }

  double distance(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    require_size(p, 1, "SO2.distance", "p");
    require_size(q, 1, "SO2.distance", "q");
    M1 p1(p), q1(q);
    return impl_->distance(p1, q1);
  }

  Eigen::VectorXd geodesic(const Eigen::VectorXd& p, const Eigen::VectorXd& q, double t) const {
    require_size(p, 1, "SO2.geodesic", "p");
    require_size(q, 1, "SO2.geodesic", "q");
    M1 p1(p), q1(q);
    return impl_->geodesic(p1, q1, t);
  }

  /// @brief Certify the periods-aware Loewner lower bound for this metric, which
  /// wraps at the cut where a raw chord would overestimate.
  heuristics::MatrixLowerBound<Eigen::Dynamic> matrix_lower_bound() const {
    return algorithm::precompute_matrix_lower_bound(*impl_).heuristic();
  }

  /// @brief The period of the angle, 2 pi.
  Eigen::VectorXd periods() const { return impl_->periods(); }

  /// @brief The metric on the angle's velocity.
  Eigen::MatrixXd coordinate_metric(const Eigen::VectorXd& q) const {
    require_size(q, 1, "SO2.coordinate_metric", "q");
    return impl_->coordinate_metric(M1(q));
  }

  DynamicManifold to_dynamic_manifold() const {
    auto shared = impl_;
    DynamicManifold dm{
        [shared]() { return shared->dim(); },
        [shared]() -> Eigen::VectorXd { return shared->random_point(); },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
          M1 p1(p), v1(v);
          return shared->exp(p1, v1);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          M1 p1(p), q1(q);
          return shared->log(p1, q1);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) -> double {
          M1 p1(p), u1(u), v1(v);
          return shared->inner(p1, u1, v1);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          M1 p1(p), v1(v);
          return shared->norm(p1, v1);
        },
        // The SO(2) tangent space is the Lie algebra so(2) = R^1, and projection is the identity.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; },
        [shared]() { return shared->unit_cube_dim(); },
        [shared](Eigen::Ref<const Eigen::VectorXd> u) -> Eigen::VectorXd {
          return shared->from_unit_cube(u);
        }};
    dm.set_sampler_fn([shared]() { return shared->sampler(); });
    dm.set_riemannian_log(geodex::is_riemannian_log(*shared));
    return dm;
  }

  std::string repr() const { return "SO2(weight=" + std::to_string(weight_) + ")"; }

 private:
  /// The wrapped manifold, shared with every DynamicManifold this wrapper hands out. A plan
  /// advances this manifold's own sampler.
  std::shared_ptr<Impl> impl_;
  double weight_;
};

}  // namespace geodex::python

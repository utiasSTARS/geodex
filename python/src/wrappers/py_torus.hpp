/// @file py_torus.hpp
/// @brief Python wrapper for geodex::Torus with dynamic dimension.

#pragma once

#include <memory>
#include <string>

#include <Eigen/Core>

#include "geodex/algorithm/precompute_matrix_lower_bound.hpp"
#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/manifold/torus.hpp"

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"
#include "sizes.hpp"

namespace geodex::python {

class PyTorus {
 public:
  using Impl = Torus<Eigen::Dynamic, TorusFlatMetric<Eigen::Dynamic>, DynamicSampler>;

  explicit PyTorus(int n, const std::string& sampler = "scrambled")
      : impl_(std::make_shared<Impl>(n)) {
    if (sampler != "scrambled") impl_->set_sampler(make_sampler(sampler));
  }

  int dim() const { return impl_->dim(); }

  Eigen::VectorXd random_point() const { return impl_->random_point(); }

  /// @brief Reseed the sampler for reproducible sampling.
  void seed(std::uint64_t s) { impl_->seed(s); }

  /// @brief Switch the sampler kind at runtime.
  void set_sampler(const std::string& kind) { impl_->set_sampler(make_sampler(kind)); }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    require_size(p, impl_->dim(), "Torus.inner", "p");
    require_size(u, impl_->dim(), "Torus.inner", "u");
    require_size(v, impl_->dim(), "Torus.inner", "v");
    return impl_->inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, impl_->dim(), "Torus.norm", "p");
    require_size(v, impl_->dim(), "Torus.norm", "v");
    return impl_->norm(p, v);
  }

  Eigen::VectorXd exp(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, impl_->dim(), "Torus.exp", "p");
    require_size(v, impl_->dim(), "Torus.exp", "v");
    return impl_->exp(p, v);
  }

  Eigen::VectorXd log(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    require_size(p, impl_->dim(), "Torus.log", "p");
    require_size(q, impl_->dim(), "Torus.log", "q");
    return impl_->log(p, q);
  }

  double distance(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    require_size(p, impl_->dim(), "Torus.distance", "p");
    require_size(q, impl_->dim(), "Torus.distance", "q");
    return impl_->distance(p, q);
  }

  Eigen::VectorXd geodesic(const Eigen::VectorXd& p, const Eigen::VectorXd& q, double t) const {
    require_size(p, impl_->dim(), "Torus.geodesic", "p");
    require_size(q, impl_->dim(), "Torus.geodesic", "q");
    return impl_->geodesic(p, q, t);
  }

  /// @brief Certify the periods-aware Loewner lower bound for this metric, which
  /// wraps at the cut where a raw chord would overestimate.
  heuristics::MatrixLowerBound<Eigen::Dynamic> matrix_lower_bound() const {
    return algorithm::precompute_matrix_lower_bound(*impl_).heuristic();
  }

  /// @brief The period of every angle, 2 pi.
  Eigen::VectorXd periods() const { return impl_->periods(); }

  /// @brief The metric on angle velocities.
  Eigen::MatrixXd coordinate_metric(const Eigen::VectorXd& q) const {
    require_size(q, impl_->dim(), "Torus.coordinate_metric", "q");
    return impl_->coordinate_metric(q);
  }

  DynamicManifold to_dynamic_manifold() const {
    auto shared = impl_;
    DynamicManifold dm{
        [shared]() { return shared->dim(); },
        [shared]() -> Eigen::VectorXd { return shared->random_point(); },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
          return shared->exp(p, v);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          return shared->log(p, q);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) {
          return shared->inner(p, u, v);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) { return shared->norm(p, v); },
        // The torus tangent space is the ambient R^n of the angles, and projection is identity.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; },
        [shared]() { return shared->unit_cube_dim(); },
        [shared](Eigen::Ref<const Eigen::VectorXd> u) -> Eigen::VectorXd {
          return shared->from_unit_cube(u);
        }};
    dm.set_sampler_fn([shared]() { return shared->sampler(); });
    dm.set_riemannian_log(geodex::is_riemannian_log(*shared));
    return dm;
  }

  std::string repr() const { return "Torus(dim=" + std::to_string(impl_->dim()) + ")"; }

 private:
  /// The wrapped manifold, shared with every DynamicManifold this wrapper hands out. A plan
  /// advances this manifold's own sampler.
  std::shared_ptr<Impl> impl_;
};

}  // namespace geodex::python

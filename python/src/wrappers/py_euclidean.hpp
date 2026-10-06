/// @file py_euclidean.hpp
/// @brief Python wrapper for geodex::Euclidean with dynamic dimension.

#pragma once

#include <memory>
#include <string>

#include <Eigen/Core>

#include "geodex/manifold/euclidean.hpp"

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"
#include "sizes.hpp"

namespace geodex::python {

class PyEuclidean {
 public:
  using Impl = Euclidean<Eigen::Dynamic, EuclideanStandardMetric<Eigen::Dynamic>, DynamicSampler>;

  explicit PyEuclidean(int n, const std::string& sampler = "scrambled")
      : impl_(std::make_shared<Impl>(n)) {
    if (sampler != "scrambled") impl_->set_sampler(make_sampler(sampler));
  }

  int dim() const { return impl_->dim(); }

  Eigen::VectorXd random_point() const { return impl_->random_point(); }

  /// @brief Reseed the sampler for reproducible sampling.
  void seed(std::uint64_t s) { impl_->seed(s); }

  /// @brief Switch the sampler kind at runtime.
  void set_sampler(const std::string& kind) { impl_->set_sampler(make_sampler(kind)); }

  /// @brief Set the per-dimension sampling bounds (default [-1, 1]^n).
  void set_sampling_bounds(const Eigen::VectorXd& lo, const Eigen::VectorXd& hi) {
    impl_->set_sampling_bounds(lo, hi);
  }

  /// @brief Lower per-dimension sampling bound.
  const Eigen::VectorXd& lo() const { return impl_->lo(); }

  /// @brief Upper per-dimension sampling bound.
  const Eigen::VectorXd& hi() const { return impl_->hi(); }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    require_size(p, impl_->dim(), "Euclidean.inner", "p");
    require_size(u, impl_->dim(), "Euclidean.inner", "u");
    require_size(v, impl_->dim(), "Euclidean.inner", "v");
    return impl_->inner(p, u, v);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, impl_->dim(), "Euclidean.norm", "p");
    require_size(v, impl_->dim(), "Euclidean.norm", "v");
    return impl_->norm(p, v);
  }

  Eigen::VectorXd exp(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, impl_->dim(), "Euclidean.exp", "p");
    require_size(v, impl_->dim(), "Euclidean.exp", "v");
    return impl_->exp(p, v);
  }

  Eigen::VectorXd log(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    require_size(p, impl_->dim(), "Euclidean.log", "p");
    require_size(q, impl_->dim(), "Euclidean.log", "q");
    return impl_->log(p, q);
  }

  double distance(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    require_size(p, impl_->dim(), "Euclidean.distance", "p");
    require_size(q, impl_->dim(), "Euclidean.distance", "q");
    return impl_->distance(p, q);
  }

  Eigen::VectorXd geodesic(const Eigen::VectorXd& p, const Eigen::VectorXd& q, double t) const {
    require_size(p, impl_->dim(), "Euclidean.geodesic", "p");
    require_size(q, impl_->dim(), "Euclidean.geodesic", "q");
    return impl_->geodesic(p, q, t);
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
        // The Euclidean tangent space is the ambient space, and projection is the identity.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; },
        [shared]() { return shared->unit_cube_dim(); },
        [shared](Eigen::Ref<const Eigen::VectorXd> u) -> Eigen::VectorXd {
          return shared->from_unit_cube(u);
        }};
    dm.set_geodesic_fn([shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q,
                                const double t) -> Eigen::VectorXd {
      return shared->geodesic(p, q, t);
    });
    dm.set_distance_fn([shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
      return shared->distance(p, q);
    });
    dm.set_sampler_fn([shared]() { return shared->sampler(); });
    dm.set_riemannian_log(geodex::is_riemannian_log(*shared));
    return dm;
  }

  std::string repr() const { return "Euclidean(dim=" + std::to_string(impl_->dim()) + ")"; }

 private:
  /// The wrapped manifold, shared with every DynamicManifold this wrapper hands out. A plan
  /// advances this manifold's own sampler.
  std::shared_ptr<Impl> impl_;
};

}  // namespace geodex::python

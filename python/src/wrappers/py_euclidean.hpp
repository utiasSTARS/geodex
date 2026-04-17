/// @file py_euclidean.hpp
/// @brief Python wrapper for geodex::Euclidean with dynamic dimension.

#pragma once

#include <memory>
#include <string>

#include <Eigen/Core>

#include "geodex/manifold/euclidean.hpp"

#include "dynamic_manifold.hpp"

namespace geodex::python {

class PyEuclidean {
 public:
  using Impl = Euclidean<Eigen::Dynamic, EuclideanStandardMetric<Eigen::Dynamic>>;

  explicit PyEuclidean(int n) : impl_(n) {}

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
    auto shared = std::make_shared<Impl>(impl_);
    return DynamicManifold{
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
        // Euclidean tangent space = ambient space; projection is the identity.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; }};
  }

  std::string repr() const { return "Euclidean(dim=" + std::to_string(impl_.dim()) + ")"; }

 private:
  Impl impl_;
};

}  // namespace geodex::python

/// @file py_so2.hpp
/// @brief Python wrapper for geodex::SO2 (the circle group).

#pragma once

#include <memory>
#include <string>

#include <Eigen/Core>

#include "geodex/manifold/so2.hpp"

#include "dynamic_manifold.hpp"

namespace geodex::python {

/// @brief Python-facing wrapper for the abelian manifold SO(2).
class PySO2 {
 public:
  /// @brief The wrapped C++ manifold: canonical metric with the true exp/log map.
  using Impl = SO2<>;

  /// @brief Fixed-size 1-vector used internally by the C++ manifold.
  using M1 = Eigen::Matrix<double, 1, 1>;

  /// @brief Construct SO(2) with a scalar rotational weight.
  /// @param weight Positive metric weight; the norm scales as sqrt(weight).
  PySO2(double weight = 1.0) : impl_{SO2CanonicalMetric{weight}}, weight_(weight) {}

  int dim() const { return impl_.dim(); }

  Eigen::VectorXd random_point() const { return impl_.random_point(); }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    M1 p1(p), u1(u), v1(v);
    return impl_.inner(p1, u1, v1);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    M1 p1(p), v1(v);
    return impl_.norm(p1, v1);
  }

  Eigen::VectorXd exp(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    M1 p1(p), v1(v);
    return impl_.exp(p1, v1);
  }

  Eigen::VectorXd log(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    M1 p1(p), q1(q);
    return impl_.log(p1, q1);
  }

  double distance(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    M1 p1(p), q1(q);
    return impl_.distance(p1, q1);
  }

  Eigen::VectorXd geodesic(const Eigen::VectorXd& p, const Eigen::VectorXd& q, double t) const {
    M1 p1(p), q1(q);
    return impl_.geodesic(p1, q1, t);
  }

  DynamicManifold to_dynamic_manifold() const {
    auto shared = std::make_shared<Impl>(impl_);
    return DynamicManifold{
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
        // SO(2) tangent space is the Lie algebra so(2) = R^1; projection is the identity.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; }};
  }

  std::string repr() const { return "SO2(weight=" + std::to_string(weight_) + ")"; }

 private:
  Impl impl_;
  double weight_;
};

}  // namespace geodex::python

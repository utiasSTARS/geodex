/// @file py_sphere_n.hpp
/// @brief Python wrapper for geodex::Sphere<Eigen::Dynamic> (n-dimensional sphere).

#pragma once

#include <Eigen/Core>
#include <geodex/manifold/sphere.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include "dynamic_manifold.hpp"

namespace geodex::python {

/// @brief Python wrapper for the n-dimensional sphere \f$ S^n \f$.
///
/// @details Wraps `Sphere<Eigen::Dynamic>` — points are `VectorXd` of size
/// `n+1`. The dimension `n` is set at construction time.
class PySphereN {
 public:
  using SphereExp = Sphere<Eigen::Dynamic, IdentityMetric<Eigen::Dynamic>, SphereExponentialMap>;
  using SphereProj = Sphere<Eigen::Dynamic, IdentityMetric<Eigen::Dynamic>, SphereProjectionRetraction>;
  using V = std::variant<SphereExp, SphereProj>;

  PySphereN(int n, const std::string& retraction = "exponential")
      : impl_(SphereExp(n)), dim_(n), retraction_name_(retraction) {
    if (n < 1) {
      throw std::invalid_argument("Sphere dimension must be >= 1, got " + std::to_string(n));
    }
    if (retraction == "exponential") {
      impl_.emplace<SphereExp>(n);
    } else if (retraction == "projection") {
      impl_.emplace<SphereProj>(n);
    } else {
      throw std::invalid_argument("Unknown retraction: '" + retraction +
                                  "'. Options: 'exponential', 'projection'");
    }
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, impl_);
  }

  Eigen::VectorXd random_point() const {
    return std::visit([](const auto& m) -> Eigen::VectorXd { return m.random_point(); }, impl_);
  }

  Eigen::VectorXd project(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.project(p, v); }, impl_);
  }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u,
               const Eigen::VectorXd& v) const {
    return std::visit([&](const auto& m) { return m.inner(p, u, v); }, impl_);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::visit([&](const auto& m) { return m.norm(p, v); }, impl_);
  }

  Eigen::VectorXd exp(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.exp(p, v); }, impl_);
  }

  Eigen::VectorXd log(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.log(p, q); }, impl_);
  }

  double distance(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    return std::visit([&](const auto& m) { return m.distance(p, q); }, impl_);
  }

  Eigen::VectorXd geodesic(const Eigen::VectorXd& p, const Eigen::VectorXd& q,
                            double t) const {
    return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.geodesic(p, q, t); }, impl_);
  }

  DynamicManifold to_dynamic_manifold() const {
    auto shared = std::make_shared<V>(impl_);
    return DynamicManifold{
        [shared]() { return std::visit([](const auto& m) { return m.dim(); }, *shared); },
        [shared]() -> Eigen::VectorXd {
          return std::visit([](const auto& m) -> Eigen::VectorXd { return m.random_point(); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
          return std::visit(
              [&](const auto& m) -> Eigen::VectorXd { return m.exp(p, v); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          return std::visit(
              [&](const auto& m) -> Eigen::VectorXd { return m.log(p, q); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) -> double {
          return std::visit([&](const auto& m) { return m.inner(p, u, v); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          return std::visit([&](const auto& m) { return m.norm(p, v); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
          return std::visit(
              [&](const auto& m) -> Eigen::VectorXd { return m.project(p, v); }, *shared);
        }};
  }

  std::string repr() const {
    return "Sphere(dim=" + std::to_string(dim_) + ", retraction='" + retraction_name_ + "')";
  }

 private:
  V impl_;
  int dim_;
  std::string retraction_name_;
};

}  // namespace geodex::python

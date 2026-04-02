/// @file py_sphere.hpp
/// @brief Python wrapper for geodex::Sphere with variant-based dispatch.

#pragma once

#include <Eigen/Core>
#include <geodex/manifold/sphere.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include "dynamic_manifold.hpp"

namespace geodex::python {

class PySphere {
 public:
  using V = std::variant<Sphere<SphereRoundMetric, SphereExponentialMap>,
                         Sphere<SphereRoundMetric, SphereProjectionRetraction>>;

  PySphere(const std::string& retraction = "exponential")
      : retraction_name_(retraction) {
    if (retraction == "exponential") {
      impl_.emplace<Sphere<SphereRoundMetric, SphereExponentialMap>>();
    } else if (retraction == "projection") {
      impl_.emplace<Sphere<SphereRoundMetric, SphereProjectionRetraction>>();
    } else {
      throw std::invalid_argument("Unknown retraction: '" + retraction +
                                  "'. Options: 'exponential', 'projection'");
    }
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, impl_);
  }

  Eigen::Vector3d random_point() const {
    return std::visit([](const auto& m) { return m.random_point(); }, impl_);
  }

  Eigen::Vector3d project(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.project(p, v); }, impl_);
  }

  double inner(const Eigen::Vector3d& p, const Eigen::Vector3d& u,
               const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.inner(p, u, v); }, impl_);
  }

  double norm(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.norm(p, v); }, impl_);
  }

  Eigen::Vector3d exp(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.exp(p, v); }, impl_);
  }

  Eigen::Vector3d log(const Eigen::Vector3d& p, const Eigen::Vector3d& q) const {
    return std::visit([&](const auto& m) { return m.log(p, q); }, impl_);
  }

  double distance(const Eigen::Vector3d& p, const Eigen::Vector3d& q) const {
    return std::visit([&](const auto& m) { return m.distance(p, q); }, impl_);
  }

  Eigen::Vector3d geodesic(const Eigen::Vector3d& p, const Eigen::Vector3d& q,
                           double t) const {
    return std::visit([&](const auto& m) { return m.geodesic(p, q, t); }, impl_);
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
          Eigen::Vector3d p3(p), v3(v);
          return std::visit(
              [&](const auto& m) -> Eigen::VectorXd { return m.exp(p3, v3); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          Eigen::Vector3d p3(p), q3(q);
          return std::visit(
              [&](const auto& m) -> Eigen::VectorXd { return m.log(p3, q3); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) -> double {
          Eigen::Vector3d p3(p), u3(u), v3(v);
          return std::visit([&](const auto& m) { return m.inner(p3, u3, v3); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          Eigen::Vector3d p3(p), v3(v);
          return std::visit([&](const auto& m) { return m.norm(p3, v3); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
          Eigen::Vector3d p3(p), v3(v);
          return std::visit(
              [&](const auto& m) -> Eigen::VectorXd { return m.project(p3, v3); }, *shared);
        }};
  }

  std::string repr() const {
    return "Sphere(retraction='" + retraction_name_ + "')";
  }

 private:
  V impl_;
  std::string retraction_name_;
};

}  // namespace geodex::python

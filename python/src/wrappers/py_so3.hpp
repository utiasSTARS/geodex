/// @file py_so3.hpp
/// @brief Python wrapper for geodex::SO3 with variant-based retraction-frame dispatch.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include <Eigen/Core>

#include "geodex/manifold/so3.hpp"

#include "dynamic_manifold.hpp"

namespace geodex::python {

class PySO3 {
 public:
  using V = std::variant<SO3<SO3CanonicalMetric, SO3LeftExponentialMap>,
                         SO3<SO3CanonicalMetric, SO3RightExponentialMap>>;

  PySO3(const std::string& frame = "body", double weight = 1.0) : frame_(frame) {
    SO3CanonicalMetric metric(weight);
    if (frame == "body") {
      impl_.emplace<SO3<SO3CanonicalMetric, SO3LeftExponentialMap>>(metric,
                                                                    SO3LeftExponentialMap{});
    } else if (frame == "world") {
      impl_.emplace<SO3<SO3CanonicalMetric, SO3RightExponentialMap>>(metric,
                                                                     SO3RightExponentialMap{});
    } else {
      throw std::invalid_argument("frame must be 'body' or 'world'");
    }
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, impl_);
  }

  Eigen::Vector4d random_point() const {
    return std::visit([](const auto& m) { return m.random_point(); }, impl_);
  }

  double inner(const Eigen::Vector4d& p, const Eigen::Vector3d& u, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.inner(p, u, v); }, impl_);
  }

  double norm(const Eigen::Vector4d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.norm(p, v); }, impl_);
  }

  Eigen::Vector4d exp(const Eigen::Vector4d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.exp(p, v); }, impl_);
  }

  Eigen::Vector3d log(const Eigen::Vector4d& p, const Eigen::Vector4d& q) const {
    return std::visit([&](const auto& m) { return m.log(p, q); }, impl_);
  }

  double distance(const Eigen::Vector4d& p, const Eigen::Vector4d& q) const {
    return std::visit([&](const auto& m) { return m.distance(p, q); }, impl_);
  }

  Eigen::Vector4d geodesic(const Eigen::Vector4d& p, const Eigen::Vector4d& q, double t) const {
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
          Eigen::Vector4d p4(p);
          Eigen::Vector3d v3(v);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.exp(p4, v3); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          Eigen::Vector4d p4(p), q4(q);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.log(p4, q4); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) -> double {
          Eigen::Vector4d p4(p);
          Eigen::Vector3d u3(u), v3(v);
          return std::visit([&](const auto& m) { return m.inner(p4, u3, v3); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          Eigen::Vector4d p4(p);
          Eigen::Vector3d v3(v);
          return std::visit([&](const auto& m) { return m.norm(p4, v3); }, *shared);
        },
        // SO(3) tangents are already the minimal body algebra so(3) ~= R^3; projection is identity.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; }};
  }

  std::string repr() const { return "SO3(frame='" + frame_ + "')"; }

 private:
  V impl_;
  std::string frame_;
};

}  // namespace geodex::python

/// @file py_se2.hpp
/// @brief Python wrapper for geodex::SE2 with variant-based retraction dispatch.

#pragma once

#include <Eigen/Core>
#include <geodex/manifold/se2.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include "dynamic_manifold.hpp"

namespace geodex::python {

class PySE2 {
 public:
  using V = std::variant<SE2<SE2LeftInvariantMetric, SE2ExponentialMap>,
                         SE2<SE2LeftInvariantMetric, SE2EulerRetraction>>;

  PySE2(double wx = 1.0, double wy = 1.0, double wtheta = 1.0,
        const std::string& retraction = "exponential",
        double x_lo = 0.0, double x_hi = 10.0,
        double y_lo = 0.0, double y_hi = 10.0)
      : retraction_name_(retraction) {
    SE2LeftInvariantMetric metric(wx, wy, wtheta);
    if (retraction == "exponential") {
      impl_.emplace<SE2<SE2LeftInvariantMetric, SE2ExponentialMap>>(
          metric, SE2ExponentialMap{}, x_lo, x_hi, y_lo, y_hi);
    } else if (retraction == "euler") {
      impl_.emplace<SE2<SE2LeftInvariantMetric, SE2EulerRetraction>>(
          metric, SE2EulerRetraction{}, x_lo, x_hi, y_lo, y_hi);
    } else {
      throw std::invalid_argument(
          "Unknown retraction: '" + retraction +
          "'. Options: 'exponential', 'euler'");
    }
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, impl_);
  }

  Eigen::Vector3d random_point() const {
    return std::visit([](const auto& m) { return m.random_point(); }, impl_);
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
        // SE2 is parameterized as (x, y, theta) ∈ R^3; tangent space = ambient R^3.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; }};
  }

  std::string repr() const {
    return "SE2(retraction='" + retraction_name_ + "')";
  }

 private:
  V impl_;
  std::string retraction_name_;
};

}  // namespace geodex::python

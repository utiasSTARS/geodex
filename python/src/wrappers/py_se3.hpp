/// @file py_se3.hpp
/// @brief Python wrapper for geodex::SE3 with variant-based frame (retraction) dispatch.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include <Eigen/Core>

#include "geodex/manifold/se3.hpp"

#include "dynamic_manifold.hpp"

namespace geodex::python {

class PySE3 {
 public:
  /// @brief Pose \f$ [t_x, t_y, t_z,\; q_x, q_y, q_z, q_w] \in \mathbb{R}^7 \f$.
  using Point = Eigen::Matrix<double, 7, 1>;
  /// @brief Body/spatial twist \f$ [v;\,\omega] \in \mathbb{R}^6 \f$.
  using Tangent = Eigen::Matrix<double, 6, 1>;

  /// @brief Left/right group-exponential variants (body vs. world frame).
  using V = std::variant<SE3<SE3InvariantMetric, SE3LeftExponentialMap>,
                         SE3<SE3InvariantMetric, SE3RightExponentialMap>>;

  PySE3(const std::string& frame = "body", double w_trans = 1.0, double w_rot = 1.0,
        double x_lo = 0.0, double x_hi = 10.0, double y_lo = 0.0, double y_hi = 10.0,
        double z_lo = 0.0, double z_hi = 10.0)
      : frame_name_(frame) {
    SE3InvariantMetric metric(w_trans, w_rot);
    const Eigen::Vector3d lo(x_lo, y_lo, z_lo);
    const Eigen::Vector3d hi(x_hi, y_hi, z_hi);
    if (frame == "body") {
      impl_.emplace<SE3<SE3InvariantMetric, SE3LeftExponentialMap>>(metric, SE3LeftExponentialMap{},
                                                                    lo, hi);
    } else if (frame == "world") {
      impl_.emplace<SE3<SE3InvariantMetric, SE3RightExponentialMap>>(
          metric, SE3RightExponentialMap{}, lo, hi);
    } else {
      throw std::invalid_argument("Unknown frame: '" + frame + "'. Options: 'body', 'world'");
    }
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, impl_);
  }

  Point random_point() const {
    return std::visit([](const auto& m) { return m.random_point(); }, impl_);
  }

  double inner(const Point& p, const Tangent& u, const Tangent& v) const {
    return std::visit([&](const auto& m) { return m.inner(p, u, v); }, impl_);
  }

  double norm(const Point& p, const Tangent& v) const {
    return std::visit([&](const auto& m) { return m.norm(p, v); }, impl_);
  }

  Point exp(const Point& p, const Tangent& v) const {
    return std::visit([&](const auto& m) { return m.exp(p, v); }, impl_);
  }

  Tangent log(const Point& p, const Point& q) const {
    return std::visit([&](const auto& m) { return m.log(p, q); }, impl_);
  }

  double distance(const Point& p, const Point& q) const {
    return std::visit([&](const auto& m) { return m.distance(p, q); }, impl_);
  }

  Point geodesic(const Point& p, const Point& q, double t) const {
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
          Eigen::Matrix<double, 7, 1> p7(p);
          Eigen::Matrix<double, 6, 1> v6(v);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.exp(p7, v6); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          Eigen::Matrix<double, 7, 1> p7(p), q7(q);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.log(p7, q7); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) -> double {
          Eigen::Matrix<double, 7, 1> p7(p);
          Eigen::Matrix<double, 6, 1> u6(u), v6(v);
          return std::visit([&](const auto& m) { return m.inner(p7, u6, v6); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          Eigen::Matrix<double, 7, 1> p7(p);
          Eigen::Matrix<double, 6, 1> v6(v);
          return std::visit([&](const auto& m) { return m.norm(p7, v6); }, *shared);
        },
        // SE(3)'s tangent space is the Lie algebra se(3) ≅ R^6; project is the identity.
        [](const Eigen::VectorXd& /*p*/, const Eigen::VectorXd& v) { return v; }};
  }

  std::string repr() const { return "SE3(frame='" + frame_name_ + "')"; }

 private:
  V impl_;
  std::string frame_name_;
};

}  // namespace geodex::python

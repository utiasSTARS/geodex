/// @file py_sphere.hpp
/// @brief Python wrapper for geodex::Sphere with variant-based dispatch.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include <Eigen/Core>

#include "geodex/manifold/sphere.hpp"

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"

namespace geodex::python {

class PySphere {
 public:
  using Exp = Sphere<2, SphereRoundMetric, SphereExponentialMap, DynamicSampler>;
  using Proj = Sphere<2, SphereRoundMetric, SphereProjectionRetraction, DynamicSampler>;
  using V = std::variant<Exp, Proj>;

  PySphere(const std::string& retraction = "exponential",
           const std::string& sampler = "scrambled")
      : retraction_name_(retraction) {
    if (retraction == "exponential") {
      impl_ = std::make_shared<V>(std::in_place_type<Exp>);
    } else if (retraction == "projection") {
      impl_ = std::make_shared<V>(std::in_place_type<Proj>);
    } else {
      throw std::invalid_argument("Unknown retraction: '" + retraction +
                                  "'. Options: 'exponential', 'projection'");
    }
    if (sampler != "scrambled") set_sampler(sampler);
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, *impl_);
  }

  Eigen::Vector3d random_point() const {
    return std::visit([](const auto& m) { return m.random_point(); }, *impl_);
  }

  /// @brief Reseed the sampler for reproducible sampling.
  void seed(std::uint64_t s) { std::visit([&](auto& m) { m.seed(s); }, *impl_); }

  /// @brief Switch the sampler kind at runtime.
  void set_sampler(const std::string& kind) {
    std::visit([&](auto& m) { m.set_sampler(make_sampler(kind)); }, *impl_);
  }

  Eigen::Vector3d project(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.project(p, v); }, *impl_);
  }

  double inner(const Eigen::Vector3d& p, const Eigen::Vector3d& u, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.inner(p, u, v); }, *impl_);
  }

  double norm(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.norm(p, v); }, *impl_);
  }

  Eigen::Vector3d exp(const Eigen::Vector3d& p, const Eigen::Vector3d& v) const {
    return std::visit([&](const auto& m) { return m.exp(p, v); }, *impl_);
  }

  Eigen::Vector3d log(const Eigen::Vector3d& p, const Eigen::Vector3d& q) const {
    return std::visit([&](const auto& m) { return m.log(p, q); }, *impl_);
  }

  double distance(const Eigen::Vector3d& p, const Eigen::Vector3d& q) const {
    return std::visit([&](const auto& m) { return m.distance(p, q); }, *impl_);
  }

  Eigen::Vector3d geodesic(const Eigen::Vector3d& p, const Eigen::Vector3d& q, double t) const {
    return std::visit([&](const auto& m) { return m.geodesic(p, q, t); }, *impl_);
  }

  DynamicManifold to_dynamic_manifold() const {
    auto shared = impl_;
    DynamicManifold dm{
        [shared]() { return std::visit([](const auto& m) { return m.dim(); }, *shared); },
        [shared]() -> Eigen::VectorXd {
          return std::visit([](const auto& m) -> Eigen::VectorXd { return m.random_point(); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
          Eigen::Vector3d p3(p), v3(v);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.exp(p3, v3); },
                            *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          Eigen::Vector3d p3(p), q3(q);
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.log(p3, q3); },
                            *shared);
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
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.project(p3, v3); },
                            *shared);
        },
        [shared]() { return std::visit([](const auto& m) { return m.unit_cube_dim(); }, *shared); },
        [shared](Eigen::Ref<const Eigen::VectorXd> u) -> Eigen::VectorXd {
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.from_unit_cube(u); },
                            *shared);
        }};
    dm.set_geodesic_fn([shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q,
                                const double t) -> Eigen::VectorXd {
      const Eigen::Vector3d p3(p), q3(q);
      return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.geodesic(p3, q3, t); },
                        *shared);
    });
    dm.set_distance_fn([shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
      const Eigen::Vector3d p3(p), q3(q);
      return std::visit([&](const auto& m) { return m.distance(p3, q3); }, *shared);
    });
    dm.set_sampler_fn([shared]() {
      return std::visit([](const auto& m) { return DynamicSampler(m.sampler()); }, *shared);
    });
    dm.set_riemannian_log(
        std::visit([](const auto& m) { return geodex::is_riemannian_log(m); }, *shared));
    return dm;
  }

  std::string repr() const { return "Sphere(retraction='" + retraction_name_ + "')"; }

 private:
  /// The wrapped manifold, shared with every DynamicManifold this wrapper hands out. A plan
  /// advances this manifold's own sampler.
  std::shared_ptr<V> impl_;
  std::string retraction_name_;
};

}  // namespace geodex::python

/// @file py_sphere_n.hpp
/// @brief Python wrapper for geodex::Sphere<Eigen::Dynamic> (n-dimensional sphere).

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include <Eigen/Core>

#include "geodex/manifold/sphere.hpp"

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"
#include "sizes.hpp"

namespace geodex::python {

/// @brief Python wrapper for the n-dimensional sphere \f$ S^n \f$.
///
/// @details Wraps `Sphere<Eigen::Dynamic>`. Points are `VectorXd` of size `n+1`. The
/// dimension `n` is set at construction time.
class PySphereN {
 public:
  using SphereExp =
      Sphere<Eigen::Dynamic, IdentityMetric<Eigen::Dynamic>, SphereExponentialMap, DynamicSampler>;
  using SphereProj =
      Sphere<Eigen::Dynamic, IdentityMetric<Eigen::Dynamic>, SphereProjectionRetraction,
             DynamicSampler>;
  using V = std::variant<SphereExp, SphereProj>;

  PySphereN(int n, const std::string& retraction = "exponential",
            const std::string& sampler = "scrambled")
      : dim_(n), retraction_name_(retraction) {
    if (n < 1) {
      throw std::invalid_argument("Sphere dimension must be >= 1, got " + std::to_string(n));
    }
    if (retraction == "exponential") {
      impl_ = std::make_shared<V>(std::in_place_type<SphereExp>, n);
    } else if (retraction == "projection") {
      impl_ = std::make_shared<V>(std::in_place_type<SphereProj>, n);
    } else {
      throw std::invalid_argument("Unknown retraction: '" + retraction +
                                  "'. Options: 'exponential', 'projection'");
    }
    if (sampler != "scrambled") set_sampler(sampler);
  }

  int dim() const {
    return std::visit([](const auto& m) { return m.dim(); }, *impl_);
  }

  Eigen::VectorXd random_point() const {
    return std::visit([](const auto& m) -> Eigen::VectorXd { return m.random_point(); }, *impl_);
  }

  /// @brief Reseed the sampler for reproducible sampling.
  void seed(std::uint64_t s) {
    std::visit([&](auto& m) { m.seed(s); }, *impl_);
  }

  /// @brief Switch the sampler kind at runtime.
  void set_sampler(const std::string& kind) {
    std::visit([&](auto& m) { m.set_sampler(make_sampler(kind)); }, *impl_);
  }

  Eigen::VectorXd project(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, dim_ + 1, "SphereN.project", "p");
    require_size(v, dim_ + 1, "SphereN.project", "v");
    return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.project(p, v); }, *impl_);
  }

  double inner(const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) const {
    require_size(p, dim_ + 1, "SphereN.inner", "p");
    require_size(u, dim_ + 1, "SphereN.inner", "u");
    require_size(v, dim_ + 1, "SphereN.inner", "v");
    return std::visit([&](const auto& m) { return m.inner(p, u, v); }, *impl_);
  }

  double norm(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, dim_ + 1, "SphereN.norm", "p");
    require_size(v, dim_ + 1, "SphereN.norm", "v");
    return std::visit([&](const auto& m) { return m.norm(p, v); }, *impl_);
  }

  Eigen::VectorXd exp(const Eigen::VectorXd& p, const Eigen::VectorXd& v) const {
    require_size(p, dim_ + 1, "SphereN.exp", "p");
    require_size(v, dim_ + 1, "SphereN.exp", "v");
    return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.exp(p, v); }, *impl_);
  }

  Eigen::VectorXd log(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    require_size(p, dim_ + 1, "SphereN.log", "p");
    require_size(q, dim_ + 1, "SphereN.log", "q");
    return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.log(p, q); }, *impl_);
  }

  double distance(const Eigen::VectorXd& p, const Eigen::VectorXd& q) const {
    require_size(p, dim_ + 1, "SphereN.distance", "p");
    require_size(q, dim_ + 1, "SphereN.distance", "q");
    return std::visit([&](const auto& m) { return m.distance(p, q); }, *impl_);
  }

  Eigen::VectorXd geodesic(const Eigen::VectorXd& p, const Eigen::VectorXd& q, double t) const {
    require_size(p, dim_ + 1, "SphereN.geodesic", "p");
    require_size(q, dim_ + 1, "SphereN.geodesic", "q");
    return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.geodesic(p, q, t); },
                      *impl_);
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
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.exp(p, v); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.log(p, q); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                 const Eigen::VectorXd& v) -> double {
          return std::visit([&](const auto& m) { return m.inner(p, u, v); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          return std::visit([&](const auto& m) { return m.norm(p, v); }, *shared);
        },
        [shared](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.project(p, v); },
                            *shared);
        },
        [shared]() { return std::visit([](const auto& m) { return m.unit_cube_dim(); }, *shared); },
        [shared](Eigen::Ref<const Eigen::VectorXd> u) -> Eigen::VectorXd {
          return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.from_unit_cube(u); },
                            *shared);
        }};
    dm.set_geodesic_fn([shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q,
                                const double t) -> Eigen::VectorXd {
      return std::visit([&](const auto& m) -> Eigen::VectorXd { return m.geodesic(p, q, t); },
                        *shared);
    });
    dm.set_distance_fn([shared](const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
      return std::visit([&](const auto& m) { return m.distance(p, q); }, *shared);
    });
    dm.set_sampler_fn([shared]() {
      return std::visit([](const auto& m) { return DynamicSampler(m.sampler()); }, *shared);
    });
    dm.set_riemannian_log(
        std::visit([](const auto& m) { return geodex::is_riemannian_log(m); }, *shared));
    return dm;
  }

  std::string repr() const {
    return "Sphere(dim=" + std::to_string(dim_) + ", retraction='" + retraction_name_ + "')";
  }

 private:
  /// The wrapped manifold, shared with every DynamicManifold this wrapper hands out. A plan
  /// advances this manifold's own sampler.
  std::shared_ptr<V> impl_;
  int dim_;
  std::string retraction_name_;
};

}  // namespace geodex::python

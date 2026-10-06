/// @file py_product.hpp
/// @brief Python wrapper for the Riemannian product of several manifolds.
///
/// @details Composed at runtime from a list of type-erased sub-manifolds. Points and
/// tangents concatenate the blocks' points and tangents. exp, log and geodesic act
/// block-wise, and distance is the L2 combination of the block distances. Point and tangent
/// segments use separate offset tables. A block's point size can differ from its tangent
/// size, as in SO3 with a 4-vector point and a 3-vector tangent. When every block exposes
/// from_unit_cube, one joint sampler spans the summed unit cube and keeps the whole product
/// well distributed.

#pragma once

#include <cmath>
#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "dynamic_manifold.hpp"
#include "sampler_kind.hpp"

namespace geodex::python {

class PyProduct {
 public:
  /// @brief Compose a product from pre-extracted sub-manifolds.
  explicit PyProduct(std::vector<DynamicManifold> blocks)
      : n_blocks_(blocks.size()), sampler_(std::make_shared<DynamicSampler>()) {
    std::vector<int> point_off, point_size, tan_off, tan_size, cube_off, cube_size;
    int point_dim = 0, tan_dim = 0, cube_dim = 0, intrinsic_dim = 0;
    bool all_have_cube = !blocks.empty();
    for (const auto& b : blocks) {
      // A point of the block, taken through the cube map when there is one. The probe does
      // not advance the block's sampler, which its Python object shares.
      const Eigen::VectorXd rp =
          b.has_from_unit_cube()
              ? b.from_unit_cube(Eigen::VectorXd::Constant(b.unit_cube_dim(), 0.5))
              : b.random_point();
      const int ps = static_cast<int>(rp.size());
      // Probe the tangent ambient size via log(p, p) (the zero tangent at p).
      const int ts = static_cast<int>(b.log(rp, rp).size());
      const bool has_cube = b.has_from_unit_cube();
      all_have_cube = all_have_cube && has_cube;
      const int cs = has_cube ? b.unit_cube_dim() : 0;
      point_off.push_back(point_dim);
      point_size.push_back(ps);
      point_dim += ps;
      tan_off.push_back(tan_dim);
      tan_size.push_back(ts);
      tan_dim += ts;
      cube_off.push_back(cube_dim);
      cube_size.push_back(cs);
      cube_dim += cs;
      intrinsic_dim += b.dim();
    }

    bool riemannian_log = true;
    for (const auto& b : blocks) riemannian_log = riemannian_log && b.has_riemannian_log_runtime();
    auto shared = std::make_shared<std::vector<DynamicManifold>>(std::move(blocks));
    const auto po = point_off, ps = point_size, to = tan_off, ts = tan_size;
    const auto co = cube_off, cs = cube_size;
    const std::size_t n = n_blocks_;

    auto dim_fn = [intrinsic_dim]() { return intrinsic_dim; };
    auto exp_fn = [shared, po, ps, to, ts, point_dim, n](
                      const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
      Eigen::VectorXd out(point_dim);
      for (std::size_t i = 0; i < n; ++i)
        out.segment(po[i], ps[i]) = (*shared)[i].exp(p.segment(po[i], ps[i]), v.segment(to[i], ts[i]));
      return out;
    };
    auto log_fn = [shared, po, ps, to, ts, tan_dim, n](
                      const Eigen::VectorXd& p, const Eigen::VectorXd& q) -> Eigen::VectorXd {
      Eigen::VectorXd out(tan_dim);
      for (std::size_t i = 0; i < n; ++i)
        out.segment(to[i], ts[i]) = (*shared)[i].log(p.segment(po[i], ps[i]), q.segment(po[i], ps[i]));
      return out;
    };
    auto inner_fn = [shared, po, ps, to, ts, n](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                                const Eigen::VectorXd& v) -> double {
      double acc = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        acc += (*shared)[i].inner(p.segment(po[i], ps[i]), u.segment(to[i], ts[i]),
                                  v.segment(to[i], ts[i]));
      return acc;
    };
    auto norm_fn = [inner_fn](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
      return std::sqrt(inner_fn(p, v, v));
    };
    auto geodesic_fn = [shared, po, ps, point_dim, n](const Eigen::VectorXd& p,
                                                      const Eigen::VectorXd& q,
                                                      const double t) -> Eigen::VectorXd {
      Eigen::VectorXd out(point_dim);
      for (std::size_t i = 0; i < n; ++i)
        out.segment(po[i], ps[i]) =
            (*shared)[i].geodesic(p.segment(po[i], ps[i]), q.segment(po[i], ps[i]), t);
      return out;
    };
    auto distance_fn = [shared, po, ps, n](const Eigen::VectorXd& p,
                                           const Eigen::VectorXd& q) -> double {
      double s2 = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double d = (*shared)[i].distance(p.segment(po[i], ps[i]), q.segment(po[i], ps[i]));
        s2 += d * d;
      }
      return std::sqrt(s2);
    };
    auto project_fn = [shared, po, ps, to, ts, tan_dim, n](
                          const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> Eigen::VectorXd {
      Eigen::VectorXd out(tan_dim);
      for (std::size_t i = 0; i < n; ++i) {
        const auto& blk = (*shared)[i];
        const Eigen::VectorXd vi = v.segment(to[i], ts[i]);
        out.segment(to[i], ts[i]) = blk.has_project() ? blk.project(p.segment(po[i], ps[i]), vi) : vi;
      }
      return out;
    };

    if (all_have_cube) {
      // One joint sampler over the summed cube keeps the product well distributed.
      // Per-block low-discrepancy samplers would reuse the small primes and collapse
      // the joint samples toward a diagonal of the product space.
      auto from_cube = [shared, po, ps, co, cs, point_dim, n](
                           Eigen::Ref<const Eigen::VectorXd> u) -> Eigen::VectorXd {
        Eigen::VectorXd out(point_dim);
        for (std::size_t i = 0; i < n; ++i)
          out.segment(po[i], ps[i]) = (*shared)[i].from_unit_cube(u.segment(co[i], cs[i]));
        return out;
      };
      auto sampler = sampler_;
      auto rand = [sampler, cube_dim, from_cube]() -> Eigen::VectorXd {
        Eigen::VectorXd u(cube_dim);
        sampler->sample(cube_dim, u);
        return from_cube(u);
      };
      impl_ = DynamicManifold{dim_fn,  rand,      exp_fn,
                              log_fn,  inner_fn,  norm_fn,
                              project_fn, [cube_dim]() { return cube_dim; }, from_cube};
      impl_.set_sampler_fn([sampler]() { return *sampler; });
      impl_.set_geodesic_fn(geodesic_fn);
      impl_.set_distance_fn(distance_fn);
      joint_ = true;
      impl_.set_riemannian_log(riemannian_log);
      impl_.set_sizes(point_dim, tan_dim);
    } else {
      // When a block does not have a cube map, sample each block independently. The samples
      // are valid points but do not form a joint low-discrepancy sequence.
      auto rand = [shared, po, ps, point_dim, n]() -> Eigen::VectorXd {
        Eigen::VectorXd out(point_dim);
        for (std::size_t i = 0; i < n; ++i) out.segment(po[i], ps[i]) = (*shared)[i].random_point();
        return out;
      };
      impl_ = DynamicManifold{dim_fn, rand, exp_fn, log_fn, inner_fn, norm_fn, project_fn};
      impl_.set_geodesic_fn(geodesic_fn);
      impl_.set_distance_fn(distance_fn);
      joint_ = false;
      impl_.set_riemannian_log(riemannian_log);
      impl_.set_sizes(point_dim, tan_dim);
    }
  }

  int dim() const { return impl_.dim(); }
  Eigen::VectorXd random_point() const { return impl_.random_point(); }

  /// @brief Reseed the joint sampler, which exists when every block has a cube map.
  void seed(std::uint64_t s) {
    if (joint_) sampler_->seed(s);
  }

  /// @brief Switch the kind of the joint sampler, which exists when every block has a cube map.
  void set_sampler(const std::string& kind) {
    if (joint_) *sampler_ = make_sampler(kind);
  }

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

  DynamicManifold to_dynamic_manifold() const { return impl_; }

  std::string repr() const { return "Product(" + std::to_string(n_blocks_) + " manifolds)"; }

 private:
  DynamicManifold impl_;
  std::size_t n_blocks_;
  std::shared_ptr<DynamicSampler> sampler_;  ///< Joint sampler when all blocks have a cube map.
  bool joint_ = false;
};

}  // namespace geodex::python

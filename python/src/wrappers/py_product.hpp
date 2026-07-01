/// @file py_product.hpp
/// @brief Python wrapper for the Riemannian product of several manifolds.
///
/// @details A product manifold is composed at runtime from a list of already
/// type-erased sub-manifolds (each a `DynamicManifold`). Points and tangents are
/// the block-concatenation of the sub-manifold points/tangents; exp/log/geodesic
/// act block-wise and distance is the L2 combination of the block distances.
///
/// Point and tangent segments are tracked with *separate* offset tables, because
/// a block's point size may differ from its tangent size (e.g. an embedded
/// `Sphere` block: 3-vector point, 3-vector ambient tangent but intrinsic dim 2;
/// a genuine-Lie `SO3` block: 4-vector point, 3-vector tangent). For the common
/// mobile-manipulator products (R^n x SE2, R^3 x SO3, ...) the two tables
/// coincide.

#pragma once

#include <cmath>

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "dynamic_manifold.hpp"

namespace geodex::python {

class PyProduct {
 public:
  /// @brief Compose a product from pre-extracted sub-manifolds.
  explicit PyProduct(std::vector<DynamicManifold> blocks) : n_blocks_(blocks.size()) {
    std::vector<int> point_off, point_size, tan_off, tan_size;
    int point_dim = 0, tan_dim = 0, intrinsic_dim = 0;
    for (const auto& b : blocks) {
      const Eigen::VectorXd rp = b.random_point();
      const int ps = static_cast<int>(rp.size());
      // Probe the tangent ambient size via log(p, p) — the zero tangent at p,
      // whose length is the block's tangent representation size (== dim for
      // minimal-tangent Lie groups, > dim for embedded manifolds like Sphere).
      const int ts = static_cast<int>(b.log(rp, rp).size());
      point_off.push_back(point_dim);
      point_size.push_back(ps);
      point_dim += ps;
      tan_off.push_back(tan_dim);
      tan_size.push_back(ts);
      tan_dim += ts;
      intrinsic_dim += b.dim();
    }

    auto shared = std::make_shared<std::vector<DynamicManifold>>(std::move(blocks));
    const auto po = point_off, ps = point_size, to = tan_off, ts = tan_size;
    const std::size_t n = n_blocks_;

    impl_ = DynamicManifold{
        [intrinsic_dim]() { return intrinsic_dim; },
        [shared, po, ps, point_dim, n]() -> Eigen::VectorXd {
          Eigen::VectorXd out(point_dim);
          for (std::size_t i = 0; i < n; ++i) out.segment(po[i], ps[i]) = (*shared)[i].random_point();
          return out;
        },
        [shared, po, ps, to, ts, point_dim, n](const Eigen::VectorXd& p,
                                               const Eigen::VectorXd& v) -> Eigen::VectorXd {
          Eigen::VectorXd out(point_dim);
          for (std::size_t i = 0; i < n; ++i)
            out.segment(po[i], ps[i]) =
                (*shared)[i].exp(p.segment(po[i], ps[i]), v.segment(to[i], ts[i]));
          return out;
        },
        [shared, po, ps, to, ts, tan_dim, n](const Eigen::VectorXd& p,
                                             const Eigen::VectorXd& q) -> Eigen::VectorXd {
          Eigen::VectorXd out(tan_dim);
          for (std::size_t i = 0; i < n; ++i)
            out.segment(to[i], ts[i]) =
                (*shared)[i].log(p.segment(po[i], ps[i]), q.segment(po[i], ps[i]));
          return out;
        },
        [shared, po, ps, to, ts, n](const Eigen::VectorXd& p, const Eigen::VectorXd& u,
                                    const Eigen::VectorXd& v) -> double {
          double acc = 0.0;
          for (std::size_t i = 0; i < n; ++i)
            acc += (*shared)[i].inner(p.segment(po[i], ps[i]), u.segment(to[i], ts[i]),
                                      v.segment(to[i], ts[i]));
          return acc;
        },
        [shared, po, ps, to, ts, n](const Eigen::VectorXd& p, const Eigen::VectorXd& v) -> double {
          double acc = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            const double ni = (*shared)[i].norm(p.segment(po[i], ps[i]), v.segment(to[i], ts[i]));
            acc += ni * ni;
          }
          return std::sqrt(acc);
        },
        [shared, po, ps, to, ts, tan_dim, n](const Eigen::VectorXd& p,
                                             const Eigen::VectorXd& v) -> Eigen::VectorXd {
          Eigen::VectorXd out(tan_dim);
          for (std::size_t i = 0; i < n; ++i) {
            const auto& blk = (*shared)[i];
            const Eigen::VectorXd vi = v.segment(to[i], ts[i]);
            out.segment(to[i], ts[i]) =
                blk.has_project() ? blk.project(p.segment(po[i], ps[i]), vi) : vi;
          }
          return out;
        }};
  }

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

  DynamicManifold to_dynamic_manifold() const { return impl_; }

  std::string repr() const { return "Product(" + std::to_string(n_blocks_) + " manifolds)"; }

 private:
  DynamicManifold impl_;
  std::size_t n_blocks_;
};

}  // namespace geodex::python

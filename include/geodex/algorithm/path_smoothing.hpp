/// @file path_smoothing.hpp
/// @brief Metric-aware path smoothing: shortcutting + L-BFGS energy minimization.
///
/// @details Two-phase path improvement for planner output:
/// 1. **Shortcutting**: randomly remove redundant vertices while maintaining or
///    reducing metric energy, with collision checking along manifold geodesics.
/// 2. **L-BFGS smoothing**: minimize the discrete path energy with collision
///    constraints in the Armijo line search.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <geodex/algorithm/interpolation.hpp>
#include <geodex/core/concepts.hpp>
#include <geodex/core/metric.hpp>
#include <random>
#include <vector>

namespace geodex {

// ---------------------------------------------------------------------------
// Settings and result types
// ---------------------------------------------------------------------------

/// @brief Settings for metric-aware path smoothing.
struct PathSmoothingSettings {
  // --- Shortcutting phase ---
  int max_shortcut_attempts = 200;  ///< Random shortcut attempts.
  int edge_collision_samples = 10;  ///< Minimum geodesic samples per edge for collision.
  double collision_resolution = 0.1;  ///< Max spacing (meters) between collision checks.

  // --- L-BFGS energy smoothing phase ---
  int lbfgs_target_segments = 64;   ///< Upsample resolution for L-BFGS.
  int lbfgs_max_iterations = 200;   ///< Max L-BFGS iterations.
  double grad_tol = 1e-8;           ///< Convergence: gradient infinity norm.
  double energy_tol = 1e-10;        ///< Convergence: relative energy change.
  double fd_epsilon = 1e-7;         ///< Finite difference step for dM/dq.
  int lbfgs_memory = 7;             ///< L-BFGS history size.
  double armijo_c = 1e-4;           ///< Armijo sufficient decrease parameter.
  double max_displacement = 0.0;    ///< Trust region radius per waypoint (0 = disabled).

  // --- Discrete geodesic densification ---
  InterpolationSettings interp;     ///< Settings for discrete_geodesic between waypoints.
};

/// @brief Result of path smoothing.
template <typename PointT>
struct PathSmoothingResult {
  std::vector<PointT> path;    ///< Smoothed path (including endpoints).
  double energy = 0.0;         ///< Discrete energy of the result.
  double distance = 0.0;       ///< Geodesic distance estimate (sqrt(energy)).
  int vertices_removed = 0;    ///< Vertices removed in shortcutting phase.
  int smooth_iterations = 0;   ///< L-BFGS iterations in smoothing phase.
  bool collision_free = true;  ///< Whether final path passed validation.
};

namespace detail {

// ---------------------------------------------------------------------------
// Mass matrix synthesis from manifold inner product
// ---------------------------------------------------------------------------

/// @brief Build the Gram matrix (mass matrix) from the manifold's inner product.
///
/// @details Evaluates \f$ G_{ij} = \langle e_i, e_j \rangle_q \f$ where
/// \f$ e_i \f$ are the standard basis vectors.
template <RiemannianManifold M>
Eigen::MatrixXd gram_matrix(const M& manifold, const Eigen::VectorXd& q_vec) {
  const int d = manifold.dim();
  typename M::Point q;
  if constexpr (M::Point::SizeAtCompileTime == Eigen::Dynamic) {
    q.resize(d);
  }
  for (int i = 0; i < d; ++i) q[i] = q_vec[i];

  if constexpr (HasBatchInnerMatrix<M>) {
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(d, d);
    Eigen::MatrixXd G = manifold.inner_matrix(q, I, I);
    return 0.5 * (G + G.transpose());  // symmetrize against FP noise
  } else {
    Eigen::MatrixXd G(d, d);
    for (int i = 0; i < d; ++i) {
      typename M::Tangent ei;
      if constexpr (M::Tangent::SizeAtCompileTime == Eigen::Dynamic) {
        ei = M::Tangent::Zero(d);
      } else {
        ei = M::Tangent::Zero();
      }
      ei[i] = 1.0;
      for (int j = i; j < d; ++j) {
        typename M::Tangent ej;
        if constexpr (M::Tangent::SizeAtCompileTime == Eigen::Dynamic) {
          ej = M::Tangent::Zero(d);
        } else {
          ej = M::Tangent::Zero();
        }
        ej[j] = 1.0;
        G(i, j) = G(j, i) = manifold.inner(q, ei, ej);
      }
    }
    return G;
  }
}

// ---------------------------------------------------------------------------
// L-BFGS infrastructure (ported from geodesic_bvp.hpp)
// ---------------------------------------------------------------------------

/// @brief Compute energy of a single segment using manifold log.
template <RiemannianManifold M>
double segment_energy(const M& manifold, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
  const int d = manifold.dim();
  typename M::Point pa, pb;
  if constexpr (M::Point::SizeAtCompileTime == Eigen::Dynamic) {
    pa.resize(d);
    pb.resize(d);
  }
  for (int i = 0; i < d; ++i) {
    pa[i] = a[i];
    pb[i] = b[i];
  }
  auto v = manifold.log(pa, pb);
  return manifold.inner(pa, v, v);
}

/// @brief Compute discrete path energy using manifold log: \f$ N \sum_k \|log(q_k, q_{k+1})\|^2 \f$.
template <RiemannianManifold M>
double compute_energy(const M& manifold, const std::vector<Eigen::VectorXd>& path) {
  const int N = static_cast<int>(path.size()) - 1;
  double E = 0.0;
  for (int k = 0; k < N; ++k) {
    E += segment_energy(manifold, path[k], path[k + 1]);
  }
  return N * E;
}

/// @brief Compute energy gradient via FD, using manifold log for energy.
template <RiemannianManifold M>
Eigen::VectorXd compute_gradient(const M& manifold, std::vector<Eigen::VectorXd>& path,
                                 double fd_eps) {
  const int N = static_cast<int>(path.size()) - 1;
  const int n = static_cast<int>(path[0].size());
  const int n_interior = N - 1;
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(n_interior * n);

  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < n; ++j) {
      // Only segments (i-1,i) and (i,i+1) depend on path[i].
      path[i][j] += fd_eps;
      double E_plus = segment_energy(manifold, path[i - 1], path[i]) +
                      segment_energy(manifold, path[i], path[i + 1]);
      path[i][j] -= 2.0 * fd_eps;
      double E_minus = segment_energy(manifold, path[i - 1], path[i]) +
                       segment_energy(manifold, path[i], path[i + 1]);
      path[i][j] += fd_eps;  // restore

      grad[(i - 1) * n + j] = N * (E_plus - E_minus) / (2.0 * fd_eps);
    }

    // grad.segment((i - 1) * n, n) already set above.
  }
  return grad;
}

inline Eigen::VectorXd pack_interior(const std::vector<Eigen::VectorXd>& path) {
  const int N = static_cast<int>(path.size()) - 1;
  const int n = static_cast<int>(path[0].size());
  Eigen::VectorXd x((N - 1) * n);
  for (int i = 1; i < N; ++i) {
    x.segment((i - 1) * n, n) = path[i];
  }
  return x;
}

inline void unpack_interior(const Eigen::VectorXd& x, std::vector<Eigen::VectorXd>& path) {
  const int N = static_cast<int>(path.size()) - 1;
  const int n = static_cast<int>(path[0].size());
  for (int i = 1; i < N; ++i) {
    path[i] = x.segment((i - 1) * n, n);
  }
}

/// @brief L-BFGS two-loop recursion for search direction.
inline Eigen::VectorXd lbfgs_direction(const Eigen::VectorXd& grad,
                                       const std::vector<Eigen::VectorXd>& s_hist,
                                       const std::vector<Eigen::VectorXd>& y_hist) {
  const int m = static_cast<int>(s_hist.size());
  if (m == 0) return -grad;

  Eigen::VectorXd q = grad;
  std::vector<double> alpha(m), rho(m);

  for (int i = m - 1; i >= 0; --i) {
    rho[i] = 1.0 / y_hist[i].dot(s_hist[i]);
    alpha[i] = rho[i] * s_hist[i].dot(q);
    q -= alpha[i] * y_hist[i];
  }

  double gamma = s_hist.back().dot(y_hist.back()) / y_hist.back().dot(y_hist.back());
  Eigen::VectorXd r = gamma * q;

  for (int i = 0; i < m; ++i) {
    double beta = rho[i] * y_hist[i].dot(r);
    r += (alpha[i] - beta) * s_hist[i];
  }
  return -r;
}

/// @brief Insert geodesic midpoints to double resolution.
template <RiemannianManifold M>
std::vector<Eigen::VectorXd> upsample(const M& manifold,
                                       const std::vector<Eigen::VectorXd>& path) {
  const int d = manifold.dim();
  std::vector<Eigen::VectorXd> refined;
  refined.reserve(2 * path.size() - 1);
  for (std::size_t i = 0; i < path.size(); ++i) {
    refined.push_back(path[i]);
    if (i + 1 < path.size()) {
      typename M::Point pa, pb;
      if constexpr (M::Point::SizeAtCompileTime == Eigen::Dynamic) {
        pa.resize(d);
        pb.resize(d);
      }
      for (int j = 0; j < d; ++j) {
        pa[j] = path[i][j];
        pb[j] = path[i + 1][j];
      }
      auto mid = manifold.geodesic(pa, pb, 0.5);
      Eigen::VectorXd mid_vec(d);
      for (int j = 0; j < d; ++j) mid_vec[j] = mid[j];
      refined.push_back(mid_vec);
    }
  }
  return refined;
}

/// @brief Collision-constrained Armijo line search.
template <RiemannianManifold M, typename ValidityFn>
double armijo_constrained(const M& manifold, const ValidityFn& validity_fn,
                          std::vector<Eigen::VectorXd>& path, const Eigen::VectorXd& x,
                          const Eigen::VectorXd& dir, const Eigen::VectorXd& grad, double f0,
                          double armijo_c, int edge_samples, double max_disp,
                          const Eigen::VectorXd& ref_x) {
  double step = 1.0;
  const double slope = grad.dot(dir);
  if (slope >= 0) return 0.0;

  const int N = static_cast<int>(path.size()) - 1;
  const int n = static_cast<int>(path[0].size());
  const int n_interior = N - 1;

  for (int iter = 0; iter < 30; ++iter) {
    Eigen::VectorXd x_new = x + step * dir;

    // Trust region check.
    if (max_disp > 0.0) {
      bool within = true;
      for (int k = 0; k < n_interior && within; ++k) {
        double disp = (x_new.segment(k * n, n) - ref_x.segment(k * n, n)).norm();
        if (disp > max_disp) within = false;
      }
      if (!within) {
        step *= 0.5;
        continue;
      }
    }

    unpack_interior(x_new, path);

    // Point collision check.
    bool valid = true;
    for (const auto& q : path) {
      typename M::Point p;
      if constexpr (M::Point::SizeAtCompileTime == Eigen::Dynamic) {
        p.resize(n);
      }
      for (int i = 0; i < n; ++i) p[i] = q[i];
      if (!validity_fn(p)) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      step *= 0.5;
      continue;
    }

    // Edge collision check via manifold geodesic.
    for (int k = 0; k < N && valid; ++k) {
      typename M::Point a, b;
      if constexpr (M::Point::SizeAtCompileTime == Eigen::Dynamic) {
        a.resize(n);
        b.resize(n);
      }
      for (int i = 0; i < n; ++i) {
        a[i] = path[k][i];
        b[i] = path[k + 1][i];
      }
      for (int s = 1; s <= edge_samples; ++s) {
        double t = static_cast<double>(s) / (edge_samples + 1);
        auto mid = manifold.geodesic(a, b, t);
        if (!validity_fn(mid)) {
          valid = false;
          break;
        }
      }
    }
    if (!valid) {
      step *= 0.5;
      continue;
    }

    // Armijo energy decrease.
    double f_new = compute_energy(manifold, path);
    if (f_new <= f0 + armijo_c * step * slope) {
      return step;
    }
    step *= 0.5;
  }

  unpack_interior(x, path);
  return 0.0;
}

/// @brief Collision-constrained L-BFGS energy minimization.
template <RiemannianManifold M, typename ValidityFn>
int optimize_constrained(const M& manifold, const ValidityFn& validity_fn,
                         std::vector<Eigen::VectorXd>& path,
                         const PathSmoothingSettings& settings,
                         const Eigen::VectorXd& ref_x) {
  const int N = static_cast<int>(path.size()) - 1;
  const int n = static_cast<int>(path[0].size());
  const int n_vars = (N - 1) * n;
  if (n_vars == 0) return 0;

  Eigen::VectorXd x = pack_interior(path);
  double f = compute_energy(manifold, path);
  Eigen::VectorXd grad = compute_gradient(manifold, path, settings.fd_epsilon);

  std::vector<Eigen::VectorXd> s_hist, y_hist;
  int iter = 0;

  for (; iter < settings.lbfgs_max_iterations; ++iter) {
    const double grad_norm = grad.cwiseAbs().maxCoeff();
    if (grad_norm < settings.grad_tol) break;

    Eigen::VectorXd dir = lbfgs_direction(grad, s_hist, y_hist);

    Eigen::VectorXd x_old = x;
    const double f_old = f;
    Eigen::VectorXd grad_old = grad;

    const double step = armijo_constrained(manifold, validity_fn, path, x, dir, grad, f,
                                           settings.armijo_c, settings.edge_collision_samples,
                                           settings.max_displacement, ref_x);
    if (step == 0.0) break;

    x = x_old + step * dir;
    unpack_interior(x, path);
    f = compute_energy(manifold, path);
    grad = compute_gradient(manifold, path, settings.fd_epsilon);

    if (std::abs(f - f_old) < settings.energy_tol * std::abs(f_old) && f_old > 0) break;

    // Update L-BFGS history.
    Eigen::VectorXd s = x - x_old;
    Eigen::VectorXd y = grad - grad_old;
    const double sy = s.dot(y);
    if (sy > 1e-16) {
      if (static_cast<int>(s_hist.size()) >= settings.lbfgs_memory) {
        s_hist.erase(s_hist.begin());
        y_hist.erase(y_hist.begin());
      }
      s_hist.push_back(std::move(s));
      y_hist.push_back(std::move(y));
    }
  }
  return iter;
}

// ---------------------------------------------------------------------------
// Shortcutting
// ---------------------------------------------------------------------------

/// @brief Metric-aware random shortcutting with collision checking along geodesics.
template <RiemannianManifold M, typename ValidityFn>
int shortcut(const M& manifold, const ValidityFn& validity_fn,
             std::vector<typename M::Point>& path, int min_edge_samples,
             double collision_resolution, int max_attempts) {
  int total_removed = 0;
  std::mt19937 rng(42);

  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    const int n = static_cast<int>(path.size());
    if (n <= 2) break;

    std::uniform_int_distribution<int> dist(0, n - 1);
    int i = dist(rng), j = dist(rng);
    if (i > j) std::swap(i, j);
    if (j - i < 2) continue;

    // Compare sub-path energy vs direct connection.
    double E_sub = 0.0;
    for (int k = i; k < j; ++k) {
      double d = manifold.distance(path[k], path[k + 1]);
      E_sub += d * d;
    }
    E_sub *= (j - i);

    double d_direct = manifold.distance(path[i], path[j]);
    double E_direct = d_direct * d_direct;

    if (E_direct >= E_sub) continue;

    // Scale collision checks with edge length to avoid missing obstacles.
    double coord_dist = (path[j] - path[i]).template head<2>().norm();
    int n_checks = std::max(min_edge_samples,
                            static_cast<int>(std::ceil(coord_dist / collision_resolution)));

    // Collision check along geodesic shortcut.
    bool valid = true;
    for (int s = 1; s <= n_checks; ++s) {
      double t = static_cast<double>(s) / (n_checks + 1);
      auto mid = manifold.geodesic(path[i], path[j], t);
      if (!validity_fn(mid)) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;

    // Accept: remove intermediate vertices.
    const int removed = j - i - 1;
    path.erase(path.begin() + i + 1, path.begin() + j);
    total_removed += removed;
  }
  return total_removed;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// @brief Smooth a collision-free path using metric-aware shortcutting and
///        collision-constrained L-BFGS energy minimization.
///
/// @details Two-phase pipeline:
/// 1. **Shortcutting**: randomly picks non-adjacent vertex pairs and replaces
///    the sub-path with a direct geodesic when the shortcut has lower energy
///    and is collision-free.
/// 2. **L-BFGS smoothing**: upsamples the path and minimizes discrete energy
///    \f$ N \sum_k d_k^\top M(m_k) d_k \f$ with collision constraints in the
///    Armijo line search.
///
/// @tparam M A type satisfying `RiemannianManifold`.
/// @tparam ValidityFn Callable with signature `bool(const Point&)`. Returns
///                    true if the configuration is collision-free.
/// @param manifold The manifold instance.
/// @param validity_fn Collision checker.
/// @param initial_path Collision-free path from planner (>= 2 waypoints).
/// @param settings Smoothing parameters.
/// @return PathSmoothingResult with smoothed path, energy, and diagnostics.
template <RiemannianManifold M, typename ValidityFn>
PathSmoothingResult<typename M::Point> smooth_path(const M& manifold,
                                                    const ValidityFn& validity_fn,
                                                    const std::vector<typename M::Point>& initial_path,
                                                    PathSmoothingSettings settings = {}) {
  using Point = typename M::Point;
  assert(initial_path.size() >= 2);

  PathSmoothingResult<Point> result;

  if (initial_path.size() < 3) {
    result.path = initial_path;
    return result;
  }

  const int d = manifold.dim();

  // Phase 1: Shortcutting — remove redundant vertices along manifold geodesics.
  std::vector<Point> shortcut_path = initial_path;
  result.vertices_removed = detail::shortcut(manifold, validity_fn, shortcut_path,
                                              settings.edge_collision_samples,
                                              settings.collision_resolution,
                                              settings.max_shortcut_attempts);

  // Convert to VectorXd.
  std::vector<Eigen::VectorXd> vpath(shortcut_path.size());
  for (std::size_t i = 0; i < shortcut_path.size(); ++i) {
    vpath[i] = Eigen::VectorXd(d);
    for (int j = 0; j < d; ++j) vpath[i][j] = shortcut_path[i][j];
  }

  // Upsample with manifold geodesic midpoints.
  while (static_cast<int>(vpath.size()) - 1 < settings.lbfgs_target_segments) {
    vpath = detail::upsample(manifold, vpath);
  }

  // Save reference for trust region — keeps result close to raw path.
  Eigen::VectorXd ref_x = detail::pack_interior(vpath);

  // L-BFGS energy minimization with collision constraints.
  result.smooth_iterations =
      detail::optimize_constrained(manifold, validity_fn, vpath, settings, ref_x);

  // Convert back.
  std::vector<Point> path(vpath.size());
  for (std::size_t i = 0; i < vpath.size(); ++i) {
    if constexpr (Point::SizeAtCompileTime == Eigen::Dynamic) {
      path[i].resize(d);
    }
    for (int j = 0; j < d; ++j) path[i][j] = vpath[i][j];
  }

  result.energy = 0.0;
  for (std::size_t k = 0; k + 1 < path.size(); ++k) {
    double dd = manifold.distance(path[k], path[k + 1]);
    result.energy += dd * dd;
  }
  result.energy *= static_cast<double>(path.size() - 1);
  result.distance = std::sqrt(std::max(result.energy, 0.0));
  result.path = std::move(path);

  return result;
}

}  // namespace geodex

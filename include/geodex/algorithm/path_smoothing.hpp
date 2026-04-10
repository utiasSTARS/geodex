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
  int edge_collision_samples = 10;  ///< Geodesic samples per edge for collision.

  // --- L-BFGS energy smoothing phase ---
  int lbfgs_target_segments = 64;   ///< Upsample resolution for L-BFGS.
  int lbfgs_max_iterations = 200;   ///< Max L-BFGS iterations.
  double grad_tol = 1e-8;           ///< Convergence: gradient infinity norm.
  double energy_tol = 1e-10;        ///< Convergence: relative energy change.
  double fd_epsilon = 1e-7;         ///< Finite difference step for dM/dq.
  int lbfgs_memory = 7;             ///< L-BFGS history size.
  double armijo_c = 1e-4;           ///< Armijo sufficient decrease parameter.
  double max_displacement = 0.0;    ///< Trust region radius per waypoint (0 = disabled).
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

/// @brief Compute discrete path energy: \f$ N \sum_k d_k^T M(m_k) d_k \f$.
template <RiemannianManifold M>
double compute_energy(const M& manifold, const std::vector<Eigen::VectorXd>& path) {
  const int N = static_cast<int>(path.size()) - 1;
  double E = 0.0;
  for (int k = 0; k < N; ++k) {
    Eigen::VectorXd mid = 0.5 * (path[k] + path[k + 1]);
    Eigen::VectorXd diff = path[k + 1] - path[k];
    Eigen::MatrixXd G = gram_matrix(manifold, mid);
    E += diff.dot(G * diff);
  }
  return N * E;
}

/// @brief Compute energy gradient w.r.t. interior waypoints.
template <RiemannianManifold M>
Eigen::VectorXd compute_gradient(const M& manifold, const std::vector<Eigen::VectorXd>& path,
                                 double fd_eps) {
  const int N = static_cast<int>(path.size()) - 1;
  const int n = static_cast<int>(path[0].size());
  const int n_interior = N - 1;
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(n_interior * n);

  // Precompute midpoints, diffs, mass matrices.
  std::vector<Eigen::VectorXd> mids(N);
  std::vector<Eigen::VectorXd> diffs(N);
  std::vector<Eigen::MatrixXd> Ms(N);
  for (int k = 0; k < N; ++k) {
    mids[k] = 0.5 * (path[k] + path[k + 1]);
    diffs[k] = path[k + 1] - path[k];
    Ms[k] = gram_matrix(manifold, mids[k]);
  }

  for (int i = 1; i < N; ++i) {
    Eigen::VectorXd gi = Eigen::VectorXd::Zero(n);

    // Analytic part: from quadratic form in segments (i-1,i) and (i,i+1).
    gi += 2.0 * N * (Ms[i - 1] * diffs[i - 1]);
    gi -= 2.0 * N * (Ms[i] * diffs[i]);

    // Metric sensitivity via FD: waypoint q_i affects midpoints of both segments.
    for (int j = 0; j < n; ++j) {
      // Segment (i-1, i)
      {
        Eigen::VectorXd mid_p = mids[i - 1], mid_m = mids[i - 1];
        mid_p[j] += fd_eps;
        mid_m[j] -= fd_eps;
        double dtMpd = diffs[i - 1].dot(gram_matrix(manifold, mid_p) * diffs[i - 1]);
        double dtMmd = diffs[i - 1].dot(gram_matrix(manifold, mid_m) * diffs[i - 1]);
        gi[j] += N * 0.5 * (dtMpd - dtMmd) / (2.0 * fd_eps);
      }
      // Segment (i, i+1)
      {
        Eigen::VectorXd mid_p = mids[i], mid_m = mids[i];
        mid_p[j] += fd_eps;
        mid_m[j] -= fd_eps;
        double dtMpd = diffs[i].dot(gram_matrix(manifold, mid_p) * diffs[i]);
        double dtMmd = diffs[i].dot(gram_matrix(manifold, mid_m) * diffs[i]);
        gi[j] += N * 0.5 * (dtMpd - dtMmd) / (2.0 * fd_eps);
      }
    }

    grad.segment((i - 1) * n, n) = gi;
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

/// @brief Insert midpoints to double resolution.
inline std::vector<Eigen::VectorXd> upsample(const std::vector<Eigen::VectorXd>& path) {
  std::vector<Eigen::VectorXd> refined;
  refined.reserve(2 * path.size() - 1);
  for (std::size_t i = 0; i < path.size(); ++i) {
    refined.push_back(path[i]);
    if (i + 1 < path.size()) {
      refined.push_back(0.5 * (path[i] + path[i + 1]));
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
             std::vector<typename M::Point>& path, int max_attempts, int edge_samples) {
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

    // Collision check along geodesic shortcut.
    bool valid = true;
    for (int s = 1; s <= edge_samples; ++s) {
      double t = static_cast<double>(s) / (edge_samples + 1);
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

  // --- Phase 1: Metric-aware shortcutting ---
  std::vector<Point> path = initial_path;
  result.vertices_removed =
      detail::shortcut(manifold, validity_fn, path, settings.max_shortcut_attempts,
                       settings.edge_collision_samples);

  // --- Phase 2: L-BFGS energy smoothing ---
  if (path.size() >= 3) {
    const int d = manifold.dim();

    // Convert to VectorXd for L-BFGS.
    std::vector<Eigen::VectorXd> vpath(path.size());
    for (std::size_t i = 0; i < path.size(); ++i) {
      vpath[i] = Eigen::VectorXd(d);
      for (int j = 0; j < d; ++j) vpath[i][j] = path[i][j];
    }

    // Upsample to target resolution.
    while (static_cast<int>(vpath.size()) - 1 < settings.lbfgs_target_segments) {
      vpath = detail::upsample(vpath);
    }

    Eigen::VectorXd ref_x = detail::pack_interior(vpath);

    result.smooth_iterations =
        detail::optimize_constrained(manifold, validity_fn, vpath, settings, ref_x);

    // Convert back.
    path.resize(vpath.size());
    for (std::size_t i = 0; i < vpath.size(); ++i) {
      if constexpr (Point::SizeAtCompileTime == Eigen::Dynamic) {
        path[i].resize(d);
      }
      for (int j = 0; j < d; ++j) path[i][j] = vpath[i][j];
    }
  }

  // --- Final validation ---
  result.collision_free = true;
  for (const auto& q : path) {
    if (!validity_fn(q)) {
      result.collision_free = false;
      break;
    }
  }
  if (result.collision_free) {
    const int N = static_cast<int>(path.size()) - 1;
    for (int k = 0; k < N; ++k) {
      for (int s = 1; s <= settings.edge_collision_samples; ++s) {
        double t = static_cast<double>(s) / (settings.edge_collision_samples + 1);
        auto mid = manifold.geodesic(path[k], path[k + 1], t);
        if (!validity_fn(mid)) {
          result.collision_free = false;
          break;
        }
      }
      if (!result.collision_free) break;
    }
  }

  // Compute final energy via manifold distance.
  result.energy = 0.0;
  {
    const int N = static_cast<int>(path.size()) - 1;
    for (int k = 0; k < N; ++k) {
      double d = manifold.distance(path[k], path[k + 1]);
      result.energy += d * d;
    }
    result.energy *= N;
  }
  result.distance = std::sqrt(std::max(result.energy, 0.0));
  result.path = std::move(path);

  return result;
}

}  // namespace geodex

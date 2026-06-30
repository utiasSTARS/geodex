/// @file simplify_path.hpp
/// @brief Energy-aware, collision-constrained path simplification on a Riemannian manifold.
///
/// @details Two-phase pipeline for collision-free planner output:
///   1. **Energy-aware shortcutting**: random non-adjacent waypoint pairs are
///      tried as direct connections; a shortcut is accepted only when it
///      strictly lowers the discrete Dirichlet energy and stays collision-free.
///   2. **Collision-constrained smoothing**: L-BFGS minimization of the same
///      Dirichlet energy with point and edge collision checks inside the Armijo
///      line search, plus an optional trust region.
///
/// Lives alongside `algorithm/path_smoothing.hpp`. Both are generic over
/// `RiemannianManifold`; `simplify_path` uses an **energy-based** shortcut
/// criterion where `path_smoothing.hpp` uses an arc-length criterion.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "geodex/core/concepts.hpp"

namespace geodex::algorithm {

/// @brief Settings for `simplify_path`.
struct SimplifyPathSettings {
  // --- Shortcutting phase ---
  int max_shortcut_attempts = 200;  ///< Random shortcut attempts.
  int edge_collision_samples = 10;  ///< Geodesic samples per edge for collision checks.
  uint64_t shortcut_seed = 42;      ///< RNG seed for shortcut sampling.

  // --- Smoothing phase ---
  int smooth_target_segments = 128;  ///< Upsample resolution for L-BFGS.
  int max_iter_per_level = 1000;     ///< Max L-BFGS iterations.
  double grad_tol = 1e-8;            ///< Convergence: gradient infinity norm.
  double energy_tol = 1e-10;         ///< Convergence: relative energy change.
  double fd_epsilon = 1e-7;          ///< Finite-difference step for the gradient.
  int lbfgs_memory = 7;              ///< L-BFGS history size.
  double armijo_c = 1e-4;            ///< Armijo sufficient decrease parameter.

  /// @brief Trust region radius per waypoint (coordinate units). 0 disables.
  ///
  /// @details Bounds how far each interior waypoint may drift from its initial
  /// (upsampled) position during smoothing.
  double max_displacement = 0.0;

  bool verbose = false;  ///< Print per-iteration info to stderr.
};

/// @brief Result of `simplify_path`.
template <typename PointT>
struct SimplifyPathResult {
  std::vector<PointT> path;    ///< Simplified path (including endpoints).
  double energy = 0.0;         ///< Discrete energy of the result.
  double distance = 0.0;       ///< \f$ \sqrt{\mathrm{energy}} \f$.
  int vertices_removed = 0;    ///< Vertices removed in the shortcutting phase.
  int smooth_iterations = 0;   ///< L-BFGS iterations in the smoothing phase.
  bool collision_free = true;  ///< Whether the final path passed validation.
};

namespace detail {

/// @brief Internal-only implementation of `simplify_path`. All helpers are
/// private static methods; only `run` is reachable from outside, and only via
/// the public `geodex::algorithm::simplify_path` free function.
struct SimplifyPathImpl {
 public:
  template <RiemannianManifold M, typename ValidityFn>
  static auto run(const M& manifold, const ValidityFn& validity_fn,
                  const std::vector<typename M::Point>& initial_path,
                  const SimplifyPathSettings& settings)
      -> SimplifyPathResult<typename M::Point> {
    using Point = typename M::Point;
    assert(initial_path.size() >= 2);

    SimplifyPathResult<Point> result;
    const int d = manifold.dim();

    // Phase 1: shortcutting on Point-valued path.
    std::vector<Point> path = initial_path;

    if (settings.verbose) {
      const double E0 = path_energy(manifold, path);
      std::cerr << "simplify_path: " << path.size() << " waypoints, initial E=" << E0
                << ", d=" << std::sqrt(std::max(E0, 0.0)) << "\n";
    }

    result.vertices_removed = shortcut(manifold, validity_fn, path, settings);

    if (settings.verbose) {
      const double E1 = path_energy(manifold, path);
      std::cerr << "  after shortcutting: " << path.size() << " waypoints ("
                << result.vertices_removed << " removed), E=" << E1
                << ", d=" << std::sqrt(std::max(E1, 0.0)) << "\n";
    }

    // Phase 2: collision-constrained L-BFGS on a flat VectorXd representation.
    if (path.size() >= 3) {
      std::vector<Eigen::VectorXd> vpath(path.size());
      for (std::size_t i = 0; i < path.size(); ++i) vpath[i] = point_to_vector(path[i], d);

      while (static_cast<int>(vpath.size()) - 1 < settings.smooth_target_segments) {
        vpath = upsample(manifold, vpath);
      }
      const Eigen::VectorXd reference_x = pack_interior(vpath);

      result.smooth_iterations =
          optimize_constrained(manifold, validity_fn, vpath, settings, reference_x);

      path.resize(vpath.size());
      for (std::size_t i = 0; i < vpath.size(); ++i) {
        path[i] = vector_to_point<Point>(vpath[i], d);
      }
    }

    // Final validation.
    result.collision_free = true;
    for (const auto& p : path) {
      if (!validity_fn(p)) {
        result.collision_free = false;
        break;
      }
    }
    if (result.collision_free) {
      const int N = static_cast<int>(path.size()) - 1;
      for (int k = 0; k < N; ++k) {
        if (!edge_collision_free(manifold, validity_fn, path[k], path[k + 1],
                                 settings.edge_collision_samples)) {
          result.collision_free = false;
          break;
        }
      }
    }

    result.energy = path_energy(manifold, path);
    result.distance = std::sqrt(std::max(result.energy, 0.0));
    result.path = std::move(path);

    if (settings.verbose) {
      std::cerr << "  result: E=" << result.energy << ", d=" << result.distance << ", "
                << result.smooth_iterations << " L-BFGS iters"
                << (result.collision_free ? " (collision-free)" : " (COLLISION)") << "\n";
    }
    return result;
  }

 private:
  // --- Point ↔ Eigen::VectorXd conversions -----------------------------------

  template <typename Point>
  static Eigen::VectorXd point_to_vector(const Point& p, int d) {
    Eigen::VectorXd v(d);
    for (int i = 0; i < d; ++i) v[i] = p[i];
    return v;
  }

  template <typename Point>
  static Point vector_to_point(const Eigen::VectorXd& v, int d) {
    Point p;
    if constexpr (Point::SizeAtCompileTime == Eigen::Dynamic) p.resize(d);
    for (int i = 0; i < d; ++i) p[i] = v[i];
    return p;
  }

  // --- Energies (left-endpoint quadrature) -----------------------------------

  template <RiemannianManifold M>
  static double segment_energy(const M& manifold, const typename M::Point& a,
                               const typename M::Point& b) {
    const auto v = manifold.log(a, b);
    return manifold.inner(a, v, v);
  }

  template <RiemannianManifold M>
  static double segment_energy_vec(const M& manifold, const Eigen::VectorXd& a,
                                   const Eigen::VectorXd& b) {
    const int d = manifold.dim();
    using Point = typename M::Point;
    return segment_energy(manifold, vector_to_point<Point>(a, d), vector_to_point<Point>(b, d));
  }

  template <RiemannianManifold M>
  static double path_energy(const M& manifold,
                            const std::vector<typename M::Point>& path) {
    const int N = static_cast<int>(path.size()) - 1;
    if (N <= 0) return 0.0;
    double E = 0.0;
    for (int k = 0; k < N; ++k) E += segment_energy(manifold, path[k], path[k + 1]);
    return N * E;
  }

  template <RiemannianManifold M>
  static double dirichlet_energy(const M& manifold,
                                 const std::vector<Eigen::VectorXd>& path) {
    const int N = static_cast<int>(path.size()) - 1;
    if (N <= 0) return 0.0;
    double E = 0.0;
    for (int k = 0; k < N; ++k) E += segment_energy_vec(manifold, path[k], path[k + 1]);
    return N * E;
  }

  template <RiemannianManifold M>
  static double subpath_energy(const M& manifold,
                               const std::vector<typename M::Point>& path, int i, int j) {
    const int K = j - i;
    if (K <= 0) return 0.0;
    double E = 0.0;
    for (int k = i; k < j; ++k) E += segment_energy(manifold, path[k], path[k + 1]);
    return K * E;
  }

  // --- Collision check along a manifold geodesic -----------------------------

  template <RiemannianManifold M, typename ValidityFn>
  static bool edge_collision_free(const M& manifold, const ValidityFn& validity_fn,
                                  const typename M::Point& a, const typename M::Point& b,
                                  int n_checks) {
    for (int k = 1; k <= n_checks; ++k) {
      const double t = static_cast<double>(k) / (n_checks + 1);
      if (!validity_fn(manifold.geodesic(a, b, t))) return false;
    }
    return true;
  }

  // --- Phase 1: energy-aware shortcutting ------------------------------------

  template <RiemannianManifold M, typename ValidityFn>
  static int shortcut(const M& manifold, const ValidityFn& validity_fn,
                      std::vector<typename M::Point>& path,
                      const SimplifyPathSettings& settings) {
    int total_removed = 0;
    std::mt19937 rng(settings.shortcut_seed);

    for (int attempt = 0; attempt < settings.max_shortcut_attempts; ++attempt) {
      const int n = static_cast<int>(path.size());
      if (n <= 2) break;

      std::uniform_int_distribution<int> dist(0, n - 1);
      int i = dist(rng), j = dist(rng);
      if (i > j) std::swap(i, j);
      if (j - i < 2) continue;

      const double E_subpath = subpath_energy(manifold, path, i, j);
      const double E_shortcut = segment_energy(manifold, path[i], path[j]);
      if (E_shortcut >= E_subpath) continue;

      if (!validity_fn(path[i]) || !validity_fn(path[j])) continue;
      if (!edge_collision_free(manifold, validity_fn, path[i], path[j],
                               settings.edge_collision_samples)) {
        continue;
      }

      const int removed = j - i - 1;
      path.erase(path.begin() + i + 1, path.begin() + j);
      total_removed += removed;

      if (settings.verbose) {
        std::cerr << "  shortcut: removed " << removed << " vertices between [" << i << ", " << j
                  << "], E: " << E_subpath << " -> " << E_shortcut << "\n";
      }
    }
    return total_removed;
  }

  // --- L-BFGS infrastructure (interior-only, flat representation) ------------

  static Eigen::VectorXd pack_interior(const std::vector<Eigen::VectorXd>& path) {
    const int N = static_cast<int>(path.size()) - 1;
    const int n = static_cast<int>(path[0].size());
    Eigen::VectorXd x((N - 1) * n);
    for (int i = 1; i < N; ++i) x.segment((i - 1) * n, n) = path[i];
    return x;
  }

  static void unpack_interior(const Eigen::VectorXd& x, std::vector<Eigen::VectorXd>& path) {
    const int N = static_cast<int>(path.size()) - 1;
    const int n = static_cast<int>(path[0].size());
    for (int i = 1; i < N; ++i) path[i] = x.segment((i - 1) * n, n);
  }

  static Eigen::VectorXd lbfgs_two_loop(const Eigen::VectorXd& grad,
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
    const double gamma =
        s_hist.back().dot(y_hist.back()) / y_hist.back().dot(y_hist.back());
    Eigen::VectorXd r = gamma * q;
    for (int i = 0; i < m; ++i) {
      const double beta = rho[i] * y_hist[i].dot(r);
      r += (alpha[i] - beta) * s_hist[i];
    }
    return -r;
  }

  template <RiemannianManifold M>
  static Eigen::VectorXd compute_gradient(const M& manifold,
                                          std::vector<Eigen::VectorXd>& path, double fd_eps) {
    const int N = static_cast<int>(path.size()) - 1;
    const int n = static_cast<int>(path[0].size());
    Eigen::VectorXd grad = Eigen::VectorXd::Zero((N - 1) * n);

    for (int i = 1; i < N; ++i) {
      for (int j = 0; j < n; ++j) {
        path[i][j] += fd_eps;
        const double E_plus = segment_energy_vec(manifold, path[i - 1], path[i]) +
                              segment_energy_vec(manifold, path[i], path[i + 1]);
        path[i][j] -= 2.0 * fd_eps;
        const double E_minus = segment_energy_vec(manifold, path[i - 1], path[i]) +
                               segment_energy_vec(manifold, path[i], path[i + 1]);
        path[i][j] += fd_eps;  // restore
        grad[(i - 1) * n + j] = N * (E_plus - E_minus) / (2.0 * fd_eps);
      }
    }
    return grad;
  }

  // --- Phase-2 path refinement via manifold geodesic midpoints ---------------

  template <RiemannianManifold M>
  static std::vector<Eigen::VectorXd> upsample(const M& manifold,
                                               const std::vector<Eigen::VectorXd>& path) {
    const int d = manifold.dim();
    using Point = typename M::Point;
    std::vector<Eigen::VectorXd> refined;
    refined.reserve(2 * path.size() - 1);
    for (std::size_t i = 0; i < path.size(); ++i) {
      refined.push_back(path[i]);
      if (i + 1 < path.size()) {
        const Point pa = vector_to_point<Point>(path[i], d);
        const Point pb = vector_to_point<Point>(path[i + 1], d);
        refined.push_back(point_to_vector(manifold.geodesic(pa, pb, 0.5), d));
      }
    }
    return refined;
  }

  // --- Phase 2: collision-constrained L-BFGS ---------------------------------

  template <RiemannianManifold M, typename ValidityFn>
  static double armijo_constrained(const M& manifold, const ValidityFn& validity_fn,
                                   std::vector<Eigen::VectorXd>& path,
                                   const Eigen::VectorXd& x, const Eigen::VectorXd& dir,
                                   const Eigen::VectorXd& grad, double f0,
                                   const SimplifyPathSettings& settings,
                                   const Eigen::VectorXd& ref_x) {
    using Point = typename M::Point;
    double step = 1.0;
    const double slope = grad.dot(dir);
    if (slope >= 0) return 0.0;

    const int N = static_cast<int>(path.size()) - 1;
    const int n = static_cast<int>(path[0].size());
    const int n_interior = N - 1;
    const int d = manifold.dim();

    for (int iter = 0; iter < 30; ++iter) {
      const Eigen::VectorXd x_new = x + step * dir;

      if (settings.max_displacement > 0.0) {
        bool within = true;
        for (int k = 0; k < n_interior; ++k) {
          const double drift = (x_new.segment(k * n, n) - ref_x.segment(k * n, n)).norm();
          if (drift > settings.max_displacement) {
            within = false;
            break;
          }
        }
        if (!within) {
          step *= 0.5;
          continue;
        }
      }

      unpack_interior(x_new, path);

      bool valid = true;
      for (const auto& q : path) {
        if (!validity_fn(vector_to_point<Point>(q, d))) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        step *= 0.5;
        continue;
      }

      for (int k = 0; k < N; ++k) {
        const Point a = vector_to_point<Point>(path[k], d);
        const Point b = vector_to_point<Point>(path[k + 1], d);
        if (!edge_collision_free(manifold, validity_fn, a, b, settings.edge_collision_samples)) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        step *= 0.5;
        continue;
      }

      const double f_new = dirichlet_energy(manifold, path);
      if (f_new <= f0 + settings.armijo_c * step * slope) return step;
      step *= 0.5;
    }

    unpack_interior(x, path);
    return 0.0;
  }

  template <RiemannianManifold M, typename ValidityFn>
  static int optimize_constrained(const M& manifold, const ValidityFn& validity_fn,
                                  std::vector<Eigen::VectorXd>& path,
                                  const SimplifyPathSettings& settings,
                                  const Eigen::VectorXd& ref_x) {
    const int N = static_cast<int>(path.size()) - 1;
    const int n = static_cast<int>(path[0].size());
    const int n_vars = (N - 1) * n;
    if (n_vars == 0) return 0;

    Eigen::VectorXd x = pack_interior(path);
    double f = dirichlet_energy(manifold, path);
    Eigen::VectorXd grad = compute_gradient(manifold, path, settings.fd_epsilon);

    std::vector<Eigen::VectorXd> s_hist, y_hist;
    int iter = 0;
    for (; iter < settings.max_iter_per_level; ++iter) {
      const double grad_norm = grad.cwiseAbs().maxCoeff();
      if (grad_norm < settings.grad_tol) break;

      const Eigen::VectorXd dir = lbfgs_two_loop(grad, s_hist, y_hist);
      const Eigen::VectorXd x_old = x;
      const double f_old = f;
      const Eigen::VectorXd grad_old = grad;

      const double step =
          armijo_constrained(manifold, validity_fn, path, x, dir, grad, f, settings, ref_x);
      if (step == 0.0) break;

      x = x_old + step * dir;
      unpack_interior(x, path);
      f = dirichlet_energy(manifold, path);
      grad = compute_gradient(manifold, path, settings.fd_epsilon);

      if (std::abs(f - f_old) < settings.energy_tol * std::abs(f_old) && f_old > 0) break;

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

      if (settings.verbose) {
        std::cerr << "    [N=" << N << "] iter=" << iter << " E=" << f
                  << " |grad|=" << grad_norm << " step=" << step << "\n";
      }
    }
    return iter;
  }
};

}  // namespace detail

/// @brief Simplify a collision-free path on a Riemannian manifold using
///        energy-aware shortcutting and collision-constrained L-BFGS smoothing.
///
/// @tparam M A type satisfying `RiemannianManifold`.
/// @tparam ValidityFn Callable with signature `bool(const Point&)` returning
///   true when a configuration is collision-free.
/// @param manifold The manifold instance.
/// @param validity_fn Collision validity callable.
/// @param initial_path Collision-free path from a planner (>= 2 waypoints).
/// @param settings Simplification parameters.
template <RiemannianManifold M, typename ValidityFn>
auto simplify_path(const M& manifold, const ValidityFn& validity_fn,
                   const std::vector<typename M::Point>& initial_path,
                   SimplifyPathSettings settings = {})
    -> SimplifyPathResult<typename M::Point> {
  return detail::SimplifyPathImpl::run(manifold, validity_fn, initial_path, settings);
}

}  // namespace geodex::algorithm

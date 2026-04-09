/// @file interpolation.hpp
/// @brief Discrete geodesic interpolation via Riemannian natural gradient descent.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <geodex/core/concepts.hpp>
#include <geodex/core/debug.hpp>
#include <geodex/core/metric.hpp>
#include <limits>
#include <type_traits>
#include <vector>

namespace geodex {

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/// @brief Termination status for the discrete geodesic walk.
enum class InterpolationStatus {
  Converged,         ///< Distance to target fell below convergence tolerance.
  MaxStepsReached,   ///< Iteration budget exhausted without reaching tolerance.
  GradientVanished,  ///< Riemannian gradient norm is ~0 at a non-target point.
  CutLocus,  ///< `log` collapsed to ~0 while start and target are distinct (e.g. antipodal on a
             ///< sphere).
  StepShrunkToZero,  ///< Distortion halving drove step size below `min_step_size`.
  DegenerateInput,   ///< `start == target` on entry; returned a single-point path immediately.
};

/// @brief Return a human-readable name for an `InterpolationStatus`.
inline const char* to_string(InterpolationStatus s) {
  switch (s) {
    case InterpolationStatus::Converged:
      return "Converged";
    case InterpolationStatus::MaxStepsReached:
      return "MaxStepsReached";
    case InterpolationStatus::GradientVanished:
      return "GradientVanished";
    case InterpolationStatus::CutLocus:
      return "CutLocus";
    case InterpolationStatus::StepShrunkToZero:
      return "StepShrunkToZero";
    case InterpolationStatus::DegenerateInput:
      return "DegenerateInput";
  }
  return "Unknown";
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// @brief Settings for the discrete geodesic walk.
///
/// @details Each iteration takes a Riemannian step of length
/// \f$\min(\texttt{step\_size}, \text{remaining distance})\f$ in the descent direction.
/// Iteration count and returned-path size therefore scale as approximately
/// \f$\text{initial\_distance} / \texttt{step\_size}\f$, so `step_size` also functions
/// as the effective path resolution (the maximum Riemannian distance between
/// consecutive returned points).
///
/// **Fast path**: the algorithm first tries the Riemannian logarithm as the descent
/// direction, exploiting the identity
/// \f$\nabla_g(\tfrac{1}{2}\, d_g^2(\cdot, q))(x) = -\log_x^g(q)\f$
/// which holds on any Riemannian manifold when `x` is strictly inside `q`'s
/// injectivity radius and `log` is the Riemannian logarithm of the metric in use.
/// A progress check on each step verifies that the proposed point actually decreased
/// the distance to target. When the check fails (e.g., the retraction is not the
/// true exponential map, or the metric differs from the one implied by the
/// retraction), the algorithm falls back **for that step only** to a central
/// finite-difference natural gradient computed from the manifold's `inner` product.
struct InterpolationSettings {
  /// @brief Max Riemannian step per iteration; also the effective path resolution.
  double step_size = 0.5;

  /// @brief Absolute stop threshold on \f$|\log(\text{current}, \text{target})|_R\f$.
  double convergence_tol = 1e-4;

  /// @brief Relative stop threshold: also stop when distance drops below
  /// `convergence_rel * initial_distance`.
  double convergence_rel = 1e-3;

  /// @brief Maximum number of successful gradient-descent steps.
  int max_steps = 100;

  /// @brief Central finite-difference step for the fallback gradient. Set to 0
  /// to auto-select as `max(1e-8, 1e-5 * max(1, initial_distance))`.
  double fd_epsilon = 0.0;

  /// @brief Max ratio \f$|\log(\text{current}, \text{proposed})|_R / \text{step\_used}\f$
  /// before the retraction is considered to have over-shot. If violated, the
  /// step cap is halved and the iteration retries. This guards against
  /// retractions that blow up under curvature and is usually 1.5 (modest slack
  /// over a perfect isometry).
  double distortion_ratio = 1.5;

  /// @brief Factor by which the current step cap grows back toward `step_size`
  /// after each successful iteration. Set to 1.0 to disable growth.
  double growth_factor = 1.5;

  /// @brief Floor below which the step cap triggers a `StepShrunkToZero` failure.
  double min_step_size = 1e-12;

  /// @brief Riemannian-norm threshold below which the gradient is considered
  /// vanished (triggers `GradientVanished`).
  double gradient_eps = 1e-12;

  /// @brief \f$|\log|_R\f$ threshold that, combined with a nonzero ambient gap,
  /// flags a cut-locus situation (e.g., antipodal points on the sphere where
  /// `log` returns zero).
  double cut_locus_eps = 1e-10;
};

// ---------------------------------------------------------------------------
// Local traits
// ---------------------------------------------------------------------------

namespace detail {

/// @brief True when the manifold provides a `project(p, v)` method mapping an
/// ambient vector to the tangent space.
template <typename M>
concept HasProject = requires(const M m, const typename M::Point p, const typename M::Tangent v) {
  { m.project(p, v) } -> std::same_as<typename M::Tangent>;
};

}  // namespace detail

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// @brief Reusable scratch buffers for `discrete_geodesic`.
///
/// @details Passing a cache to `discrete_geodesic` eliminates per-iteration
/// heap allocations (for fixed-size manifolds) and per-call allocation beyond the
/// first call (for dynamic-size manifolds). Intended for use in hot loops
/// such as steering functions in sampling-based planners:
///
/// ```cpp
/// geodex::InterpolationCache<Sphere<>> cache;
/// for (auto& edge : edges) {
///   auto r = geodex::discrete_geodesic(sphere, edge.a, edge.b, settings, &cache);
///   ...
/// }
/// ```
///
/// Users who do not need this optimization can omit the cache argument and
/// a stack-local one will be used automatically.
template <RiemannianManifold M>
struct InterpolationCache {
  using Point = typename M::Point;      ///< Manifold point type.
  using Tangent = typename M::Tangent;  ///< Manifold tangent vector type.

 private:
  static constexpr int N = Tangent::SizeAtCompileTime;
  static constexpr int MaxN = (N == Eigen::Dynamic) ? Eigen::Dynamic : N;

 public:
  /// @brief Scratch FD-path ambient tangent (natural gradient reconstructed in ambient space).
  Tangent v_fd;

  /// @brief Basis matrix (FD path): columns are tangent basis vectors at the current point.
  Eigen::Matrix<double, N, Eigen::Dynamic, 0, MaxN, MaxN> basis_mat;

  /// @brief Metric tensor \f$G_{ij} = \langle e_i, e_j\rangle_p\f$ in the current basis.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, MaxN, MaxN> G;

  /// @brief Coordinate-space gradient of \f$\tfrac{1}{2}\, d^2(\cdot, \text{target})\f$.
  Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxN, 1> grad;

  /// @brief Natural-gradient coefficients \f$\alpha = -G^{-1} g\f$.
  Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxN, 1> alpha;

  /// @brief Resize all buffers for the given ambient and intrinsic dimensions.
  /// @param ambient Ambient-space dimension of a tangent vector (typically `Tangent::size()`).
  /// @param d Intrinsic manifold dimension (`manifold.dim()`).
  void reset(int ambient, int d) {
    basis_mat.resize(ambient, d);
    G.resize(d, d);
    grad.resize(d);
    alpha.resize(d);
    if constexpr (N == Eigen::Dynamic) {
      v_fd.resize(ambient);
    }
  }
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// @brief Output of the discrete geodesic interpolation.
///
/// @tparam PointT Point type of the manifold (e.g. `Eigen::Vector3d`).
template <typename PointT>
struct InterpolationResult {
  /// @brief Sequence of iterates from `start` toward `target` (always starts with `start`).
  std::vector<PointT> path;

  /// @brief Termination reason — always check this before using the path for downstream work.
  InterpolationStatus status = InterpolationStatus::Converged;

  /// @brief Number of successful gradient steps taken (distortion retries do not count).
  int iterations = 0;

  /// @brief Number of times the step cap was halved due to distortion / progress failure.
  int distortion_halvings = 0;

  /// @brief Riemannian distance from `start` to `target` at entry.
  double initial_distance = 0.0;

  /// @brief Riemannian distance from the final iterate to `target` at exit.
  double final_distance = 0.0;
};

// ---------------------------------------------------------------------------
// Implementation helpers
// ---------------------------------------------------------------------------

namespace detail {

/// @brief First-order Riemannian distance via \f$|\log(a, b)|_R\f$.
///
/// @details For manifolds whose `log` is the actual Riemannian logarithm, this
/// is the exact geodesic distance. For retraction-based logs it is a first-order
/// approximation. Used inside `discrete_geodesic` hot loops in place of
/// `distance_midpoint` because it is very cheap and sufficient for
/// convergence and progress checks.
template <RiemannianManifold M>
inline auto distance_via_log(const M& m, const typename M::Point& a, const typename M::Point& b) ->
    typename M::Scalar {
  auto v = m.log(a, b);
  return m.norm(a, v);
}

/// @brief Build an orthonormal tangent basis at `p` into `cache.basis_mat`.
///
/// @details Seeds columns with ambient unit vectors (projected onto the tangent
/// space if the manifold provides `project`) and orthonormalizes via Euclidean
/// Gram-Schmidt. Returns the number of linearly independent basis vectors found (≤ `d`).
template <RiemannianManifold M>
int build_tangent_basis(const M& m, const typename M::Point& p, int d,
                        InterpolationCache<M>& cache) {
  using Tangent = typename M::Tangent;
  constexpr int N = Tangent::SizeAtCompileTime;

  const int ambient_dim = static_cast<int>(p.size());
  cache.basis_mat.resize(ambient_dim, d);

  int col = 0;
  for (int i = 0; i < ambient_dim && col < d; ++i) {
    Tangent e_i;
    if constexpr (N == Eigen::Dynamic) {
      e_i = Tangent::Zero(ambient_dim);
    } else {
      e_i = Tangent::Zero();
    }
    e_i[i] = 1.0;

    if constexpr (HasProject<M>) {
      e_i = m.project(p, e_i);
    }

    // Euclidean Gram-Schmidt against existing basis columns.
    for (int j = 0; j < col; ++j) {
      const double dot = e_i.dot(cache.basis_mat.col(j));
      e_i -= dot * cache.basis_mat.col(j);
    }

    const double nrm = e_i.norm();
    if (nrm > 1e-12) {
      cache.basis_mat.col(col) = e_i / nrm;
      ++col;
    }
  }

  if (col < d) {
    cache.basis_mat.conservativeResize(Eigen::NoChange, col);
  }
  return col;
}

/// @brief Compute the Riemannian natural gradient of \f$\tfrac{1}{2}\, d^2(\cdot, \text{target})\f$
/// at `p` via finite differences. Writes the ambient-space gradient into `cache.v_fd`.
///
/// @return `true` on success, `false` on Cholesky failure or zero-rank basis.
template <RiemannianManifold M>
bool natural_gradient_fd(const M& m, const typename M::Point& p, const typename M::Point& target,
                         double h, InterpolationCache<M>& cache) {
  using Tangent = typename M::Tangent;
  constexpr int N = Tangent::SizeAtCompileTime;

  const int d = build_tangent_basis(m, p, m.dim(), cache);
  if (d == 0) {
    if constexpr (N == Eigen::Dynamic) {
      cache.v_fd = Tangent::Zero(p.size());
    } else {
      cache.v_fd = Tangent::Zero();
    }
    return false;
  }

  cache.grad.resize(d);
  cache.G.resize(d, d);
  cache.alpha.resize(d);

  // 1) Coordinate gradient via central finite differences.
  for (int i = 0; i < d; ++i) {
    const auto p_plus = m.exp(p, h * cache.basis_mat.col(i));
    const auto p_minus = m.exp(p, -h * cache.basis_mat.col(i));
    const double d_plus = distance_via_log(m, p_plus, target);
    const double d_minus = distance_via_log(m, p_minus, target);
    cache.grad(i) = (0.5 * d_plus * d_plus - 0.5 * d_minus * d_minus) / (2.0 * h);
  }
  GEODEX_LOG("  natural_gradient_fd grad=" << cache.grad.transpose());

  // 2) Metric tensor G_ij = <e_i, e_j>_p. Use batch path if the manifold
  // provides `inner_matrix` (e.g., KineticEnergyMetric: one mass-matrix eval
  // instead of d^2 scalar calls).
  if constexpr (HasBatchInnerMatrix<M>) {
    const Eigen::MatrixXd B = cache.basis_mat.leftCols(d);
    const Eigen::MatrixXd G_full = m.inner_matrix(p, B, B);
    cache.G = 0.5 * (G_full + G_full.transpose());  // symmetrize against FP noise
  } else {
    for (int i = 0; i < d; ++i) {
      for (int j = i; j < d; ++j) {
        cache.G(i, j) = m.inner(p, cache.basis_mat.col(i), cache.basis_mat.col(j));
        cache.G(j, i) = cache.G(i, j);
      }
    }
  }
  GEODEX_LOG("  natural_gradient_fd G=\n" << cache.G);

  // 3) Scale-relative Tikhonov regularization + LLT solve.
  const double trace = cache.G.diagonal().sum();
  const double reg = 1e-12 * std::max(1.0, trace / static_cast<double>(d));
  cache.G.diagonal().array() += reg;

  auto solver = cache.G.llt();
  if (solver.info() != Eigen::Success) {
    GEODEX_LOG("  natural_gradient_fd: LLT failed");
    if constexpr (N == Eigen::Dynamic) {
      cache.v_fd = Tangent::Zero(p.size());
    } else {
      cache.v_fd = Tangent::Zero();
    }
    return false;
  }

  cache.alpha = solver.solve(-cache.grad);
  GEODEX_LOG("  natural_gradient_fd alpha=" << cache.alpha.transpose());

  // 4) Reconstruct in ambient space: v_fd = B * alpha.
  if constexpr (N == Eigen::Dynamic) {
    cache.v_fd = cache.basis_mat.leftCols(d) * cache.alpha;
  } else {
    cache.v_fd.noalias() = cache.basis_mat.leftCols(d) * cache.alpha;
  }
  return true;
}

/// @brief Resolve the initial step cap, optionally bounded by injectivity radius.
template <RiemannianManifold M>
double initial_step_cap(const M& m, double requested_step_size) {
  if constexpr (HasInjectivityRadius<M>) {
    const double inj = static_cast<double>(m.injectivity_radius());
    if (std::isfinite(inj) && inj > 0.0) {
      return std::min(requested_step_size, 0.5 * inj);
    }
  }
  return requested_step_size;
}

/// @brief Auto-select the FD central-difference step when the user passed 0.
///
/// @details The optimal central-FD step for double precision is approximately
/// \f$\varepsilon_{\mathrm{mach}}^{1/3} \approx 6\times 10^{-6}\f$. We scale
/// gently with the initial distance so tiny workspaces don't get an FD step
/// that dwarfs the geometry.
inline double resolve_fd_epsilon(double user_value, double initial_distance) {
  if (user_value > 0.0) return user_value;
  return std::max(1e-8, 1e-5 * std::max(1.0, initial_distance));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// discrete_geodesic
// ---------------------------------------------------------------------------

/// @brief Walk from `start` toward `target` via Riemannian natural gradient descent.
///
/// @details The algorithm iteratively descends on
/// \f$\varphi(x) = \tfrac{1}{2}\, d^2(x, \text{target})\f$ with step length capped
/// by `settings.step_size` per iteration. Each iteration tries the Riemannian
/// logarithm direction first (exploiting \f$\nabla\varphi = -\log_x(\text{target})\f$
/// on any Riemannian manifold away from the cut locus) and verifies via a
/// progress check. When the log direction does not produce enough distance
/// decrease (e.g., because the retraction is a projection, or the metric is
/// custom), the algorithm falls back for that step to a central finite-difference
/// natural gradient computed from the manifold's `inner` product.
///
/// The returned `InterpolationResult` carries the path, a termination status
/// enum, iteration count, and the initial/final distances — allowing callers to
/// distinguish successful convergence from `MaxStepsReached`, `CutLocus`,
/// `GradientVanished`, `StepShrunkToZero`, and `DegenerateInput`.
///
/// **Walk semantics**: iteration count and path size both scale as
/// \f$\approx \text{initial\_distance} / \texttt{step\_size}\f$. Reduce
/// `step_size` for higher path resolution.
///
/// @note See Kyaw, P. T., & Kelly, J. (2026). *Geometry-Aware Sampling-Based
/// Motion Planning on Riemannian Manifolds.* arXiv:2602.00992. The identity
/// \f$\nabla_g(\tfrac{1}{2}\, d_g^2(\cdot, q))(x) = -\log_x^g(q)\f$ is standard;
/// see Sakai, *Riemannian Geometry*, §IV.5 and do Carmo, *Riemannian Geometry*,
/// Ch 13 Prop 3.6.
///
/// @tparam M A type satisfying `RiemannianManifold`.
/// @param manifold The manifold instance.
/// @param start Starting point.
/// @param target Target point to walk toward.
/// @param settings Algorithm parameters.
/// @param cache Optional reusable cache. If null, a stack-local one is used.
/// @return An `InterpolationResult` carrying the path and termination diagnostics.
template <RiemannianManifold M>
auto discrete_geodesic(const M& manifold, const typename M::Point& start,
                       const typename M::Point& target, InterpolationSettings settings = {},
                       InterpolationCache<M>* cache = nullptr)
    -> InterpolationResult<typename M::Point> {
  using Point = typename M::Point;
  using Tangent = typename M::Tangent;
  using Result = InterpolationResult<Point>;

  Result R;
  /// Default reserve cap: avoid over-allocating when max_steps is very large.
  static constexpr int kDefaultPathReserve = 128;
  R.path.reserve(std::min(settings.max_steps + 1, kDefaultPathReserve));
  R.path.push_back(start);

  // Cache: either the caller-supplied one or a stack-local default.
  InterpolationCache<M> stack_cache;
  InterpolationCache<M>& C = cache ? *cache : stack_cache;
  C.reset(static_cast<int>(start.size()), manifold.dim());

  GEODEX_LOG("=== discrete_geodesic start ===");
  GEODEX_LOG("start=" << start.transpose() << "  target=" << target.transpose());

  // Initial distance via |log(start, target)|_R — exact for Riemannian-log manifolds,
  // first-order approximation otherwise.
  Tangent v_log = manifold.log(start, target);
  double dist = manifold.norm(start, v_log);
  R.initial_distance = dist;
  R.final_distance = dist;

  // Early exits: degenerate input, tolerance already met, or cut locus.
  {
    const double ambient_gap = (target - start).norm();
    if (ambient_gap == 0.0) {
      R.status = InterpolationStatus::DegenerateInput;
      GEODEX_LOG("=== discrete_geodesic done (DegenerateInput) ===");
      return R;
    }
    if (dist < settings.cut_locus_eps && ambient_gap > 1e-10) {
      R.status = InterpolationStatus::CutLocus;
      GEODEX_LOG("=== discrete_geodesic done (CutLocus) ===");
      return R;
    }
    if (dist <= settings.convergence_tol) {
      R.status = InterpolationStatus::Converged;
      GEODEX_LOG("=== discrete_geodesic done (already within tol) ===");
      return R;
    }
  }

  const double fd_eps = detail::resolve_fd_epsilon(settings.fd_epsilon, dist);
  double step_cap = detail::initial_step_cap(manifold, settings.step_size);
  const double initial_distance = dist;
  // Resolved once per call: is the base log the Riemannian log of the metric?
  // `is_riemannian_log` collapses the compile-time (`M::has_riemannian_log`)
  // and runtime (`m.has_riemannian_log_runtime()`) signals into one bool.
  const bool fast_path_enabled = geodex::is_riemannian_log(manifold);

  Point current = start;

  for (int i = 0; i < settings.max_steps; ++i) {
    GEODEX_LOG("--- step " << i << ": current=" << current.transpose() << "  dist=" << dist
                           << "  step_cap=" << step_cap);

    // Convergence on absolute and relative thresholds.
    if (dist <= settings.convergence_tol || dist <= settings.convergence_rel * initial_distance) {
      R.status = InterpolationStatus::Converged;
      break;
    }

    // log may vanish at a stationary point that's not the target (cut locus or
    // symmetry). Either way, we can't descend further.
    if (dist < settings.gradient_eps) {
      R.status = InterpolationStatus::CutLocus;
      break;
    }

    const double step_used = std::min(step_cap, dist);

    Tangent direction;
    Point proposed;
    Tangent new_v_log;
    double new_dist = 0.0;
    bool accepted = false;

    // Runtime branch: when `fast_path_enabled` is true the base `log` is the
    // Riemannian log of the metric and `-log` is the natural gradient of
    // (1/2) d^2. Otherwise we use finite differences to compute the correct
    // natural gradient under the (possibly custom) metric. For manifolds with
    // compile-time opt-in the branch predictor collapses this into a
    // branchless fast path.
    if (fast_path_enabled) {
      direction = (1.0 / dist) * v_log;
      proposed = manifold.exp(current, step_used * direction);
      new_v_log = manifold.log(proposed, target);
      new_dist = manifold.norm(proposed, new_v_log);

      // Verify: did we actually get closer, and did the retraction deliver a
      // step close to the intended length?
      const bool progress_ok = (new_dist < dist);
      const double actual_step = detail::distance_via_log(manifold, current, proposed);
      const bool fidelity_ok = (actual_step <= settings.distortion_ratio * step_used);
      accepted = (progress_ok && fidelity_ok);

      if (!accepted) {
        GEODEX_LOG("  log step rejected (progress_ok=" << progress_ok << " fidelity_ok="
                                                       << fidelity_ok << "); trying FD");
      }
    }

    // --- FD natural gradient. Always used when the manifold does not provide a
    // Riemannian log; used as a fallback when the log-step verification fails. ---
    if (!accepted) {
      if (!detail::natural_gradient_fd(manifold, current, target, fd_eps, C)) {
        R.status = InterpolationStatus::GradientVanished;
        break;
      }
      const double fd_norm = manifold.norm(current, C.v_fd);
      if (fd_norm < settings.gradient_eps) {
        R.status = InterpolationStatus::GradientVanished;
        break;
      }
      direction = (1.0 / fd_norm) * C.v_fd;
      proposed = manifold.exp(current, step_used * direction);
      new_v_log = manifold.log(proposed, target);
      new_dist = manifold.norm(proposed, new_v_log);

      const bool progress_ok = (new_dist < dist);
      const double actual_step = detail::distance_via_log(manifold, current, proposed);
      const bool fidelity_ok = (actual_step <= settings.distortion_ratio * step_used);

      if (!progress_ok || !fidelity_ok) {
        // FD descent at this step cap still over-shoots or makes no progress —
        // halve the cap and retry the iteration.
        step_cap *= 0.5;
        ++R.distortion_halvings;
        GEODEX_LOG("  FD rejected; halving step_cap -> " << step_cap);
        if (step_cap < settings.min_step_size) {
          R.status = InterpolationStatus::StepShrunkToZero;
          break;
        }
        --i;
        continue;
      }
    }

    // Accept the step.
    current = proposed;
    R.path.push_back(current);
    ++R.iterations;

    // Reuse the log computation we already did for the NEXT iteration.
    v_log = std::move(new_v_log);
    dist = new_dist;
    R.final_distance = dist;

    // Regrow step cap after a successful iteration.
    step_cap = std::min(settings.step_size, settings.growth_factor * step_cap);
  }

  // Post-loop status resolution. If we exhausted max_steps without ever setting
  // a terminal status, report MaxStepsReached.
  if (R.status == InterpolationStatus::Converged && R.iterations == settings.max_steps &&
      dist > settings.convergence_tol && dist > settings.convergence_rel * initial_distance) {
    R.status = InterpolationStatus::MaxStepsReached;
  }

  GEODEX_LOG("=== discrete_geodesic done, " << R.path.size()
                                            << " points, status=" << to_string(R.status) << " ===");
  return R;
}

}  // namespace geodex

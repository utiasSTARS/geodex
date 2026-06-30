/// @file precompute_matrix_lower_bound.hpp
/// @brief Optimization-based Loewner lower bound via constraint generation.
///
/// @details Multi-start gradient descent finds the worst-case configuration
/// \f$ q^* \f$ that minimizes \f$ \lambda_{\min}(L^{-1} M(q) L^{-\top}) \f$,
/// then tightens \f$ M_{\mathrm{lower}} \f$ via Loewner meet against
/// \f$ M(q^*) \f$. Converges when \f$ \lambda_{\min} \ge 1 - \mathrm{tol} \f$
/// over the configuration space, providing a convergence certificate.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

#include "geodex/core/concepts.hpp"
#include "geodex/core/metric.hpp"
#include "geodex/heuristics/matrix_lower_bound.hpp"

namespace geodex::algorithm {

/// @brief Settings for `precompute_matrix_lower_bound`.
struct PrecomputeMatrixLowerBoundSettings {
  int max_outer = 50;             ///< Maximum outer constraint-generation iterations.
  double tol = 1e-6;              ///< Stop when \f$ \lambda_{\min} \ge 1 - \mathrm{tol} \f$.
  int n_starts_per_iter = 0;      ///< Multi-start seeds per outer iter (0 = auto: max(20, 10*dim)).
  int max_iters_per_start = 200;  ///< Max gradient-descent iterations per start.
  double grad_tol = 1e-7;         ///< Gradient-norm convergence for inner gradient descent.
  double fd_eps = 1e-5;           ///< Finite-difference step for \f$ \nabla \lambda_{\min} \f$.
  uint64_t seed = 42;             ///< RNG seed for multi-start initial points.
  bool verbose = false;           ///< Print outer-iteration diagnostics to stderr.
};

/// @brief Result of `precompute_matrix_lower_bound`.
struct PrecomputeMatrixLowerBoundResult {
  Eigen::MatrixXd M_lower;              ///< The certified Loewner lower bound on \f$ M(q) \f$.
  double lambda_min_certificate = 0.0;  ///< Final worst-case \f$ \lambda_{\min} \f$.
  int n_outer_iters = 0;                ///< Outer constraint-generation iterations executed.
  int n_metric_evals = 0;               ///< Total \f$ M(q) \f$ evaluations.
  bool converged = false;               ///< True when `lambda_min_certificate >= 1 - tol`.
  double elapsed_ms = 0.0;              ///< Wall-clock duration of the precompute.
};

namespace detail {

/// @brief Internal-only implementation of `precompute_matrix_lower_bound`. All
/// helpers are private static methods; only `run` is reachable from outside,
/// and only via the public `geodex::algorithm::precompute_matrix_lower_bound`
/// free function.
struct PrecomputeMatrixLowerBoundImpl {
 public:
  template <typename ManifoldT>
  static auto run(const ManifoldT& manifold,
                  const PrecomputeMatrixLowerBoundSettings& settings)
      -> PrecomputeMatrixLowerBoundResult {
    const int dim = manifold.dim();
    const Eigen::VectorXd lo = manifold.lo();
    const Eigen::VectorXd hi = manifold.hi();
    std::mt19937 rng(settings.seed);

    const int n_starts = settings.n_starts_per_iter > 0 ? settings.n_starts_per_iter
                                                        : std::max(20, 10 * dim);

    int eval_count = 0;
    const Eigen::VectorXd q_center = 0.5 * (lo + hi);
    const Eigen::MatrixXd M0 = metric_matrix(manifold, q_center);
    ++eval_count;
    geodex::heuristics::MatrixLowerBound<> heuristic(M0);

    const auto t_start = std::chrono::steady_clock::now();
    const double convergence_tol = 1.0 - settings.tol;
    int outer = 0;
    double last_lambda_min = 0.0;

    for (; outer < settings.max_outer; ++outer) {
      auto [q_star, lambda_min] =
          find_worst_case(manifold, heuristic.llt(), lo, hi, n_starts,
                          settings.max_iters_per_start, settings.grad_tol, settings.fd_eps, rng,
                          eval_count);
      last_lambda_min = lambda_min;

      if (settings.verbose) {
        std::cerr << "  iter=" << outer << " lambda_min=" << lambda_min
                  << " det=" << heuristic.det() << " evals=" << eval_count << "\n";
      }

      if (lambda_min >= convergence_tol) break;

      const Eigen::MatrixXd M_star = metric_matrix(manifold, q_star);
      ++eval_count;
      heuristic.update(M_star);
    }

    const auto t_end = std::chrono::steady_clock::now();

    PrecomputeMatrixLowerBoundResult result;
    result.M_lower = heuristic.matrix();
    result.lambda_min_certificate = last_lambda_min;
    result.n_outer_iters = outer;
    result.n_metric_evals = eval_count;
    result.converged = (last_lambda_min >= convergence_tol);
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    return result;
  }

 private:
  /// @brief Form the metric tensor \f$ M(q) \f$ as a dense \f$ d \times d \f$ matrix.
  template <typename ManifoldT>
  static Eigen::MatrixXd metric_matrix(const ManifoldT& manifold,
                                       const Eigen::VectorXd& q_vec) {
    const int d = manifold.dim();
    using Point = typename ManifoldT::Point;
    Point q;
    if constexpr (Point::SizeAtCompileTime == Eigen::Dynamic) q.resize(d);
    for (int i = 0; i < d; ++i) q[i] = q_vec[i];

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(d, d);
    Eigen::MatrixXd M = manifold.inner_matrix(q, I, I);
    return 0.5 * (M + M.transpose());  // symmetrize against FP noise
  }

  /// @brief Evaluate \f$ \lambda_{\min}(L^{-1} M(q) L^{-\top}) \f$.
  template <typename ManifoldT>
  static double eval_lambda_min(const ManifoldT& manifold,
                                const Eigen::LLT<Eigen::MatrixXd>& llt,
                                const Eigen::VectorXd& q, int& eval_count) {
    ++eval_count;
    const Eigen::MatrixXd M = metric_matrix(manifold, q);
    const Eigen::MatrixXd L = llt.matrixL();
    const Eigen::MatrixXd Linv_M = L.template triangularView<Eigen::Lower>().solve(M);
    const Eigen::MatrixXd S =
        L.template triangularView<Eigen::Lower>().solve(Linv_M.transpose()).transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S, Eigen::EigenvaluesOnly);
    return solver.eigenvalues().minCoeff();
  }

  /// @brief Central-FD gradient of \f$ \lambda_{\min}(S(q)) \f$ w.r.t. \f$ q \f$.
  template <typename ManifoldT>
  static Eigen::VectorXd grad_lambda_min(const ManifoldT& manifold,
                                         const Eigen::LLT<Eigen::MatrixXd>& llt,
                                         const Eigen::VectorXd& q, double fd_eps,
                                         int& eval_count) {
    const int dim = static_cast<int>(q.size());
    Eigen::VectorXd g(dim);
    for (int i = 0; i < dim; ++i) {
      Eigen::VectorXd qp = q, qm = q;
      qp[i] += fd_eps;
      qm[i] -= fd_eps;
      g[i] = (eval_lambda_min(manifold, llt, qp, eval_count) -
              eval_lambda_min(manifold, llt, qm, eval_count)) /
             (2.0 * fd_eps);
    }
    return g;
  }

  /// @brief Per-dimension box clamp.
  static void clamp_to_bounds(Eigen::VectorXd& q, const Eigen::VectorXd& lo,
                              const Eigen::VectorXd& hi) {
    for (int i = 0; i < q.size(); ++i) q[i] = std::clamp(q[i], lo[i], hi[i]);
  }

  /// @brief Single gradient-descent run with Armijo backtracking and box clamping.
  template <typename ManifoldT>
  static std::pair<Eigen::VectorXd, double> gradient_descent(
      const ManifoldT& manifold, const Eigen::LLT<Eigen::MatrixXd>& llt,
      const Eigen::VectorXd& q0, const Eigen::VectorXd& lo, const Eigen::VectorXd& hi,
      int max_iters, double grad_tol, double fd_eps, int& eval_count) {
    Eigen::VectorXd q = q0;
    double f = eval_lambda_min(manifold, llt, q, eval_count);

    for (int iter = 0; iter < max_iters; ++iter) {
      const Eigen::VectorXd g = grad_lambda_min(manifold, llt, q, fd_eps, eval_count);
      if (g.norm() < grad_tol) break;

      double alpha = 1.0;
      constexpr double c1 = 1e-4;
      const Eigen::VectorXd dir = -g;

      for (int ls = 0; ls < 20; ++ls) {
        Eigen::VectorXd q_new = q + alpha * dir;
        clamp_to_bounds(q_new, lo, hi);
        const double f_new = eval_lambda_min(manifold, llt, q_new, eval_count);
        if (f_new <= f + c1 * alpha * g.dot(dir)) {
          q = q_new;
          f = f_new;
          break;
        }
        alpha *= 0.5;
      }
    }
    return {q, f};
  }

  /// @brief Multi-start gradient descent to find the worst-case configuration.
  template <typename ManifoldT>
  static std::pair<Eigen::VectorXd, double> find_worst_case(
      const ManifoldT& manifold, const Eigen::LLT<Eigen::MatrixXd>& llt,
      const Eigen::VectorXd& lo, const Eigen::VectorXd& hi, int n_starts,
      int max_iters_per_start, double grad_tol, double fd_eps, std::mt19937& rng,
      int& eval_count) {
    const int dim = static_cast<int>(lo.size());
    Eigen::VectorXd best_q;
    double best_f = std::numeric_limits<double>::infinity();

    for (int s = 0; s < n_starts; ++s) {
      Eigen::VectorXd q0(dim);
      for (int i = 0; i < dim; ++i) {
        std::uniform_real_distribution<double> dist(lo[i], hi[i]);
        q0[i] = dist(rng);
      }
      auto [q, f] = gradient_descent(manifold, llt, q0, lo, hi, max_iters_per_start, grad_tol,
                                     fd_eps, eval_count);
      if (f < best_f) {
        best_f = f;
        best_q = q;
      }
    }
    return {best_q, best_f};
  }
};

}  // namespace detail

/// @brief Compute a constant SPD Loewner lower bound on \f$ M(q) \f$ via constraint generation.
///
/// @details Iteratively tightens \f$ M_{\mathrm{lower}} \f$ by:
///   1. solving \f$ q^* = \arg\min_{q \in [\mathrm{lo}, \mathrm{hi}]^d}
///      \lambda_{\min}(L^{-1} M(q) L^{-\top}) \f$ via multi-start gradient descent;
///   2. updating \f$ M_{\mathrm{lower}} \f$ with the Loewner meet against
///      \f$ M(q^*) \f$ (`heuristics::MatrixLowerBound::update`).
///
/// Terminates when the worst-case eigenvalue stays \f$ \ge 1 - \mathrm{tol} \f$ across the
/// configuration space, providing a convergence certificate. The bound is then ready
/// for use by `geodex::heuristics::MatrixLowerBound`.
///
/// @see Phone Thiha Kyaw, Jonathan Kelly. "Direct Informed Sampling on
///   Riemannian Manifolds via Loewner Order Lower Bounds." arXiv:2606.02879
///   (2026). Derives the constraint-generation scheme and its convergence
///   certificate.
///
/// @tparam ManifoldT A `RiemannianManifold` that provides the batched
///   `inner_matrix(p, U, V)` (`HasBatchInnerMatrix`) and per-dimension sampling
///   bounds via `lo()` / `hi()` (i.e., a bounded Euclidean-like config space).
/// @param manifold The configuration manifold.
/// @param settings Solver settings.
/// @return PrecomputeMatrixLowerBoundResult with the certified bound and diagnostics.
template <typename ManifoldT>
  requires RiemannianManifold<ManifoldT> && HasBatchInnerMatrix<ManifoldT> &&
           requires(const ManifoldT& m) {
             { m.lo() } -> std::convertible_to<Eigen::VectorXd>;
             { m.hi() } -> std::convertible_to<Eigen::VectorXd>;
           }
PrecomputeMatrixLowerBoundResult precompute_matrix_lower_bound(
    const ManifoldT& manifold, const PrecomputeMatrixLowerBoundSettings& settings = {}) {
  return detail::PrecomputeMatrixLowerBoundImpl::run(manifold, settings);
}

}  // namespace geodex::algorithm

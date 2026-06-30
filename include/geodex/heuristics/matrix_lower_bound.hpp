/// @file matrix_lower_bound.hpp
/// @brief Matrix-lower-bound heuristic using a constant SPD Loewner lower bound.

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace geodex::heuristics {

/// @brief Matrix lower-bound heuristic using a constant SPD Loewner lower bound.
///
/// @details For a configuration-dependent metric \f$ M(q) \f$, if there exists
/// a constant SPD matrix \f$ M_{\mathrm{lower}} \f$ with
/// \f$ M(q) \succeq M_{\mathrm{lower}} \f$ in the Loewner order for every
/// \f$ q \in \mathcal{Q} \f$, then the geodesic distance satisfies
/// \f[
///   d_M(a,b) \ge \sqrt{(a - b)^\top M_{\mathrm{lower}} (a - b)}.
/// \f]
/// This bound is tighter than the scalar eigenvalue bound because it
/// preserves directional information. The Cholesky factor \f$ L \f$ of
/// \f$ M_{\mathrm{lower}} = L L^\top \f$ is cached, and the heuristic
/// evaluates \f$ \|L^\top (a - b)\|_2 \f$.
///
/// An optional eigenvalue floor \f$ \lambda_{\min} \f$ can be supplied to
/// guarantee dominance over the scalar eigenvalue bound in every direction:
/// when set, the heuristic returns
/// \f$ \max(\|L^\top \Delta\|,\; \sqrt{\lambda_{\min}}\,\|\Delta\|) \f$. This
/// matters because a Loewner meet is conservative in each eigendirection
/// independently, so \f$ M_{\mathrm{lower}} \f$ can have eigenvalues below
/// \f$ \lambda_{\min} \f$ even when every observed \f$ M(q_i) \f$ exceeds
/// \f$ \lambda_{\min}\,I \f$ along that direction.
///
/// @see Phone Thiha Kyaw, Jonathan Kelly. "Direct Informed Sampling on
///   Riemannian Manifolds via Loewner Order Lower Bounds." arXiv:2606.02879
///   (2026).
///
/// @tparam Dim Static dimension (default: Eigen::Dynamic).
template <int Dim = Eigen::Dynamic>
class MatrixLowerBound {
 public:
  using MatrixType = Eigen::Matrix<double, Dim, Dim>;
  using VectorType = Eigen::Vector<double, Dim>;

  /// @brief Construct from a constant SPD lower bound.
  /// @param M_lower SPD matrix satisfying \f$ M(q) \succeq M_{\mathrm{lower}} \f$.
  explicit MatrixLowerBound(const MatrixType& M_lower)
      : llt_(M_lower), sqrt_lambda_min_floor_(0.0), n_updates_(0) {}

  /// @brief Construct with an eigenvalue floor.
  /// @todo Remove this option; floor will not guarantee to be admissible
  /// @param M_lower SPD matrix satisfying \f$ M(q) \succeq M_{\mathrm{lower}} \f$.
  /// @param lambda_min Global minimum eigenvalue of \f$ M(q) \f$ over \f$ \mathcal{Q} \f$.
  MatrixLowerBound(const MatrixType& M_lower, double lambda_min)
      : llt_(M_lower), sqrt_lambda_min_floor_(std::sqrt(lambda_min)), n_updates_(0) {}

  /// @brief Compute the admissible lower bound on geodesic distance.
  /// @details Returns \f$ \|L^\top (a - b)\|_2 \f$, or if an eigenvalue floor
  /// was provided, \f$ \max(\|L^\top \Delta\|,\; \sqrt{\lambda_{\min}}\,\|\Delta\|) \f$.
  template <typename PointA, typename PointB>
  auto operator()(const PointA& a, const PointB& b) const -> double {
    const VectorType diff = a - b;
    const VectorType Ldiff = llt_.matrixL().transpose() * diff;
    const double h_mlb = Ldiff.norm();
    if (sqrt_lambda_min_floor_ > 0.0) {
      const double h_elb = sqrt_lambda_min_floor_ * diff.norm();
      return std::max(h_mlb, h_elb);
    }
    return h_mlb;
  }

  /// @brief Perform one incremental Loewner-meet update with a new observation.
  ///
  /// @details Given an SPD observation \f$ M_{\mathrm{new}} \f$ (e.g. the
  /// metric at a newly explored worst-case configuration), tightens
  /// \f$ M_{\mathrm{lower}} \f$ so that \f$ M_{\mathrm{lower}} \preceq M_{\mathrm{new}} \f$:
  ///   1. compute \f$ S = L^{-1} M_{\mathrm{new}} L^{-\top} \f$ (SPD);
  ///   2. if \f$ \min(\lambda_k(S)) \ge 1 \f$, skip (already dominated);
  ///   3. clamp eigenvalues of \f$ S \f$ to \f$ \min(\lambda_k, 1) \f$;
  ///   4. reconstruct \f$ M_{\mathrm{lower}} \leftarrow L V \tilde\Lambda V^\top L^\top \f$;
  ///   5. re-Cholesky.
  ///
  /// @param M_new New SPD metric observation.
  /// @return True if the bound was loosened, false if it already dominated.
  bool update(const MatrixType& M_new) {
    const MatrixType L = llt_.matrixL();

    // S = L^{-1} M_new L^{-T}
    const MatrixType Linv_Mnew =
        L.template triangularView<Eigen::Lower>().solve(M_new);
    const MatrixType S =
        L.template triangularView<Eigen::Lower>()
            .solve(Linv_Mnew.transpose())
            .transpose();

    Eigen::SelfAdjointEigenSolver<MatrixType> solver(S);
    auto evals = solver.eigenvalues();

    if (evals.minCoeff() >= 1.0) {
      return false;
    }

    const auto evecs = solver.eigenvectors();
    for (int k = 0; k < evals.size(); ++k) {
      if (evals[k] > 1.0) evals[k] = 1.0;
    }

    const MatrixType D = evals.asDiagonal();
    MatrixType M_lower = L * evecs * D * evecs.transpose() * L.transpose();
    M_lower = (M_lower + M_lower.transpose()) / 2.0;  // symmetrize to prevent drift

    llt_.compute(M_lower);
    ++n_updates_;
    return true;
  }

  /// @brief Reconstruct the current \f$ M_{\mathrm{lower}} \f$ from its Cholesky factor.
  auto matrix() const -> MatrixType {
    const MatrixType L = llt_.matrixL();
    return L * L.transpose();
  }

  /// @brief Determinant of the current \f$ M_{\mathrm{lower}} \f$.
  /// @details Computed from Cholesky diagonals as \f$ \prod_i L_{ii}^2 \f$ via
  /// log-sum-exp for numerical stability.
  auto det() const -> double {
    const auto L = llt_.matrixL();
    double log_det = 0.0;
    for (int i = 0; i < L.rows(); ++i) log_det += std::log(L.coeff(i, i));
    return std::exp(2.0 * log_det);
  }

  /// @brief Eigenvalues of the current \f$ M_{\mathrm{lower}} \f$, ascending.
  auto eigenvalues() const -> Eigen::VectorXd {
    const MatrixType M = matrix();
    Eigen::SelfAdjointEigenSolver<MatrixType> solver(M, Eigen::EigenvaluesOnly);
    return solver.eigenvalues();
  }

  /// @brief Number of times `update()` actually loosened the bound.
  auto update_count() const -> int { return n_updates_; }

  /// @brief Access the underlying Cholesky factorization of \f$ M_{\mathrm{lower}} \f$.
  auto llt() const -> const Eigen::LLT<MatrixType>& { return llt_; }

  /// @brief Whether this heuristic has an eigenvalue floor set.
  auto has_eigenvalue_floor() const -> bool { return sqrt_lambda_min_floor_ > 0.0; }

 private:
  Eigen::LLT<MatrixType> llt_;
  double sqrt_lambda_min_floor_;
  int n_updates_;
};

}  // namespace geodex::heuristics

/// @file matrix_lower_bound.hpp
/// @brief Matrix-lower-bound heuristic using a constant SPD Loewner lower bound.

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "geodex/utils/angle.hpp"

namespace geodex::heuristics {

/// @brief Matrix lower-bound heuristic using a constant SPD Loewner lower bound.
///
/// @details If \f$ M(q) \succeq M_{\mathrm{lower}} \f$ in the Loewner order for every
/// \f$ q \in \mathcal{Q} \f$, then
/// \f$ d_M(a,b) \ge \sqrt{(a - b)^\top M_{\mathrm{lower}} (a - b)} \f$. The heuristic
/// caches the Cholesky factor of \f$ M_{\mathrm{lower}} = L L^\top \f$ and evaluates
/// \f$ \|L^\top (a - b)\|_2 \f$. It keeps the directional information that the scalar
/// eigenvalue bound loses.
///
/// @see Phone Thiha Kyaw, Jonathan Kelly. "Direct Informed Sampling on
///   Riemannian Manifolds via Loewner Order Lower Bounds." IEEE Robotics and
///   Automation Letters (RA-L), 2026. arXiv:2606.02879.
///
/// @tparam Dim Static dimension, Eigen::Dynamic by default.
template <int Dim = Eigen::Dynamic>
class MatrixLowerBound {
 public:
  using MatrixType = Eigen::Matrix<double, Dim, Dim>;  ///< Type of the bound matrix.
  using VectorType = Eigen::Vector<double, Dim>;       ///< Type of a coordinate difference.

  /// @brief Construct from a constant SPD lower bound.
  /// @param M_lower SPD matrix satisfying \f$ M(q) \succeq M_{\mathrm{lower}} \f$.
  explicit MatrixLowerBound(const MatrixType& M_lower)
      : llt_(M_lower), sqrt_lambda_min_floor_(0.0) {}

  /// @brief Construct with an eigenvalue floor.
  /// @details The heuristic returns
  /// \f$ \max(\|L^\top \Delta\|,\; \sqrt{\lambda_{\min}}\,\|\Delta\|) \f$ and
  /// dominates the scalar eigenvalue bound in every direction. A Loewner meet is
  /// conservative in each eigendirection and can leave eigenvalues below the floor.
  /// @warning The floor is admissible only if `lambda_min` bounds every eigenvalue of
  /// \f$ M(q) \f$ over \f$ \mathcal{Q} \f$. The constructor does not check it. A floor
  /// that is too high makes the heuristic overestimate.
  /// @param M_lower SPD matrix satisfying \f$ M(q) \succeq M_{\mathrm{lower}} \f$.
  /// @param lambda_min Global minimum eigenvalue of \f$ M(q) \f$ over \f$ \mathcal{Q} \f$.
  MatrixLowerBound(const MatrixType& M_lower, double lambda_min)
      : llt_(M_lower), sqrt_lambda_min_floor_(std::sqrt(lambda_min)) {}

  /// @brief Construct with per-axis deck-group generators for periodic coordinates.
  /// @details On a flat quotient (SE(2), SO(2), \f$ T^n \f$) the heuristic reduces each
  /// periodic axis to \f$ [-P_i/2, P_i/2) \f$ before it applies the bound. This equals
  /// the deck-group minimum only when no periodic axis is coupled to another in
  /// \f$ M_{\mathrm{lower}} \f$. SE(2) satisfies the condition by structure.
  /// @param M_lower SPD matrix satisfying \f$ M(q) \succeq M_{\mathrm{lower}} \f$.
  /// @param periods The manifold's `periods()`, or empty to disable wrapping.
  /// @throws std::invalid_argument on a size mismatch, or on a periodic axis coupled
  ///         to another axis in `M_lower`.
  MatrixLowerBound(const MatrixType& M_lower, const Eigen::VectorXd& periods)
      : llt_(M_lower), sqrt_lambda_min_floor_(0.0) {
    if (periods.size() == 0) return;
    if (periods.size() != M_lower.rows()) {
      throw std::invalid_argument(
          "MatrixLowerBound: periods has " + std::to_string(periods.size()) +
          " entries but M_lower is " + std::to_string(M_lower.rows()) + "-dimensional.");
    }
    require_periodic_axes_decoupled(M_lower, periods);
    periods_ = periods;
  }

  /// @brief Compute the admissible lower bound on geodesic distance.
  /// @details Returns \f$ \|L^\top \Delta\|_2 \f$, or the floored maximum, where
  /// \f$ \Delta \f$ is \f$ a - b \f$ with every periodic axis reduced to its nearest
  /// representative. `a` and `b` must hold `size()` entries. This hot-path call does
  /// not check it.
  template <typename PointA, typename PointB>
  auto operator()(const PointA& a, const PointB& b) const -> double {
    VectorType diff = a - b;
    for (int i = 0; i < periods_.size(); ++i) {
      diff[i] = utils::wrap_delta_period(diff[i], periods_[i]);
    }
    const VectorType Ldiff = llt_.matrixL().transpose() * diff;
    const double h_mlb = Ldiff.norm();
    if (sqrt_lambda_min_floor_ > 0.0) {
      const double h_elb = sqrt_lambda_min_floor_ * diff.norm();
      return std::max(h_mlb, h_elb);
    }
    return h_mlb;
  }

  /// @brief Per-axis deck-group generators, empty when no axis is periodic.
  auto periods() const -> const Eigen::VectorXd& { return periods_; }

  /// @brief Number of coordinates the bound measures.
  auto size() const -> Eigen::Index { return llt_.matrixLLT().rows(); }

  /// @brief Perform one incremental Loewner-meet update with a new observation.
  ///
  /// @details Tightens \f$ M_{\mathrm{lower}} \f$ until
  /// \f$ M_{\mathrm{lower}} \preceq M_{\mathrm{new}} \f$. It clamps the eigenvalues of
  /// \f$ S = L^{-1} M_{\mathrm{new}} L^{-\top} \f$ at 1, sets
  /// \f$ M_{\mathrm{lower}} \leftarrow L V \tilde\Lambda V^\top L^\top \f$ and refactors it.
  ///
  /// @param M_new New SPD metric observation.
  /// @return True if the bound was loosened, false if it already dominated.
  /// @throws std::invalid_argument if periods are set and the meet couples a
  ///         periodic axis. Attach periods to a converged bound instead.
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

    if (periods_.size() != 0) require_periodic_axes_decoupled(M_lower, periods_);

    llt_.compute(M_lower);
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

  /// @brief Access the underlying Cholesky factorization of \f$ M_{\mathrm{lower}} \f$.
  auto llt() const -> const Eigen::LLT<MatrixType>& { return llt_; }

  /// @brief Whether this heuristic has an eigenvalue floor set.
  auto has_eigenvalue_floor() const -> bool { return sqrt_lambda_min_floor_ > 0.0; }

 private:
  /// @brief Reject a bound whose periodic axes are coupled to any other axis.
  ///
  /// @details With coupling, a lattice translate along the periodic axis can cancel
  /// part of another component after \f$ L^\top \f$ is applied. No per-axis choice
  /// sees this, and the heuristic would overestimate.
  static void require_periodic_axes_decoupled(const MatrixType& M,
                                              const Eigen::VectorXd& periods) {
    constexpr double kRelTol = 1e-8;
    const int n = static_cast<int>(M.rows());
    for (int i = 0; i < n; ++i) {
      if (periods[i] <= 0.0) continue;
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        const double scale = std::sqrt(std::abs(M(i, i) * M(j, j)));
        if (std::abs(M(i, j)) > kRelTol * std::max(scale, 1.0)) {
          throw std::invalid_argument(
              "MatrixLowerBound: periodic axis " + std::to_string(i) +
              " is coupled to axis " + std::to_string(j) +
              " in M_lower, so per-axis wrapping is not the deck-group minimum and the "
              "heuristic would be inadmissible.");
        }
      }
    }
  }

  Eigen::LLT<MatrixType> llt_;
  Eigen::VectorXd periods_;  ///< empty, or one period per coordinate (0 = aperiodic)
  double sqrt_lambda_min_floor_;
};

}  // namespace geodex::heuristics

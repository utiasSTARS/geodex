/// @file
/// @brief Precompiled Loewner lower bounds for the built-in robots' CRBA metric.
///
/// `geodex::robots::MassLowerBound<Robot::Panda>::matrix()` returns the constant
/// SPD matrix \f$ M_{\mathrm{lower}} \f$ with \f$ M(q) \succeq M_{\mathrm{lower}} \f$
/// (Loewner order) for every \f$ q \f$ in the robot's joint-limit box, where
/// \f$ M(q) \f$ is the CRBA joint-space mass matrix (`robots::MassMatrix<R>`).
/// This is exactly the admissible bound consumed by
/// `geodex::heuristics::MatrixLowerBound`, but certified once at build time and
/// shipped in the generated `<robot>_bound.hpp` headers; so planners load a
/// constant instead of re-running `algorithm::precompute_matrix_lower_bound` at
/// startup.
///
/// The bound is metric-specific (CRBA kinetic energy) and box-specific (the URDF
/// joint limits). For a different metric, a tighter sub-box, or a runtime-URDF
/// robot, use `algorithm::precompute_matrix_lower_bound` directly instead. (A
/// bound certified over the full joint-limit box remains admissible on any
/// sub-box, just possibly looser.)
///
/// This header is consumer-only: it is NOT compiled into `geodex_robots`, so the
/// generated bound headers are never a build-order prerequisite for the library
/// or for the `precompute_robot_bound` tool that produces them.
///
/// @see Phone Thiha Kyaw, Jonathan Kelly. "Direct Informed Sampling on
///   Riemannian Manifolds via Loewner Order Lower Bounds." arXiv:2606.02879 (2026).

#pragma once

#include <Eigen/Core>

#include "geodex/robots/mass_matrix.hpp"

// GEODEX_ROBOT_BOUND_INCLUDES_BEGIN
#include "generated/panda_bound.hpp"
#include "generated/ur5_bound.hpp"
#include "generated/fetch_bound.hpp"
#include "generated/baxter_bound.hpp"
#include "generated/pr2_bound.hpp"
// GEODEX_ROBOT_BOUND_INCLUDES_END

namespace geodex::robots {

namespace detail {

/// @brief Compile-time access to a robot's precompiled Loewner bound. Specialize for
/// each robot that ships a `<robot>_bound.hpp`. Instantiating `MassLowerBound<R>`
/// for an `R` without a specialization fails at compile time with an
/// "incomplete type" error — the same contract as `RobotTraits`.
template <Robot R>
struct RobotBoundTraits;

// GEODEX_ROBOT_BOUND_TRAITS_BEGIN
template <>
struct RobotBoundTraits<Robot::Panda> {
  static constexpr const double* data = generated::panda_mass_lower_bound;
  static constexpr int count = generated::panda_lower_bound_count;
  static constexpr double certificate = generated::panda_mass_lower_bound_certificate;
  static constexpr bool converged = generated::panda_mass_lower_bound_converged;
};

template <>
struct RobotBoundTraits<Robot::Ur5> {
  static constexpr const double* data = generated::ur5_mass_lower_bound;
  static constexpr int count = generated::ur5_lower_bound_count;
  static constexpr double certificate = generated::ur5_mass_lower_bound_certificate;
  static constexpr bool converged = generated::ur5_mass_lower_bound_converged;
};

template <>
struct RobotBoundTraits<Robot::Fetch> {
  static constexpr const double* data = generated::fetch_mass_lower_bound;
  static constexpr int count = generated::fetch_lower_bound_count;
  static constexpr double certificate = generated::fetch_mass_lower_bound_certificate;
  static constexpr bool converged = generated::fetch_mass_lower_bound_converged;
};

template <>
struct RobotBoundTraits<Robot::Baxter> {
  static constexpr const double* data = generated::baxter_mass_lower_bound;
  static constexpr int count = generated::baxter_lower_bound_count;
  static constexpr double certificate = generated::baxter_mass_lower_bound_certificate;
  static constexpr bool converged = generated::baxter_mass_lower_bound_converged;
};

template <>
struct RobotBoundTraits<Robot::Pr2> {
  static constexpr const double* data = generated::pr2_mass_lower_bound;
  static constexpr int count = generated::pr2_lower_bound_count;
  static constexpr double certificate = generated::pr2_mass_lower_bound_certificate;
  static constexpr bool converged = generated::pr2_mass_lower_bound_converged;
};
// GEODEX_ROBOT_BOUND_TRAITS_END

}  // namespace detail

/// @brief Precompiled Loewner lower bound for robot @p R's CRBA kinetic-energy metric.
///
/// @tparam R A `Robot` enumerator that ships a `<robot>_bound.hpp`.
template <Robot R>
struct MassLowerBound {
 private:
  using Bound = detail::RobotBoundTraits<R>;

 public:
  /// @brief Velocity-space dimension (square size of `M_lower`).
  static constexpr int Nv = MassMatrix<R>::Nv;

  /// @brief Fixed-size matrix type for `M_lower`.
  using Mat = Eigen::Matrix<double, Nv, Nv>;

  /// @brief True when a precompiled bound ships for this robot. Trivially true here:
  /// the primary `RobotBoundTraits` is undefined, so a new robot fails to
  /// instantiate. Exposed so consumers can `static_assert(...::available, ...)`.
  static constexpr bool available = true;

  /// @brief Worst-case certificate `min_q lambda_min(L^-1 M(q) L^-T)` from the
  /// precompute. Values `>= 1 - tol` certify admissibility over the box.
  static constexpr double certificate = Bound::certificate;

  /// @brief Whether the certifying precompute reached its convergence tolerance.
  static constexpr bool converged = Bound::converged;

  /// @brief Reconstruct the symmetric SPD `M_lower` from its row-major upper
  /// triangle (same unpack as `MassMatrix::operator()`).
  static auto matrix() -> Mat {
    Mat M;
    int k = 0;
    for (int i = 0; i < Nv; ++i) {
      M(i, i) = Bound::data[k++];
      for (int j = i + 1; j < Nv; ++j) {
        const double v = Bound::data[k++];
        M(i, j) = v;
        M(j, i) = v;
      }
    }
    return M;
  }
};

}  // namespace geodex::robots

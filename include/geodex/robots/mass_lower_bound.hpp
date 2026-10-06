/// @file
/// @brief Precompiled Loewner lower bounds for the built-in robots' CRBA metric.
///
/// `geodex::robots::MassLowerBound<Robot::Panda>::matrix()` returns the constant
/// SPD matrix \f$ M_{\mathrm{lower}} \f$ with \f$ M(q) \succeq M_{\mathrm{lower}} \f$
/// (Loewner order) for every \f$ q \f$ in the robot's joint-limit box, where
/// \f$ M(q) \f$ is the CRBA joint-space mass matrix (`robots::MassMatrix<R>`).
/// This is the admissible bound that `geodex::heuristics::MatrixLowerBound` takes,
/// certified once and shipped in the generated `<robot>_bound.hpp` headers. Planners
/// load a constant and do not run `algorithm::precompute_matrix_lower_bound` at
/// startup.
///
/// The bound is specific to the metric (CRBA kinetic energy) and to the box (the URDF
/// joint limits). For a different metric, a tighter sub-box or a runtime-URDF robot,
/// use `algorithm::precompute_matrix_lower_bound`. A bound certified over the full
/// joint-limit box stays admissible, and possibly looser, on any sub-box.
///
/// @see Phone Thiha Kyaw, Jonathan Kelly. "Direct Informed Sampling on
///   Riemannian Manifolds via Loewner Order Lower Bounds." IEEE Robotics and
///   Automation Letters (RA-L), 2026. arXiv:2606.02879.

#pragma once

#include <Eigen/Core>

#include "geodex/robots/mass_matrix.hpp"

// GEODEX_ROBOT_BOUND_INCLUDES_BEGIN
#include "generated/baxter_bound.hpp"
#include "generated/fr3_gripper_bound.hpp"
#include "generated/husky_ur5e_bound.hpp"
#include "generated/panda_bound.hpp"
#include "generated/pr2_bound.hpp"
#include "generated/ridgeback_ur5e_bound.hpp"
#include "generated/stretch3_bound.hpp"
#include "generated/stretch4_bound.hpp"
#include "generated/ur5_bound.hpp"
// GEODEX_ROBOT_BOUND_INCLUDES_END

namespace geodex::robots {

namespace detail {

/// @brief Compile-time access to a robot's precompiled Loewner bound. Specialize for
/// each robot that ships a `<robot>_bound.hpp`. Instantiating `MassLowerBound<R>`
/// for an `R` without a specialization fails at compile time with an
/// "incomplete type" error, the same contract as `RobotTraits`.
template <Robot R>
struct RobotBoundTraits;

// GEODEX_ROBOT_BOUND_TRAITS_BEGIN
template <>
struct RobotBoundTraits<Robot::Baxter> {
  static constexpr const double* data = generated::baxter_mass_lower_bound;
  static constexpr int count = generated::baxter_lower_bound_count;
  static constexpr double certificate = generated::baxter_mass_lower_bound_certificate;
  static constexpr bool converged = generated::baxter_mass_lower_bound_converged;
  static constexpr bool proved = generated::baxter_mass_lower_bound_proved;
};

template <>
struct RobotBoundTraits<Robot::Fr3Gripper> {
  static constexpr const double* data = generated::fr3_gripper_mass_lower_bound;
  static constexpr int count = generated::fr3_gripper_lower_bound_count;
  static constexpr double certificate = generated::fr3_gripper_mass_lower_bound_certificate;
  static constexpr bool converged = generated::fr3_gripper_mass_lower_bound_converged;
  static constexpr bool proved = generated::fr3_gripper_mass_lower_bound_proved;
};

template <>
struct RobotBoundTraits<Robot::HuskyUr5e> {
  static constexpr const double* data = generated::husky_ur5e_mass_lower_bound;
  static constexpr int count = generated::husky_ur5e_lower_bound_count;
  static constexpr double certificate = generated::husky_ur5e_mass_lower_bound_certificate;
  static constexpr bool converged = generated::husky_ur5e_mass_lower_bound_converged;
  static constexpr bool proved = generated::husky_ur5e_mass_lower_bound_proved;
};

template <>
struct RobotBoundTraits<Robot::Panda> {
  static constexpr const double* data = generated::panda_mass_lower_bound;
  static constexpr int count = generated::panda_lower_bound_count;
  static constexpr double certificate = generated::panda_mass_lower_bound_certificate;
  static constexpr bool converged = generated::panda_mass_lower_bound_converged;
  static constexpr bool proved = generated::panda_mass_lower_bound_proved;
};

template <>
struct RobotBoundTraits<Robot::Pr2> {
  static constexpr const double* data = generated::pr2_mass_lower_bound;
  static constexpr int count = generated::pr2_lower_bound_count;
  static constexpr double certificate = generated::pr2_mass_lower_bound_certificate;
  static constexpr bool converged = generated::pr2_mass_lower_bound_converged;
  static constexpr bool proved = generated::pr2_mass_lower_bound_proved;
};

template <>
struct RobotBoundTraits<Robot::RidgebackUr5e> {
  static constexpr const double* data = generated::ridgeback_ur5e_mass_lower_bound;
  static constexpr int count = generated::ridgeback_ur5e_lower_bound_count;
  static constexpr double certificate = generated::ridgeback_ur5e_mass_lower_bound_certificate;
  static constexpr bool converged = generated::ridgeback_ur5e_mass_lower_bound_converged;
  static constexpr bool proved = generated::ridgeback_ur5e_mass_lower_bound_proved;
};

template <>
struct RobotBoundTraits<Robot::Stretch3> {
  static constexpr const double* data = generated::stretch3_mass_lower_bound;
  static constexpr int count = generated::stretch3_lower_bound_count;
  static constexpr double certificate = generated::stretch3_mass_lower_bound_certificate;
  static constexpr bool converged = generated::stretch3_mass_lower_bound_converged;
  static constexpr bool proved = generated::stretch3_mass_lower_bound_proved;
};

template <>
struct RobotBoundTraits<Robot::Stretch4> {
  static constexpr const double* data = generated::stretch4_mass_lower_bound;
  static constexpr int count = generated::stretch4_lower_bound_count;
  static constexpr double certificate = generated::stretch4_mass_lower_bound_certificate;
  static constexpr bool converged = generated::stretch4_mass_lower_bound_converged;
  static constexpr bool proved = generated::stretch4_mass_lower_bound_proved;
};

template <>
struct RobotBoundTraits<Robot::Ur5> {
  static constexpr const double* data = generated::ur5_mass_lower_bound;
  static constexpr int count = generated::ur5_lower_bound_count;
  static constexpr double certificate = generated::ur5_mass_lower_bound_certificate;
  static constexpr bool converged = generated::ur5_mass_lower_bound_converged;
  static constexpr bool proved = generated::ur5_mass_lower_bound_proved;
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

  /// @brief True when a precompiled bound ships for this robot, always true here. The
  /// primary `RobotBoundTraits` is undefined, and a robot without a bound fails to
  /// instantiate. Consumers can `static_assert(...::available, ...)`.
  static constexpr bool available = true;

  /// @brief Lower bound of `lambda_min(M_lower^-1 M(q))` over the joint box, at least 1.
  /// When `proved` is true, an interval branch and bound over the generated CRBA
  /// expression proves it, and `M(q) >= M_lower` holds at every `q` in the box.
  /// Otherwise it is the smallest value a sampled and pattern search found, and
  /// `M_lower` sits 3 percent below that.
  static constexpr double certificate = Bound::certificate;

  /// @brief Whether the shaping of `M_lower` converged.
  static constexpr bool converged = Bound::converged;

  /// @brief Whether `M(q) >= M_lower` is proved over the whole joint box.
  static constexpr bool proved = Bound::proved;

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

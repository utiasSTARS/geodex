/// @file sweep_model.hpp
/// @brief Per-kernel bounds on sphere travel and the kernel's joint limits (internal).
///
/// `robotgen.py sweep` writes one `SweepModel` per VAMP kernel into
/// `robots/generated/<robot>_sweep.hh`. The certified motion validator turns a motion's
/// coordinate change into a bound on how far any sphere center moves, and the checkers
/// test the joint limits. The header does not use VAMP or SIMD types.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace geodex::integration::vamp::detail {

/// @brief Travel bounds and joint limits of one kernel with @p N coordinates.
///
/// For coordinate `j`, a sphere center moves at most `reach[j]` per unit change of that
/// coordinate, and the end-effector frame origin at most `ee_reach[j]`. A point held at
/// distance `a` from the end-effector origin moves at most
/// `ee_reach[j] + a * ee_rotation[j]`. With a planar base, coordinates 0 to 2 are
/// `(x, y, theta)` and a center lies at most `base_reach` from the base's yaw axis.
template <std::size_t N>
struct SweepModel {
  bool planar_base = false;      ///< coordinates 0 to 2 are an SE(2) base pose
  double base_reach = 0.0;       ///< largest distance of a sphere center from the yaw axis
  double ee_base_reach = 0.0;    ///< distance of the end-effector origin from the yaw axis
  std::array<double, N> reach{};        ///< sphere-center speed per unit of each coordinate
  std::array<double, N> ee_reach{};     ///< end-effector-origin speed per unit
  std::array<double, N> ee_rotation{};  ///< rotation rate of the end-effector frame per unit
  std::array<double, N> lower{};        ///< joint lower limits
  std::array<double, N> upper{};        ///< joint upper limits
};

/// @brief Bound on how far any sphere center moves along a motion, with the spheres of an
/// attached body folded in.
///
/// A motion with base twist `(vx, vy, omega)` and joint change `dq` moves a center at most
/// `|(vx, vy)| + |omega| base + sum_j joint[j] |dq_j|`. Along a curve with a constant twist and
/// linear joints, the bound holds for every part of the motion in proportion to its share.
struct TravelBound {
  bool planar_base = false;   ///< the tangent starts with the base twist
  double base = 0.0;          ///< largest distance of a center from the base's yaw axis
  std::vector<double> joint;  ///< center speed per unit of each joint after the base

  /// @brief Bound for the motion with tangent @p v, the base twist first.
  double operator()(const double* v) const {
    double d = 0.0;
    std::size_t first = 0;
    if (planar_base) {
      d += std::hypot(v[0], v[1]) + std::abs(v[2]) * base;
      first = 3;
    }
    for (std::size_t j = 0; j < joint.size(); ++j) d += joint[j] * std::abs(v[first + j]);
    return d;
  }

  /// @brief Bound per unit coordinate norm of the tangent, the norm of the coefficients.
  double speed() const {
    double sum = planar_base ? 1.0 + base * base : 0.0;
    for (const double r : joint) sum += r * r;
    return std::sqrt(sum);
  }
};

}  // namespace geodex::integration::vamp::detail

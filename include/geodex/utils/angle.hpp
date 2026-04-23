/// @file angle.hpp
/// @brief Angle wrapping utilities for periodic coordinates.

#pragma once

#include <numbers>

#include <Eigen/Core>

namespace geodex::utils {

/// @brief Wrap angle to \f$ [-\pi, \pi) \f$.
///
/// @note Uses while-loop wrapping instead of std::fmod for performance (3-4x
/// faster on typical inputs). Assumes inputs are within a few multiples of
/// \f$ 2\pi \f$, which holds for all geodex exp/log operations. Do not call
/// with extremely large values (e.g. > 1e6) as the loop count grows linearly.
inline double wrap_to_pi(double theta) {
  constexpr double two_pi = 2.0 * std::numbers::pi;
  while (theta >= std::numbers::pi) theta -= two_pi;
  while (theta < -std::numbers::pi) theta += two_pi;
  return theta;
}

/// @brief Wrap angle to \f$ [0, 2\pi) \f$.
inline double wrap_to_2pi(double theta) {
  constexpr double two_pi = 2.0 * std::numbers::pi;
  while (theta >= two_pi) theta -= two_pi;
  while (theta < 0.0) theta += two_pi;
  return theta;
}

/// @brief Wrap each component of a vector to \f$ [0, 2\pi) \f$.
template <int Dim>
Eigen::Vector<double, Dim> wrap_point(const Eigen::Vector<double, Dim>& p) {
  return p.unaryExpr([](double x) { return wrap_to_2pi(x); });
}

/// @brief Wrap each component of a difference vector to \f$ [-\pi, \pi) \f$.
template <int Dim>
Eigen::Vector<double, Dim> wrap_delta(const Eigen::Vector<double, Dim>& d) {
  return d.unaryExpr([](double x) { return wrap_to_pi(x); });
}

}  // namespace geodex::utils

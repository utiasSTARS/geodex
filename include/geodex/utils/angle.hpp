/// @file angle.hpp
/// @brief Angle wrapping utilities for periodic coordinates.

#pragma once

#include <cmath>

#include <numbers>

#include <Eigen/Core>

namespace geodex::utils {

/// @brief The full turn \f$ 2\pi \f$, the period of a circle coordinate.
inline constexpr double two_pi = 2.0 * std::numbers::pi;

/// @brief Wrap angle to \f$ [-\pi, \pi) \f$.
///
/// @note Wraps with a loop. Inputs must lie within a few periods of the range, which
/// holds for all geodex exp/log operations. The loop count grows linearly with the
/// input magnitude.
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

/// @brief Wrap a coordinate difference to \f$ [-P/2, P/2) \f$ for a general period \f$ P \f$.
///
/// @details The nearest representative of \f$ d \f$ under the deck group
/// \f$ P\mathbb{Z} \f$, i.e. \f$ d - P\,\mathrm{round}(d/P) \f$. Correct for any
/// magnitude of \f$ d \f$, and a no-op when \f$ d \f$ already lies in range.
/// A non-positive period means the axis is not periodic and \f$ d \f$ is returned
/// unchanged.
inline double wrap_delta_period(double d, const double period) {
  if (period <= 0.0) return d;
  const double half = 0.5 * period;
  if (d < -half || d >= half) d -= period * std::round(d / period);
  return d;
}

}  // namespace geodex::utils

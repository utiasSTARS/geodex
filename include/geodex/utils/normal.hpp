/// @file normal.hpp
/// @brief Inverse standard normal CDF for turning unit-cube samples into normal variates.

#pragma once

#include <cmath>

#include <numbers>

namespace geodex::utils {

/// @brief Inverse standard normal CDF (probit), Acklam's rational approximation
/// refined by one Halley step against `std::erfc` to full double precision.
///
/// @details Used wherever a low-discrepancy unit-cube coordinate must become a
/// standard normal variate, such as a direction uniform on a sphere or in a ball.
/// @param p Probability, clamped to \f$ [10^{-300}, 1 - 10^{-16}] \f$. The
///   endpoints of the unit interval map to finite values.
/// @return The \f$ x \f$ with \f$ \Phi(x) = p \f$.
inline double normal_quantile(double p) {
  if (p < 1e-300) p = 1e-300;
  if (p > 1.0 - 1e-16) p = 1.0 - 1e-16;
  static constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                 -2.759285104469687e+02, 1.383577518672690e+02,
                                 -3.066479806614716e+01, 2.506628277459239e+00};
  static constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                 -1.556989798598866e+02, 6.680131188771972e+01,
                                 -1.328068155288572e+01};
  static constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                 -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00,  2.938163982698783e+00};
  static constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                                 2.445134137142996e+00, 3.754408661907416e+00};
  constexpr double plow = 0.02425;
  double x;
  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else if (p <= 1.0 - plow) {
    const double q = p - 0.5;
    const double r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  } else {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  const double e = 0.5 * std::erfc(-x / std::numbers::sqrt2) - p;
  const double u = e * std::sqrt(2.0 * std::numbers::pi) * std::exp(0.5 * x * x);
  return x - u / (1.0 + 0.5 * x * u);
}

}  // namespace geodex::utils

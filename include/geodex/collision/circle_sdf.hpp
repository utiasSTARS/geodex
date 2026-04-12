/// @file circle_sdf.hpp
/// @brief Signed distance fields for circular obstacles.
///
/// Provides a single-circle SDF and a log-sum-exp smooth-min combiner for
/// multiple circles. The smooth SDF is suitable for use with SDFConformalMetric
/// to bias planning away from obstacles.

#pragma once

#include <cmath>
#include <geodex/utils/math.hpp>
#include <limits>
#include <vector>

namespace geodex::collision {

/// @brief Signed distance to a single circular obstacle.
///
/// Returns positive values in free space and negative inside the circle:
/// \f$ \mathrm{sdf}(q) = \|q_{xy} - c\| - r \f$
class CircleSDF {
 public:
  CircleSDF(const double cx, const double cy, const double radius)
      : cx_(cx), cy_(cy), radius_(radius), radius_sq_(radius * radius) {}

  template <typename Point>
  double operator()(const Point& q) const {
    const double dx = q[0] - cx_, dy = q[1] - cy_;
    return std::sqrt(dx * dx + dy * dy) - radius_;
  }

  double cx() const { return cx_; }
  double cy() const { return cy_; }
  double radius() const { return radius_; }
  double radius_sq() const { return radius_sq_; }

 private:
  double cx_, cy_, radius_, radius_sq_;
};

/// @brief Log-sum-exp smooth-min SDF over a collection of circular obstacles.
///
/// Produces a smooth approximation to the minimum signed distance across all
/// circles. The smoothing parameter @p beta controls how closely the smooth-min
/// approximates the hard minimum (higher values give a tighter approximation
/// but sharper gradient transitions).
///
/// The numerically stable computation is:
/// \f[
///   \mathrm{sdf}(q) = -\frac{1}{\beta}\left(m + \ln\sum_i
///   \exp(-\beta\,d_i - m)\right), \quad m = \max_i(-\beta\,d_i)
/// \f]
class CircleSmoothSDF {
 public:
  CircleSmoothSDF(std::vector<CircleSDF> circles, const double beta = 20.0)
      : circles_(std::move(circles)), beta_(beta) {}

  template <typename Point>
  double operator()(const Point& q) const {
    if (circles_.empty()) return std::numeric_limits<double>::max();

    const auto n = circles_.size();

    // Single-pass: compute and cache -beta * d_i, track max for stability.
    thread_local std::vector<double> neg_bd;
    neg_bd.resize(n);

    double max_neg = std::numeric_limits<double>::lowest();
    for (size_t i = 0; i < n; ++i) {
      neg_bd[i] = -beta_ * circles_[i](q);
      max_neg = std::max(max_neg, neg_bd[i]);
    }

    // Accumulate exp(-beta * d_i - max_neg) from cached values.
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      sum += utils::fast_exp(neg_bd[i] - max_neg);
    }

    return -(max_neg + std::log(sum)) / beta_;
  }

  /// @brief Check if a point is outside all circles (binary collision test).
  template <typename Point>
  bool is_free(const Point& q) const {
    for (const auto& c : circles_) {
      const double dx = q[0] - c.cx(), dy = q[1] - c.cy();
      if (dx * dx + dy * dy <= c.radius_sq()) return false;
    }
    return true;
  }

  const std::vector<CircleSDF>& circles() const { return circles_; }
  double beta() const { return beta_; }

 private:
  std::vector<CircleSDF> circles_;
  double beta_;
};

}  // namespace geodex::collision

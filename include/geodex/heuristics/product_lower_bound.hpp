/// @file product_lower_bound.hpp
/// @brief Matrix lower-bound heuristic of a product metric from the bounds of its factors.

#pragma once

#include <cstddef>

#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "geodex/heuristics/matrix_lower_bound.hpp"

namespace geodex::heuristics {

/// @brief Loewner lower bound of one factor of a product metric.
struct FactorBound {
  /// @brief Constant SPD matrix below the factor's metric on its coordinate velocities.
  Eigen::MatrixXd matrix;

  /// @brief The factor's coordinate periods, 0 on an aperiodic coordinate. Empty when no
  /// coordinate of the factor is periodic.
  Eigen::VectorXd periods;
};

/// @brief Matrix lower-bound heuristic of a product metric from the bounds of its factors.
///
/// @details The metric of `ProductManifold<M1, ..., MN>` is the direct sum
/// \f$ g = g_1 \oplus \cdots \oplus g_N \f$. If \f$ g_i \succeq M_i \f$ for every factor, the
/// block-diagonal matrix \f$ M_1 \oplus \cdots \oplus M_N \f$ bounds \f$ g \f$ from below.
/// The heuristic takes that matrix and the factors' periods in the same order, and it
/// wraps every periodic coordinate to its nearest representative.
///
/// @code
/// // A differential-drive base and a 5-joint arm on SE(2) x R^5.
/// const auto metric = geodex::SE2LeftInvariantMetric::differential_drive();
/// const geodex::SE2<> base(metric, lo, hi);
/// const auto heuristic = geodex::heuristics::product_lower_bound(
///     {{metric.coordinate_lower_bound(), base.periods()}, {arm_bound}});
/// @endcode
///
/// @param factors One bound per factor, in the order of the product's coordinates.
/// @return The heuristic of the block-diagonal bound. It carries periods when a factor
///         has them.
/// @throws std::invalid_argument when @p factors is empty, a matrix is not square, or a
///         factor's periods do not match its matrix.
inline MatrixLowerBound<Eigen::Dynamic> product_lower_bound(
    const std::vector<FactorBound>& factors) {
  if (factors.empty()) throw std::invalid_argument("product_lower_bound: no factors.");
  Eigen::Index n = 0;
  bool periodic = false;
  for (std::size_t i = 0; i < factors.size(); ++i) {
    const FactorBound& f = factors[i];
    if (f.matrix.rows() != f.matrix.cols()) {
      throw std::invalid_argument("product_lower_bound: the matrix of factor " + std::to_string(i) +
                                  " is not square.");
    }
    if (f.periods.size() != 0 && f.periods.size() != f.matrix.rows()) {
      throw std::invalid_argument("product_lower_bound: factor " + std::to_string(i) + " has " +
                                  std::to_string(f.periods.size()) + " periods for a " +
                                  std::to_string(f.matrix.rows()) + "-dimensional matrix.");
    }
    n += f.matrix.rows();
    periodic = periodic || f.periods.size() != 0;
  }

  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd periods;
  if (periodic) periods = Eigen::VectorXd::Zero(n);
  Eigen::Index offset = 0;
  for (const FactorBound& f : factors) {
    const Eigen::Index k = f.matrix.rows();
    M.block(offset, offset, k, k) = f.matrix;
    if (f.periods.size() != 0) periods.segment(offset, k) = f.periods;
    offset += k;
  }
  return MatrixLowerBound<Eigen::Dynamic>(M, periods);
}

}  // namespace geodex::heuristics

/// @file ordered_sum.hpp
/// @brief Dot products, norms and quadratic forms that add their terms in index order.
///
/// @details Eigen chooses the summation order of a reduction from the compile-time size
/// of its operands. A fixed-size vector and a dynamic-size vector with the same entries
/// can give sums that differ in the last bit. These functions add their terms in index
/// order for every storage. Each product is a statement of its own, and a compiler that
/// contracts only within one expression, as Clang does by default, rounds it before the
/// sum. The smoother, the Euclidean heuristic and the kinetic-energy metric use them. A
/// plan on a fixed-size manifold computes the same bits as the same plan through a
/// dynamic-size wrapper, such as the Python bindings.

#pragma once

#include <cmath>

#include <Eigen/Core>

namespace geodex::utils {

/// @brief \f$ u^\top v \f$, summed in index order.
/// @param u First vector.
/// @param v Second vector, the size of @p u.
/// @return The dot product.
template <typename U, typename V>
double ordered_dot(const Eigen::MatrixBase<U>& u, const Eigen::MatrixBase<V>& v) {
  double s = 0.0;
  for (Eigen::Index i = 0; i < u.size(); ++i) {
    const double term = static_cast<double>(u.coeff(i)) * static_cast<double>(v.coeff(i));
    s += term;
  }
  return s;
}

/// @brief \f$ \|v\|_2^2 \f$, summed in index order.
/// @param v The vector.
/// @return The squared Euclidean norm.
template <typename V>
double ordered_squared_norm(const Eigen::MatrixBase<V>& v) {
  return ordered_dot(v, v);
}

/// @brief \f$ \|v\|_2 \f$, the square root of `ordered_squared_norm`.
/// @param v The vector.
/// @return The Euclidean norm.
template <typename V>
double ordered_norm(const Eigen::MatrixBase<V>& v) {
  return std::sqrt(ordered_squared_norm(v));
}

/// @brief \f$ u^\top A v \f$. Each entry of \f$ A v \f$ is summed in column order, and
/// the dot product with @p u in row order.
/// @param u Left vector.
/// @param a Square matrix.
/// @param v Right vector.
/// @return The quadratic form.
template <typename U, typename A, typename V>
double ordered_quadratic_form(const Eigen::MatrixBase<U>& u, const Eigen::MatrixBase<A>& a,
                              const Eigen::MatrixBase<V>& v) {
  double s = 0.0;
  for (Eigen::Index i = 0; i < a.rows(); ++i) {
    double row = 0.0;
    for (Eigen::Index j = 0; j < a.cols(); ++j) {
      const double term = static_cast<double>(a.coeff(i, j)) * static_cast<double>(v.coeff(j));
      row += term;
    }
    const double term = static_cast<double>(u.coeff(i)) * row;
    s += term;
  }
  return s;
}

}  // namespace geodex::utils

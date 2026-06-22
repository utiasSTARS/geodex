/// @file traits.hpp
/// @brief Type traits for detecting admissible-heuristic kinds.

#pragma once

#include <type_traits>

namespace geodex::heuristics {

// Forward declarations — full definitions in the per-class headers.
template <typename BaseHeuristicT>
class EigenvalueLowerBound;

template <int Dim>
class MatrixLowerBound;

/// @brief Detect any `MatrixLowerBound<Dim>` specialization.
template <typename T>
struct is_matrix_lower_bound : std::false_type {};

/// @copydoc is_matrix_lower_bound
template <int Dim>
struct is_matrix_lower_bound<MatrixLowerBound<Dim>> : std::true_type {};

/// @brief Helper variable template: true iff `T` is a `MatrixLowerBound<Dim>`.
template <typename T>
inline constexpr bool is_matrix_lower_bound_v = is_matrix_lower_bound<T>::value;

/// @brief Detect any `EigenvalueLowerBound<BaseH>` specialization.
template <typename T>
struct is_eigenvalue_lower_bound : std::false_type {};

/// @copydoc is_eigenvalue_lower_bound
template <typename BaseH>
struct is_eigenvalue_lower_bound<EigenvalueLowerBound<BaseH>> : std::true_type {};

/// @brief Helper variable template: true iff `T` is an `EigenvalueLowerBound<BaseH>`.
template <typename T>
inline constexpr bool is_eigenvalue_lower_bound_v = is_eigenvalue_lower_bound<T>::value;

}  // namespace geodex::heuristics

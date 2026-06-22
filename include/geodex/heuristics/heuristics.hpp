/// @file heuristics.hpp
/// @brief Umbrella header for `geodex::heuristics::` — admissible heuristics for
/// motion planning on Riemannian manifolds.
///
/// @details Pulls in `Zero`, `Euclidean`, `EigenvalueLowerBound<Base>`,
/// `MatrixLowerBound<Dim>`, and the detection traits.

#pragma once

#include "geodex/heuristics/eigenvalue_lower_bound.hpp"
#include "geodex/heuristics/euclidean.hpp"
#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/heuristics/traits.hpp"
#include "geodex/heuristics/zero.hpp"

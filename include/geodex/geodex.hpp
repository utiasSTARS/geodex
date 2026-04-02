/// @file geodex.hpp
/// @brief Umbrella header for the geodex library.
/// @details Includes all core concepts, manifold implementations, and algorithms.

#pragma once

#include "algorithm/distance.hpp"
#include "core/concepts.hpp"
#include "core/distance.hpp"
#include "core/interpolation.hpp"
#include "core/metric.hpp"
#include "core/retraction.hpp"
#include "manifold/configuration_space.hpp"
#include "manifold/euclidean.hpp"
#include "manifold/se2.hpp"
#include "manifold/sphere.hpp"
#include "manifold/torus.hpp"
#include "metrics/constant_spd.hpp"
#include "metrics/jacobi.hpp"
#include "metrics/kinetic_energy.hpp"
#include "metrics/pullback.hpp"
#include "metrics/se2_left_invariant.hpp"
#include "metrics/weighted.hpp"

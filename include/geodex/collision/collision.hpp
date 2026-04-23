/// @file collision.hpp
/// @brief Umbrella include for the geodex::collision module.
///
/// Includes all collision primitives:
///   - CircleSDF, CircleSmoothSDF
///   - RectObstacle, RectSmoothSDF, rects_overlap
///   - DistanceGrid, GridSDF, InflatedSDF
///   - PolygonFootprint
///   - FootprintGridChecker

#pragma once

#include "geodex/collision/circle_sdf.hpp"
#include "geodex/collision/distance_grid.hpp"
#include "geodex/collision/footprint_grid_checker.hpp"
#include "geodex/collision/polygon_footprint.hpp"
#include "geodex/collision/rectangle_sdf.hpp"

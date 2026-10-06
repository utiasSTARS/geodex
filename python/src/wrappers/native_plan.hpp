/// @file native_plan.hpp
/// @brief plan() on the typed space of a ConfigurationSpace over SE2 with a clearance metric.

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include <Eigen/Core>

#include "geodex/heuristics/heuristics.hpp"
#include "geodex/planning/plan.hpp"

#include "se2_clearance_space.hpp"

namespace geodex::python {

/// @brief The heuristics that plan() takes from Python for a ConfigurationSpace.
using PlanHeuristic = std::variant<heuristics::Zero, heuristics::Euclidean,
                                   heuristics::EigenvalueLowerBound<heuristics::Euclidean>,
                                   heuristics::MatrixLowerBound<Eigen::Dynamic>>;

/// @brief `planning::plan()` on `SE2ClearanceSpace`, the typed space of `parts`.
///
/// @details The result equals the result of the same plan on the type-erased space.
/// @param parts The space.
/// @param start Start pose.
/// @param goal Goal pose.
/// @param is_valid Validity of a pose. Empty means free space.
/// @param settings Plan settings.
/// @param heuristic Heuristic of the informed planner.
/// @param max_reverse_length Reverse budget of a DirectionalMotionValidator, or nothing
///        without one.
planning::PlanResult<Eigen::VectorXd> plan_se2_clearance(
    const SE2ClearanceParts& parts, const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
    const std::function<bool(const Eigen::Vector3d&)>& is_valid,
    const planning::PlanSettings& settings, const PlanHeuristic& heuristic,
    std::optional<double> max_reverse_length);

}  // namespace geodex::python

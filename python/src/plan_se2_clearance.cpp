/// @file plan_se2_clearance.cpp
/// @brief plan() on the typed space of a ConfigurationSpace over SE2 with the clearance
/// metric of a bound C++ SDF. The build compiles this file only with OMPL.

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <ompl/base/DiscreteMotionValidator.h>

#include "geodex/integration/ompl/directional_motion_validator.hpp"

#include "wrappers/native_plan.hpp"

namespace geodex::python {

namespace {

namespace ob = ::ompl::base;

/// @brief The result with dynamic-size points, as the Python binding holds it.
planning::PlanResult<Eigen::VectorXd> to_dynamic(const planning::PlanResult<Eigen::Vector3d>& r) {
  // The structured binding fails to compile when PlanResult gains a member. Copy every
  // member below.
  [[maybe_unused]] const auto& [solved, smoothed, path, raw_path, cost, time_ms, smooth_ms,
                                first_solution_ms, first_solution_iterations, informed_samples,
                                focused_samples, uniform_samples] = r;
  planning::PlanResult<Eigen::VectorXd> out;
  out.solved = r.solved;
  out.smoothed = r.smoothed;
  out.path.assign(r.path.begin(), r.path.end());
  out.raw_path.assign(r.raw_path.begin(), r.raw_path.end());
  out.cost = r.cost;
  out.time_ms = r.time_ms;
  out.smooth_ms = r.smooth_ms;
  out.first_solution_ms = r.first_solution_ms;
  out.first_solution_iterations = r.first_solution_iterations;
  out.informed_samples = r.informed_samples;
  out.focused_samples = r.focused_samples;
  out.uniform_samples = r.uniform_samples;
  return out;
}

}  // namespace

planning::PlanResult<Eigen::VectorXd> plan_se2_clearance(
    const SE2ClearanceParts& parts, const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
    const std::function<bool(const Eigen::Vector3d&)>& is_valid,
    const planning::PlanSettings& settings, const PlanHeuristic& heuristic,
    const std::optional<double> max_reverse_length) {
  return std::visit(
      [&](const auto& h) {
        using Heuristic = std::remove_cvref_t<decltype(h)>;
        return visit_se2_clearance(parts, [&](const auto& space) {
          using Space = std::remove_cvref_t<decltype(space)>;
          // The DirectionalMotionValidator of the type-erased plan, over the typed space.
          std::function<std::shared_ptr<ob::MotionValidator>(const ob::SpaceInformationPtr&)>
              motion_validator;
          if (max_reverse_length) {
            const double budget = *max_reverse_length;
            motion_validator = [space, budget](const ob::SpaceInformationPtr& si) {
              return std::make_shared<integration::ompl::DirectionalMotionValidator<Space>>(
                  si.get(), space, std::make_shared<ob::DiscreteMotionValidator>(si.get()), budget);
            };
          }
          return to_dynamic(planning::plan<Space, Heuristic>(space, start, goal, is_valid, settings,
                                                             h, motion_validator));
        });
      },
      heuristic);
}

}  // namespace geodex::python

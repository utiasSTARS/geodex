/// @file validity_checker.hpp
/// @brief Generic OMPL StateValidityChecker adapter for any callable.
///
/// Replaces per-example checker classes (SE2ObstacleChecker, GridObstacleChecker,
/// SE2PolygonCollisionChecker) with a single template that wraps any callable
/// returning bool from a manifold point.

#pragma once

#include <geodex/integration/ompl/geodex_state_space.hpp>
#include <memory>
#include <ompl/base/StateValidityChecker.h>

namespace geodex::ompl_integration {

/// @brief Generic OMPL StateValidityChecker from any bool(Point) callable.
///
/// Usage:
/// @code
///   auto validity = [&](const auto& q) { return checker.is_valid(q); };
///   ss.setStateValidityChecker(
///       make_validity_checker<ManifoldT>(si, validity));
/// @endcode
template <typename ManifoldT, typename ValidityFn>
class ValidityChecker : public ompl::base::StateValidityChecker {
 public:
  ValidityChecker(const ompl::base::SpaceInformationPtr& si, ValidityFn fn)
      : ompl::base::StateValidityChecker(si), fn_(std::move(fn)) {}

  bool isValid(const ompl::base::State* state) const override {
    const auto* s = state->as<GeodexState<ManifoldT>>();
    return fn_(s->asEigen());
  }

 private:
  ValidityFn fn_;
};

/// @brief Convenience factory (deduces ValidityFn from the argument).
template <typename ManifoldT, typename ValidityFn>
auto make_validity_checker(const ompl::base::SpaceInformationPtr& si, ValidityFn fn) {
  return std::make_shared<ValidityChecker<ManifoldT, ValidityFn>>(si, std::move(fn));
}

}  // namespace geodex::ompl_integration

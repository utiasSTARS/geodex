/// @file directional_motion_validator.hpp
/// @brief Motion validator that additionally requires forward drivability.
///
/// On SE(2) with a group retraction, the screw between two poses has constant
/// body velocity, and the forward component of `log(a, b)` decides the gear of
/// the whole edge. The validator rejects an edge whose body-forward motion is below
/// `-max_reverse_length`. Pure rotation passes.

#pragma once

#include <memory>
#include <utility>

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>

#include "geodex/integration/ompl/geodex_state_space.hpp"

namespace geodex::integration::ompl {

namespace ob = ::ompl::base;

/// @brief Forward-drivability constraint stacked on an inner motion validator.
///
/// @tparam ManifoldT Manifold whose `log(a, b)` returns a body twist with the
///         forward component at index 0.
template <typename ManifoldT>
class DirectionalMotionValidator : public ob::MotionValidator {
 public:
  using StateType = GeodexState<ManifoldT>;  ///< OMPL state type of the space.

  /// @brief Wrap `inner`, which checks collisions, with the forward-motion test.
  /// @param si Space information of the planner.
  /// @param manifold Manifold whose `log` gives the edge's body twist.
  /// @param inner Motion validator that decides collisions along forward edges.
  /// @param max_reverse_length Reverse budget per edge in body-forward units.
  ///        0 forbids any net reverse motion.
  DirectionalMotionValidator(ob::SpaceInformation* si, ManifoldT manifold,
                             ob::MotionValidatorPtr inner, double max_reverse_length = 0.0)
      : ob::MotionValidator(si),
        manifold_(std::move(manifold)),
        inner_(std::move(inner)),
        max_reverse_length_(max_reverse_length) {}

  /// @brief True when the edge drives forward within the budget and `inner` accepts it.
  bool checkMotion(const ob::State* s1, const ob::State* s2) const override {
    if (!forward(s1, s2)) return false;
    return inner_->checkMotion(s1, s2);
  }

  /// @brief `checkMotion` that also reports the last valid state. A reverse edge
  /// reports `s1` at fraction 0.
  bool checkMotion(const ob::State* s1, const ob::State* s2,
                   std::pair<ob::State*, double>& last_valid) const override {
    if (!forward(s1, s2)) {
      if (last_valid.first != nullptr) si_->copyState(last_valid.first, s1);
      last_valid.second = 0.0;
      return false;
    }
    return inner_->checkMotion(s1, s2, last_valid);
  }

 private:
  bool forward(const ob::State* s1, const ob::State* s2) const {
    const auto a = s1->as<StateType>()->asEigen();
    const auto b = s2->as<StateType>()->asEigen();
    const auto twist = manifold_.log(typename ManifoldT::Point(a), typename ManifoldT::Point(b));
    return twist[0] >= -max_reverse_length_;
  }

  ManifoldT manifold_;
  ob::MotionValidatorPtr inner_;
  double max_reverse_length_;
};

}  // namespace geodex::integration::ompl

/// @file native_validity.hpp
/// @brief Bound `is_valid` methods that plan() calls in C++, without Python.
///
/// @details plan() takes a validity callable. When the callable is the bound `is_valid`
/// method of a FootprintGridChecker or a geodex.vamp CollisionChecker, plan() calls the
/// checker in place. A Python subclass that overrides `is_valid`, and any other callable,
/// runs through Python.

#pragma once

#include <optional>
#include <variant>

#include <Eigen/Core>
#include <nanobind/nanobind.h>

#include "geodex/collision/footprint_grid_checker.hpp"

#include "native_collision.hpp"

#ifdef GEODEX_PYTHON_HAS_VAMP
#include "geodex/integration/vamp/registry.hpp"
#endif

namespace geodex::python {

/// @brief A bound `is_valid` method of a geodex collision checker, called in place.
///
/// @details It points into the checker and does not keep it alive. The caller holds the
/// method object, which holds the checker.
class NativeValidity {
 public:
  /// @brief The checker whose bound `is_valid` method `fn` is, or nothing for any other
  /// callable.
  static std::optional<NativeValidity> find(nanobind::handle fn) {
    const auto method = detail::bound_method(fn);
    if (!method) return std::nullopt;
    const auto& [self, func] = *method;
    if (detail::is_bound_method<collision::FootprintGridChecker>(self, func, "is_valid")) {
      return NativeValidity(nanobind::cast<const collision::FootprintGridChecker*>(self));
    }
#ifdef GEODEX_PYTHON_HAS_VAMP
    namespace gvamp = geodex::integration::vamp;
    if (detail::is_bound_method<gvamp::CollisionChecker>(self, func, "is_valid")) {
      return NativeValidity(nanobind::cast<const gvamp::CollisionChecker*>(self));
    }
#endif
    return std::nullopt;
  }

  /// @brief Whether the bound method takes a point of `n` entries. FootprintGridChecker
  /// takes exactly 3, and Python raises TypeError for another size.
  bool takes(const Eigen::Index n) const {
    return !std::holds_alternative<const collision::FootprintGridChecker*>(target_) || n == 3;
  }

  /// @brief The value of the bound `is_valid(q)`. `takes(q.size())` holds.
  template <typename Q>
  bool operator()(const Q& q) const {
    if (const auto* c = std::get_if<const collision::FootprintGridChecker*>(&target_)) {
      return (*c)->is_valid(Eigen::Vector3d(q[0], q[1], q[2]));
    }
#ifdef GEODEX_PYTHON_HAS_VAMP
    const auto* checker = std::get<const geodex::integration::vamp::CollisionChecker*>(target_);
    return checker->is_valid(q.data(), static_cast<int>(q.size()));
#else
    return false;
#endif
  }

 private:
#ifdef GEODEX_PYTHON_HAS_VAMP
  using Target = std::variant<const collision::FootprintGridChecker*,
                              const geodex::integration::vamp::CollisionChecker*>;
#else
  using Target = std::variant<const collision::FootprintGridChecker*>;
#endif

  explicit NativeValidity(Target target) : target_(target) {}

  Target target_;
};

}  // namespace geodex::python

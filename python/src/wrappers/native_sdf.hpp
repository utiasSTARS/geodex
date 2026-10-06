/// @file native_sdf.hpp
/// @brief The bound geodex SDFs, as their Python `__call__` evaluates them.
///
/// @details The bindings of the SDF classes evaluate through `call_sdf`. `NativeSdf` points
/// to an SDF inside a Python object and evaluates it the same way without Python. The file
/// is plain C++ and does not include nanobind.

#pragma once

#include <functional>
#include <type_traits>
#include <variant>

#include <Eigen/Core>

#include "geodex/collision/collision.hpp"

#include "sizes.hpp"

namespace geodex::python {

/// @brief An SDF over configuration points, as InflatedSDF and MemoizedSDF hold it.
using SDFCallable = std::function<double(const Eigen::VectorXd&)>;

/// @brief The InflatedSDF bound to Python.
using InflatedSDFCallable = collision::InflatedSDF<SDFCallable>;

/// @brief The MemoizedSDF bound to Python, keyed on an SE(2) pose (x, y, theta).
using MemoizedSDFCallable = collision::MemoizedSDF<SDFCallable, 3>;

/// @brief How the `__call__(q)` of a bound SDF evaluates, and the fewest entries of q it
/// takes. A smaller q raises ValueError.
template <typename T>
struct SdfCall;

template <>
struct SdfCall<collision::CircleSDF> {
  static constexpr const char* name = "CircleSDF";
  static constexpr Eigen::Index min_size = 2;
  template <typename Q>
  static double eval(const collision::CircleSDF& s, const Q& q) {
    return s(Eigen::Vector2d(q[0], q[1]));
  }
};

template <>
struct SdfCall<collision::CircleSmoothSDF> {
  static constexpr const char* name = "CircleSmoothSDF";
  static constexpr Eigen::Index min_size = 2;
  template <typename Q>
  static double eval(const collision::CircleSmoothSDF& s, const Q& q) {
    return s(Eigen::Vector2d(q[0], q[1]));
  }
};

template <>
struct SdfCall<collision::RectSmoothSDF> {
  static constexpr const char* name = "RectSmoothSDF";
  static constexpr Eigen::Index min_size = 2;
  template <typename Q>
  static double eval(const collision::RectSmoothSDF& s, const Q& q) {
    return s(Eigen::Vector2d(q[0], q[1]));
  }
};

template <>
struct SdfCall<collision::GridSDF> {
  static constexpr const char* name = "GridSDF";
  static constexpr Eigen::Index min_size = 2;
  template <typename Q>
  static double eval(const collision::GridSDF& s, const Q& q) {
    return s(q);
  }
};

template <>
struct SdfCall<collision::FootprintGridChecker> {
  static constexpr const char* name = "FootprintGridChecker";
  static constexpr Eigen::Index min_size = 3;
  template <typename Q>
  static double eval(const collision::FootprintGridChecker& s, const Q& q) {
    return s(q);
  }
};

/// @brief InflatedSDF does not check q. The wrapped SDF checks it.
template <>
struct SdfCall<InflatedSDFCallable> {
  static constexpr const char* name = "InflatedSDF";
  static constexpr Eigen::Index min_size = 0;
  template <typename Q>
  static double eval(const InflatedSDFCallable& s, const Q& q) {
    return s(q);
  }
};

template <>
struct SdfCall<MemoizedSDFCallable> {
  static constexpr const char* name = "MemoizedSDF";
  static constexpr Eigen::Index min_size = 3;
  template <typename Q>
  static double eval(const MemoizedSDFCallable& s, const Q& q) {
    return s(q);
  }
};

/// @brief The body of the `__call__(q)` binding of a bound SDF.
template <typename T>
double call_sdf(const T& sdf, const Eigen::VectorXd& q) {
  using C = SdfCall<T>;
  if constexpr (C::min_size > 0) require_min_size(q, C::min_size, C::name, "q");
  return C::eval(sdf, q);
}

/// @brief A bound geodex SDF, called in place.
///
/// @details It points into the Python object and does not keep it alive. Its holder keeps a
/// reference to the object. `find_native_sdf` in native_collision.hpp finds it.
class NativeSdf {
 public:
  /// @brief The SDF types that a Python object can hold.
  using Target = std::variant<const collision::FootprintGridChecker*, const collision::GridSDF*,
                              const collision::CircleSDF*, const collision::CircleSmoothSDF*,
                              const collision::RectSmoothSDF*, const InflatedSDFCallable*,
                              const MemoizedSDFCallable*>;

  /// @param target The SDF inside a Python object.
  explicit NativeSdf(const Target target) : target_(target) {}

  /// @brief The fewest entries of q that the bound call takes.
  Eigen::Index min_size() const {
    return std::visit(
        [](const auto* s) { return SdfCall<std::remove_cvref_t<decltype(*s)>>::min_size; },
        target_);
  }

  /// @brief The SDF at q, the value `obj(q)` returns. q holds at least `min_size()` entries.
  template <typename Q>
  double operator()(const Q& q) const {
    return std::visit(
        [&](const auto* s) { return SdfCall<std::remove_cvref_t<decltype(*s)>>::eval(*s, q); },
        target_);
  }

 private:
  Target target_;
};

}  // namespace geodex::python

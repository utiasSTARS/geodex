/// @file
/// @brief Built-in mass matrices of known robots, generated from their CRBA.
///
/// `geodex::robots::MassMatrix<Robot::Panda>` evaluates the joint-space mass
/// matrix \f$ M(q) \f$ with a generated per-robot CRBA function. The implementation
/// does not depend on Pinocchio at run time or on `GEODEX_PINOCCHIO`.
///
/// The class is templated on the `Robot` enum, and its storage is fixed-size at
/// compile time. `operator()` returns a `const Eigen::Matrix<double, Nq, Nq>&`, and
/// downstream Eigen expressions (`u.dot(mm(q) * v)`, `U.transpose() * mm(q) * V`)
/// specialize the matvec and matmul at compile time.
///
/// To load an arbitrary URDF at runtime, use
/// `geodex::integration::pinocchio::MassMatrix(urdf_path)`, which requires
/// `GEODEX_PINOCCHIO=ON` and links against Pinocchio.

#pragma once

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <Eigen/Core>

// GEODEX_ROBOT_INCLUDES_BEGIN
#include "generated/baxter_crba.hpp"
#include "generated/fr3_gripper_crba.hpp"
#include "generated/husky_ur5e_crba.hpp"
#include "generated/panda_crba.hpp"
#include "generated/pr2_crba.hpp"
#include "generated/ridgeback_ur5e_crba.hpp"
#include "generated/stretch3_crba.hpp"
#include "generated/stretch4_crba.hpp"
#include "generated/ur5_crba.hpp"
// GEODEX_ROBOT_INCLUDES_END

namespace geodex::robots {

/// @brief Catalog of robots shipped with precompiled CRBA symbols.
enum class Robot {
// GEODEX_ROBOT_ENUM_BEGIN
  Baxter,
  Fr3Gripper,
  HuskyUr5e,
  Panda,
  Pr2,
  RidgebackUr5e,
  Stretch3,
  Stretch4,
  Ur5,
// GEODEX_ROBOT_ENUM_END
};

/// @brief Drive of a robot's mobile base, `None` for a fixed-base robot.
///
/// A robot with a mobile base plans on SE(2) x R^n. Its whole-body configuration is the
/// base pose `(x, y, theta)` followed by the arm joints, and `MassMatrix<R>` covers the
/// arm joints with the base held still.
enum class BaseDrive { None, Holonomic, Differential };

namespace detail {

/// @brief Compile-time per-robot information. Specialize for each entry in
/// the `Robot` enum that ships a precompiled CRBA symbol. Trying to
/// instantiate `MassMatrix<R>` for an `R` without a specialization fails
/// at compile time with an "incomplete type" error.
template <Robot R>
struct RobotTraits;

// GEODEX_ROBOT_TRAITS_BEGIN
template <>
struct RobotTraits<Robot::Baxter> {
  static constexpr std::string_view name = "baxter";
  static constexpr BaseDrive drive = BaseDrive::None;
  static constexpr int Nq = generated::baxter_nq;
  static constexpr int Nv = generated::baxter_nv;
  static constexpr int UpperCount = generated::baxter_upper_count;
  static constexpr const double* lower_limit = generated::baxter_lower_limit;
  static constexpr const double* upper_limit = generated::baxter_upper_limit;
  static void crba(const double* q, double* M_upper) { ::baxter_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Fr3Gripper> {
  static constexpr std::string_view name = "fr3_arm_gripper";
  static constexpr BaseDrive drive = BaseDrive::None;
  static constexpr int Nq = generated::fr3_gripper_nq;
  static constexpr int Nv = generated::fr3_gripper_nv;
  static constexpr int UpperCount = generated::fr3_gripper_upper_count;
  static constexpr const double* lower_limit = generated::fr3_gripper_lower_limit;
  static constexpr const double* upper_limit = generated::fr3_gripper_upper_limit;
  static void crba(const double* q, double* M_upper) { ::fr3_gripper_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::HuskyUr5e> {
  static constexpr std::string_view name = "husky_ur5e";
  static constexpr BaseDrive drive = BaseDrive::Differential;
  static constexpr int Nq = generated::husky_ur5e_nq;
  static constexpr int Nv = generated::husky_ur5e_nv;
  static constexpr int UpperCount = generated::husky_ur5e_upper_count;
  static constexpr const double* lower_limit = generated::husky_ur5e_lower_limit;
  static constexpr const double* upper_limit = generated::husky_ur5e_upper_limit;
  static void crba(const double* q, double* M_upper) { ::husky_ur5e_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Panda> {
  static constexpr std::string_view name = "panda";
  static constexpr BaseDrive drive = BaseDrive::None;
  static constexpr int Nq = generated::panda_nq;
  static constexpr int Nv = generated::panda_nv;
  static constexpr int UpperCount = generated::panda_upper_count;
  static constexpr const double* lower_limit = generated::panda_lower_limit;
  static constexpr const double* upper_limit = generated::panda_upper_limit;
  static void crba(const double* q, double* M_upper) { ::panda_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Pr2> {
  static constexpr std::string_view name = "pr2";
  static constexpr BaseDrive drive = BaseDrive::None;
  static constexpr int Nq = generated::pr2_nq;
  static constexpr int Nv = generated::pr2_nv;
  static constexpr int UpperCount = generated::pr2_upper_count;
  static constexpr const double* lower_limit = generated::pr2_lower_limit;
  static constexpr const double* upper_limit = generated::pr2_upper_limit;
  static void crba(const double* q, double* M_upper) { ::pr2_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::RidgebackUr5e> {
  static constexpr std::string_view name = "ridgeback_ur5e";
  static constexpr BaseDrive drive = BaseDrive::Holonomic;
  static constexpr int Nq = generated::ridgeback_ur5e_nq;
  static constexpr int Nv = generated::ridgeback_ur5e_nv;
  static constexpr int UpperCount = generated::ridgeback_ur5e_upper_count;
  static constexpr const double* lower_limit = generated::ridgeback_ur5e_lower_limit;
  static constexpr const double* upper_limit = generated::ridgeback_ur5e_upper_limit;
  static void crba(const double* q, double* M_upper) { ::ridgeback_ur5e_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Stretch3> {
  static constexpr std::string_view name = "stretch3";
  static constexpr BaseDrive drive = BaseDrive::Differential;
  static constexpr int Nq = generated::stretch3_nq;
  static constexpr int Nv = generated::stretch3_nv;
  static constexpr int UpperCount = generated::stretch3_upper_count;
  static constexpr const double* lower_limit = generated::stretch3_lower_limit;
  static constexpr const double* upper_limit = generated::stretch3_upper_limit;
  static void crba(const double* q, double* M_upper) { ::stretch3_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Stretch4> {
  static constexpr std::string_view name = "stretch4";
  static constexpr BaseDrive drive = BaseDrive::Holonomic;
  static constexpr int Nq = generated::stretch4_nq;
  static constexpr int Nv = generated::stretch4_nv;
  static constexpr int UpperCount = generated::stretch4_upper_count;
  static constexpr const double* lower_limit = generated::stretch4_lower_limit;
  static constexpr const double* upper_limit = generated::stretch4_upper_limit;
  static void crba(const double* q, double* M_upper) { ::stretch4_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Ur5> {
  static constexpr std::string_view name = "ur5";
  static constexpr BaseDrive drive = BaseDrive::None;
  static constexpr int Nq = generated::ur5_nq;
  static constexpr int Nv = generated::ur5_nv;
  static constexpr int UpperCount = generated::ur5_upper_count;
  static constexpr const double* lower_limit = generated::ur5_lower_limit;
  static constexpr const double* upper_limit = generated::ur5_upper_limit;
  static void crba(const double* q, double* M_upper) { ::ur5_crba(q, M_upper); }
};
// GEODEX_ROBOT_TRAITS_END

}  // namespace detail

/// @brief Joint-space mass matrix \f$ M(q) \f$ for a known robot.
///
/// Holds fixed-size per-instance buffers (no heap allocation). Copying
/// produces an independent instance with its own buffers. Not thread-safe
/// across concurrent calls on one instance. Use one instance per thread.
/// Construction is trivial (defaulted) and `constexpr`-friendly.
template <Robot R>
class MassMatrix {
  using Traits = detail::RobotTraits<R>;

 public:
  static constexpr int Nq = Traits::Nq;       ///< Number of configuration coordinates.
  static constexpr int Nv = Traits::Nv;       ///< Number of velocity coordinates.
  using Vec = Eigen::Matrix<double, Nq, 1>;   ///< A configuration.
  using Mat = Eigen::Matrix<double, Nq, Nq>;  ///< A mass matrix.

  MassMatrix() = default;
  ~MassMatrix() = default;

  /// @brief Copy into an instance with its own buffers.
  MassMatrix(const MassMatrix&) = default;

  /// @brief Copy into this instance's buffers.
  MassMatrix& operator=(const MassMatrix&) = default;

  /// @brief Move, which copies the fixed-size buffers.
  MassMatrix(MassMatrix&&) noexcept = default;

  /// @brief Move, which copies the fixed-size buffers.
  MassMatrix& operator=(MassMatrix&&) noexcept = default;

  /// @brief Evaluate \f$ M(q) \f$. The returned reference is valid until the
  /// next call on this instance.
  auto operator()(const Vec& q) const -> const Mat& {
    Traits::crba(q.data(), upper_buf_.data());
    int k = 0;
    for (int i = 0; i < Nq; ++i) {
      M_(i, i) = upper_buf_[k++];
      for (int j = i + 1; j < Nq; ++j) {
        const double v = upper_buf_[k++];
        M_(i, j) = v;
        M_(j, i) = v;
      }
    }
    return M_;
  }

  /// @brief Configuration-space dimension (compile-time constant).
  static constexpr auto nq() -> int { return Nq; }

  /// @brief Per-joint position limits `(lower, upper)`.
  static auto joint_limits() -> std::pair<Vec, Vec> {
    Vec lo, hi;
    for (int i = 0; i < Nq; ++i) {
      lo[i] = Traits::lower_limit[i];
      hi[i] = Traits::upper_limit[i];
    }
    return {lo, hi};
  }

 private:
  mutable Eigen::Matrix<double, Traits::UpperCount, 1> upper_buf_{};
  mutable Mat M_{Mat::Zero()};
};

/// @brief Drive of robot @p R's mobile base.
template <Robot R>
inline constexpr BaseDrive base_drive = detail::RobotTraits<R>::drive;

/// @brief Whether robot @p R has a planar mobile base ahead of its arm joints.
template <Robot R>
inline constexpr bool has_planar_base = base_drive<R> != BaseDrive::None;

/// @brief Robots with a precompiled CRBA, in alphabetical order.
auto registered_robots() -> std::span<const Robot>;

/// @brief Invoke `f.template operator()<R>()` with the compile-time robot matching @p r.
///
/// Turns a runtime robot id (a parameter, a command-line flag) into the template
/// argument that `MassMatrix<R>` and `MassLowerBound<R>` need. The switch does not
/// have a `default`, and a new enumerator without a case raises a `-Wswitch`
/// diagnostic.
template <class F>
decltype(auto) visit(Robot r, F&& f) {
  switch (r) {
// GEODEX_ROBOT_VISIT_BEGIN
    case Robot::Baxter: return f.template operator()<Robot::Baxter>();
    case Robot::Fr3Gripper: return f.template operator()<Robot::Fr3Gripper>();
    case Robot::HuskyUr5e: return f.template operator()<Robot::HuskyUr5e>();
    case Robot::Panda: return f.template operator()<Robot::Panda>();
    case Robot::Pr2: return f.template operator()<Robot::Pr2>();
    case Robot::RidgebackUr5e: return f.template operator()<Robot::RidgebackUr5e>();
    case Robot::Stretch3: return f.template operator()<Robot::Stretch3>();
    case Robot::Stretch4: return f.template operator()<Robot::Stretch4>();
    case Robot::Ur5: return f.template operator()<Robot::Ur5>();
// GEODEX_ROBOT_VISIT_END
  }
  throw std::logic_error("geodex::robots::visit: unhandled Robot");
}

/// @brief Canonical lowercase name of @p r, shared with the VAMP registry ("panda", ...).
inline auto name(Robot r) -> std::string_view {
  return visit(r, []<Robot R>() { return detail::RobotTraits<R>::name; });
}

/// @brief Inverse of name(). Returns `std::nullopt` for an empty or unknown name.
auto robot_from_name(std::string_view s) -> std::optional<Robot>;

}  // namespace geodex::robots

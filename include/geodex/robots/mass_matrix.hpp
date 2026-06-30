/// @file
/// @brief Built-in CRBA-driven mass matrices for known robots.
///
/// `geodex::robots::MassMatrix<Robot::Panda>` evaluates the joint-space mass
/// matrix \f$ M(q) \f$ via a precompiled per-robot symbol (CppAD::CG-generated
/// from the robot's URDF, then post-processed for SIMD-friendly trig).
/// The implementation has zero Pinocchio runtime dependency — it ships with
/// geodex unconditionally and is independent of `GEODEX_PINOCCHIO`.
///
/// Templated on the `Robot` enum so the storage is fully fixed-size at
/// compile time. `operator()` returns a `const Eigen::Matrix<double, Nq, Nq>&`
/// which lets downstream Eigen expressions (`u.dot(mm(q) * v)`,
/// `U.transpose() * mm(q) * V`) specialize the matvec / matmul at compile time.
///
/// Use `geodex::integration::pinocchio::MassMatrix(urdf_path)` instead when
/// you need to load an arbitrary URDF at runtime — that path requires
/// `GEODEX_PINOCCHIO=ON` and links against Pinocchio.

#pragma once

#include <Eigen/Core>
#include <utility>
#include <vector>

// GEODEX_ROBOT_INCLUDES_BEGIN
#include "generated/panda_crba.hpp"
#include "generated/ur5_crba.hpp"
#include "generated/fetch_crba.hpp"
#include "generated/baxter_crba.hpp"
#include "generated/pr2_crba.hpp"
// GEODEX_ROBOT_INCLUDES_END

namespace geodex::robots {

/// @brief Catalog of robots shipped with precompiled CRBA symbols.
enum class Robot {
// GEODEX_ROBOT_ENUM_BEGIN
  Panda,
  Ur5,
  Fetch,
  Baxter,
  Pr2,
// GEODEX_ROBOT_ENUM_END
};

namespace detail {

/// @brief Compile-time per-robot information. Specialize for each entry in
/// the `Robot` enum that ships a precompiled CRBA symbol. Trying to
/// instantiate `MassMatrix<R>` for an `R` without a specialization fails
/// at compile time with a "incomplete type" error.
template <Robot R>
struct RobotTraits;

// GEODEX_ROBOT_TRAITS_BEGIN
template <>
struct RobotTraits<Robot::Panda> {
  static constexpr int Nq = generated::panda_nq;
  static constexpr int Nv = generated::panda_nv;
  static constexpr int UpperCount = generated::panda_upper_count;
  static constexpr const double* lower_limit = generated::panda_lower_limit;
  static constexpr const double* upper_limit = generated::panda_upper_limit;
  static void crba(const double* q, double* M_upper) { ::panda_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Ur5> {
  static constexpr int Nq = generated::ur5_nq;
  static constexpr int Nv = generated::ur5_nv;
  static constexpr int UpperCount = generated::ur5_upper_count;
  static constexpr const double* lower_limit = generated::ur5_lower_limit;
  static constexpr const double* upper_limit = generated::ur5_upper_limit;
  static void crba(const double* q, double* M_upper) { ::ur5_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Fetch> {
  static constexpr int Nq = generated::fetch_nq;
  static constexpr int Nv = generated::fetch_nv;
  static constexpr int UpperCount = generated::fetch_upper_count;
  static constexpr const double* lower_limit = generated::fetch_lower_limit;
  static constexpr const double* upper_limit = generated::fetch_upper_limit;
  static void crba(const double* q, double* M_upper) { ::fetch_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Baxter> {
  static constexpr int Nq = generated::baxter_nq;
  static constexpr int Nv = generated::baxter_nv;
  static constexpr int UpperCount = generated::baxter_upper_count;
  static constexpr const double* lower_limit = generated::baxter_lower_limit;
  static constexpr const double* upper_limit = generated::baxter_upper_limit;
  static void crba(const double* q, double* M_upper) { ::baxter_crba(q, M_upper); }
};

template <>
struct RobotTraits<Robot::Pr2> {
  static constexpr int Nq = generated::pr2_nq;
  static constexpr int Nv = generated::pr2_nv;
  static constexpr int UpperCount = generated::pr2_upper_count;
  static constexpr const double* lower_limit = generated::pr2_lower_limit;
  static constexpr const double* upper_limit = generated::pr2_upper_limit;
  static void crba(const double* q, double* M_upper) { ::pr2_crba(q, M_upper); }
};
// GEODEX_ROBOT_TRAITS_END

}  // namespace detail

/// @brief Joint-space mass matrix \f$ M(q) \f$ for a known robot.
///
/// Holds fixed-size per-instance buffers (no heap allocation). Copying
/// produces an independent instance with its own buffers. Not thread-safe
/// across concurrent calls on the same instance — use one instance per
/// thread. Construction is trivial (defaulted) and `constexpr`-friendly.
template <Robot R>
class MassMatrix {
  using Traits = detail::RobotTraits<R>;

 public:
  static constexpr int Nq = Traits::Nq;
  static constexpr int Nv = Traits::Nv;
  using Vec = Eigen::Matrix<double, Nq, 1>;
  using Mat = Eigen::Matrix<double, Nq, Nq>;

  MassMatrix() = default;
  ~MassMatrix() = default;
  MassMatrix(const MassMatrix&) = default;
  MassMatrix& operator=(const MassMatrix&) = default;
  MassMatrix(MassMatrix&&) noexcept = default;
  MassMatrix& operator=(MassMatrix&&) noexcept = default;

  /// @brief Evaluate \f$ M(q) \f$. Returned reference is valid until the
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

/// @brief List of robots for which a precompiled CRBA is available.
auto registered_robots() -> std::vector<Robot>;

}  // namespace geodex::robots

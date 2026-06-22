/// @file registry.hpp
/// @brief Public API declarations for VAMP-accelerated collision checking.
///
/// This header is intentionally lightweight: pure forward declarations plus
/// the opaque @ref EnvHandle struct and the @ref CollisionChecker virtual
/// base. It does not pull in VAMP, Eigen, or any SIMD intrinsic — consumer
/// translation units stay free of AVX2/FMA flags. The function bodies live
/// in a single thin compilation unit (@c src/integration/vamp_impl.cpp)
/// inside the @c geodex_vamp static archive that ships with this project;
/// the linker resolves consumer calls there.
///
/// Linking @c geodex (or its alias @c geodex::geodex) automatically pulls
/// in @c geodex_vamp when the project was built with
/// @c -DGEODEX_VAMP=ON; users do not need to reference any integration
/// target explicitly.
///
/// Typical usage:
/// @code
/// #include <geodex/integration/vamp/registry.hpp>
///
/// auto env = geodex::integration::vamp::load_scene("scene.yaml");
/// auto checker = geodex::integration::vamp::make_vamp_checker("panda", env);
/// bool ok = checker->is_valid(q.data(), 7);
/// @endcode
///
/// @ref make_vamp_motion_validator returns an OMPL @c MotionValidator that
/// performs SIMD batch edge validation, suitable for plugging into an
/// @c ompl::base::SpaceInformation.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>

namespace geodex::integration::vamp {

/// Opaque handle to a VAMP scene environment.
///
/// The actual VAMP environment type is hidden behind a @c shared_ptr<void>
/// so the public API does not expose VAMP types in its signatures. Construct
/// via @ref load_scene; copy freely.
struct EnvHandle {
  std::shared_ptr<void> impl;
};

/// Per-robot point-validity collision checker.
///
/// Instances are produced by @ref make_vamp_checker.
class CollisionChecker {
 public:
  virtual ~CollisionChecker() = default;

  /// Check whether the configuration @p values is collision-free.
  ///
  /// @param values  Pointer to a configuration of @p dim joint angles (radians).
  /// @param dim     Configuration dimension; must match the robot's DOF.
  /// @return @c true if the configuration is valid (no collision).
  virtual bool is_valid(const double* values, int dim) const = 0;
};

/// Load a MotionBenchMaker (MBM) style scene YAML into an opaque VAMP
/// environment handle.
///
/// Supports primitive collision objects (boxes, cylinders, spheres) and mesh
/// objects (axis-aligned bounding-box approximation). The loaded environment
/// is sorted by VAMP for cache-friendly traversal.
auto load_scene(const std::string& yaml_path) -> EnvHandle;

/// Build a per-robot collision checker bound to @p env.
///
/// Supported robot names: @c "baxter", @c "fetch", @c "panda", @c "pr2", @c "ur5".
/// @throw std::runtime_error if @p robot_name is not one of the above.
auto make_vamp_checker(const std::string& robot_name, EnvHandle env)
    -> std::unique_ptr<CollisionChecker>;

/// Build an OMPL motion validator (SIMD batch edge check) for the named robot.
///
/// @throw std::runtime_error if @p robot_name is not registered.
auto make_vamp_motion_validator(const std::string& robot_name,
                                const ::ompl::base::SpaceInformationPtr& si,
                                EnvHandle env)
    -> std::unique_ptr<::ompl::base::MotionValidator>;

/// Names of robots compiled into the @c geodex_vamp archive, sorted
/// lexicographically.
auto registered_robots() -> std::vector<std::string>;

}  // namespace geodex::integration::vamp

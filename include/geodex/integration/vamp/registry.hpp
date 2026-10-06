/// @file registry.hpp
/// @brief Public API declarations for VAMP-accelerated collision checking.
///
/// This header holds forward declarations, the opaque `EnvHandle` struct and the
/// `CollisionChecker` virtual base. It pulls in Eigen for the in-memory
/// scene-builder types but not VAMP or SIMD intrinsics, and consumer translation units
/// do not need AVX2/FMA flags. The function bodies live in
/// @c src/integration/vamp_impl.cpp and one translation unit per robot kernel, all
/// inside the @c geodex_vamp static archive.
///
/// With @c -DGEODEX_VAMP=ON, linking @c geodex (alias @c geodex::geodex) pulls in
/// @c geodex_vamp. Users do not reference an integration target.
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
/// `make_vamp_motion_validator` returns an OMPL @c MotionValidator for an
/// @c ompl::base::SpaceInformation that validates edges in SIMD batches.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>

namespace geodex::integration::vamp {

/// Opaque handle to a VAMP scene environment.
///
/// A @c shared_ptr<void> hides the VAMP environment type, and the public signatures
/// do not expose VAMP types. Construct it with @ref load_scene and copy it freely.
struct EnvHandle {
  std::shared_ptr<void> impl;  ///< the VAMP environment, shared by copies of the handle
};

/// Per-robot point-validity collision checker.
///
/// Instances are produced by @ref make_vamp_checker.
class CollisionChecker {
 public:
  /// Destroy the checker and its copy of an attached body.
  virtual ~CollisionChecker() = default;

  /// Check whether the configuration @p values is collision-free.
  ///
  /// @param values  Pointer to a configuration of @p dim joint angles (radians).
  /// @param dim     Configuration dimension, which must match the robot's DOF.
  /// @return @c true if the configuration lies inside the robot's joint box and is
  ///         collision-free. A wrong @p dim returns @c false.
  virtual bool is_valid(const double* values, int dim) const = 0;

  /// Number of configurations the backend evaluates in one SIMD pass.
  ///
  /// Callers holding several configurations (samples along one edge, say) should pass
  /// them together through @ref all_valid.
  virtual int batch_width() const { return 1; }

  /// Check whether all of @p count configurations are valid.
  ///
  /// @param values  Row-major @p count x @p dim array of configurations.
  /// @param dim     Configuration dimension, which must match the robot's DOF.
  /// @param count   Number of configurations.
  /// @return @c true when every configuration is valid. The result does not say which
  ///         configuration failed.
  /// @throw std::invalid_argument if @p dim does not match the robot.
  virtual bool all_valid(const double* values, int dim, int count) const {
    for (int i = 0; i < count; ++i) {
      if (!is_valid(values + static_cast<std::ptrdiff_t>(i) * dim, dim)) return false;
    }
    return true;
  }
};

/// Load a MotionBenchMaker (MBM) style scene YAML into an opaque VAMP
/// environment handle.
///
/// Supports primitive collision objects (boxes, cylinders, spheres) and mesh
/// objects (axis-aligned bounding-box approximation). The loaded environment
/// is sorted by VAMP for cache-friendly traversal.
auto load_scene(const std::string& yaml_path) -> EnvHandle;

/// Opaque in-memory collision-scene builder.
///
/// Accumulates primitive obstacles programmatically as an alternative to
/// @ref load_scene. The underlying VAMP environment stays hidden behind a
/// @c shared_ptr<void>. Create with @ref make_scene_builder, add primitives,
/// then finalize with @ref build_scene.
struct SceneBuilder {
  std::shared_ptr<void> impl;  ///< the scalar VAMP environment being filled
};

/// Create an empty in-memory scene builder.
auto make_scene_builder() -> SceneBuilder;

/// Add an oriented box to @p s.
///
/// @param s        Scene builder to add to.
/// @param center   Box center in world coordinates.
/// @param size     Full extents along the box's local x, y, z axes, as in a MoveIt
///                 `SolidPrimitive` box and an MBM scene file.
/// @param rotation World-from-local box rotation.
void scene_add_box(SceneBuilder& s, const Eigen::Vector3d& center, const Eigen::Vector3d& size,
                   const Eigen::Matrix3d& rotation);

/// Add a sphere to @p s.
void scene_add_sphere(SceneBuilder& s, const Eigen::Vector3d& center,
                      double radius);

/// Add an oriented cylinder to @p s. Its local z axis is the cylinder axis.
///
/// VAMP checks the capsule on the same axis and radius, which contains the cylinder.
void scene_add_cylinder(SceneBuilder& s, const Eigen::Vector3d& center, double radius,
                        double height, const Eigen::Matrix3d& rotation);

// The quaternion overloads convert in the caller's translation unit. Eigen::Quaterniond
// is a fixed-size vectorizable type whose alignment differs between the AVX-compiled
// geodex_vamp archive and a default-compiled caller. It does not cross into the archive.

/// Add an oriented box to @p s, orientation given as a world-from-local quaternion.
inline void scene_add_box(SceneBuilder& s, const Eigen::Vector3d& center,
                          const Eigen::Vector3d& size, const Eigen::Quaterniond& orientation) {
  const Eigen::Matrix3d rotation = orientation.normalized().toRotationMatrix();
  scene_add_box(s, center, size, rotation);
}

/// Add an oriented cylinder to @p s, orientation given as a world-from-local quaternion.
inline void scene_add_cylinder(SceneBuilder& s, const Eigen::Vector3d& center, double radius,
                               double height, const Eigen::Quaterniond& orientation) {
  const Eigen::Matrix3d rotation = orientation.normalized().toRotationMatrix();
  scene_add_cylinder(s, center, radius, height, rotation);
}

/// Finalize @p s into a sorted, opaque VAMP environment handle.
auto build_scene(const SceneBuilder& s) -> EnvHandle;

/// Build a per-robot collision checker bound to @p env.
///
/// @p robot_name is one of @ref registered_robots(). Mobile manipulators take the
/// whole-body configuration, the planar base pose `(x, y, theta)` followed by the arm
/// joints.
/// @throw std::runtime_error if @p robot_name is not registered.
auto make_vamp_checker(const std::string& robot_name, EnvHandle env)
    -> std::unique_ptr<CollisionChecker>;

/// Build an OMPL motion validator (SIMD batch edge check) for the named robot.
///
/// Fixed-base arms check the straight joint-space chord. Mobile manipulators sample
/// the state space's own interpolation and check an SE(2) base edge along the
/// geodesic the planner follows.
/// @throw std::runtime_error if @p robot_name is not registered.
auto make_vamp_motion_validator(const std::string& robot_name,
                                const ::ompl::base::SpaceInformationPtr& si,
                                EnvHandle env)
    -> std::unique_ptr<::ompl::base::MotionValidator>;

/// Default largest sphere-center travel between two edge checks, in meters.
inline constexpr double kDefaultMaxSphereStep = 0.005;

/// Build an OMPL motion validator that proves each edge clear of the obstacles.
///
/// Baked sphere-speed bounds cut each edge into pieces along which no sphere center
/// moves more than @p max_sphere_step meters, and the piece ends are checked against the
/// obstacles grown by half that step. An accepted edge keeps every robot and attached
/// sphere out of every obstacle along the whole motion, up to VAMP's float arithmetic.
///
/// @note A piece whose end touches the grown obstacles is halved, with half the padding,
/// up to 8 times. An edge within `max_sphere_step / 256` of an obstacle can be rejected
/// without touching it. Self-collision and contact with an attached body are checked at
/// the piece ends only, and two robot spheres can overlap by at most @p max_sphere_step
/// between them. With a planar base, the base must stay inside the space's bounds along
/// the whole arc.
/// @param robot_name A name from `registered_robots()`.
/// @param si Space information of the planning problem.
/// @param env Scene to check against.
/// @param max_sphere_step Largest sphere-center travel between two checks, in meters.
/// @throw std::runtime_error if @p robot_name is not registered.
/// @throw std::invalid_argument if @p max_sphere_step is not positive and finite.
auto make_vamp_certified_motion_validator(const std::string& robot_name,
                                          const ::ompl::base::SpaceInformationPtr& si,
                                          EnvHandle env,
                                          double max_sphere_step = kDefaultMaxSphereStep)
    -> std::unique_ptr<::ompl::base::MotionValidator>;

/// A copy of @p env with every obstacle grown by @p padding meters on every side. The copy
/// keeps the attached spheres.
///
/// @throw std::invalid_argument if @p padding is negative or not finite, or if @p env holds
/// point clouds, height fields or finite cylinders.
auto pad_scene(const EnvHandle& env, double padding) -> EnvHandle;

/// Bound on how far any sphere center of a registered robot, and of the spheres attached in
/// @p env, moves per unit coordinate norm of a motion, in meters. For a mobile manipulator, the
/// norm covers the base twist and the arm joints.
///
/// @throw std::runtime_error if @p robot_name is not registered.
auto sphere_speed(const std::string& robot_name, const EnvHandle& env) -> double;

/// @brief Bound in meters on how far any sphere center moves along a motion, from the
/// motion's tangent. A mobile manipulator's tangent holds the base twist `(vx, vy, omega)`
/// before the joint changes, as its space's `log` returns it.
using SphereTravel = std::function<double(const Eigen::Ref<const Eigen::VectorXd>& tangent)>;

/// Sphere travel bound of a registered robot, with the spheres attached in @p env. Along a
/// motion with a constant base twist and linear joints, as the robot spaces interpolate, no
/// sphere center moves more than `|(vx, vy)| + |omega| rho + sum_j r_j |dq_j|`, and every part
/// of the motion moves at most its share of it. `rho` is the largest distance of a center from
/// the base's yaw axis, and `r_j` bounds a center's speed per unit of joint `j`. `sphere_speed`
/// is the norm of the coefficients.
///
/// @throw std::runtime_error if @p robot_name is not registered. The returned function throws
/// std::invalid_argument for a tangent whose size is not the robot's dimension.
auto make_sphere_travel(const std::string& robot_name, const EnvHandle& env) -> SphereTravel;

/// Collision spheres of a registered robot's VAMP model at @p q, as
/// `{x, y, z, radius}` in the world frame, in the model's order.
///
/// @throw std::runtime_error if @p robot_name is not registered.
/// @throw std::invalid_argument if @p dim is not the robot's dimension.
auto robot_spheres(const std::string& robot_name, const double* q, int dim)
    -> std::vector<std::array<double, 4>>;

/// Attach a rigid sphere set to the robot's end-effector frame.
///
/// Spheres are `{x, y, z, radius}` in the generated model's end-effector frame and
/// move with it. The kernel's attachment test checks them against the environment and
/// the robot's own spheres. An empty vector removes the attachment. A checker or
/// validator copies the attachment when it is built. Attach or detach before building
/// it. A sphere cover under-approximates a box at its corners. Validate a held box
/// exactly where that matters.
void attach_spheres(EnvHandle& env, const std::vector<std::array<double, 4>>& spheres);

/// Configuration dimension of a registered robot's VAMP model.
///
/// @throw std::runtime_error if @p robot_name is not registered.
auto robot_dimension(const std::string& robot_name) -> int;

/// Joint names of a registered robot's VAMP model, in configuration order.
///
/// A configuration passed to the robot's checker or validator lists these joints in
/// this order. Mobile manipulators start with `base_x_joint`, `base_y_joint` and
/// `base_theta_joint`. The names match `data/robots/<robot>/robot.yaml`.
/// @throw std::runtime_error if @p robot_name is not registered.
auto robot_joint_names(const std::string& robot_name) -> std::vector<std::string>;

/// Name of the link whose frame a registered robot's VAMP model poses attached spheres
/// in, the frame of @ref attach_spheres.
///
/// @throw std::runtime_error if @p robot_name is not registered.
auto robot_end_effector(const std::string& robot_name) -> std::string;

/// Names of robots compiled into the @c geodex_vamp archive, sorted
/// lexicographically.
auto registered_robots() -> std::vector<std::string>;

}  // namespace geodex::integration::vamp

/// @file bind_robots.cpp
/// @brief Python bindings for the geodex.robots submodule.
///
/// @details Exposes the built-in robots as configuration spaces under `geodex.robots`.
/// `RobotModel` is the shared type. Per-robot classes such as `Panda` and `Stretch3` subclass
/// it, and `geodex.robots.Panda()` builds a model. A fixed-base robot takes the arm metric. A
/// mobile robot also takes the base metric and the region its base samples.

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "wrappers/py_robot_model.hpp"

namespace nb = nanobind;
namespace gr = geodex::robots;
using namespace geodex::python;

namespace {

using Weights = std::array<double, 3>;
using Workspace = std::array<std::array<double, 2>, 2>;

gr::ArmMetric parse_metric(const std::string& metric) {
  if (metric == "kinetic_energy") return gr::ArmMetric::KineticEnergy;
  if (metric == "euclidean") return gr::ArmMetric::Euclidean;
  throw std::invalid_argument("metric must be 'kinetic_energy' or 'euclidean'; got '" + metric +
                              "'.");
}

RobotModelOptions mobile_options(const std::string& metric,
                                 const std::optional<std::string>& base,
                                 const std::optional<Weights>& base_weights,
                                 const Workspace& workspace) {
  RobotModelOptions o;
  o.metric = parse_metric(metric);
  if (base_weights) {
    o.base =
        geodex::SE2LeftInvariantMetric((*base_weights)[0], (*base_weights)[1], (*base_weights)[2]);
  } else if (base) {
    if (*base == "holonomic") {
      o.base = geodex::SE2LeftInvariantMetric::holonomic();
    } else if (*base == "differential_drive") {
      o.base = geodex::SE2LeftInvariantMetric::differential_drive();
    } else {
      throw std::invalid_argument("base must be 'holonomic' or 'differential_drive'; got '" +
                                  *base + "'.");
    }
  }
  o.xy_lo = Eigen::Vector2d(workspace[0][0], workspace[1][0]);
  o.xy_hi = Eigen::Vector2d(workspace[0][1], workspace[1][1]);
  if ((o.xy_hi.array() <= o.xy_lo.array()).any()) {
    throw std::invalid_argument("workspace must be ((x_lo, x_hi), (y_lo, y_hi)) with lo < hi.");
  }
  return o;
}

const char* drive_name(gr::BaseDrive d) {
  switch (d) {
    case gr::BaseDrive::Holonomic:
      return "holonomic";
    case gr::BaseDrive::Differential:
      return "differential_drive";
    case gr::BaseDrive::None:
      break;
  }
  return "none";
}

// Per-robot classes. They do not add state to PyRobotModel.
struct FixedBaseModel : PyRobotModel {
  FixedBaseModel(const char* name, const std::string& metric)
      : PyRobotModel(make_robot_model(name, RobotModelOptions{parse_metric(metric)})) {}
};
struct PandaModel : FixedBaseModel {
  explicit PandaModel(const std::string& metric) : FixedBaseModel("panda", metric) {}
};
struct UR5Model : FixedBaseModel {
  explicit UR5Model(const std::string& metric) : FixedBaseModel("ur5", metric) {}
};
struct BaxterModel : FixedBaseModel {
  explicit BaxterModel(const std::string& metric) : FixedBaseModel("baxter", metric) {}
};
struct PR2Model : FixedBaseModel {
  explicit PR2Model(const std::string& metric) : FixedBaseModel("pr2", metric) {}
};
struct Fr3GripperModel : FixedBaseModel {
  explicit Fr3GripperModel(const std::string& metric)
      : FixedBaseModel("fr3_arm_gripper", metric) {}
};

struct MobileModel : PyRobotModel {
  MobileModel(const char* name, const RobotModelOptions& options)
      : PyRobotModel(make_robot_model(name, options)) {}
};
struct Stretch3Model : MobileModel {
  explicit Stretch3Model(const RobotModelOptions& o) : MobileModel("stretch3", o) {}
};
struct Stretch4Model : MobileModel {
  explicit Stretch4Model(const RobotModelOptions& o) : MobileModel("stretch4", o) {}
};
struct RidgebackUR5eModel : MobileModel {
  explicit RidgebackUR5eModel(const RobotModelOptions& o) : MobileModel("ridgeback_ur5e", o) {}
};
struct HuskyUR5eModel : MobileModel {
  explicit HuskyUR5eModel(const RobotModelOptions& o) : MobileModel("husky_ur5e", o) {}
};

constexpr const char* kArmDoc =
    "    metric: Metric on the arm joints, 'kinetic_energy' (the robot's mass matrix\n"
    "        M(q), default) or 'euclidean'.\n";

constexpr const char* kMobileDoc =
    "The configuration is (x, y, theta, arm joints...) on SE(2) x R^n. Only the metric\n"
    "changes between a holonomic and a differential-drive base.\n\n"
    "Args:\n"
    "    metric: Metric on the arm joints, 'kinetic_energy' (default) or 'euclidean'.\n"
    "    base: 'holonomic' or 'differential_drive' (left-invariant SE(2) metric with\n"
    "        lateral weight 100). None matches the robot's drive.\n"
    "    base_weights: Body weights (wx, wy, wtheta) of the SE(2) metric. They override\n"
    "        base.\n"
    "    workspace: ((x_lo, x_hi), (y_lo, y_hi)), the region the base samples.";

template <typename Model>
void bind_fixed(nb::module_& robots, const char* cls, const std::string& doc) {
  static const std::string full = doc + "\n\nArgs:\n" + kArmDoc;
  nb::class_<Model, PyRobotModel>(robots, cls, full.c_str())
      .def(nb::init<const std::string&>(), nb::arg("metric") = "kinetic_energy");
}

template <typename Model>
void bind_mobile(nb::module_& robots, const char* cls, const std::string& doc) {
  static const std::string full = doc + "\n\n" + kMobileDoc;
  nb::class_<Model, PyRobotModel>(robots, cls, full.c_str())
      .def(
          "__init__",
          [](Model* self, const std::string& metric, const std::optional<std::string>& base,
             const std::optional<Weights>& base_weights, const Workspace& workspace) {
            new (self) Model(mobile_options(metric, base, base_weights, workspace));
          },
          nb::arg("metric") = "kinetic_energy", nb::arg("base") = nb::none(),
          nb::arg("base_weights") = nb::none(),
          nb::arg("workspace") = Workspace{{{-5.0, 5.0}, {-5.0, 5.0}}});
}

}  // namespace

void bind_robots(nb::module_& m) {
  auto robots = m.def_submodule("robots", "Built-in robots as configuration spaces.");

  nb::class_<PyRobotModel>(
      robots, "RobotModel",
      "A built-in robot as a configuration space.\n\n"
      "A fixed-base robot is R^dof with joint-limit bounds. A mobile robot is\n"
      "SE(2) x R^n, the base pose (x, y, theta) followed by the arm joints. Pass an\n"
      "instance anywhere geodex accepts a manifold, including plan() with a Scene.")
      .def("name", &PyRobotModel::name,
           "Robot name, such as 'panda'. It is also the VAMP kernel name.")
      .def("dof", &PyRobotModel::dof, "Number of configuration coordinates, base included.")
      .def("dim", &PyRobotModel::dim, "Configuration-space dimension, equal to dof.")
      .def("joint_limits", &PyRobotModel::joint_limits,
           "Per-coordinate (lower, upper) bounds, the base region first, then joint limits.")
      .def("random_point", &PyRobotModel::random_point,
           "Sample a configuration uniformly within the bounds.")
      .def("seed", &PyRobotModel::seed, nb::arg("seed"),
           "Reseed the sampler behind random_point() and unseeded plans.")
      .def("set_sampler", &PyRobotModel::set_sampler, nb::arg("sampler"),
           "Switch the sampler kind to 'scrambled', 'halton' or 'random'.")
      .def("mass_matrix", &PyRobotModel::mass_matrix, nb::arg("q"),
           "Metric tensor on coordinate velocities at q.\n\n"
           "For a fixed-base robot under the kinetic-energy metric, it is the joint-space\n"
           "mass matrix M(q). A mobile robot adds the SE(2) block on (x, y, theta). Use it\n"
           "to score a path under the metric of its plan.")
      .def("has_mass_matrix", &PyRobotModel::has_mass_matrix,
           "Whether mass_matrix() is available for this model.")
      .def("periods", &PyRobotModel::periods,
           "Per-coordinate periods, 2 pi on the base heading and empty for a fixed base.")
      .def("heuristic", &PyRobotModel::heuristic,
           "Admissible Loewner-bound heuristic of this model's metric, with the base\n"
           "heading wrapped. plan() uses it by default for a robot.")
      .def(
          "drive", [](const PyRobotModel& r) { return std::string(drive_name(r.drive())); },
          "The robot's base drive, 'holonomic', 'differential_drive' or 'none'. It sets\n"
          "the default base metric. A model built with another base metric keeps that metric.")
      .def("__repr__", &PyRobotModel::repr);

  bind_fixed<PandaModel>(robots, "Panda", "Franka Emika Panda, 7-DoF arm.");
  bind_fixed<UR5Model>(robots, "UR5", "Universal Robots UR5, 6-DoF arm.");
  bind_fixed<BaxterModel>(robots, "Baxter", "Rethink Baxter, 14-DoF dual arm.");
  bind_fixed<PR2Model>(robots, "PR2", "Willow Garage PR2, 14-DoF dual arm.");
  bind_fixed<Fr3GripperModel>(
      robots, "Fr3Gripper",
      "Franka FR3 with a Robotiq 2F-85 on the flange, 7-DoF arm. Its VAMP model is "
      "'fr3_arm_gripper'.");
  bind_mobile<Stretch3Model>(
      robots, "Stretch3",
      "Hello Robot Stretch 3, differential-drive base, 8 coordinates (x, y, theta, lift,\n"
      "arm extension, wrist yaw, pitch, roll). The arm extension moves the four\n"
      "nested arm segments by equal amounts.");
  bind_mobile<Stretch4Model>(
      robots, "Stretch4",
      "Hello Robot Stretch 4, three-omniwheel holonomic base, 8 coordinates (x, y,\n"
      "theta, lift, arm extension, wrist yaw, pitch, roll).");
  bind_mobile<RidgebackUR5eModel>(
      robots, "RidgebackUR5e",
      "Clearpath Ridgeback with a Universal Robots UR5e on its default mount, mecanum\n"
      "holonomic base, 9 coordinates (x, y, theta, six arm joints).");
  bind_mobile<HuskyUR5eModel>(
      robots, "HuskyUR5e",
      "Clearpath Husky with a Universal Robots UR5e on its default top plate, skid-steer\n"
      "differential-drive base, 9 coordinates (x, y, theta, six arm joints).");

  robots.def(
      "available",
      [] {
        std::vector<std::string> names;
        for (const auto r : gr::registered_robots()) names.emplace_back(gr::name(r));
        return names;
      },
      "Names of the built-in robots, in alphabetical order.");
}

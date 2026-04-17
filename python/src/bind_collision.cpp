/// @file bind_collision.cpp
/// @brief Python bindings for geodex collision detection primitives.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "geodex/collision/collision.hpp"

namespace nb = nanobind;
using namespace geodex::collision;

void bind_collision(nb::module_& m) {
  auto col = m.def_submodule("collision", "Collision detection primitives.");

  // --- CircleSDF ---
  nb::class_<CircleSDF>(col, "CircleSDF", "Signed distance function for a circle obstacle.")
      .def(nb::init<double, double, double>(), nb::arg("cx"), nb::arg("cy"), nb::arg("radius"),
           "Create a circle SDF.\n\n"
           "Args:\n"
           "    cx, cy: Center coordinates.\n"
           "    radius: Circle radius.")
      .def(
          "__call__",
          [](const CircleSDF& self, const double x, const double y) {
            const Eigen::Vector2d q(x, y);
            return self(q);
          },
          nb::arg("x"), nb::arg("y"), "Evaluate signed distance at (x, y).")
      .def_prop_ro("cx", &CircleSDF::cx, "X-coordinate of the circle center.")
      .def_prop_ro("cy", &CircleSDF::cy, "Y-coordinate of the circle center.")
      .def_prop_ro("radius", &CircleSDF::radius, "Radius of the circle.");

  // --- CircleSmoothSDF ---
  nb::class_<CircleSmoothSDF>(col, "CircleSmoothSDF",
                              "Smooth-min SDF over multiple circle obstacles.")
      .def(nb::init<std::vector<CircleSDF>, double>(), nb::arg("circles"), nb::arg("beta") = 20.0,
           "Create from circles with smoothing parameter beta.")
      .def(
          "__call__",
          [](const CircleSmoothSDF& self, const double x, const double y) {
            const Eigen::Vector2d q(x, y);
            return self(q);
          },
          nb::arg("x"), nb::arg("y"), "Evaluate smooth signed distance at (x, y).")
      .def(
          "is_free",
          [](const CircleSmoothSDF& self, const double x, const double y) {
            const Eigen::Vector2d q(x, y);
            return self.is_free(q);
          },
          nb::arg("x"), nb::arg("y"), "Check if (x, y) is outside all circles.")
      .def_prop_ro("beta", &CircleSmoothSDF::beta, "Log-sum-exp smoothing parameter.");

  // --- RectObstacle ---
  nb::class_<RectObstacle>(col, "RectObstacle", "An oriented rectangle obstacle.")
      .def(nb::init<>())
      .def_rw("cx", &RectObstacle::cx, "Center x-coordinate.")
      .def_rw("cy", &RectObstacle::cy, "Center y-coordinate.")
      .def_rw("theta", &RectObstacle::theta, "Orientation angle (radians).")
      .def_rw("half_length", &RectObstacle::half_length, "Half-extent along local x-axis.")
      .def_rw("half_width", &RectObstacle::half_width, "Half-extent along local y-axis.");

  // --- RectSmoothSDF ---
  nb::class_<RectSmoothSDF>(col, "RectSmoothSDF",
                            "Smooth-min SDF over oriented rectangle obstacles.")
      .def(nb::init<std::vector<RectObstacle>, double, double>(), nb::arg("obstacles"),
           nb::arg("beta") = 20.0, nb::arg("inflation") = 0.0,
           "Create from rectangle obstacles with smoothing and optional inflation.")
      .def(
          "__call__",
          [](const RectSmoothSDF& self, const double x, const double y) {
            const Eigen::Vector2d q(x, y);
            return self(q);
          },
          nb::arg("x"), nb::arg("y"), "Evaluate smooth signed distance at (x, y).")
      .def_prop_ro("beta", &RectSmoothSDF::beta, "Log-sum-exp smoothing parameter.")
      .def_prop_ro("inflation", &RectSmoothSDF::inflation, "Inflation radius.");

  // --- PolygonFootprint ---
  nb::class_<PolygonFootprint>(col, "PolygonFootprint",
                               "Polygon footprint for swept-volume collision checking.")
      .def_static("rectangle", &PolygonFootprint::rectangle, nb::arg("half_length"),
                  nb::arg("half_width"), nb::arg("samples_per_edge") = 8,
                  "Create a rectangular footprint.")
      .def("sample_count", &PolygonFootprint::sample_count, "Number of perimeter samples.")
      .def("bounding_radius", &PolygonFootprint::bounding_radius,
           "Max distance from origin to any sample.");
}

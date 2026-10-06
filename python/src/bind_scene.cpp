/// @file bind_scene.cpp
/// @brief Python bindings for the in-memory collision-scene builder.
///
/// @details Exposes @c geodex.Scene, the programmatic obstacle builder, and a top-level
/// @c geodex.load_scene that returns a @c geodex.vamp.EnvHandle. plan() reads the scene's
/// environment in C++. @c env() returns the handle, and a held object attaches to it before
/// a checker binds to it.

#include <array>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#include "geodex/integration/vamp/registry.hpp"
#include "wrappers/py_scene.hpp"

namespace nb = nanobind;
namespace gvamp = geodex::integration::vamp;
using geodex::python::PyScene;

void bind_scene(nb::module_& m) {
  nb::class_<PyScene>(
      m, "Scene",
      "In-memory collision-scene builder. Add primitive obstacles, then pass "
      "the scene to a planner.\n\n"
      "Example:\n"
      "    Build a scene and plan a UR5 path through it::\n"
      "\n"
      "        scene = geodex.Scene()\n"
      "        scene.add_box(position=[0.5, 0.0, 0.3], size=[0.4, 0.6, 0.05])\n"
      "        scene.add_sphere(center=[0.3, 0.2, 0.5], radius=0.1)\n"
      "        result = geodex.plan(geodex.robots.UR5(), q0, q1, collision=scene)")
      .def(nb::init<>())
      .def("add_box", &PyScene::add_box, nb::arg("position"), nb::arg("size"),
           nb::arg("orientation") = std::array<double, 4>{0.0, 0.0, 0.0, 1.0},
           "Add an oriented box. size holds the full extents, and orientation is "
           "[qx, qy, qz, qw].")
      .def("add_sphere", &PyScene::add_sphere, nb::arg("center"), nb::arg("radius"),
           "Add a sphere centered at center.")
      .def("add_cylinder", &PyScene::add_cylinder, nb::arg("position"),
           nb::arg("radius"), nb::arg("height"),
           nb::arg("orientation") = std::array<double, 4>{0.0, 0.0, 0.0, 1.0},
           "Add an oriented cylinder. Its local z axis is the cylinder axis, and "
           "orientation is [qx, qy, qz, qw].")
      .def("env", &PyScene::env,
           "Build and return the VAMP environment handle for this scene.\n\n"
           "Attach a held object with geodex.vamp.attach_spheres(handle, spheres), then\n"
           "pass the handle as plan(..., collision=handle).")
      .def("__repr__", [](const PyScene&) { return std::string("<geodex.Scene>"); });

  m.def(
      "load_scene",
      [](const std::string& path) { return gvamp::load_scene(path); },
      nb::arg("path"),
      "Load an MBM-style scene YAML into an opaque VAMP environment handle.\n\n"
      "Supports primitive collision objects (boxes, cylinders, spheres) and mesh\n"
      "objects (axis-aligned bounding-box approximation).");
}

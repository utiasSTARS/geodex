/// @file bind_collision.cpp
/// @brief Python bindings for geodex collision detection primitives.

#include <functional>
#include <stdexcept>
#include <string>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "geodex/collision/collision.hpp"

#include "wrappers/native_collision.hpp"
#include "wrappers/py_callable.hpp"

namespace nb = nanobind;
using namespace geodex::collision;
using geodex::python::call_sdf;
using geodex::python::callable_owner_slots;
using geodex::python::InflatedSDFCallable;
using geodex::python::MemoizedSDFCallable;
using geodex::python::SdfFunction;

namespace {

// Checks the perimeter sample index `i` of a footprint and raises IndexError when it is
// out of range.
int sample_index(const PolygonFootprint& fp, const int i) {
  if (i < 0 || i >= fp.sample_count_raw()) {
    throw nb::index_error(("sample index " + std::to_string(i) + " is outside [0, " +
                           std::to_string(fp.sample_count_raw()) + ")")
                              .c_str());
  }
  return i;
}

}  // namespace

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
      .def(
          "__call__",
          [](const CircleSDF& self, const Eigen::VectorXd& q) { return call_sdf(self, q); },
          nb::arg("q"), "Evaluate signed distance at the point q = (x, y, ...).")
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
          "__call__",
          [](const CircleSmoothSDF& self, const Eigen::VectorXd& q) { return call_sdf(self, q); },
          nb::arg("q"), "Evaluate smooth signed distance at the point q = (x, y, ...).")
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
      .def(
          "__init__",
          [](RectObstacle* self, double cx, double cy, double theta, double half_length,
             double half_width) {
            new (self) RectObstacle{cx, cy, theta, half_length, half_width};
          },
          nb::arg("cx"), nb::arg("cy"), nb::arg("theta"), nb::arg("half_length"),
          nb::arg("half_width"),
          "Create an oriented rectangle from center, orientation, and half-extents.")
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
      .def(
          "__call__",
          [](const RectSmoothSDF& self, const Eigen::VectorXd& q) { return call_sdf(self, q); },
          nb::arg("q"), "Evaluate smooth signed distance at the point q = (x, y, ...).")
      .def_prop_ro("beta", &RectSmoothSDF::beta, "Log-sum-exp smoothing parameter.")
      .def_prop_ro("inflation", &RectSmoothSDF::inflation, "Inflation radius.");

  // --- PolygonFootprint ---
  nb::class_<PolygonFootprint>(col, "PolygonFootprint",
                               "Polygon footprint for swept-volume collision checking.")
      .def(nb::init<const std::vector<Eigen::Vector2d>&, int>(), nb::arg("vertices"),
           nb::arg("samples_per_edge") = 8,
           "Create a convex polygon footprint from ordered body-frame vertices.\n\n"
           "Args:\n"
           "    vertices: Convex polygon vertices (counter-clockwise), centered on the origin.\n"
           "    samples_per_edge: Perimeter samples placed uniformly along each edge.")
      .def_static("rectangle", &PolygonFootprint::rectangle, nb::arg("half_length"),
                  nb::arg("half_width"), nb::arg("samples_per_edge") = 8,
                  "Create a rectangular footprint.")
      .def("sample_count", &PolygonFootprint::sample_count,
           "Number of perimeter samples, padded to even for SIMD.")
      .def("sample_count_raw", &PolygonFootprint::sample_count_raw,
           "Number of perimeter samples before padding.")
      .def("bounding_radius", &PolygonFootprint::bounding_radius,
           "Max distance from origin to any sample.")
      .def("max_sample_gap", &PolygonFootprint::max_sample_gap,
           "Largest distance between two consecutive perimeter samples.")
      .def(
          "body_x",
          [](const PolygonFootprint& fp, const int i) { return fp.body_x(sample_index(fp, i)); },
          nb::arg("i"), "Body-frame x of perimeter sample i, 0 <= i < sample_count_raw().")
      .def(
          "body_y",
          [](const PolygonFootprint& fp, const int i) { return fp.body_y(sample_index(fp, i)); },
          nb::arg("i"), "Body-frame y of perimeter sample i, 0 <= i < sample_count_raw().");

  // --- DistanceGrid ---
  nb::class_<DistanceGrid>(
      col, "DistanceGrid",
      "A 2D precomputed distance transform with bilinear interpolation.\n\n"
      "It stores obstacle distances at regular grid points and answers queries in world\n"
      "meters. Positive values are free space. Zero and negative values are obstacles.")
      .def(nb::init<>())
      .def(nb::init<int, int, double, std::vector<double>>(), nb::arg("width"), nb::arg("height"),
           nb::arg("resolution"), nb::arg("data"),
           "Create from raw row-major distance values (data[r * width + c]).")
      .def("load", &DistanceGrid::load, nb::arg("filename"),
           "Load from the geodex distance-transform text format. Returns True on success.")
      .def(
          "reset",
          [](nb::pointer_and_handle<DistanceGrid> g, const int width, const int height,
             const double resolution) {
            if (width <= 0 || height <= 0 || !(resolution > 0.0)) {
              throw std::invalid_argument(
                  "DistanceGrid.reset: width and height must be positive and resolution > 0");
            }
            std::vector<double>& data = g.p->reset(width, height, resolution);
            return nb::ndarray<nb::numpy, double, nb::ndim<1>>(data.data(), {data.size()}, g.h);
          },
          nb::arg("width"), nb::arg("height"), nb::arg("resolution"),
          "Resize the grid for a rebuild and return a writable view of its width * height\n"
          "values in row-major order. The values are unspecified until written. The view\n"
          "keeps the grid alive and stays valid until the next reset or load.")
      .def("distance_at", &DistanceGrid::distance_at, nb::arg("x"), nb::arg("y"),
           "Bilinear-interpolated signed distance at world coordinates (x, y).")
      .def("width", &DistanceGrid::width, "Grid width in cells.")
      .def("height", &DistanceGrid::height, "Grid height in cells.")
      .def("resolution", &DistanceGrid::resolution, "Cell size in meters.")
      .def("lipschitz_slack", &DistanceGrid::lipschitz_slack,
           "Additive slack over a 1-Lipschitz bound. The interpolated field changes by at\n"
           "most |p - q| + lipschitz_slack() between two points.\n\n"
           "The slack is sqrt(2) times the resolution for an unsigned distance transform. A\n"
           "grid with a negative node counts as the signed transform of an occupancy grid,\n"
           "the distance to the nearest occupied node minus the distance to the nearest free\n"
           "one, and its slack is twice as large.");

  // --- GridSDF ---
  nb::class_<GridSDF>(col, "GridSDF",
                      "SDF callable wrapping a DistanceGrid, for use as a ClearanceMetric sdf.")
      .def(nb::init<const DistanceGrid*>(), nb::arg("grid"), nb::keep_alive<1, 2>(),
           "Wrap a DistanceGrid as an SDF callable. The grid must outlive this object.")
      .def(
          "__call__",
          [](const GridSDF& self, const Eigen::VectorXd& q) { return call_sdf(self, q); },
          nb::arg("q"), "Grid-interpolated signed distance at the point q = (x, y, ...).");

  // --- InflatedSDF ---
  nb::class_<InflatedSDFCallable>(
      col, "InflatedSDF",
      "Wraps any SDF callable and subtracts a constant inflation radius.\n\n"
      "The result is the SDF of the obstacles grown by the inflation radius, for example\n"
      "by the radius of a circular robot.",
      nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](InflatedSDFCallable* self, nb::callable sdf, double inflation) {
            new (self) InflatedSDFCallable(SdfFunction(std::move(sdf), self), inflation);
          },
          nb::arg("sdf"), nb::arg("inflation"),
          nb::sig("def __init__(self, sdf: collections.abc.Callable[[numpy.ndarray[dtype=float64, "
                  "shape=(*), order='C']], float], inflation: float) -> None"),
          "Wrap an SDF callable, subtracting the inflation radius from every query. A\n"
          "geodex.collision SDF runs in C++ without calling into Python.")
      .def(
          "__call__",
          [](const InflatedSDFCallable& self, const Eigen::VectorXd& q) {
            return call_sdf(self, q);
          },
          nb::arg("q"), "Inflated signed distance at the point q.")
      .def_prop_ro("inflation", &InflatedSDFCallable::inflation, "The inflation radius.");

  // --- MemoizedSDF ---
  nb::class_<MemoizedSDFCallable>(
      col, "MemoizedSDF",
      "Wraps an SDF callable with a small table of recently queried poses.\n\n"
      "Entries are keyed on the exact bits of the pose (x, y, theta). A hit returns\n"
      "exactly what the wrapped SDF returned. Copies share one table. Only the thread that\n"
      "created the wrapper uses the table. Other threads call the SDF directly.",
      nb::type_slots(callable_owner_slots))
      .def(
          "__init__",
          [](MemoizedSDFCallable* self, nb::callable sdf) {
            new (self) MemoizedSDFCallable(SdfFunction(std::move(sdf), self));
          },
          nb::arg("sdf"),
          nb::sig("def __init__(self, sdf: collections.abc.Callable[[numpy.ndarray[dtype=float64, "
                  "shape=(*), order='C']], float]) -> None"),
          "Wrap an SDF callable over SE(2) poses (x, y, theta). A geodex.collision SDF runs\n"
          "in C++ without calling into Python.")
      .def(
          "__call__",
          [](const MemoizedSDFCallable& self, const Eigen::VectorXd& q) {
            return call_sdf(self, q);
          },
          nb::arg("q"), "Signed distance at the pose q = (x, y, theta, ...).")
      .def("clear", &MemoizedSDFCallable::clear,
           "Forget every cached value, for example after a rebuild of the wrapped grid.");

  // --- FootprintGridChecker ---
  nb::class_<FootprintGridChecker>(
      col, "FootprintGridChecker",
      "Collision checker of a polygon footprint against a distance grid.\n\n"
      "is_valid(q) is a binary test. Calling the object returns a continuous signed\n"
      "distance, the smallest grid clearance over the footprint minus the safety margin.\n"
      "One object serves as the planner's validity check and as a ClearanceMetric sdf.\n"
      "plan(..., checker.is_valid) and ClearanceMetric(..., checker) call it in C++\n"
      "without calling into Python.")
      .def(nb::init<const DistanceGrid*, PolygonFootprint, double>(), nb::arg("grid"),
           nb::arg("footprint"), nb::arg("safety_margin") = 0.0, nb::keep_alive<1, 2>(),
           "Create a footprint checker. The grid must outlive this object.")
      .def(
          "is_valid",
          [](const FootprintGridChecker& self, const Eigen::Vector3d& q) {
            return self.is_valid(q);
          },
          nb::arg("q"), "Binary collision test at pose q = (x, y, theta). True if collision-free.")
      .def(
          "__call__",
          [](const FootprintGridChecker& self, const Eigen::VectorXd& q) {
            return call_sdf(self, q);
          },
          nb::arg("q"),
          "Continuous clearance at pose q = (x, y, theta), the smallest footprint distance "
          "minus the safety margin.")
      .def(
          "min_distance_capped",
          [](const FootprintGridChecker& self, const Eigen::Vector3d& q, const double cap) {
            return self.min_distance_capped(q, cap);
          },
          nb::arg("q"), nb::arg("cap"),
          "Footprint clearance that is exact below cap.\n\n"
          "It equals the checker's value whenever that value lies in (0, cap). Above cap, it\n"
          "is a lower bound of at least cap. When the footprint collides, it is at most 0,\n"
          "and only its sign is meaningful.")
      .def_prop_ro("safety_margin", &FootprintGridChecker::safety_margin, "The safety margin.");

  // --- rects_overlap ---
  col.def("rects_overlap", &rects_overlap, nb::arg("a"), nb::arg("b"),
          "Separating-axis overlap test for two oriented rectangles. True if they collide.");
}

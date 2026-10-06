from collections.abc import Callable, Sequence
from typing import Annotated, overload

from numpy.typing import ArrayLike


class CircleSDF:
    """Signed distance function for a circle obstacle."""

    def __init__(self, cx: float, cy: float, radius: float) -> None:
        """
        Create a circle SDF.

        Args:
            cx, cy: Center coordinates.
            radius: Circle radius.
        """

    @overload
    def __call__(self, x: float, y: float) -> float:
        """Evaluate signed distance at (x, y)."""

    @overload
    def __call__(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Evaluate signed distance at the point q = (x, y, ...)."""

    @property
    def cx(self) -> float:
        """X-coordinate of the circle center."""

    @property
    def cy(self) -> float:
        """Y-coordinate of the circle center."""

    @property
    def radius(self) -> float:
        """Radius of the circle."""

class CircleSmoothSDF:
    """Smooth-min SDF over multiple circle obstacles."""

    def __init__(self, circles: Sequence[CircleSDF], beta: float = 20.0) -> None:
        """Create from circles with smoothing parameter beta."""

    @overload
    def __call__(self, x: float, y: float) -> float:
        """Evaluate smooth signed distance at (x, y)."""

    @overload
    def __call__(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Evaluate smooth signed distance at the point q = (x, y, ...)."""

    def is_free(self, x: float, y: float) -> bool:
        """Check if (x, y) is outside all circles."""

    @property
    def beta(self) -> float:
        """Log-sum-exp smoothing parameter."""

class DistanceGrid:
    """
    A 2D precomputed distance transform with bilinear interpolation.

    It stores obstacle distances at regular grid points and answers queries in world
    meters. Positive values are free space. Zero and negative values are obstacles.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, width: int, height: int, resolution: float, data: Sequence[float]) -> None:
        """Create from raw row-major distance values (data[r * width + c])."""

    def load(self, filename: str) -> bool:
        """
        Load from the geodex distance-transform text format. Returns True on success.
        """

    def reset(self, width: int, height: int, resolution: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None))]:
        """
        Resize the grid for a rebuild and return a writable view of its width * height
        values in row-major order. The values are unspecified until written. The view
        keeps the grid alive and stays valid until the next reset or load.
        """

    def distance_at(self, x: float, y: float) -> float:
        """Bilinear-interpolated signed distance at world coordinates (x, y)."""

    def width(self) -> int:
        """Grid width in cells."""

    def height(self) -> int:
        """Grid height in cells."""

    def resolution(self) -> float:
        """Cell size in meters."""

    def lipschitz_slack(self) -> float:
        """
        Additive slack over a 1-Lipschitz bound. The interpolated field changes by at
        most |p - q| + lipschitz_slack() between two points.

        The slack is sqrt(2) times the resolution for an unsigned distance transform. A
        grid with a negative node counts as the signed transform of an occupancy grid,
        the distance to the nearest occupied node minus the distance to the nearest free
        one, and its slack is twice as large.
        """

class FootprintGridChecker:
    """
    Collision checker of a polygon footprint against a distance grid.

    is_valid(q) is a binary test. Calling the object returns a continuous signed
    distance, the smallest grid clearance over the footprint minus the safety margin.
    One object serves as the planner's validity check and as a ClearanceMetric sdf.
    plan(..., checker.is_valid) and ClearanceMetric(..., checker) call it in C++
    without calling into Python.
    """

    def __init__(self, grid: DistanceGrid, footprint: PolygonFootprint, safety_margin: float = 0.0) -> None:
        """Create a footprint checker. The grid must outlive this object."""

    def is_valid(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> bool:
        """
        Binary collision test at pose q = (x, y, theta). True if collision-free.
        """

    def __call__(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """
        Continuous clearance at pose q = (x, y, theta), the smallest footprint distance minus the safety margin.
        """

    def min_distance_capped(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], cap: float) -> float:
        """
        Footprint clearance that is exact below cap.

        It equals the checker's value whenever that value lies in (0, cap). Above cap, it
        is a lower bound of at least cap. When the footprint collides, it is at most 0,
        and only its sign is meaningful.
        """

    @property
    def safety_margin(self) -> float:
        """The safety margin."""

class GridSDF:
    """
    SDF callable wrapping a DistanceGrid, for use as a ClearanceMetric sdf.
    """

    def __init__(self, grid: DistanceGrid) -> None:
        """
        Wrap a DistanceGrid as an SDF callable. The grid must outlive this object.
        """

    def __call__(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Grid-interpolated signed distance at the point q = (x, y, ...)."""

class InflatedSDF:
    """
    Wraps any SDF callable and subtracts a constant inflation radius.

    The result is the SDF of the obstacles grown by the inflation radius, for example
    by the radius of a circular robot.
    """

    def __init__(self, sdf: Callable[[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]], float], inflation: float) -> None:
        """
        Wrap an SDF callable, subtracting the inflation radius from every query. A
        geodex.collision SDF runs in C++ without calling into Python.
        """

    def __call__(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Inflated signed distance at the point q."""

    @property
    def inflation(self) -> float:
        """The inflation radius."""

class MemoizedSDF:
    """
    Wraps an SDF callable with a small table of recently queried poses.

    Entries are keyed on the exact bits of the pose (x, y, theta). A hit returns
    exactly what the wrapped SDF returned. Copies share one table. Only the thread that
    created the wrapper uses the table. Other threads call the SDF directly.
    """

    def __init__(self, sdf: Callable[[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]], float]) -> None:
        """
        Wrap an SDF callable over SE(2) poses (x, y, theta). A geodex.collision SDF runs
        in C++ without calling into Python.
        """

    def __call__(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Signed distance at the pose q = (x, y, theta, ...)."""

    def clear(self) -> None:
        """
        Forget every cached value, for example after a rebuild of the wrapped grid.
        """

class PolygonFootprint:
    """Polygon footprint for swept-volume collision checking."""

    def __init__(self, vertices: Sequence[Annotated[ArrayLike, dict(dtype='float64', shape=(2), order='C')]], samples_per_edge: int = 8) -> None:
        """
        Create a convex polygon footprint from ordered body-frame vertices.

        Args:
            vertices: Convex polygon vertices (counter-clockwise), centered on the origin.
            samples_per_edge: Perimeter samples placed uniformly along each edge.
        """

    @staticmethod
    def rectangle(half_length: float, half_width: float, samples_per_edge: int = 8) -> PolygonFootprint:
        """Create a rectangular footprint."""

    def sample_count(self) -> int:
        """Number of perimeter samples, padded to even for SIMD."""

    def sample_count_raw(self) -> int:
        """Number of perimeter samples before padding."""

    def bounding_radius(self) -> float:
        """Max distance from origin to any sample."""

    def max_sample_gap(self) -> float:
        """Largest distance between two consecutive perimeter samples."""

    def body_x(self, i: int) -> float:
        """Body-frame x of perimeter sample i, 0 <= i < sample_count_raw()."""

    def body_y(self, i: int) -> float:
        """Body-frame y of perimeter sample i, 0 <= i < sample_count_raw()."""

class RectObstacle:
    """An oriented rectangle obstacle."""

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, cx: float, cy: float, theta: float, half_length: float, half_width: float) -> None:
        """
        Create an oriented rectangle from center, orientation, and half-extents.
        """

    @property
    def cx(self) -> float:
        """Center x-coordinate."""

    @cx.setter
    def cx(self, arg: float, /) -> None: ...

    @property
    def cy(self) -> float:
        """Center y-coordinate."""

    @cy.setter
    def cy(self, arg: float, /) -> None: ...

    @property
    def theta(self) -> float:
        """Orientation angle (radians)."""

    @theta.setter
    def theta(self, arg: float, /) -> None: ...

    @property
    def half_length(self) -> float:
        """Half-extent along local x-axis."""

    @half_length.setter
    def half_length(self, arg: float, /) -> None: ...

    @property
    def half_width(self) -> float:
        """Half-extent along local y-axis."""

    @half_width.setter
    def half_width(self, arg: float, /) -> None: ...

class RectSmoothSDF:
    """Smooth-min SDF over oriented rectangle obstacles."""

    def __init__(self, obstacles: Sequence[RectObstacle], beta: float = 20.0, inflation: float = 0.0) -> None:
        """Create from rectangle obstacles with smoothing and optional inflation."""

    @overload
    def __call__(self, x: float, y: float) -> float:
        """Evaluate smooth signed distance at (x, y)."""

    @overload
    def __call__(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Evaluate smooth signed distance at the point q = (x, y, ...)."""

    @property
    def beta(self) -> float:
        """Log-sum-exp smoothing parameter."""

    @property
    def inflation(self) -> float:
        """Inflation radius."""

def rects_overlap(a: RectObstacle, b: RectObstacle) -> bool:
    """
    Separating-axis overlap test for two oriented rectangles. True if they collide.
    """

from collections.abc import Callable, Sequence
import enum
from typing import Annotated, overload

from numpy.typing import ArrayLike

from . import (
    collision as collision,
    heuristics as heuristics,
    planners as planners,
    robots as robots,
    vamp as vamp
)


class AffineCombinedMetric:
    """
    Positive linear combination of N Riemannian metric policies.

    Composes N metric policies g_1, ..., g_N with non-negative coefficients
    c_1, ..., c_N into the metric <u, v>_p = sum_k c_k <u, v>_p^{g_k}.
    Use it for composite metrics such as 'pullback + beta * kinetic-energy'. It needs
    at least one summand, non-negative coefficients and at least one positive one.
    """

    def __init__(self, metrics: list, coeffs: Sequence[float]) -> None:
        """
        Create an AffineCombinedMetric from a list of metrics and a list of
        non-negative coefficients (matching length, at least one > 0).
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Combined inner product sum_k c_k <u, v>_p^{g_k}."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Combined Riemannian norm."""

    @property
    def coeffs(self) -> list[float]:
        """The coefficient list."""

    @property
    def size(self) -> int:
        """Number of summands."""

    def __repr__(self) -> str: ...

class ClearanceMetric:
    """
    SDF-based conformal metric that scales a base metric by obstacle proximity.

    The inner product is <u,v>_q = (1 + kappa * exp(-beta * sdf(q))) * <u,v>^base_q.
    """

    def __init__(self, base_metric: object, sdf: Callable, kappa: float = 5.0, beta: float = 3.0) -> None:
        """
        Create an SDF-based conformal metric.

        Args:
            base_metric: Any geodex metric to scale.
            sdf: Callable(q) -> float returning signed distance (positive = free). A
                geodex.collision SDF, such as a FootprintGridChecker or a GridSDF, runs
                in C++ without calling into Python.
            kappa: Strength of obstacle repulsion (default 5.0).
            beta: Falloff rate (default 3.0).
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Conformally scaled inner product."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Conformally scaled norm."""

    @property
    def kappa(self) -> float:
        """Obstacle repulsion strength."""

    @property
    def beta(self) -> float:
        """Falloff rate."""

    def __repr__(self) -> str: ...

class ConfigurationSpace:
    """
    A configuration space combining a base manifold's topology with a custom metric.

    Topology operations (exp, log, dim, random_point) come from the base manifold.
    Geometry operations (inner, norm, distance) come from the custom metric.
    """

    def __init__(self, base_manifold: object, metric: object) -> None:
        """
        Create a configuration space.

        Args:
            base_manifold: Base manifold (Sphere, Euclidean, Torus, SE2, etc.).
            metric: Custom metric (KineticEnergyMetric, ConstantSPDMetric, etc.).
        """

    def dim(self) -> int:
        """Return the intrinsic dimension."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample a random point from the base manifold."""

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian inner product from the custom metric."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian norm from the custom metric."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Exponential map from the base manifold."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Logarithmic map from the base manifold."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """
        Geodesic distance using the midpoint approximation with the custom metric.
        """

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Geodesic interpolation at parameter t in [0, 1]."""

    def __repr__(self) -> str: ...

class ConstantSPDMetric:
    """
    Point-independent Riemannian metric defined by a constant SPD matrix.

    The inner product is <u, v> = u^T A v where A is a constant SPD matrix.
    """

    def __init__(self, matrix: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]) -> None:
        """
        Create a constant SPD metric.

        Args:
            matrix: Symmetric positive-definite weight matrix.
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian inner product u^T A v."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian norm sqrt(v^T A v)."""

    def __repr__(self) -> str: ...

class DirectionalMotionValidator:
    """
    Forward-drivability constraint for plan() on SE(2)-like spaces.

    It rejects tree edges whose net body-forward motion is below -max_reverse_length,
    and planned paths drive forward. The manifold's log must return a body twist with
    the forward component at index 0. For SE2, that means retraction='exponential' with
    frame='body', and plan() refuses any other SE2. Pass it as
    plan(..., motion_validator=...).
    """

    def __init__(self, max_reverse_length: float = 0.0) -> None:
        """
        Create the constraint.

        Args:
            max_reverse_length: Reverse budget per edge in body-forward units. 0 forbids
                any net reverse motion.
        """

    @property
    def max_reverse_length(self) -> float:
        """Reverse budget per edge in body-forward units."""

    @max_reverse_length.setter
    def max_reverse_length(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

class Euclidean:
    """
    Euclidean manifold R^n with the standard flat metric.

    Exp/log are trivial (addition/subtraction).
    """

    def __init__(self, dim: int, sampler: str = 'scrambled') -> None:
        """
        Create a Euclidean space of the given dimension.

        Args:
            dim: Dimension n.
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

    def dim(self) -> int:
        """Return the dimension."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample a random point uniformly in the sampling bounds."""

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Inner product <u, v> = u . v."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Euclidean norm ||v||."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Exponential map: p + v."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Logarithmic map: q - p."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Euclidean distance ||p - q||."""

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Linear interpolation (1-t)*p + t*q."""

    def seed(self, seed: int) -> None:
        """Reseed the sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler: 'scrambled', 'halton', or 'random'."""

    def set_sampling_bounds(self, lo: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], hi: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> None:
        """
        Set the per-dimension sampling bounds (default [-1, 1]^n).

        Bounds affect random_point() and the search domain that plan() derives; the
        exp/log/metric operations are unchanged. A ConfigurationSpace built on this
        manifold inherits these bounds.
        """

    def lo(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Lower per-dimension sampling bound."""

    def hi(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Upper per-dimension sampling bound."""

    def __repr__(self) -> str: ...

class HaltonSampler:
    """
    Deterministic Halton low-discrepancy sampler.

    Samples the same sequence in [0, 1)^n every run, with no randomization.
    """

    def __init__(self) -> None: ...

    def sample(self, n: int) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample the next n-dimensional point in [0, 1)^n."""

    def seed(self, seed: int) -> None:
        """Reset the sequence to start from the given index."""

class InterpolationResult:
    """
    Result of discrete_geodesic. It holds the path, the termination
    status, the iteration count and the initial and final Riemannian
    distances to the target.
    """

    @property
    def path(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """
        (N, d) float64 ndarray of the points from start toward target, start first.
        """

    @property
    def waypoints(self) -> list[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]]:
        """The path as a list of np.ndarray."""

    @property
    def status(self) -> InterpolationStatus:
        """InterpolationStatus of the walk. Check it before using `path`."""

    @property
    def iterations(self) -> int:
        """Number of successful gradient steps, excluding distortion retries."""

    @property
    def distortion_halvings(self) -> int:
        """Number of times a failed progress check halved the step cap."""

    @property
    def fd_midpoint_fallbacks(self) -> int:
        """
        Number of finite-difference samples that used |log|_R after the guard rejected the midpoint surrogate. A nonzero value flags a non-Riemannian retraction, a cut-locus crossing or a non-smooth metric near the point.
        """

    @property
    def initial_distance(self) -> float:
        """Riemannian distance from start to target at entry."""

    @property
    def final_distance(self) -> float:
        """Riemannian distance from the final iterate to target at exit."""

    def __repr__(self) -> str: ...

class InterpolationSettings:
    """
    Settings for the discrete geodesic walk.

    Each iteration takes a Riemannian step of length min(step_size, remaining_distance)
    in the descent direction. step_size also sets the path resolution. The iteration
    count and the path size scale as initial_distance / step_size.
    """

    def __init__(self, step_size: float = 0.5, convergence_tol: float = 0.0001, convergence_rel: float = 0.001, max_steps: int = 100, fd_epsilon: float = 0.0, distortion_ratio: float = 1.5, growth_factor: float = 1.5, min_step_size: float = 1e-12, gradient_eps: float = 1e-12, cut_locus_eps: float = 1e-10, force_log_direction: bool = False, fd_midpoint_guard_tau: float = 0.25) -> None:
        """
        Create interpolation settings.

        Args:
            step_size: Largest Riemannian step per iteration and the path resolution.
            convergence_tol: Absolute stop threshold on |log(current, target)|_R.
            convergence_rel: Relative stop threshold (distance < rel * initial_distance).
            max_steps: Maximum number of successful gradient-descent steps.
            fd_epsilon: Central finite-difference step of the fallback gradient. 0 selects
                it automatically.
            distortion_ratio: Largest accepted ratio of the realized step length to the
                intended step length. A larger step halves the step cap and retries.
            growth_factor: Factor that regrows the step cap after a successful step.
            min_step_size: Failure threshold after repeated distortion halvings.
            gradient_eps: Gradient norm threshold of the GradientVanished status.
            cut_locus_eps: |log|_R threshold that flags the CutLocus status.
            force_log_direction: If True, always descend along -log(current, target) and
                skip the finite-difference fallback. The path follows the base
                retraction's geodesic instead of the metric's Riemannian geodesic.
            fd_midpoint_guard_tau: Relative-error threshold of the midpoint distance
                surrogate in the finite-difference gradient. Above it, the sample uses
                |log|_R for that basis direction. 0 always uses |log|_R.
        """

    @property
    def step_size(self) -> float:
        """Largest Riemannian step per iteration and the path resolution."""

    @step_size.setter
    def step_size(self, arg: float, /) -> None: ...

    @property
    def convergence_tol(self) -> float:
        """Absolute stop threshold on |log(current, target)|_R."""

    @convergence_tol.setter
    def convergence_tol(self, arg: float, /) -> None: ...

    @property
    def convergence_rel(self) -> float:
        """Relative stop threshold (distance < rel * initial_distance)."""

    @convergence_rel.setter
    def convergence_rel(self, arg: float, /) -> None: ...

    @property
    def max_steps(self) -> int:
        """Maximum number of successful gradient-descent steps."""

    @max_steps.setter
    def max_steps(self, arg: int, /) -> None: ...

    @property
    def fd_epsilon(self) -> float:
        """
        Central finite-difference step of the fallback gradient. 0 selects it automatically.
        """

    @fd_epsilon.setter
    def fd_epsilon(self, arg: float, /) -> None: ...

    @property
    def distortion_ratio(self) -> float:
        """
        Largest accepted ratio of the realized step length to the intended step length.
        """

    @distortion_ratio.setter
    def distortion_ratio(self, arg: float, /) -> None: ...

    @property
    def growth_factor(self) -> float:
        """Factor by which the step cap grows back after a successful iteration."""

    @growth_factor.setter
    def growth_factor(self, arg: float, /) -> None: ...

    @property
    def min_step_size(self) -> float:
        """Failure threshold after repeated distortion halvings."""

    @min_step_size.setter
    def min_step_size(self, arg: float, /) -> None: ...

    @property
    def gradient_eps(self) -> float:
        """Gradient norm threshold of the GradientVanished status."""

    @gradient_eps.setter
    def gradient_eps(self, arg: float, /) -> None: ...

    @property
    def cut_locus_eps(self) -> float:
        """|log|_R threshold that flags CutLocus."""

    @cut_locus_eps.setter
    def cut_locus_eps(self, arg: float, /) -> None: ...

    @property
    def force_log_direction(self) -> bool:
        """
        If True, always descend along -log(current, target) and skip the finite-difference fallback. The path follows the base retraction's geodesic instead of the metric's Riemannian geodesic.
        """

    @force_log_direction.setter
    def force_log_direction(self, arg: bool, /) -> None: ...

    @property
    def fd_midpoint_guard_tau(self) -> float:
        """
        Relative-error threshold of the midpoint distance surrogate in the finite-difference gradient. Above it, the sample uses |log|_R.
        """

    @fd_midpoint_guard_tau.setter
    def fd_midpoint_guard_tau(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

class InterpolationStatus(enum.Enum):
    """Termination status of the discrete geodesic walk."""

    Converged = 0
    """The distance to the target fell below the convergence tolerance."""

    MaxStepsReached = 1
    """The walk used its iteration budget before reaching the tolerance."""

    GradientVanished = 2
    """The Riemannian gradient vanished at a point other than the target."""

    CutLocus = 3
    """
    log returned about zero for distinct points, for example antipodal ones.
    """

    StepShrunkToZero = 4
    """Distortion halvings drove the step size below min_step_size."""

    DegenerateInput = 5
    """Start and target are equal. The path has a single point."""

class JacobiMetric:
    """
    Jacobi metric for minimum-time geodesics under a potential field.

    The inner product at q is <u, v>_q = 2(H - P(q)) u^T M(q) v
    where H is the total energy and P(q) is the potential energy.
    """

    def __init__(self, mass_matrix_fn: Callable, potential_fn: Callable, total_energy: float) -> None:
        """
        Create a Jacobi metric.

        Args:
            mass_matrix_fn: Callable(q) -> np.ndarray returning the SPD mass matrix.
            potential_fn: Callable(q) -> float returning the potential energy.
            total_energy: Total energy H (must satisfy H > P(q) everywhere).
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian inner product 2(H - P(p)) u^T M(p) v."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian norm."""

    def __repr__(self) -> str: ...

class KineticEnergyMetric:
    """
    Kinetic-energy metric g(q) = M(q).

    The inner product at q is <u, v>_q = u^T M(q) v where M(q) is a
    symmetric positive-definite mass matrix returned by the callable.
    """

    def __init__(self, mass_matrix_fn: Callable) -> None:
        """
        Create a kinetic-energy metric.

        Args:
            mass_matrix_fn: Callable(q) -> np.ndarray returning the SPD mass matrix.
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian inner product <u, v>_p = u^T M(p) v."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian norm ||v||_p = sqrt(v^T M(p) v)."""

    def __repr__(self) -> str: ...

class LogLevel(enum.Enum):
    """How much plan() lets its planners print."""

    Debug = 0

    Info = 1

    Warn = 2

    Error = 3

    Off = 4

class PathSmoothingProfile:
    """Time and work counters of one smooth_path call."""

    @property
    def total_ms(self) -> float:
        """Whole call, milliseconds."""

    @property
    def shortcut_ms(self) -> float:
        """Shortcut rounds, milliseconds."""

    @property
    def descent_ms(self) -> float:
        """Subdivision and local energy descent, milliseconds."""

    @property
    def resample_ms(self) -> float:
        """Even output spacing, milliseconds."""

    @property
    def rounding_ms(self) -> float:
        """Corner rounding, milliseconds."""

    @property
    def certify_ms(self) -> float:
        """Final check, fallbacks included, milliseconds."""

    @property
    def point_checks(self) -> int:
        """Configurations handed to the validity oracle."""

    @property
    def batch_calls(self) -> int:
        """Batched validity calls."""

    @property
    def edge_checks(self) -> int:
        """Edges tested."""

    @property
    def edge_proofs(self) -> int:
        """Edges settled by edge_provably_clear."""

    @property
    def shortcut_attempts(self) -> int:
        """Shortcuts tried."""

    @property
    def shortcuts(self) -> int:
        """Shortcuts accepted."""

    @property
    def relax_visits(self) -> int:
        """Waypoint visits of the energy descent."""

    @property
    def relax_moves(self) -> int:
        """Waypoint moves that passed the edge checks."""

    @property
    def rounded_corners(self) -> int:
        """Corners replaced by a curve."""

    @property
    def cusps(self) -> int:
        """Corners that turn more than corner_max_angle."""

    @property
    def kept_corners(self) -> int:
        """Corners that stay after every curve size failed a check."""

    @property
    def split_corners(self) -> int:
        """Rounded corners that keep the corner in the leading sharp_coordinates."""

    @property
    def rounding_retries(self) -> int:
        """Curve sizes that failed a check."""

    @property
    def input_waypoints(self) -> int:
        """Size of the input path."""

    @property
    def output_waypoints(self) -> int:
        """Size of the returned path."""

    @property
    def fallback(self) -> int:
        """
        Returned stage. 0 is the smoothed path, 1 the optimized waypoints before resampling, 2 the first shortcut round and 3 the input.
        """

class PathSmoothingResult:
    """Result of smooth_path."""

    @property
    def path(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """(N, d) float64 ndarray, the returned path."""

    @property
    def waypoints(self) -> list[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]]:
        """list[np.ndarray], the returned path as a Python list."""

    @property
    def length(self) -> float:
        """Metric length of the returned path."""

    @property
    def collision_free(self) -> bool:
        """
        True when every waypoint and every edge of the smoothed path passed the check. The returned waypoints lie on that path, and the edges between them stay within corner_tolerance of it.
        """

    @property
    def first_invalid_index(self) -> int | None:
        """
        Index of the first waypoint that fails or starts a failing edge, None when the path passes.
        """

    @property
    def profile(self) -> PathSmoothingProfile:
        """Timing and work counters."""

    def __repr__(self) -> str: ...

class PathSmoothingSettings:
    """
    Settings for smooth_path. The defaults work in meters for SE(2) and in radians
    for an arm.
    """

    def __init__(self, collision_check_resolution: float = 0.0, output_spacing: float = 0.0, seed: int = 42, edge_provably_clear: object | None = None, edge_validator: object | None = None, path_predicate: object | None = None, round_corners: bool = True, corner_tolerance: float = 0.0001, corner_max_angle: float = 1.5707963267948966, sharp_coordinates: int = 0, edge_travel: object | None = None) -> None:
        """
        Create smoothing settings.

        Args:
            collision_check_resolution: Largest spacing between validity samples along an
                edge, measured as the coordinate norm of log(a, b), or in the units of
                edge_travel when that is set. 0 uses one hundredth of the input path's
                coordinate length and ignores edge_travel. smooth_path raises ValueError
                when it is negative, not finite, or asks for more than 1e7 samples on one
                edge.
            output_spacing: Longest step between the returned waypoints, as the coordinate
                norm of log. 0 leaves the step without a limit. The waypoints lie on the
                smoothed path at equal steps between the ends and the corners without a
                curve, at the longest step whose edges stay within corner_tolerance of it.
            seed: Seed of the shortcut sampler. The result is a pure function of the input.
            edge_provably_clear: Optional callable (a, b) -> bool. True proves the whole
                geodesic from a to b valid and skips its samples. False proves nothing.
            edge_validator: Optional callable (a, b) -> bool that decides every edge of
                the smoothed path in place of the sampled test, for example a planner's
                motion validator. The evenly spaced edges do not pass through it. Give an
                asymmetric constraint also as path_predicate.
            path_predicate: Optional callable (list of waypoints) -> bool. smooth_path
                rejects a shortcut, a waypoint move or a rounded corner that fails it, and
                returns the smoothed path's own waypoints when the evenly spaced path fails.
            round_corners: Round the corners of the smoothed path into C2 curves. False
                keeps the corners.
            corner_tolerance: Largest distance between a rounding curve and the edges
                between its samples, and between the smoothed path and the evenly spaced
                edges, as the coordinate norm of log.
            corner_max_angle: Largest turning angle of a rounded corner in radians, under
                the metric. A sharper corner stays.
            sharp_coordinates: Number of leading tangent coordinates whose path may keep a
                corner, such as the pose of a differential-drive base before its arm's
                joints. Where the curve through every coordinate stays smaller than its full
                size, and at a corner sharper than corner_max_angle, smooth_path also tries a
                curve that keeps the corner in these coordinates and rounds the others, and
                the larger curve stays. exp and log must act on these coordinates apart from
                the others, as on a product space. 0 rounds every coordinate together, and a
                negative value raises ValueError.
            edge_travel: Optional callable (a, b) -> float, a bound on how far the checked
                geometry moves along the edge from a to b, in the units of a positive
                collision_check_resolution. The checks then space the edge by it instead of
                the coordinate norm of log(a, b). The bound must hold for every part of the
                edge in proportion to its share, as a bound on the speed does. smooth_path
                raises ValueError when it returns a negative or non-finite value.
        """

    @property
    def collision_check_resolution(self) -> float:
        """
        Largest spacing between validity samples along an edge. 0 derives it from the input path.
        """

    @collision_check_resolution.setter
    def collision_check_resolution(self, arg: float, /) -> None: ...

    @property
    def output_spacing(self) -> float:
        """
        Longest step between the evenly spaced waypoints. 0 leaves the step without a limit.
        """

    @output_spacing.setter
    def output_spacing(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int:
        """Seed of the shortcut sampler."""

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def round_corners(self) -> bool:
        """Round the corners of the smoothed path into C2 curves."""

    @round_corners.setter
    def round_corners(self, arg: bool, /) -> None: ...

    @property
    def corner_tolerance(self) -> float:
        """
        Largest distance between a rounding curve and the edges between its samples, and between the smoothed path and the evenly spaced edges, as the coordinate norm of log.
        """

    @corner_tolerance.setter
    def corner_tolerance(self, arg: float, /) -> None: ...

    @property
    def corner_max_angle(self) -> float:
        """
        Largest turning angle of a rounded corner in radians, under the metric.
        """

    @corner_max_angle.setter
    def corner_max_angle(self, arg: float, /) -> None: ...

    @property
    def sharp_coordinates(self) -> int:
        """
        Number of leading tangent coordinates whose path may keep a corner while the others are rounded.
        """

    @sharp_coordinates.setter
    def sharp_coordinates(self, arg: int, /) -> None: ...

    @property
    def edge_provably_clear(self) -> Callable[[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]], bool] | None:
        """Optional sufficient test (a, b) -> bool for a whole edge, or None."""

    @edge_provably_clear.setter
    def edge_provably_clear(self, fn: object | None) -> None: ...

    @property
    def edge_validator(self) -> Callable[[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]], bool] | None:
        """
        Optional edge test (a, b) -> bool that replaces the sampled test, or None.
        """

    @edge_validator.setter
    def edge_validator(self, fn: object | None) -> None: ...

    @property
    def edge_travel(self) -> Callable[[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]], float] | None:
        """
        Optional bound (a, b) -> float on how far the checked geometry moves along an edge, or None.
        """

    @edge_travel.setter
    def edge_travel(self, fn: object | None) -> None: ...

    @property
    def path_predicate(self) -> Callable[[list[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]]], bool] | None:
        """Optional predicate on the whole candidate path, or None."""

    @path_predicate.setter
    def path_predicate(self, fn: object | None) -> None: ...

    def __repr__(self) -> str: ...

class PlanResult:
    """Outcome of a plan() call."""

    @property
    def solved(self) -> bool:
        """True when the planner found an exact solution."""

    @property
    def path(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """
        (N, d) float64 ndarray of the final path. When smoothed is True, it is the
        smoother's output, joined by the manifold's geodesic. Otherwise, it holds
        the planner's path, densified along the planner's own interpolation. The
        planner checked those edges on its own curve and spacing.
        """

    @property
    def waypoints(self) -> list[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]]:
        """The final path as a list of np.ndarray."""

    @property
    def raw_path(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """(N, d) float64 ndarray, the planner's waypoints before smoothing."""

    @property
    def smoothed(self) -> bool:
        """True when path is the smoother's output."""

    @property
    def smooth_ms(self) -> float:
        """Wall-clock smoothing time in milliseconds."""

    @property
    def cost(self) -> float:
        """Geodesic length of the final path under the metric."""

    @property
    def time_ms(self) -> float:
        """Wall-clock solve time in milliseconds."""

    @property
    def first_solution_ms(self) -> float:
        """
        Milliseconds from the start of the search to the first exact solution, -1 without one. The rest of time_ms refines it.
        """

    @property
    def first_solution_iterations(self) -> int:
        """
        Termination checks before the first exact solution, counted like PlanSettings.iterations, 0 without one.
        """

    @property
    def informed_samples(self) -> int:
        """G-RRT* samples from an informed set, the greedy set included."""

    @property
    def focused_samples(self) -> int:
        """G-RRT* samples from the greedy set."""

    @property
    def uniform_samples(self) -> int:
        """
        G-RRT* samples from the whole space, taken before the first solution or when the informed set is empty.
        """

    def __bool__(self) -> bool: ...

    def __len__(self) -> int: ...

    def __repr__(self) -> str: ...

class PlanSettings:
    """
    Settings of a plan() call. The defaults solve most problems.

    interp selects the curve of the planner's edges by name, one of 'base_geodesic'
    (the default), 'auto' or 'riemannian_geodesic'.
    """

    def __init__(self, time: float = 1.0, iterations: int = 0, refine_time: float = 0.0, planner: planners.GreedyRRTstar | None = None, collision_check_resolution: float = 0.0, interp: str = 'base_geodesic', goal_tolerance: float = 0.0, seed: int = 0, smooth: bool = True, smoothing: object | None = None, limits: tuple[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]] | None = None) -> None:
        """
        Create plan settings.

        Args:
            time: Planning time budget in seconds.
            iterations: Iteration budget. When > 0, the planner runs this many iterations
                instead of running by time. A seeded plan with an iteration budget is
                reproducible.
            refine_time: Seconds of refinement after the first exact solution, within
                time. 0 refines for the whole budget. plan() ignores it when iterations
                is set.
            planner: A planners.GreedyRRTstar with the planner's parameters. None selects
                the defaults.
            collision_check_resolution: Spacing of the edge checks of the planner and the
                smoother, in coordinate distance. 0 keeps OMPL's default spacing for the
                planner and gives the smoother the spacing in smoothing, or one hundredth
                of the diagonal of the planning bounds when that is 0 too. plan() raises
                ValueError when it is negative, not finite, or fine enough that an edge
                across the bounds needs more than 1e7 checks. For a robot with a Scene, it
                is instead the largest distance in meters that a robot sphere moves between
                two checks of the smoother, 0.005 when 0 (see plan()).
            interp: Curve of the planner's edges ('base_geodesic', 'auto',
                'riemannian_geodesic'). 'base_geodesic', the default, uses the space's
                geodesic. 'riemannian_geodesic' uses the metric's discrete geodesic. 'auto'
                picks the base geodesic when the space's log is the Riemannian logarithm
                of its metric or a motion validator is installed, and the discrete
                geodesic otherwise.
            goal_tolerance: Accepted distance to the goal.
            seed: Seed of the plan. A nonzero seed gives the same plan in any state of the
                space. 0, the default, takes a fresh seed from the space's own sampler and
                advances it by one random_point(). Repeated plans are then independent,
                and space.seed(s) repeats the same sequence of plans. The samples follow
                the space's sampler kind and reproduce only under an iteration budget. On
                a space whose distance breaks the triangle inequality, such as SE2 with
                unequal weights, one seed can give different plans on Linux and macOS.
            smooth: Run smooth_path on the planner's path.
            smoothing: PathSmoothingSettings for smooth_path. None selects the defaults.
                A nonzero collision_check_resolution overrides the one inside, except with
                smoothing.edge_travel, whose resolution must then be positive.
            limits: Physical limits of the coordinates as (lower, upper), such as joint
                limits. The search stays inside them, and the smoother treats a point
                outside them as invalid. None takes the limits that the space declares,
                such as a robot's joint limits. Without declared limits, the search uses a
                box around the region the space samples, and the smoother does not treat
                this box as a limit. Limits that are not finite raise ValueError.
        """

    @property
    def time(self) -> float:
        """Planning time budget in seconds."""

    @time.setter
    def time(self, arg: float, /) -> None: ...

    @property
    def iterations(self) -> int:
        """
        Iteration budget. When > 0, the planner runs this many iterations instead of running by time.
        """

    @iterations.setter
    def iterations(self, arg: int, /) -> None: ...

    @property
    def refine_time(self) -> float:
        """
        Seconds of refinement after the first exact solution. 0 uses the whole budget.
        """

    @refine_time.setter
    def refine_time(self, arg: float, /) -> None: ...

    @property
    def planner(self) -> planners.GreedyRRTstar:
        """Parameters of the planner, a planners.GreedyRRTstar."""

    @planner.setter
    def planner(self, arg: planners.GreedyRRTstar, /) -> None: ...

    @property
    def collision_check_resolution(self) -> float:
        """
        Edge-check spacing of the planner and the smoother. 0 keeps OMPL's default for the planner and gives the smoother the spacing in smoothing, or one hundredth of the bounds' diagonal.
        """

    @collision_check_resolution.setter
    def collision_check_resolution(self, arg: float, /) -> None: ...

    @property
    def interp(self) -> str:
        """
        Geodesic interpolation strategy ('auto', 'base_geodesic', 'riemannian_geodesic').
        """

    @interp.setter
    def interp(self, arg: str, /) -> None: ...

    @property
    def goal_tolerance(self) -> float:
        """Accepted distance to the goal."""

    @goal_tolerance.setter
    def goal_tolerance(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int:
        """Seed of the plan. 0 takes a fresh seed from the space's own sampler."""

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def smooth(self) -> bool:
        """Run smooth_path on the planner's path."""

    @smooth.setter
    def smooth(self, arg: bool, /) -> None: ...

    @property
    def limits(self) -> tuple[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]] | None:
        """Physical (lower, upper) coordinate limits, or None."""

    @limits.setter
    def limits(self, arg: tuple[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]], /) -> None: ...

    @property
    def smoothing(self) -> PathSmoothingSettings:
        """PathSmoothingSettings forwarded to smooth_path."""

    @smoothing.setter
    def smoothing(self, arg: PathSmoothingSettings, /) -> None: ...

    def __repr__(self) -> str: ...

class PrecomputeMatrixLowerBoundResult:
    """Result of `precompute_matrix_lower_bound`."""

    @property
    def M_lower(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """Certified Loewner lower bound on M(q)."""

    @property
    def lambda_min_certificate(self) -> float:
        """Final worst-case lambda_min(L^-1 M(q) L^-T)."""

    @property
    def n_outer_iters(self) -> int:
        """Outer constraint-generation iterations executed."""

    @property
    def n_metric_evals(self) -> int:
        """Total M(q) evaluations across the precompute."""

    @property
    def converged(self) -> bool:
        """True when lambda_min_certificate >= 1 - tol."""

    @property
    def elapsed_ms(self) -> float:
        """Wall-clock duration of the precompute (ms)."""

    def __repr__(self) -> str: ...

class PrecomputeMatrixLowerBoundSettings:
    """
    Settings for `precompute_matrix_lower_bound`, which certifies a Loewner lower bound by constraint generation.
    """

    def __init__(self) -> None:
        """Create default precompute settings."""

    @property
    def max_outer(self) -> int:
        """Maximum outer constraint-generation iterations."""

    @max_outer.setter
    def max_outer(self, arg: int, /) -> None: ...

    @property
    def tol(self) -> float:
        """Stop when lambda_min >= 1 - tol over the configuration space."""

    @tol.setter
    def tol(self, arg: float, /) -> None: ...

    @property
    def n_starts_per_iter(self) -> int:
        """Multi-start seeds per outer iteration. 0 uses max(20, 10 * dim)."""

    @n_starts_per_iter.setter
    def n_starts_per_iter(self, arg: int, /) -> None: ...

    @property
    def max_iters_per_start(self) -> int:
        """Max gradient-descent iterations per start."""

    @max_iters_per_start.setter
    def max_iters_per_start(self, arg: int, /) -> None: ...

    @property
    def grad_tol(self) -> float:
        """Gradient-norm convergence for inner gradient descent."""

    @grad_tol.setter
    def grad_tol(self, arg: float, /) -> None: ...

    @property
    def fd_eps(self) -> float:
        """Finite-difference step for the lambda_min gradient."""

    @fd_eps.setter
    def fd_eps(self, arg: float, /) -> None: ...

    @property
    def seed(self) -> int:
        """RNG seed for multi-start initial points."""

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

class Product:
    """
    Riemannian product manifold M1 x M2 x ... x Mk.

    Compose a list of manifolds into their metric product. Points and tangents
    are the block-concatenation of the sub-manifold points/tangents; exp, log,
    and geodesic act block-wise and distance is sqrt(sum of squared block
    distances). Example: geodex.Product([geodex.Euclidean(3), geodex.SE2()]) is
    an R^3 x SE(2) mobile-manipulator configuration space.
    """

    def __init__(self, manifolds: list) -> None:
        """
        Create a product manifold from a list of manifolds.

        Args:
            manifolds: A non-empty list of geodex manifolds (Euclidean, SE2, SO3, ...).
        """

    def dim(self) -> int:
        """Total intrinsic dimension (sum of block dims)."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample a random product point."""

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Product inner product (sum of block inner products)."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Product norm."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Block-wise exponential map."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Block-wise logarithmic map."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Product geodesic distance sqrt(sum of squared block distances)."""

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Block-wise geodesic interpolation at t in [0, 1]."""

    def seed(self, seed: int) -> None:
        """Reseed the joint sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the joint sampler: 'scrambled', 'halton', or 'random'."""

    def __repr__(self) -> str: ...

class PseudoRandomSampler:
    """
    Pseudo-random sampler wrapping mt19937, the i.i.d. baseline.

    Pass a seed for a reproducible stream, or none to share a thread-local
    generator across default instances.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, seed: int) -> None: ...

    def sample(self, n: int) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample the next n-dimensional point in [0, 1)^n."""

    def seed(self, seed: int) -> None:
        """Reseed and switch to an owned generator."""

class PullbackMetric:
    """
    Pullback metric from task space to configuration space via the Jacobian.

    The inner product at q is <u, v>_q = u^T J(q)^T G(q) J(q) v + lambda * u^T v.
    """

    def __init__(self, jacobian_fn: Callable, task_metric_fn: Callable, regularization: float = 0.0) -> None:
        """
        Create a pullback metric.

        Args:
            jacobian_fn: Callable(q) -> np.ndarray returning the Jacobian matrix.
            task_metric_fn: Callable(q) -> np.ndarray returning the task-space SPD metric.
            regularization: Regularization parameter lambda (default 0).
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian inner product u^T J^T G J v + lambda * u^T v."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian norm."""

    def __repr__(self) -> str: ...

class SE2:
    """
    The special Euclidean group SE(2) = R^2 x SO(2).

    Poses are (x, y, theta) with theta in [-pi, pi).
    Uses a left-invariant metric with configurable weights.
    """

    def __init__(self, wx: float = 1.0, wy: float = 1.0, wtheta: float = 1.0, retraction: str = 'exponential', frame: str = 'body', x_lo: float = 0.0, x_hi: float = 10.0, y_lo: float = 0.0, y_hi: float = 10.0, sampler: str = 'scrambled') -> None:
        """
        Create an SE(2) manifold.

        Args:
            wx, wy, wtheta: Metric weights for (x, y, theta) components.
            retraction: 'exponential' or 'euler'. 'exponential' follows the screw
                motion of a constant twist. 'euler' moves in a straight line while
                turning at a constant rate, the geodesic of the metric when wx equals wy.
            frame: 'body' (left-invariant) or 'world' (right-invariant). Applies to the
                exponential retraction. The euler retraction ignores it.
            x_lo, x_hi, y_lo, y_hi: Workspace bounds for random sampling.
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

    def dim(self) -> int:
        """Return the intrinsic dimension (always 3)."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Sample a random pose in the workspace bounds."""

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> float:
        """Left-invariant inner product <u, v>_p."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> float:
        """Left-invariant norm ||v||_p."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Exponential map (or retraction) exp_p(v)."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Logarithmic map (or inverse retraction) log_p(q)."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> float:
        """
        Length of geodesic(p, q, .) under the metric. For the exponential retraction,
        it is the norm of the constant twist, which is at least the Riemannian distance.
        """

    def periods(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """
        Deck-group generators of the coordinate axes, (0, 0, 2*pi).

        Translation is unbounded and theta closes after a full turn. Pass this to
        heuristics.MatrixLowerBound to wrap the bound at the theta cut.
        """

    def coordinate_metric(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3, 3), order='F')]:
        """
        The metric on coordinate velocities (xdot, ydot, thetadot).

        inner() and norm() measure body-frame velocities. This metric is their frame
        pullback J(theta)^T M J(theta), and the two agree only at theta = 0.
        """

    def matrix_lower_bound(self) -> heuristics.MatrixLowerBound:
        """
        Certify a periods-aware Loewner lower bound for this metric.

        Runs constraint generation over the coordinate metric. The result is admissible
        for coordinate chords and wraps at theta. plan() uses it as the default
        heuristic on SE2.
        """

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """
        The curve exp_p(t log_p(q)) at t in [0, 1]. For the exponential retraction,
        it is the screw motion of the constant twist, not a geodesic of the metric.
        """

    def seed(self, seed: int) -> None:
        """Reseed the sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler to 'scrambled', 'halton' or 'random'."""

    def __repr__(self) -> str: ...

    @staticmethod
    def car_like(turning_radius: float, lateral_penalty: float = 100.0, retraction: str = 'exponential', frame: str = 'body', x_lo: float = 0.0, x_hi: float = 10.0, y_lo: float = 0.0, y_hi: float = 10.0, sampler: str = 'scrambled') -> SE2:
        """
        Create a car-like SE(2) manifold.

        Args:
            turning_radius: Effective minimum turning radius.
            lateral_penalty: Weight suppressing sideslip (default 100).
            retraction: 'exponential' or 'euler'.
            frame: 'body' (left-invariant) or 'world' (right-invariant).
            x_lo, x_hi, y_lo, y_hi: Workspace bounds for random sampling.
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

class SE2LeftInvariantMetric:
    """
    Left-invariant metric on SE(2) with the constant diagonal inner product
    <u, v> = wx ux vx + wy uy vy + wtheta utheta vtheta on the (x, y, theta) tangent.

    High wy suppresses lateral sliding (differential-drive or car-like behavior).
    Pass it as the base metric of a ClearanceMetric for obstacle-aware SE(2) planning.
    """

    def __init__(self, wx: float = 1.0, wy: float = 1.0, wtheta: float = 1.0) -> None:
        """Create a left-invariant metric with weights (wx, wy, wtheta)."""

    @staticmethod
    def car_like(turning_radius: float, lateral_penalty: float = 100.0) -> SE2LeftInvariantMetric:
        """
        Car-like weights with wtheta = turning_radius^2 and wy = lateral_penalty.
        The geodesic turning radius is about sqrt(wtheta / wx).
        """

    @staticmethod
    def holonomic(wtheta: float = 1.0) -> SE2LeftInvariantMetric:
        """Metric of a holonomic base, weights (1, 1, wtheta)."""

    @staticmethod
    def differential_drive(lateral_weight: float = 100.0, wtheta: float = 1.0) -> SE2LeftInvariantMetric:
        """
        Metric of a differential-drive base, weights (1, lateral_weight, wtheta).
        A large lateral weight makes sideways motion expensive.
        """

    def coordinate_lower_bound(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3, 3), order='F')]:
        """
        Constant matrix below the metric on (x, y, theta) velocities at every heading,
        diag(min(wx, wy), min(wx, wy), wtheta). heuristics.MatrixLowerBound and
        heuristics.product_lower_bound take it with SE2.periods().
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Left-invariant inner product."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Left-invariant norm."""

    @property
    def weights(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """The diagonal weight vector (wx, wy, wtheta)."""

    def __repr__(self) -> str: ...

class SE3:
    """
    The special Euclidean group SE(3) = R^3 x SO(3).

    A pose is a 7-vector [tx, ty, tz, qx, qy, qz, qw], a translation followed by a
    scalar-last unit quaternion. A tangent is a twist [v; omega] of shape (6,). exp,
    log and geodesic follow the screw motion of a constant twist, which is not a
    geodesic of the metric. The invariant metric has configurable translation and
    rotation weights. frame selects 'body' (left) or 'world' (right) invariance.
    """

    def __init__(self, frame: str = 'body', w_trans: float = 1.0, w_rot: float = 1.0, x_lo: float = 0.0, x_hi: float = 10.0, y_lo: float = 0.0, y_hi: float = 10.0, z_lo: float = 0.0, z_hi: float = 10.0, sampler: str = 'scrambled') -> None:
        """
        Create an SE(3) manifold.

        Args:
            frame: 'body' (left group exponential) or 'world' (right).
            w_trans: Metric weight on each translational twist component.
            w_rot: Metric weight on each rotational twist component.
            x_lo, x_hi, y_lo, y_hi, z_lo, z_hi: Translation sampling bounds.
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

    def dim(self) -> int:
        """Return the intrinsic dimension (always 6)."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')]:
        """
        Sample a random pose with a translation uniform in the box and a Haar-uniform rotation. Returns a 7-vector with a unit quaternion part.
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(6), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(6), order='C')]) -> float:
        """
        Invariant inner product <u, v>_p of two twists (6-vectors) at pose p (7-vector).
        """

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(6), order='C')]) -> float:
        """Invariant norm ||v||_p of a twist (6-vector) at pose p (7-vector)."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(6), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')]:
        """
        Exponential map exp_p(v), the screw motion of the twist v. p is (7,), v is (6,).
        """

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(6), order='C')]:
        """Logarithmic map (inverse retraction) log_p(q). Returns a twist (6,)."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')]) -> float:
        """
        Length of geodesic(p, q, .) under the metric. It is the norm of the constant
        twist, which is at least the Riemannian distance.
        """

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(7), order='C')]:
        """
        The screw motion exp_p(t log_p(q)) at t in [0, 1], not a geodesic of the
        metric. Returns a pose (7,).
        """

    def seed(self, seed: int) -> None:
        """Reseed the sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler to 'scrambled', 'halton' or 'random'."""

    def __repr__(self) -> str: ...

class SO2:
    """
    The special orthogonal group SO(2), the circle group S^1.

    A configuration is a single angle theta in [-pi, pi) with wraparound.
    Points and tangents are shape-(1,) arrays holding the angle and angular
    velocity respectively. Uses the canonical (bi-invariant) metric with a
    configurable weight.
    """

    def __init__(self, weight: float = 1.0, sampler: str = 'scrambled') -> None:
        """
        Create an SO(2) manifold.

        Args:
            weight: Positive rotational metric weight (norm scales as sqrt(weight)).
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

    def dim(self) -> int:
        """Return the intrinsic dimension (always 1)."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample a random angle uniformly in [-pi, pi) as a shape-(1,) array."""

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Canonical inner product <u, v>_p."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Canonical norm ||v||_p."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Exponential map exp_p(v) = wrap(p + v)."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Logarithmic map log_p(q) = wrap(q - p) (shortest arc)."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Geodesic distance d(p, q)."""

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Geodesic interpolation at parameter t in [0, 1]."""

    def periods(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """
        Period of the angle, (2*pi,). Pass this to heuristics.MatrixLowerBound to wrap
        the bound at the cut.
        """

    def coordinate_metric(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """The metric on the angle's velocity, a 1 by 1 matrix."""

    def matrix_lower_bound(self) -> heuristics.MatrixLowerBound:
        """
        Certify a periods-aware Loewner lower bound for this metric. plan() uses it as
        the default heuristic on SO2.
        """

    def seed(self, seed: int) -> None:
        """Reseed the sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler to 'scrambled', 'halton' or 'random'."""

    def __repr__(self) -> str: ...

class SO3:
    """
    Special orthogonal group SO(3).

    Points are unit quaternions [x,y,z,w] (shape (4,)); tangents are body angular velocities omega (shape (3,)).
    frame='body' (left-invariant) or 'world' (right-invariant).
    """

    def __init__(self, frame: str = 'body', weight: float = 1.0, sampler: str = 'scrambled') -> None:
        """
        Create an SO(3) manifold.

        Args:
            frame: 'body' (left-invariant) or 'world' (right-invariant).
            weight: Positive isotropic metric weight; norm scales as sqrt(weight).
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

    def dim(self) -> int:
        """Return the intrinsic dimension (always 3)."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]:
        """
        Sample a rotation uniformly (Haar measure) as a unit quaternion [x,y,z,w].
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> float:
        """Riemannian inner product <u, v>_p of body angular velocities."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> float:
        """Riemannian norm ||v||_p."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]:
        """Exponential map (or retraction) exp_p(v)."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Logarithmic map (or inverse retraction) log_p(q) (shortest arc)."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]) -> float:
        """
        Geodesic distance d(p, q) (rotation angle for the bi-invariant metric).
        """

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]:
        """Geodesic interpolation at parameter t in [0, 1] (quaternion SLERP)."""

    def seed(self, seed: int) -> None:
        """Reseed the sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler: 'scrambled', 'halton', or 'random'."""

    def __repr__(self) -> str: ...

class Scene:
    """
    In-memory collision-scene builder. Add primitive obstacles, then pass the scene to a planner.

    Example:
        Build a scene and plan a UR5 path through it::

            scene = geodex.Scene()
            scene.add_box(position=[0.5, 0.0, 0.3], size=[0.4, 0.6, 0.05])
            scene.add_sphere(center=[0.3, 0.2, 0.5], radius=0.1)
            result = geodex.plan(geodex.robots.UR5(), q0, q1, collision=scene)
    """

    def __init__(self) -> None: ...

    def add_box(self, position: Sequence[float], size: Sequence[float], orientation: Sequence[float] = [0.0, 0.0, 0.0, 1.0]) -> None:
        """
        Add an oriented box. size holds the full extents, and orientation is [qx, qy, qz, qw].
        """

    def add_sphere(self, center: Sequence[float], radius: float) -> None:
        """Add a sphere centered at center."""

    def add_cylinder(self, position: Sequence[float], radius: float, height: float, orientation: Sequence[float] = [0.0, 0.0, 0.0, 1.0]) -> None:
        """
        Add an oriented cylinder. Its local z axis is the cylinder axis, and orientation is [qx, qy, qz, qw].
        """

    def env(self) -> vamp.EnvHandle:
        """
        Build and return the VAMP environment handle for this scene.

        Attach a held object with geodex.vamp.attach_spheres(handle, spheres), then
        pass the handle as plan(..., collision=handle).
        """

    def __repr__(self) -> str: ...

class ScrambledHaltonSampler:
    """
    Scrambled Halton low-discrepancy sampler, the geodex default.

    Samples points in [0, 1)^n with even coverage and a random per-seed scramble.
    Pass a seed for a reproducible sequence, or none for a fresh scramble
    from the global source.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, seed: int) -> None: ...

    def sample(self, n: int) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample the next n-dimensional point in [0, 1)^n."""

    def seed(self, seed: int) -> None:
        """Reseed with a fresh scramble and start."""

class Sphere:
    """
    The 2-sphere S^2 with interchangeable retraction policy.

    Points are unit vectors in R^3. Tangent vectors lie in the
    orthogonal complement of the base point.
    """

    def __init__(self, retraction: str = 'exponential', sampler: str = 'scrambled') -> None:
        """
        Create a Sphere with the round metric.

        Args:
            retraction: 'exponential' (true exp/log) or 'projection', which normalizes
                p + v and agrees with the exponential map to second order.
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

    def dim(self) -> int:
        """Return the intrinsic dimension (always 2)."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Sample a uniformly random point on S^2."""

    def project(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Project an ambient vector onto the tangent space at p."""

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> float:
        """Riemannian inner product <u, v>_p."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> float:
        """Riemannian norm ||v||_p."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Exponential map (or retraction) exp_p(v)."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Logarithmic map (or inverse retraction) log_p(q)."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> float:
        """Geodesic distance d(p, q)."""

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Geodesic interpolation at parameter t in [0, 1]."""

    def seed(self, seed: int) -> None:
        """Reseed the sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler to 'scrambled', 'halton' or 'random'."""

    def __repr__(self) -> str: ...

class SphereN:
    """
    The n-sphere S^n with interchangeable retraction policy.

    Points are unit vectors in R^(n+1). The dimension n is set
    at construction time.
    """

    def __init__(self, dim: int, retraction: str = 'exponential', sampler: str = 'scrambled') -> None:
        """
        Create an n-sphere with the round metric.

        Args:
            dim: Intrinsic dimension n of S^n.
            retraction: 'exponential' (true exp/log) or 'projection', which normalizes
                p + v and agrees with the exponential map to second order.
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

    def dim(self) -> int:
        """Return the intrinsic dimension n."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample a uniformly random point on S^n."""

    def project(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Project an ambient vector onto the tangent space at p."""

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian inner product <u, v>_p."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Riemannian norm ||v||_p."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Exponential map (or retraction) exp_p(v)."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Logarithmic map (or inverse retraction) log_p(q)."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Geodesic distance d(p, q)."""

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Geodesic interpolation at parameter t in [0, 1]."""

    def seed(self, seed: int) -> None:
        """Reseed the sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler to 'scrambled', 'halton' or 'random'."""

    def __repr__(self) -> str: ...

class Torus:
    """
    Flat torus T^n with periodic angle coordinates in [0, 2*pi)^n.

    Exp wraps to [0, 2*pi), log wraps differences to [-pi, pi).
    """

    def __init__(self, dim: int, sampler: str = 'scrambled') -> None:
        """
        Create a flat torus of the given dimension.

        Args:
            dim: Dimension n.
            sampler: 'scrambled' (default), 'halton', or 'random'.
        """

    def dim(self) -> int:
        """Return the dimension."""

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample a uniformly random point in [0, 2*pi)^n."""

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Flat inner product <u, v> = u . v."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Flat norm ||v||."""

    def exp(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Exponential map wrap(p + v) into [0, 2*pi)^n."""

    def log(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Logarithmic map, the shortest-path tangent in [-pi, pi)^n."""

    def distance(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Geodesic distance on the flat torus."""

    def geodesic(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], t: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Geodesic interpolation at parameter t in [0, 1]."""

    def periods(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """
        Period of every angle, 2*pi each. Pass this to heuristics.MatrixLowerBound to
        wrap the bound at the cuts.
        """

    def coordinate_metric(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """The metric on angle velocities, an n by n matrix."""

    def matrix_lower_bound(self) -> heuristics.MatrixLowerBound:
        """
        Certify a periods-aware Loewner lower bound for this metric. plan() uses it as
        the default heuristic on Torus.
        """

    def seed(self, seed: int) -> None:
        """Reseed the sampler for reproducible sampling."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler to 'scrambled', 'halton' or 'random'."""

    def __repr__(self) -> str: ...

class WeightedMetric:
    """
    Uniformly scaled metric wrapper.

    The inner product is <u, v>_q = alpha * <u, v>^base_q.
    """

    def __init__(self, base_metric: object, alpha: float) -> None:
        """
        Create a weighted metric.

        Args:
            base_metric: Any geodex metric to scale.
            alpha: Scaling factor (must be positive).
        """

    def inner(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], u: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Scaled Riemannian inner product alpha * <u, v>^base_p."""

    def norm(self, p: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], v: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Scaled Riemannian norm."""

    @property
    def alpha(self) -> float:
        """The scaling factor."""

    def __repr__(self) -> str: ...

def discrete_geodesic(manifold: object, start: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], goal: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], settings: InterpolationSettings = ...) -> InterpolationResult:
    """
    Walk from start toward goal by Riemannian natural gradient descent.

    Each iteration first tries the Riemannian logarithm direction, using the identity
    grad((1/2) d^2) = -log inside the injectivity radius, and checks the progress.
    When the check fails, that step uses a central finite-difference natural gradient
    of the manifold's inner product. The iteration count and the path size scale as
    initial_distance / settings.step_size.

    Args:
        manifold: Any geodex manifold (Sphere, Euclidean, Torus, SE2, ConfigurationSpace).
        start: Starting point (np.ndarray).
        goal: Target point (np.ndarray).
        settings: InterpolationSettings (optional).
    Returns:
        InterpolationResult with fields path, status, iterations, distortion_halvings,
        fd_midpoint_fallbacks, initial_distance, final_distance.
    """

def distance_midpoint(manifold: object, a: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], b: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
    """
    Approximate geodesic distance between two points using the midpoint method.

    The third-order approximation is d(a,b) ≈ ||log_m(b) - log_m(a)||_m with the
    geodesic midpoint m = exp_a(0.5 * log_a(b)).

    Args:
        manifold: Any geodex manifold (Sphere, Euclidean, Torus, SE2, ConfigurationSpace).
        a: First point on the manifold.
        b: Second point on the manifold.
    Returns:
        Approximate geodesic distance (float).
    """

def load_scene(path: str) -> vamp.EnvHandle:
    """
    Load an MBM-style scene YAML into an opaque VAMP environment handle.

    Supports primitive collision objects (boxes, cylinders, spheres) and mesh
    objects (axis-aligned bounding-box approximation).
    """

def log_level() -> LogLevel:
    """The level plan() runs its planners at."""

def plan(space: object, start: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], goal: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], collision: object | None = None, *, is_valid: object | None = None, settings: object | None = None, heuristic: object | None = None, motion_validator: object | None = None) -> PlanResult:
    """
    Plan a collision-free path from start to goal on a manifold.

    Args:
        space: Any geodex manifold (Sphere, Euclidean, Torus, SE2, SO2, SO3, SE3,
            ConfigurationSpace, or Product). A ConfigurationSpace over SE2 with a
            ClearanceMetric of an SE2LeftInvariantMetric and a geodex.collision SDF plans
            in C++ without calling into Python.
        start: Start point (np.ndarray).
        goal: Goal point (np.ndarray).
        collision: Optional callable q -> bool that returns True when q is
            collision-free, or a geodex.vamp Scene or EnvHandle for a robot space (SIMD
            state and edge validity). Omit it for free-space planning. With a Scene, the
            planner and the smoother check the obstacles grown by the sphere travel over
            the corner tolerance. The smoother checks its edges at steps along which no
            sphere moves more than settings.collision_check_resolution (0.005 m by default),
            and it spaces each edge by the edge's own sphere travel. The plan replaces
            settings.smoothing.collision_check_resolution and smoothing.edge_travel.
            A sphere can come closer to an obstacle between two checks than at the checks.
            Self-collision is tested only at the sampled states. The is_valid method of a
            geodex.collision.FootprintGridChecker or a geodex.vamp.CollisionChecker runs
            in C++ without calling into Python.
        is_valid: Keyword-only alias for collision. It takes precedence when both are
            given.
        settings: Optional PlanSettings (defaults when omitted). For a robot whose
            base metric makes sideways motion cost more than driving, a
            settings.smoothing.sharp_coordinates of 0 becomes 3, and the base pose may keep
            a corner while the smoother rounds the arm's joints. A value of at least the
            number of coordinates rounds every coordinate together.
        heuristic: Admissible heuristic of the informed planner, a geodex.heuristics
            instance (Zero, Euclidean, EigenvalueLowerBound or MatrixLowerBound). None
            selects the default. Robots use their precomputed Loewner lower bound, SE2, SO2
            and Torus use a Loewner bound computed at setup and wrapped at the cut, and
            other spaces use the Euclidean chord distance. Pass Zero for a custom-metric
            ConfigurationSpace, where the Euclidean bound is inadmissible.
        motion_validator: Optional geodex.DirectionalMotionValidator restricting tree
            edges to forward drivability on SE(2)-like spaces.
    Returns:
        PlanResult with the path, its cost and timings, and sampling counters.

    Examples:
        Plan a Panda path in a table-pick scene, then on an abstract manifold::

            import numpy as np
            import geodex

            robot = geodex.robots.Panda()
            scene = geodex.load_scene("table_pick.scene.yaml")
            result = geodex.plan(robot, start, goal, collision=scene)
            print(result.solved, result.path.shape, result.cost)

            S2 = geodex.Sphere()
            result = geodex.plan(S2, p0, p1, is_valid=lambda q: q[2] < 0.8)
    """

def precompute_matrix_lower_bound(metric_fn: Callable[[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]], Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]], lo: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], hi: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], settings: PrecomputeMatrixLowerBoundSettings = ...) -> PrecomputeMatrixLowerBoundResult:
    """
    Compute a constant SPD Loewner lower bound on M(q) by constraint generation.

    Args:
        metric_fn: Callable(q) -> np.ndarray returning the SPD metric tensor M(q).
        lo: Per-dimension lower bounds on the configuration space (np.ndarray, shape (d,)).
        hi: Per-dimension upper bounds on the configuration space (np.ndarray, shape (d,)).
        settings: PrecomputeMatrixLowerBoundSettings (optional).
    Returns:
        PrecomputeMatrixLowerBoundResult with the certified bound and diagnostics.
    """

def seed(seed: int) -> None:
    """
    Reseed the global source of the default samplers, making sampling reproducible for manifolds constructed afterward.
    """

def set_log_level(level: LogLevel) -> None:
    """
    Set how much plan() lets its planners print. The default, LogLevel.Warn, prints
    warnings and errors only. At LogLevel.Info, a plan that runs its planner prints its
    first-solution, refinement and smoothing times. The environment variable
    GEODEX_LOG_LEVEL (debug, info, warn, error or off, in any case) sets the starting
    level. OMPL's own level is restored after each plan.
    """

def smooth_path(manifold: object, validity_fn: Callable, path: Sequence[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]], settings: PathSmoothingSettings = ...) -> PathSmoothingResult:
    """
    Shorten and smooth a valid path under the manifold's metric.

    Alternates randomized shortcutting with local energy descent, rounds the corners
    into C2 curves when settings.round_corners is on, checks the result and spaces its
    waypoints evenly. collision_free is True exactly when every waypoint of the smoothed
    path passes validity_fn and every edge, interpolated along the manifold's geodesic,
    passes the edge test at settings.collision_check_resolution. Next to a rounding curve,
    a part of an edge of the shortened path passes at the samples of the whole edge, and an
    edge_validator tests the whole edge. The evenly spaced waypoints lie on that path, and
    the edges between them stay within corner_tolerance of it. When the smoothed path
    fails, smooth_path returns an earlier stage, down to the input itself.
    The check holds at the stated resolution only, and a path may touch an obstacle
    between two samples.

    Args:
        manifold: Any geodex manifold.
        validity_fn: Callable(q) -> bool, True when q is valid.
        path: Input path as a list of waypoints, typically a planner's output.
        settings: PathSmoothingSettings (optional).
    Returns:
        PathSmoothingResult with path, waypoints, length, collision_free,
        first_invalid_index and profile.
    """

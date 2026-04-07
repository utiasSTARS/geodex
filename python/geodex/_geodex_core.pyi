"""Type stubs for geodex._geodex_core."""

import enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_Point = NDArray[np.floating]
_Tangent = NDArray[np.floating]
_Matrix = NDArray[np.floating]

_AnyManifold = "Sphere | Euclidean | Torus | SE2 | ConfigurationSpace"
_AnyMetric = (
    "KineticEnergyMetric | JacobiMetric | PullbackMetric | ConstantSPDMetric | WeightedMetric"
)

# ---------------------------------------------------------------------------
# Manifolds
# ---------------------------------------------------------------------------

class Sphere:
    """The 2-sphere S² with interchangeable retraction policy.

    Points are unit vectors in R³. Tangent vectors lie in the orthogonal
    complement of the base point.
    """

    def __init__(self, retraction: str = "exponential") -> None:
        """Create a Sphere with the round metric.

        Args:
            retraction: ``'exponential'`` (true exp/log) or
                ``'projection'`` (first-order normalization).
        """
        ...

    def dim(self) -> int:
        """Return the intrinsic dimension (always 2)."""
        ...

    def random_point(self) -> _Point:
        """Sample a uniformly random point on S²."""
        ...

    def project(self, p: _Point, v: _Tangent) -> _Tangent:
        """Project an ambient vector onto the tangent space at *p*."""
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Riemannian inner product ⟨u, v⟩ₚ."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Riemannian norm ‖v‖ₚ."""
        ...

    def exp(self, p: _Point, v: _Tangent) -> _Point:
        """Exponential map (or retraction) expₚ(v)."""
        ...

    def log(self, p: _Point, q: _Point) -> _Tangent:
        """Logarithmic map (or inverse retraction) logₚ(q)."""
        ...

    def distance(self, p: _Point, q: _Point) -> float:
        """Geodesic distance d(p, q)."""
        ...

    def geodesic(self, p: _Point, q: _Point, t: float) -> _Point:
        """Geodesic interpolation at parameter *t* ∈ [0, 1]."""
        ...


class Euclidean:
    """Euclidean space Rⁿ with the standard inner product."""

    def __init__(self, dim: int) -> None:
        """Create Euclidean space of dimension *dim*."""
        ...

    def dim(self) -> int:
        """Return the dimension."""
        ...

    def random_point(self) -> _Point:
        """Sample from the standard normal distribution."""
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Standard dot product u · v."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Euclidean norm ‖v‖₂."""
        ...

    def exp(self, p: _Point, v: _Tangent) -> _Point:
        """Exponential map: p + v."""
        ...

    def log(self, p: _Point, q: _Point) -> _Tangent:
        """Logarithmic map: q - p."""
        ...

    def distance(self, p: _Point, q: _Point) -> float:
        """Euclidean distance ‖q - p‖₂."""
        ...

    def geodesic(self, p: _Point, q: _Point, t: float) -> _Point:
        """Linear interpolation (1 - t)*p + t*q."""
        ...


class Torus:
    """The flat n-torus Tⁿ with coordinates in [0, 2π)ⁿ."""

    def __init__(self, dim: int) -> None:
        """Create the flat n-torus of dimension *dim*."""
        ...

    def dim(self) -> int:
        """Return the dimension."""
        ...

    def random_point(self) -> _Point:
        """Sample uniformly from [0, 2π)ⁿ."""
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Flat inner product u · v."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Flat norm ‖v‖₂."""
        ...

    def exp(self, p: _Point, v: _Tangent) -> _Point:
        """Exponential map: (p + v) mod 2π."""
        ...

    def log(self, p: _Point, q: _Point) -> _Tangent:
        """Logarithmic map: shortest signed difference, wrapped to (−π, π]ⁿ."""
        ...

    def distance(self, p: _Point, q: _Point) -> float:
        """Flat torus distance (shortest path respecting wrapping)."""
        ...

    def geodesic(self, p: _Point, q: _Point, t: float) -> _Point:
        """Geodesic interpolation at parameter *t* ∈ [0, 1]."""
        ...


class SE2:
    """The special Euclidean group SE(2) = R² ⋊ SO(2).

    Poses are represented as (x, y, θ) with θ ∈ [−π, π).
    Uses a left-invariant metric with configurable diagonal weights.
    """

    def __init__(
        self,
        wx: float = 1.0,
        wy: float = 1.0,
        wtheta: float = 1.0,
        retraction: str = "exponential",
        x_lo: float = 0.0,
        x_hi: float = 10.0,
        y_lo: float = 0.0,
        y_hi: float = 10.0,
    ) -> None:
        """Create an SE(2) manifold.

        Args:
            wx: Weight for the x-translation component.
            wy: Weight for the y-translation component.
            wtheta: Weight for the rotation component.
            retraction: ``'exponential'`` (Lie group exp) or ``'euler'``
                (component-wise, 1st-order).
            x_lo, x_hi: x-axis workspace bounds for random sampling.
            y_lo, y_hi: y-axis workspace bounds for random sampling.
        """
        ...

    def dim(self) -> int:
        """Return the intrinsic dimension (always 3)."""
        ...

    def random_point(self) -> _Point:
        """Sample a random pose (x, y, θ) within the workspace bounds."""
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Left-invariant inner product ⟨u, v⟩ₚ = wₓuₓvₓ + w_y u_y v_y + w_θ u_θ v_θ."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Left-invariant norm ‖v‖ₚ."""
        ...

    def exp(self, p: _Point, v: _Tangent) -> _Point:
        """Exponential map (or retraction) expₚ(v)."""
        ...

    def log(self, p: _Point, q: _Point) -> _Tangent:
        """Logarithmic map (or inverse retraction) logₚ(q)."""
        ...

    def distance(self, p: _Point, q: _Point) -> float:
        """Left-invariant geodesic distance d(p, q)."""
        ...

    def geodesic(self, p: _Point, q: _Point, t: float) -> _Point:
        """Geodesic interpolation at parameter *t* ∈ [0, 1]."""
        ...


class ConfigurationSpace:
    """A configuration space combining a base manifold's topology with a custom metric.

    Topology operations (``exp``, ``log``, ``dim``, ``random_point``) come from the
    base manifold. Geometry operations (``inner``, ``norm``, ``distance``) come from
    the custom metric.
    """

    def __init__(
        self,
        base_manifold: "Sphere | Euclidean | Torus | SE2 | ConfigurationSpace",
        metric: "KineticEnergyMetric | JacobiMetric | PullbackMetric | ConstantSPDMetric | WeightedMetric",
    ) -> None:
        """Create a configuration space.

        Args:
            base_manifold: Base manifold supplying topology (exp/log/dim/random_point).
            metric: Custom metric supplying geometry (inner/norm/distance).
        """
        ...

    def dim(self) -> int:
        """Return the intrinsic dimension (from the base manifold)."""
        ...

    def random_point(self) -> _Point:
        """Sample a random point from the base manifold."""
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Riemannian inner product from the custom metric."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Riemannian norm from the custom metric."""
        ...

    def exp(self, p: _Point, v: _Tangent) -> _Point:
        """Exponential map from the base manifold."""
        ...

    def log(self, p: _Point, q: _Point) -> _Tangent:
        """Logarithmic map from the base manifold."""
        ...

    def distance(self, p: _Point, q: _Point) -> float:
        """Geodesic distance using the midpoint approximation with the custom metric."""
        ...

    def geodesic(self, p: _Point, q: _Point, t: float) -> _Point:
        """Geodesic interpolation at parameter *t* ∈ [0, 1]."""
        ...


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class KineticEnergyMetric:
    """Kinetic energy metric g(q) = M(q).

    The inner product at q is ⟨u, v⟩_q = uᵀ M(q) v, where M(q) is a
    symmetric positive-definite mass matrix returned by the callable.
    """

    def __init__(
        self,
        mass_matrix_fn: Callable[[_Point], _Matrix],
    ) -> None:
        """Create a kinetic energy metric.

        Args:
            mass_matrix_fn: Callable ``q → M`` returning an SPD matrix.
        """
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Riemannian inner product uᵀ M(p) v."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Riemannian norm √(vᵀ M(p) v)."""
        ...


class JacobiMetric:
    """Jacobi metric for natural trajectories of a conservative mechanical system.

    The inner product at q is ⟨u, v⟩_q = 2(H − P(q)) uᵀ M(q) v, where H
    is the total energy and P(q) is the potential energy.
    """

    def __init__(
        self,
        mass_matrix_fn: Callable[[_Point], _Matrix],
        potential_fn: Callable[[_Point], float],
        total_energy: float,
    ) -> None:
        """Create a Jacobi metric.

        Args:
            mass_matrix_fn: Callable ``q → M`` returning an SPD mass matrix.
            potential_fn: Callable ``q → P`` returning the potential energy.
            total_energy: Total energy H (must satisfy H > P(q) everywhere on the path).
        """
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Riemannian inner product 2(H − P(p)) uᵀ M(p) v."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Riemannian norm."""
        ...


class PullbackMetric:
    """Pullback metric from task space to configuration space via the Jacobian.

    The inner product at q is ⟨u, v⟩_q = uᵀ J(q)ᵀ G(q) J(q) v + λ uᵀ v.
    """

    def __init__(
        self,
        jacobian_fn: Callable[[_Point], _Matrix],
        task_metric_fn: Callable[[_Point], _Matrix],
        regularization: float = 0.0,
    ) -> None:
        """Create a pullback metric.

        Args:
            jacobian_fn: Callable ``q → J`` returning the Jacobian matrix.
            task_metric_fn: Callable ``q → G`` returning the task-space SPD metric.
            regularization: Tikhonov regularization λ (default 0).
        """
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Riemannian inner product uᵀ JᵀGJ v + λ uᵀ v."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Riemannian norm."""
        ...


class ConstantSPDMetric:
    """Point-independent Riemannian metric defined by a constant SPD matrix.

    The inner product is ⟨u, v⟩ = uᵀ A v for a fixed SPD matrix A.
    """

    def __init__(self, matrix: _Matrix) -> None:
        """Create a constant SPD metric.

        Args:
            matrix: Symmetric positive-definite weight matrix A.
        """
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Riemannian inner product uᵀ A v."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Riemannian norm √(vᵀ A v)."""
        ...


class WeightedMetric:
    """Uniformly scaled metric wrapper.

    The inner product is ⟨u, v⟩_q = α · ⟨u, v⟩^base_q.
    """

    def __init__(
        self,
        base_metric: "KineticEnergyMetric | JacobiMetric | PullbackMetric | ConstantSPDMetric | WeightedMetric",
        alpha: float,
    ) -> None:
        """Create a weighted metric.

        Args:
            base_metric: Any geodex metric to scale.
            alpha: Positive scaling factor.
        """
        ...

    @property
    def alpha(self) -> float:
        """The scaling factor (read-only)."""
        ...

    def inner(self, p: _Point, u: _Tangent, v: _Tangent) -> float:
        """Scaled inner product α · ⟨u, v⟩^base_p."""
        ...

    def norm(self, p: _Point, v: _Tangent) -> float:
        """Scaled norm √α · ‖v‖^base_p."""
        ...


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

class InterpolationStatus(enum.Enum):
    """Termination status for the discrete geodesic walk."""

    Converged = 0
    """Distance to target fell below convergence tolerance."""
    MaxStepsReached = 1
    """Iteration budget exhausted without reaching tolerance."""
    GradientVanished = 2
    """Riemannian gradient norm is ~0 at a non-target point."""
    CutLocus = 3
    """``log`` collapsed to ~0 while start and target are distinct (e.g. antipodal)."""
    StepShrunkToZero = 4
    """Distortion halving drove the step size below ``min_step_size``."""
    DegenerateInput = 5
    """``start == target`` on entry; returned a single-point path."""


class InterpolationSettings:
    """Settings for the discrete geodesic walk.

    Walk semantics: each iteration takes a Riemannian step of length
    ``min(step_size, remaining_distance)`` in the descent direction. Iteration
    count and returned-path size therefore scale as
    ``~initial_distance / step_size``, so ``step_size`` also serves as the
    effective path resolution (max Riemannian distance between consecutive
    returned points).
    """

    step_size: float
    """Max Riemannian step per iteration; also the effective path resolution."""
    convergence_tol: float
    """Absolute stop threshold on ``|log(current, target)|_R``."""
    convergence_rel: float
    """Relative stop threshold (``distance < rel * initial_distance``)."""
    max_steps: int
    """Maximum number of successful gradient-descent steps."""
    fd_epsilon: float
    """Central FD step for the fallback gradient; 0 means auto-select."""
    distortion_ratio: float
    """Progress-check tolerance; 1.5 requires ≥50% of intended-step progress."""
    growth_factor: float
    """Factor by which the step cap grows back after each successful iteration."""
    min_step_size: float
    """Failure threshold after repeated distortion halvings."""
    gradient_eps: float
    """Gradient Riemannian-norm threshold for ``GradientVanished``."""
    cut_locus_eps: float
    """``|log|_R`` threshold flagging ``CutLocus``."""

    def __init__(
        self,
        step_size: float = 0.5,
        convergence_tol: float = 1e-4,
        convergence_rel: float = 1e-3,
        max_steps: int = 100,
        fd_epsilon: float = 0.0,
        distortion_ratio: float = 1.5,
        growth_factor: float = 1.5,
        min_step_size: float = 1e-12,
        gradient_eps: float = 1e-12,
        cut_locus_eps: float = 1e-10,
    ) -> None:
        """Create interpolation settings.

        Args:
            step_size: Max Riemannian step per iteration (also effective path resolution).
            convergence_tol: Absolute stop threshold on ``|log(current, target)|_R``.
            convergence_rel: Relative stop threshold (``distance < rel * initial_distance``).
            max_steps: Maximum number of successful gradient-descent steps.
            fd_epsilon: Central FD step for the fallback gradient; 0 means auto-select.
            distortion_ratio: Progress-check tolerance (1.5 requires ≥50% progress).
            growth_factor: Factor by which the step cap grows after a successful step.
            min_step_size: Failure threshold after repeated distortion halvings.
            gradient_eps: Gradient norm threshold for ``GradientVanished``.
            cut_locus_eps: ``|log|_R`` threshold flagging ``CutLocus``.
        """
        ...


class InterpolationResult:
    """Output of :func:`discrete_geodesic`.

    Carries the discretised path, a termination :class:`InterpolationStatus`,
    iteration count, and the initial/final Riemannian distances to target.
    """

    @property
    def path(self) -> list[_Point]:
        """Sequence of points traced from start toward target (always starts with ``start``)."""
        ...

    @property
    def status(self) -> InterpolationStatus:
        """Termination reason — always check before using ``path``."""
        ...

    @property
    def iterations(self) -> int:
        """Number of successful gradient steps taken (distortion retries do not count)."""
        ...

    @property
    def distortion_halvings(self) -> int:
        """Number of times the step cap was halved due to progress failure."""
        ...

    @property
    def initial_distance(self) -> float:
        """Riemannian distance from ``start`` to ``target`` at entry."""
        ...

    @property
    def final_distance(self) -> float:
        """Riemannian distance from the final iterate to ``target`` at exit."""
        ...


def distance_midpoint(
    manifold: "Sphere | Euclidean | Torus | SE2 | ConfigurationSpace",
    a: _Point,
    b: _Point,
) -> float:
    """Approximate geodesic distance between two points using the midpoint method.

    Computes a third-order approximation:
    d(a, b) ≈ ‖log_m(b) − log_m(a)‖_m  where m = expₐ(½ logₐ(b)).

    Args:
        manifold: Any geodex manifold.
        a: First point on the manifold.
        b: Second point on the manifold.

    Returns:
        Approximate geodesic distance.
    """
    ...


def discrete_geodesic(
    manifold: "Sphere | Euclidean | Torus | SE2 | ConfigurationSpace",
    start: _Point,
    goal: _Point,
    settings: InterpolationSettings = ...,
) -> InterpolationResult:
    """Walk from *start* toward *goal* via Riemannian natural gradient descent.

    Each iteration first tries the Riemannian logarithm direction (exploiting
    the identity ``grad((1/2) d²) = -log`` at points inside the injectivity
    radius) and verifies via a progress check. When the check fails (e.g., the
    retraction is not a true exponential map or the metric does not match the
    retraction), the algorithm falls back **for that step only** to a central
    finite-difference natural gradient computed from the manifold's ``inner``
    product.

    Walk semantics: iteration count and path size both scale as
    ``~initial_distance / settings.step_size``; reduce ``step_size`` for
    higher path resolution.

    Args:
        manifold: Any geodex manifold (Sphere, Euclidean, Torus, SE2, ConfigurationSpace).
        start: Starting point.
        goal: Target point.
        settings: Algorithm settings (uses defaults if omitted).

    Returns:
        InterpolationResult with fields ``path``, ``status``, ``iterations``,
        ``distortion_halvings``, ``initial_distance``, ``final_distance``.
    """
    ...


class EuclideanHeuristic:
    """Euclidean (L2) heuristic between coordinate vectors.

    Computes the chord distance ‖a − b‖₂. Admissible for any manifold
    where geodesic distance ≥ chord distance.
    """

    def __init__(self) -> None:
        """Create a Euclidean heuristic."""
        ...

    def __call__(self, a: _Point, b: _Point) -> float:
        """Compute ‖a − b‖₂."""
        ...

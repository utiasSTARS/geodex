from collections.abc import Sequence
from typing import Annotated, overload

from numpy.typing import ArrayLike


class EigenvalueLowerBound:
    """
    Eigenvalue lower-bound heuristic for configuration-dependent metrics.

    For a Riemannian metric M(q), the geodesic distance satisfies

        d_M(a, b) >= sqrt(lambda_min) * ||a - b||_2,

    where lambda_min is a global lower bound on the eigenvalues of M(q).
    It is tighter than `Zero` and looser than `MatrixLowerBound`.
    """

    def __init__(self, lambda_min: float) -> None:
        """Construct from the global minimum eigenvalue lambda_min of M(q)."""

    def __call__(self, a: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], b: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Compute sqrt(lambda_min) * ||a - b||_2."""

    @property
    def sqrt_lambda_min(self) -> float:
        """Cached sqrt(lambda_min)."""

class Euclidean:
    """
    Euclidean (L2) chord-distance heuristic.

    Computes ||a - b||_2. It is admissible when the chord distance bounds the geodesic
    distance from below, for example when lambda_min(M(q)) >= 1 everywhere. It
    overestimates when lambda_min < 1 in some direction.
    """

    def __init__(self) -> None:
        """Create a Euclidean heuristic."""

    def __call__(self, a: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], b: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Compute ||a - b||_2."""

class MatrixLowerBound:
    """
    Matrix lower-bound heuristic via a constant SPD Loewner lower bound.

    For a metric M(q) with M(q) >= M_lower in the Loewner order, the geodesic
    distance satisfies

        d_M(a, b) >= sqrt((a - b)^T M_lower (a - b)).

    It keeps directional information and is tighter than the scalar eigenvalue
    bound. The heuristic caches the Cholesky factor L and evaluates
    ||L^T (a - b)||_2. An optional eigenvalue floor lambda_min makes it dominate
    the scalar bound in every direction.

    On a flat quotient (SE2, SO2, Torus), pass the manifold's periods. The
    difference then wraps to its nearest representative first. Without periods,
    the bound overshoots near a branch cut and is inadmissible there.
    """

    @overload
    def __init__(self, M_lower: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]) -> None:
        """Construct from an SPD matrix M_lower satisfying M(q) >= M_lower."""

    @overload
    def __init__(self, M_lower: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')], lambda_min: float) -> None:
        """
        Construct from an SPD matrix M_lower with an eigenvalue floor lambda_min.
        """

    @overload
    def __init__(self, M_lower: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')], periods: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> None:
        """
        Construct from an SPD matrix M_lower and per-axis coordinate periods.

        periods holds one entry per coordinate, 0 where the axis is not periodic
        (typically manifold.periods()). Raises ValueError on a size mismatch, or
        when a periodic axis is coupled to another axis in M_lower.
        """

    def __call__(self, a: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], b: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """
        Compute the admissible lower bound on geodesic distance. Raises ValueError
        unless a and b have the size of M_lower.
        """

    def update(self, M_new: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]) -> bool:
        """
        Incremental Loewner-meet update with a new SPD observation.

        Returns True if the update loosened the bound, and False if the current M_lower
        already dominates the new observation.
        """

    def matrix(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """Reconstruct the current M_lower from its Cholesky factor."""

    def det(self) -> float:
        """Determinant of the current M_lower."""

    def eigenvalues(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Eigenvalues of the current M_lower in ascending order."""

    @property
    def has_eigenvalue_floor(self) -> bool:
        """Whether an eigenvalue floor is set."""

    @property
    def periods(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Per-axis coordinate periods, empty when no axis is periodic."""

class Zero:
    """
    Zero heuristic, h(a, b) = 0 for every pair.

    The weakest admissible heuristic. It is admissible for any non-negative distance
    and does not add information. With an informed planner, the informed set is the full
    configuration space, sampling stays uniform and the planner does not prune any
    vertex.
    """

    def __init__(self) -> None:
        """Create a Zero heuristic."""

    def __call__(self, a: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], b: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> float:
        """Compute h(a, b) = 0."""

def product_lower_bound(factors: Sequence[Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')] | tuple[Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')], Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')] | None]]) -> MatrixLowerBound:
    """
    Matrix lower-bound heuristic of a product metric from the bounds of its factors.

    The metric of a product manifold is the direct sum of the factor metrics. If M_i
    bounds factor i from below, the block-diagonal matrix of the M_i bounds the product
    metric from below. The heuristic takes that matrix and the factors' periods in the
    same order.

    Args:
        factors: One entry per factor, in the order of the product's coordinates. An
            entry is the factor's bound matrix, or a tuple (matrix, periods) for a factor
            with periodic coordinates, such as (metric.coordinate_lower_bound(),
            se2.periods()).

    Returns:
        A MatrixLowerBound of the block-diagonal bound, with periods when a factor has
        them.

    Raises:
        ValueError: factors is empty, a matrix is not square, or a factor's periods do
            not match its matrix.
    """

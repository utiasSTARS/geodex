

class GreedyRRTstar:
    """Asymptotically optimal informed planner (G-RRT*) and its parameters."""

    def __init__(self, range: float = 0.0, greedy_ratio: float = 0.9, rewire_factor: float = 1.1, greedy_cost_for_tree_pruning: bool = True, max_neighbors: int = 0) -> None:
        """
        Create GreedyRRTstar settings.

        Args:
            range: Step size. 0 selects OMPL's automatic value.
            greedy_ratio: Fraction of samples from the greedy ellipsoid.
            rewire_factor: Rewiring radius scale.
            greedy_cost_for_tree_pruning: Prune to the greedy set when greedy_ratio > 0.
            max_neighbors: Cap on the k-nearest neighborhood. 0 leaves it unbounded.
        """

    @property
    def range(self) -> float:
        """Step size. 0 selects OMPL's automatic value."""

    @range.setter
    def range(self, arg: float, /) -> None: ...

    @property
    def greedy_ratio(self) -> float:
        """Fraction of samples from the greedy ellipsoid."""

    @greedy_ratio.setter
    def greedy_ratio(self, arg: float, /) -> None: ...

    @property
    def rewire_factor(self) -> float:
        """Rewiring radius scale."""

    @rewire_factor.setter
    def rewire_factor(self, arg: float, /) -> None: ...

    @property
    def greedy_cost_for_tree_pruning(self) -> bool:
        """Prune to the greedy set when greedy_ratio > 0."""

    @greedy_cost_for_tree_pruning.setter
    def greedy_cost_for_tree_pruning(self, arg: bool, /) -> None: ...

    @property
    def max_neighbors(self) -> int:
        """Cap on the k-nearest neighborhood. 0 leaves it unbounded."""

    @max_neighbors.setter
    def max_neighbors(self, arg: int, /) -> None: ...

    def __repr__(self) -> str: ...

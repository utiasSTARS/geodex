from collections.abc import Sequence
from typing import Annotated

from numpy.typing import ArrayLike

import geodex._geodex_core


class Baxter(RobotModel):
    """
    Rethink Baxter, 14-DoF dual arm.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (the robot's mass matrix
            M(q), default) or 'euclidean'.
    """

    def __init__(self, metric: str = 'kinetic_energy') -> None: ...

class Fr3Gripper(RobotModel):
    """
    Franka FR3 with a Robotiq 2F-85 on the flange, 7-DoF arm. Its VAMP model is 'fr3_arm_gripper'.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (the robot's mass matrix
            M(q), default) or 'euclidean'.
    """

    def __init__(self, metric: str = 'kinetic_energy') -> None: ...

class HuskyUR5e(RobotModel):
    """
    Clearpath Husky with a Universal Robots UR5e on its default top plate, skid-steer
    differential-drive base, 9 coordinates (x, y, theta, six arm joints).

    The configuration is (x, y, theta, arm joints...) on SE(2) x R^n. Only the metric
    changes between a holonomic and a differential-drive base.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (default) or 'euclidean'.
        base: 'holonomic' or 'differential_drive' (left-invariant SE(2) metric with
            lateral weight 100). None matches the robot's drive.
        base_weights: Body weights (wx, wy, wtheta) of the SE(2) metric. They override
            base.
        workspace: ((x_lo, x_hi), (y_lo, y_hi)), the region the base samples.
    """

    def __init__(self, metric: str = 'kinetic_energy', base: str | None = None, base_weights: Sequence[float] | None = None, workspace: Sequence[Sequence[float]] = [[-5.0, 5.0], [-5.0, 5.0]]) -> None: ...

class PR2(RobotModel):
    """
    Willow Garage PR2, 14-DoF dual arm.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (the robot's mass matrix
            M(q), default) or 'euclidean'.
    """

    def __init__(self, metric: str = 'kinetic_energy') -> None: ...

class Panda(RobotModel):
    """
    Franka Emika Panda, 7-DoF arm.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (the robot's mass matrix
            M(q), default) or 'euclidean'.
    """

    def __init__(self, metric: str = 'kinetic_energy') -> None: ...

class RidgebackUR5e(RobotModel):
    """
    Clearpath Ridgeback with a Universal Robots UR5e on its default mount, mecanum
    holonomic base, 9 coordinates (x, y, theta, six arm joints).

    The configuration is (x, y, theta, arm joints...) on SE(2) x R^n. Only the metric
    changes between a holonomic and a differential-drive base.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (default) or 'euclidean'.
        base: 'holonomic' or 'differential_drive' (left-invariant SE(2) metric with
            lateral weight 100). None matches the robot's drive.
        base_weights: Body weights (wx, wy, wtheta) of the SE(2) metric. They override
            base.
        workspace: ((x_lo, x_hi), (y_lo, y_hi)), the region the base samples.
    """

    def __init__(self, metric: str = 'kinetic_energy', base: str | None = None, base_weights: Sequence[float] | None = None, workspace: Sequence[Sequence[float]] = [[-5.0, 5.0], [-5.0, 5.0]]) -> None: ...

class RobotModel:
    """
    A built-in robot as a configuration space.

    A fixed-base robot is R^dof with joint-limit bounds. A mobile robot is
    SE(2) x R^n, the base pose (x, y, theta) followed by the arm joints. Pass an
    instance anywhere geodex accepts a manifold, including plan() with a Scene.
    """

    def name(self) -> str:
        """Robot name, such as 'panda'. It is also the VAMP kernel name."""

    def dof(self) -> int:
        """Number of configuration coordinates, base included."""

    def dim(self) -> int:
        """Configuration-space dimension, equal to dof."""

    def joint_limits(self) -> tuple[Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]]:
        """
        Per-coordinate (lower, upper) bounds, the base region first, then joint limits.
        """

    def random_point(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Sample a configuration uniformly within the bounds."""

    def seed(self, seed: int) -> None:
        """Reseed the sampler behind random_point() and unseeded plans."""

    def set_sampler(self, sampler: str) -> None:
        """Switch the sampler kind to 'scrambled', 'halton' or 'random'."""

    def mass_matrix(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """
        Metric tensor on coordinate velocities at q.

        For a fixed-base robot under the kinetic-energy metric, it is the joint-space
        mass matrix M(q). A mobile robot adds the SE(2) block on (x, y, theta). Use it
        to score a path under the metric of its plan.
        """

    def has_mass_matrix(self) -> bool:
        """Whether mass_matrix() is available for this model."""

    def periods(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """
        Per-coordinate periods, 2 pi on the base heading and empty for a fixed base.
        """

    def heuristic(self) -> geodex._geodex_core.heuristics.MatrixLowerBound:
        """
        Admissible Loewner-bound heuristic of this model's metric, with the base
        heading wrapped. plan() uses it by default for a robot.
        """

    def drive(self) -> str:
        """
        The robot's base drive, 'holonomic', 'differential_drive' or 'none'. It sets
        the default base metric. A model built with another base metric keeps that metric.
        """

    def __repr__(self) -> str: ...

class Stretch3(RobotModel):
    """
    Hello Robot Stretch 3, differential-drive base, 8 coordinates (x, y, theta, lift,
    arm extension, wrist yaw, pitch, roll). The arm extension moves the four
    nested arm segments by equal amounts.

    The configuration is (x, y, theta, arm joints...) on SE(2) x R^n. Only the metric
    changes between a holonomic and a differential-drive base.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (default) or 'euclidean'.
        base: 'holonomic' or 'differential_drive' (left-invariant SE(2) metric with
            lateral weight 100). None matches the robot's drive.
        base_weights: Body weights (wx, wy, wtheta) of the SE(2) metric. They override
            base.
        workspace: ((x_lo, x_hi), (y_lo, y_hi)), the region the base samples.
    """

    def __init__(self, metric: str = 'kinetic_energy', base: str | None = None, base_weights: Sequence[float] | None = None, workspace: Sequence[Sequence[float]] = [[-5.0, 5.0], [-5.0, 5.0]]) -> None: ...

class Stretch4(RobotModel):
    """
    Hello Robot Stretch 4, three-omniwheel holonomic base, 8 coordinates (x, y,
    theta, lift, arm extension, wrist yaw, pitch, roll).

    The configuration is (x, y, theta, arm joints...) on SE(2) x R^n. Only the metric
    changes between a holonomic and a differential-drive base.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (default) or 'euclidean'.
        base: 'holonomic' or 'differential_drive' (left-invariant SE(2) metric with
            lateral weight 100). None matches the robot's drive.
        base_weights: Body weights (wx, wy, wtheta) of the SE(2) metric. They override
            base.
        workspace: ((x_lo, x_hi), (y_lo, y_hi)), the region the base samples.
    """

    def __init__(self, metric: str = 'kinetic_energy', base: str | None = None, base_weights: Sequence[float] | None = None, workspace: Sequence[Sequence[float]] = [[-5.0, 5.0], [-5.0, 5.0]]) -> None: ...

class UR5(RobotModel):
    """
    Universal Robots UR5, 6-DoF arm.

    Args:
        metric: Metric on the arm joints, 'kinetic_energy' (the robot's mass matrix
            M(q), default) or 'euclidean'.
    """

    def __init__(self, metric: str = 'kinetic_energy') -> None: ...

def available() -> list[str]:
    """Names of the built-in robots, in alphabetical order."""

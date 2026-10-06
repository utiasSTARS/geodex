from collections.abc import Sequence
from typing import Annotated

from numpy.typing import ArrayLike


class CollisionChecker:
    """
    Per-robot point-validity collision checker.

    `make_vamp_checker` creates instances.
    """

    def is_valid(self, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> bool:
        """
        Check whether the configuration q is inside the joint box and collision-free.
        """

    def all_valid(self, qs: Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='C')]) -> bool:
        """
        Check whether every row of the (N, dof) array qs is valid, in SIMD batches.
        """

    def batch_width(self) -> int:
        """Number of configurations checked in one SIMD pass."""

class EnvHandle:
    """
    Opaque handle to a VAMP scene environment. Create it with `load_scene`. Copying it is safe.
    """

def attach_spheres(env: EnvHandle, spheres: Sequence[Sequence[float]]) -> None:
    """
    Attach a rigid sphere set to the robot's end-effector frame.

    Spheres are [x, y, z, radius] in the end-effector frame and move with it. They
    are checked against the environment and the robot's own spheres. An empty list
    detaches. A checker copies the attachment when it is built. Attach or detach
    before building the checker. A checker built earlier does not see the change.
    """

def load_scene(yaml_path: str) -> EnvHandle:
    """
    Load an MBM-style scene YAML into an opaque VAMP environment handle.

    Supports primitive collision objects (boxes, cylinders, spheres) and mesh
    objects (axis-aligned bounding-box approximation).
    """

def make_vamp_checker(robot_name: str, env: EnvHandle) -> CollisionChecker:
    """
    Build a per-robot CollisionChecker bound to `env`.

    robot_name is one of registered_robots(). Mobile manipulators take the
    whole-body configuration (x, y, theta, arm joints...).
    """

def pad_scene(env: EnvHandle, padding: float) -> EnvHandle:
    """
    A copy of env with every obstacle grown by padding meters on every side.

    The copy keeps the attached spheres. Raises ValueError when padding is negative or
    not finite.
    """

def registered_robots() -> list[str]:
    """Names of robots compiled into the geodex_vamp archive (sorted)."""

def robot_dimension(robot_name: str) -> int:
    """Configuration dimension of a registered robot's VAMP model."""

def robot_end_effector(robot_name: str) -> str:
    """
    Name of the link that holds attached spheres in a registered robot's VAMP
    model. attach_spheres poses spheres in its frame.
    """

def robot_joint_names(robot_name: str) -> list[str]:
    """
    Joint names of a registered robot's VAMP model, in configuration order.

    Configurations for the robot's checker and validators list these joints in this
    order. Mobile manipulators start with base_x_joint, base_y_joint and
    base_theta_joint. The names match data/robots/<robot>/robot.yaml.
    """

def robot_spheres(robot_name: str, q: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, 4), order='C')]:
    """
    Collision spheres of a registered robot's VAMP model at q, one row
    [x, y, z, radius] per sphere in the world frame.
    """

def sphere_speed(robot_name: str, env: EnvHandle) -> float:
    """
    Bound on how far any sphere center of the robot, and of the spheres attached in env,
    moves per unit coordinate norm of a motion, in meters.
    """

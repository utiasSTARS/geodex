#!/usr/bin/env python3
"""Examples of the Metrics concept page, the Python version of metrics.cpp.

Each snippet builds one kind of metric and prints the cost of the same kinds of motion
under it.

Usage:
  python examples/concepts/metrics.py [--json out.json]
"""

import json
import sys

# [docs-start:se2-weights]
import numpy as np

import geodex
# [docs-end:se2-weights]


def main():
    out = {}

    # [docs-start:se2-weights]
    pose = np.array([0.0, 0.0, 0.0])        # at the origin, heading along +x
    forward = np.array([1.0, 0.0, 0.0])     # body-frame velocity (vx, vy, omega)
    sideways = np.array([0.0, 1.0, 0.0])
    turn = np.array([0.0, 0.0, 1.0])

    bases = {
        "holonomic": geodex.SE2(wx=1.0, wy=1.0, wtheta=0.5),
        "differential drive": geodex.SE2(wx=1.0, wy=100.0, wtheta=1.0),
        "car-like": geodex.SE2.car_like(1.5, 20.0),  # turning radius 1.5 m
    }
    for name, se2 in bases.items():
        costs = [se2.norm(pose, v) for v in (forward, sideways, turn)]
        print(name, "forward {:.3f}  sideways {:.3f}  turn {:.3f}".format(*costs))
    # [docs-end:se2-weights]
    out["se2_weights"] = {name: [se2.norm(pose, v) for v in (forward, sideways, turn)]
                          for name, se2 in bases.items()}

    # [docs-start:kinetic-energy]
    def arm_mass_matrix(q):
        """Mass matrix of a two-link planar arm with unit links and masses."""
        c2 = np.cos(q[1])
        m00 = 1 / 12 + 1 / 12 + 0.25 + (1.0 + 0.25 + 2.0 * 0.5 * c2)
        m01 = 1 / 12 + (0.25 + 0.5 * c2)
        m11 = 1 / 12 + 0.25
        return np.array([[m00, m01], [m01, m11]])

    arm = geodex.ConfigurationSpace(geodex.Euclidean(2),
                                    geodex.KineticEnergyMetric(arm_mass_matrix))
    shoulder = np.array([1.0, 0.0])  # turn the shoulder at 1 rad/s
    poses = {"stretched": np.array([0.0, 0.0]), "folded": np.array([0.0, 2.8])}
    for label, q in poses.items():
        print(f"{label}: shoulder speed costs {arm.norm(q, shoulder):.3f}")
    # [docs-end:kinetic-energy]
    out["kinetic_energy"] = [arm.norm(np.array([0.0, 0.0]), shoulder),
                             arm.norm(np.array([0.0, 2.8]), shoulder)]

    # [docs-start:robot]
    panda = geodex.robots.Panda()
    q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    M = panda.mass_matrix(q)  # 7 x 7, from the robot's precompiled CRBA
    print("Panda mass matrix diagonal:", np.round(np.diag(M), 4))
    # [docs-end:robot]
    out["robot"] = np.diag(M).tolist()

    # [docs-start:jacobi]
    def potential(q, g=9.81):
        """Height of the two link centers times their weight."""
        return g * (0.5 * np.sin(q[0]) + (np.sin(q[0]) + 0.5 * np.sin(q[0] + q[1])))

    H = 1.2 * 9.81 * (0.5 + 1.5)  # 20 percent above the largest potential
    metric = geodex.JacobiMetric(arm_mass_matrix, potential, H)
    jacobi = geodex.ConfigurationSpace(geodex.Euclidean(2), metric)
    poses = {"hanging": np.array([-1.5, 0.0]), "raised": np.array([1.5, 0.0])}
    for label, q in poses.items():
        print(f"{label}: shoulder speed costs {jacobi.norm(q, shoulder):.3f}")
    # [docs-end:jacobi]
    out["jacobi"] = [jacobi.norm(np.array([-1.5, 0.0]), shoulder),
                     jacobi.norm(np.array([1.5, 0.0]), shoulder)]

    # [docs-start:clearance]
    from geodex import collision

    obstacle = collision.CircleSDF(2.0, 0.0, 0.5)  # a disc of radius 0.5 at (2, 0)
    base_metric = geodex.SE2LeftInvariantMetric(1.0, 10.0, 1.0)
    clearance = geodex.ClearanceMetric(base_metric, obstacle, 1.5, 3.0)  # kappa, beta
    se2 = geodex.SE2(x_lo=0.0, x_hi=4.0, y_lo=-2.0, y_hi=2.0)
    room = geodex.ConfigurationSpace(se2, clearance)
    for label, p in [("next to the disc", np.array([1.3, 0.0, 0.0])),
                     ("far from it", np.array([0.2, 1.8, 0.0]))]:
        print(f"{label}: forward speed costs {room.norm(p, forward):.3f}")
    # [docs-end:clearance]
    out["clearance"] = [room.norm(np.array([1.3, 0.0, 0.0]), forward),
                        room.norm(np.array([0.2, 1.8, 0.0]), forward)]

    # [docs-start:pullback]
    def jacobian(q):
        """End-effector velocity of the planar arm per joint velocity."""
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array([[-s1 - s12, -s12], [c1 + c12, c12]])

    def task_metric(q):
        return np.eye(2)  # plain Euclidean speed of the end effector

    hand = geodex.ConfigurationSpace(geodex.Euclidean(2),
                                     geodex.PullbackMetric(jacobian, task_metric, 1e-3))
    poses = {"stretched": np.array([0.0, 0.0]), "folded": np.array([0.0, 2.8])}
    for label, q in poses.items():
        print(f"{label}: shoulder speed moves the hand at {hand.norm(q, shoulder):.3f}")
    # [docs-end:pullback]
    out["pullback"] = [hand.norm(np.array([0.0, 0.0]), shoulder),
                       hand.norm(np.array([0.0, 2.8]), shoulder)]
    return out


if __name__ == "__main__":
    results = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(results, f)

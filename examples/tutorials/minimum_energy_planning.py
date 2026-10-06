#!/usr/bin/env python3
"""Minimum-energy planning for a two-link planar arm, the Python version of
minimum_energy_planning.cpp.

Plans one start and goal on [-pi, pi]^2 under the Euclidean, the kinetic-energy and the
Jacobi metric, then under the Jacobi metric at a higher energy. G-RRT* runs as an
uninformed RRT*.

Usage:
  python examples/tutorials/minimum_energy_planning.py [--json out.json]
"""

import json
import sys

# [docs-start:setup]
import numpy as np

import geodex
# [docs-end:setup]


# [docs-start:mass-matrix]
class PlanarArmMassMatrix:
    """Mass matrix M(q) of a two-link planar arm with uniform rods."""

    def __init__(self, l1=1.0, l2=1.0, m1=1.0, m2=1.0, lc1=0.5, lc2=0.5,
                 I1=1 / 12, I2=1 / 12):
        self.l1, self.m1, self.lc1, self.I1 = l1, m1, lc1, I1
        self.l2, self.m2, self.lc2, self.I2 = l2, m2, lc2, I2

    def __call__(self, q):
        c2 = np.cos(q[1])  # cos(q2), the elbow coupling term
        h = self.l1 * self.lc2 * c2  # inertial coupling coefficient
        m00 = self.I1 + self.I2 + self.m1 * self.lc1**2 + self.m2 * (
            self.l1**2 + self.lc2**2 + 2.0 * h
        )
        m01 = self.I2 + self.m2 * (self.lc2**2 + h)
        m11 = self.I2 + self.m2 * self.lc2**2
        return np.array([[m00, m01], [m01, m11]])
# [docs-end:mass-matrix]


# [docs-start:potential]
def potential(q, g=9.81, m1=1.0, m2=1.0, l1=1.0, lc1=0.5, lc2=0.5):
    """Gravitational potential P(q), the mass-weighted heights of the two link
    centers."""
    return (m1 * g * lc1 * np.sin(q[0])
            + m2 * g * (l1 * np.sin(q[0]) + lc2 * np.sin(q[0] + q[1])))
# [docs-end:potential]


def main():
    results = {"runs": []}

    # [docs-start:ke-space]
    mass_fn = PlanarArmMassMatrix()

    # Joint space [-pi, pi]^2. plan() searches the sampling bounds of the base manifold.
    base = geodex.Euclidean(2)
    base.set_sampling_bounds(np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi]))

    ke_metric = geodex.KineticEnergyMetric(mass_fn)
    cspace_ke = geodex.ConfigurationSpace(base, ke_metric)
    # [docs-end:ke-space]

    # [docs-start:jacobi-space]
    pmax = 9.81 * (1.0 * 0.5 + 1.0 * (1.0 + 0.5))  # arm straight up, about 19.62 J
    H = 1.2 * pmax  # total energy, 20 percent above the largest potential

    jacobi_metric = geodex.JacobiMetric(mass_fn, potential, H)
    cspace_jacobi = geodex.ConfigurationSpace(base, jacobi_metric)
    # [docs-end:jacobi-space]

    # [docs-start:plan]
    start = np.array([-np.pi / 4, -np.pi / 4])
    goal = np.array([3 * np.pi / 4, 3 * np.pi / 4])

    # G-RRT* with the greedy bias off and the zero heuristic runs as an uninformed
    # RRT*.
    settings = geodex.PlanSettings(
        iterations=3000,
        seed=1,
        planner=geodex.planners.GreedyRRTstar(greedy_ratio=0.0),
    )

    runs = {}
    for label, space in [("Euclidean", base), ("Kinetic energy", cspace_ke),
                         ("Jacobi", cspace_jacobi)]:
        result = geodex.plan(space, start, goal, settings=settings,
                             heuristic=geodex.heuristics.Zero())
        print(f"{label:15s} solved={result.solved} cost={result.cost:.4f} "
              f"waypoints={len(result.path)}")
        runs[label] = result
    # [docs-end:plan]

    # [docs-start:try-it]
    high_metric = geodex.JacobiMetric(mass_fn, potential, 5.0 * pmax)
    high = geodex.ConfigurationSpace(base, high_metric)
    high_result = geodex.plan(high, start, goal, settings=settings,
                              heuristic=geodex.heuristics.Zero())
    for label, result in [("Kinetic energy", runs["Kinetic energy"]),
                          ("Jacobi, H = 1.2 Pmax", runs["Jacobi"]),
                          ("Jacobi, H = 5 Pmax", high_result)]:
        elbow = np.abs(result.path[:, 1]).max()
        print(f"{label:21s} largest elbow angle {elbow:.3f} rad")
    # [docs-end:try-it]
    results["try_it"] = {"solved": bool(high_result.solved), "cost": float(high_result.cost),
                         "path": high_result.path.tolist()}

    for label, result in runs.items():
        results["runs"].append({
            "label": label,
            "solved": bool(result.solved),
            "cost": float(result.cost),
            "raw_path": result.raw_path.tolist(),
            "path": result.path.tolist(),
        })

    results.update(start=start.tolist(), goal=goal.tolist(), H=H, pmax=pmax)
    return results


if __name__ == "__main__":
    out = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f)

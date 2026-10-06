#!/usr/bin/env python3
"""Seeds and budgets, the Python version of reproducibility.cpp.

The script runs the plan of the installation page twice with one seed, then with other
seeds, with a time budget and without a seed. It also seeds the samplers outside the
planner. Usage:
  python examples/getting_started/reproducibility.py [--json out.json]
"""

import json
import sys

# [docs-start:same-seed]
import numpy as np

import geodex

# The plan of the installation page, a differential-drive robot around a disc.
space = geodex.SE2(wx=1.0, wy=100.0, wtheta=1.0,
                   x_lo=0.0, x_hi=4.0, y_lo=0.0, y_hi=4.0)
start, goal = np.array([0.5, 2.0, 0.0]), np.array([3.5, 2.0, 0.0])


def is_valid(q):
    return np.hypot(q[0] - 2.0, q[1] - 2.0) > 0.8
# [docs-end:same-seed]


def main():
    out = {}

    # [docs-start:same-seed]
    settings = geodex.PlanSettings(iterations=2000, seed=1)
    a = geodex.plan(space, start, goal, is_valid, settings=settings)
    b = geodex.plan(space, start, goal, is_valid, settings=settings)
    print("same path:", np.array_equal(a.path, b.path), "same cost:", a.cost == b.cost)
    # [docs-end:same-seed]
    out["same_seed"] = {"identical": bool(np.array_equal(a.path, b.path) and a.cost == b.cost),
                        "cost": float(a.cost), "path": a.path.tolist()}

    # [docs-start:other-seeds]
    for seed in (1, 2, 3):
        result = geodex.plan(space, start, goal, is_valid,
                             settings=geodex.PlanSettings(iterations=2000, seed=seed))
        print(f"seed {seed}: cost {result.cost:.4f}")
        # [docs-end:other-seeds]
        out[f"seed_{seed}"] = float(result.cost)

    # [docs-start:time-budget]
    # With iterations=0 (the default) the planner stops after `time` seconds.
    timed = geodex.plan(space, start, goal, is_valid,
                        settings=geodex.PlanSettings(time=0.2, seed=1))
    print(f"time budget: cost {timed.cost:.4f}")
    # [docs-end:time-budget]

    # [docs-start:sampling]
    geodex.seed(7)                # reseed the source of default samplers
    sphere = geodex.Sphere()      # constructed afterwards, it samples from seed 7
    first = sphere.random_point()

    sphere.seed(7)                # reseed this manifold's own sampler
    again = sphere.random_point()
    print("first sample:", first, "after sphere.seed(7):", again)
    # [docs-end:sampling]
    out["sampling"] = {"first": first.tolist(), "again": again.tolist()}

    # [docs-start:try-it]
    unseeded = geodex.PlanSettings(iterations=2000)  # seed 0, the default
    space.seed(3)
    first = geodex.plan(space, start, goal, is_valid, settings=unseeded)
    second = geodex.plan(space, start, goal, is_valid, settings=unseeded)
    space.seed(3)
    again = geodex.plan(space, start, goal, is_valid, settings=unseeded)
    print(f"costs {first.cost:.4f} and {second.cost:.4f}, "
          f"after space.seed(3) {again.cost:.4f}")
    # [docs-end:try-it]
    out["try_it"] = [float(first.cost), float(second.cost), float(again.cost)]
    return out


if __name__ == "__main__":
    results = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(results, f)

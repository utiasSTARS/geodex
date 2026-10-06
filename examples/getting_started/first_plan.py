#!/usr/bin/env python3
"""A first plan. A differential-drive robot on SE(2) drives around a disc.

The Python version of first_plan.cpp. Usage:
  python examples/getting_started/first_plan.py [--json out.json]
"""

import json
import sys


def main():
    # [docs-start:first-plan]
    import numpy as np
    import geodex

    # The space holds poses (x, y, heading) in a 4 m x 4 m room. The weight 100 on the
    # squared sideways speed makes a meter of sliding as long as ten meters of driving
    # forward.
    space = geodex.SE2(wx=1.0, wy=100.0, wtheta=1.0,
                       x_lo=0.0, x_hi=4.0, y_lo=0.0, y_hi=4.0)

    def is_valid(q):  # outside a disc of radius 0.8 m in the middle of the room
        return np.hypot(q[0] - 2.0, q[1] - 2.0) > 0.8

    start, goal = np.array([0.5, 2.0, 0.0]), np.array([3.5, 2.0, 0.0])
    result = geodex.plan(space, start, goal, is_valid,
                         settings=geodex.PlanSettings(iterations=2000, seed=1))
    print(result.solved, round(result.cost, 3), len(result.path))
    # [docs-end:first-plan]
    return {"solved": bool(result.solved), "cost": float(result.cost),
            "path": result.path.tolist()}


if __name__ == "__main__":
    out = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f)

#!/usr/bin/env python3
"""Examples of the Discrete Geodesic Interpolation page, the Python version of
discrete_geodesic.cpp. Usage:
  python examples/concepts/discrete_geodesic.py [--json out.json]
"""

import json
import sys


def main():
    # [docs-start:sphere]
    import numpy as np
    import geodex

    start = np.array([0.0, 0.0, 1.0])
    target = np.array([np.sin(1.3) * np.cos(0.5), np.sin(1.3) * np.sin(0.5),
                       np.cos(1.3)])

    settings = geodex.InterpolationSettings(step_size=0.05, max_steps=500)

    # 1. The round sphere takes the fast path and traces the great circle.
    round_sphere = geodex.Sphere()
    great = geodex.discrete_geodesic(round_sphere, start, target, settings)

    # 2. An anisotropic constant SPD metric takes the finite-difference path.
    A = np.diag([25.0, 1.0, 1.0])
    stretched = geodex.ConfigurationSpace(round_sphere, geodex.ConstantSPDMetric(A))
    bent = geodex.discrete_geodesic(stretched, start, target, settings)

    print(great.status.name, len(great.path), bent.status.name, len(bent.path))
    # [docs-end:sphere]
    return {"great": [p.tolist() for p in great.path], "bent": [p.tolist() for p in bent.path],
            "status": [str(great.status).split(".")[-1], str(bent.status).split(".")[-1]]}


if __name__ == "__main__":
    out = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f)

#!/usr/bin/env python3
"""Examples of the Sampling concept page, the Python version of sampling.cpp.

Usage:
  python examples/concepts/sampling.py [--json out.json]
"""

import json
import sys

# [docs-start:random-point]
import geodex
# [docs-end:random-point]


def main():
    out = {}

    # [docs-start:random-point]
    sphere = geodex.Sphere()
    r3 = geodex.Euclidean(3)
    so3 = geodex.SO3()

    s = sphere.random_point()  # uniform on the sphere
    p = r3.random_point()      # uniform in the box [-1, 1]^3
    r = so3.random_point()     # Haar-uniform rotation, a unit quaternion
    # [docs-end:random-point]
    out["random_point"] = [len(s), len(p), len(r)]

    # [docs-start:choose-sampler]
    # The sampler is "scrambled" (the default), "halton" or "random".
    r3 = geodex.Euclidean(3, sampler="halton")
    r3.set_sampler("random")                    # switch at run time
    # [docs-end:choose-sampler]

    # [docs-start:seed]
    a = geodex.Sphere()
    b = geodex.Sphere()
    a.seed(42)
    b.seed(42)
    print((a.random_point() == b.random_point()).all())  # True, the same sequence

    geodex.seed(7)  # reseed the source of every default sampler constructed afterwards
    # [docs-end:seed]
    a.seed(42)
    out["seed"] = a.random_point().tolist()

    # [docs-start:product]
    pm = geodex.Product([geodex.SO3(), geodex.Euclidean(3)])
    pm.seed(0)
    x = pm.random_point()  # one joint scrambled Halton sample over both blocks
    # [docs-end:product]
    out["product"] = x.tolist()

    # [docs-start:standalone]
    s = geodex.ScrambledHaltonSampler(5)  # a seed makes the sequence reproducible
    u = s.sample(3)  # the next point of the sequence in [0, 1)^3
    # [docs-end:standalone]
    out["standalone"] = [u.tolist(), s.sample(3).tolist()]
    return out


if __name__ == "__main__":
    results = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(results, f)

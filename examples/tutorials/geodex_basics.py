#!/usr/bin/env python3
"""Examples for the geodex basics tutorial, the Python version of geodex_basics.cpp.

Each step of the tutorial runs in turn and prints its values.

Usage:
  python examples/tutorials/geodex_basics.py [--json out.json]
"""

import json
import sys

# [docs-start:setup]
import geodex
import numpy as np
# [docs-end:setup]


def main():
    out = {}

    # [docs-start:first-manifold]
    sphere = geodex.Sphere()
    print("dim =", sphere.dim())  # 2
    # [docs-end:first-manifold]
    out["first_manifold"] = sphere.dim()

    # [docs-start:create-manifolds]
    euclidean = geodex.Euclidean(3)  # R^3 with the standard metric
    torus = geodex.Torus(2)          # 2-torus with the flat metric
    # [docs-end:create-manifolds]
    out["create_manifolds"] = [euclidean.dim(), torus.dim()]

    # [docs-start:inner-sphere]
    sphere = geodex.Sphere()

    p = np.array([0.0, 0.0, 1.0])  # north pole
    u = np.array([1.0, 0.0, 0.0])  # tangent vector pointing east
    v = np.array([0.0, 1.0, 0.0])  # tangent vector pointing south

    ip = sphere.inner(p, u, v)
    print("ip =", ip)  # 0.0 (orthogonal)
    n = sphere.norm(p, u)
    print("n =", n)  # 1.0
    # [docs-end:inner-sphere]
    out["inner_sphere"] = [ip, n]

    # [docs-start:inner-euclidean]
    euclidean = geodex.Euclidean(3)

    p = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])

    ip = euclidean.inner(p, u, v)
    print("ip =", ip)  # 0.0
    n = euclidean.norm(p, u)
    print("n =", n)  # 1.0
    # [docs-end:inner-euclidean]
    out["inner_euclidean"] = [ip, n]

    # [docs-start:exp-log-sphere]
    sphere = geodex.Sphere()

    p = np.array([0.0, 0.0, 1.0])  # north pole
    q = np.array([1.0, 0.0, 0.0])  # point on the equator

    # log gives the tangent vector at p that points toward q.
    v = sphere.log(p, q)
    print("v =", v)  # about [pi/2, 0, 0], along the great circle to the equator

    # exp follows that tangent vector and recovers q.
    q_recovered = sphere.exp(p, v)
    print("q_recovered =", q_recovered)  # about [1, 0, 0]
    # [docs-end:exp-log-sphere]
    out["exp_log_sphere"] = [v.tolist(), q_recovered.tolist()]

    # [docs-start:exp-log-euclidean]
    euclidean = geodex.Euclidean(3)

    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])

    v = euclidean.log(p, q)
    print("v =", v)  # [-1, 1, 0]
    q2 = euclidean.exp(p, v)
    print("q2 =", q2)  # [0, 1, 0], which is q
    # [docs-end:exp-log-euclidean]
    out["exp_log_euclidean"] = [v.tolist(), q2.tolist()]

    # [docs-start:distance-sphere]
    sphere = geodex.Sphere()

    p = np.array([0.0, 0.0, 1.0])  # north pole
    q = np.array([1.0, 0.0, 0.0])  # equator

    d = sphere.distance(p, q)
    print("d =", d)  # 1.5708, about pi/2
    # [docs-end:distance-sphere]
    out["distance_sphere"] = d

    # [docs-start:distance-euclidean]
    euclidean = geodex.Euclidean(3)

    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])

    d = euclidean.distance(p, q)
    print("d =", d)  # 1.41421, about sqrt(2)
    # [docs-end:distance-euclidean]
    out["distance_euclidean"] = d

    # [docs-start:distance-torus]
    circle = geodex.Torus(1)

    p = np.array([0.1])
    q = np.array([6.0])

    d = circle.distance(p, q)
    print("d =", d)  # 0.383, about 2 pi - 5.9
    # [docs-end:distance-torus]
    out["distance_torus"] = d

    # [docs-start:geodesic-sphere]
    sphere = geodex.Sphere()

    p = np.array([0.0, 0.0, 1.0])  # north pole
    q = np.array([1.0, 0.0, 0.0])  # equator

    mid = sphere.geodesic(p, q, 0.5)
    print("mid =", mid)  # about [0.707, 0, 0.707]

    # Trace the whole geodesic
    for i in range(11):
        t = i / 10.0
        pt = sphere.geodesic(p, q, t)
        print(f"t={t:.1f}: {pt}")
    # [docs-end:geodesic-sphere]
    out["geodesic_sphere"] = [mid.tolist(), pt.tolist()]

    # [docs-start:geodesic-euclidean]
    euclidean = geodex.Euclidean(3)

    p = np.array([0.0, 0.0, 0.0])
    q = np.array([2.0, 4.0, 6.0])

    mid = euclidean.geodesic(p, q, 0.5)
    print("mid =", mid)  # [1, 2, 3]
    # [docs-end:geodesic-euclidean]
    out["geodesic_euclidean"] = mid.tolist()

    # [docs-start:random-points]
    sphere = geodex.Sphere()
    euclidean = geodex.Euclidean(3)
    torus = geodex.Torus(2)

    p1 = sphere.random_point()     # uniform on the sphere
    p2 = euclidean.random_point()  # uniform in [-1, 1]^3
    p3 = torus.random_point()      # uniform in [0, 2 pi)^2
    # [docs-end:random-points]
    out["random_points"] = [len(p1), len(p2), len(p3)]

    # [docs-start:sampler-choice]
    # sampler is "scrambled" (the default), "halton" or "random".
    r3 = geodex.Euclidean(3, sampler="random")
    r3.set_sampler("halton")                    # or switch after construction
    # [docs-end:sampler-choice]

    # [docs-start:seeding]
    geodex.seed(42)   # the shared source of default samplers
    sphere.seed(42)   # or one manifold's own sampler
    # [docs-end:seeding]
    out["seeding"] = sphere.random_point().tolist()

    # [docs-start:torus-wrap]
    torus = geodex.Torus(2)

    p = np.array([0.1, 0.2])
    q = np.array([6.1, 0.5])

    # log wraps to the shortest path
    v = torus.log(p, q)
    print("v =", v)  # about [-0.283, 0.3], the difference 6.0 wrapped by 2 pi

    # exp follows the tangent vector and wraps back to [0, 2 pi)
    q_recovered = torus.exp(p, v)
    print("q_recovered =", q_recovered)  # about [6.1, 0.5]
    # [docs-end:torus-wrap]
    out["torus_wrap"] = [v.tolist(), q_recovered.tolist()]

    # [docs-start:so3]
    so3 = geodex.SO3()

    q0 = so3.random_point()  # unit quaternion [x, y, z, w]
    q1 = so3.random_point()

    w = so3.log(q0, q1)              # body angular velocity, shape (3,)
    mid = so3.geodesic(q0, q1, 0.5)  # SLERP midpoint
    # [docs-end:so3]

    # [docs-start:se2]
    se2 = geodex.SE2()

    p = np.array([1.0, 2.0, 0.0])   # pose at (1, 2), heading east
    q = np.array([3.0, 4.0, 1.57])  # pose at (3, 4), heading north

    v = se2.log(p, q)
    print("log:", v)

    q_recovered = se2.exp(p, v)
    print("recovered:", q_recovered)

    d = se2.distance(p, q)
    print("distance:", d)
    # [docs-end:se2]
    out["se2"] = [v.tolist(), q_recovered.tolist(), d]

    # [docs-start:se2-car]
    # A large w_y makes sideways motion expensive, as for a wheeled base.
    se2_car = geodex.SE2(wx=1.0, wy=100.0, wtheta=1.0)

    p = np.array([0.0, 0.0, 0.0])  # facing east
    q = np.array([2.0, 2.0, 0.0])  # same heading, offset diagonally

    d = se2_car.distance(p, q)
    print("distance:", d)
    # [docs-end:se2-car]
    out["se2_car"] = d

    # [docs-start:se3]
    se3 = geodex.SE3()

    a = se3.random_point()  # pose [tx, ty, tz, qx, qy, qz, qw], shape (7,)
    b = se3.random_point()

    xi = se3.log(a, b)             # twist [v; w], shape (6,)
    mid = se3.geodesic(a, b, 0.5)  # screw-motion midpoint
    # [docs-end:se3]

    # [docs-start:so3-frames]
    q0 = np.array([0.5, 0.0, 0.0, 0.8660])     # 60 degrees about x
    q1 = np.array([0.0, 0.0, 0.7071, 0.7071])  # 90 degrees about z

    # The metric is bi-invariant, and both frames report the same distance.
    db = geodex.SO3(frame="body").distance(q0, q1)   # 1.8235
    dw = geodex.SO3(frame="world").distance(q0, q1)  # 1.8235
    # [docs-end:so3-frames]
    out["so3_frames"] = [db, dw]

    # [docs-start:se3-frames]
    a = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.8660])  # pose [t; q]
    b = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.7071, 0.7071])

    # SE(3) does not have a bi-invariant metric, and the frames report different
    # distances.
    db = geodex.SE3(frame="body").distance(a, b)   # 3.2424
    dw = geodex.SE3(frame="world").distance(a, b)  # 2.8023
    # [docs-end:se3-frames]
    out["se3_frames"] = [db, dw]

    # [docs-start:product]
    space = geodex.Product([geodex.Euclidean(3), geodex.SE2()])
    print("dim =", space.dim())  # 6

    c = space.random_point()  # one joint low-discrepancy sample over both blocks
    # [docs-end:product]
    out["product"] = space.dim()

    # [docs-start:constant-spd]
    A = np.diag([4.0, 1.0, 1.0])
    weighted = geodex.ConstantSPDMetric(A)

    r3_weighted = geodex.ConfigurationSpace(geodex.Euclidean(3), weighted)

    p = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])

    ip = r3_weighted.inner(p, u, v)  # 0.0, still orthogonal
    n = r3_weighted.norm(p, u)       # 2.0, scaled by sqrt(4)

    q = np.array([1.0, 1.0, 1.0])
    d = r3_weighted.distance(p, q)
    print("d =", d)  # sqrt(4 + 1 + 1) = sqrt(6), about 2.449
    # [docs-end:constant-spd]
    out["constant_spd"] = [ip, n, d]

    # [docs-start:spd-sphere]
    A = np.diag([4.0, 1.0, 1.0])
    weighted = geodex.ConstantSPDMetric(A)

    sphere_weighted = geodex.ConfigurationSpace(geodex.Sphere(), weighted)

    p = np.array([0.0, 0.0, 1.0])
    u = np.array([1.0, 0.0, 0.0])

    n = sphere_weighted.norm(p, u)
    print("n =", n)  # 2.0, the same weighting
    # [docs-end:spd-sphere]
    out["spd_sphere"] = n

    # [docs-start:projection-retraction]
    sphere = geodex.Sphere(retraction="projection")

    p = np.array([0.0, 0.0, 1.0])
    q = np.array([1.0, 0.0, 0.0])

    # log and exp under the projection retraction give an approximate round trip
    v = sphere.log(p, q)
    q_approx = sphere.exp(p, v)
    print("q_approx =", q_approx)
    # [docs-end:projection-retraction]
    out["projection_retraction"] = [v.tolist(), q_approx.tolist()]

    # [docs-start:se2-euler]
    # SE(2) with the Euler retraction, the cheapest one. It matches the group
    # exponential to first order only at heading 0.
    se2_euler = geodex.SE2(retraction="euler")
    # [docs-end:se2-euler]
    out["se2_euler"] = se2_euler.dim()
    return out


if __name__ == "__main__":
    results = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(results, f)

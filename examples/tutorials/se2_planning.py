#!/usr/bin/env python3
"""SE(2) planning for a disc robot, a differential-drive robot and a car, the Python version
of se2_planning.cpp.

Run it from this directory, which holds the corridor distance grid.

Usage:
  cd examples/tutorials
  python se2_planning.py [--json out.json]
"""

import json
import sys


def main():
    out = {}

    # [docs-start:pose]
    import numpy as np

    import geodex
    from geodex import collision as gc

    se2 = geodex.SE2()
    pose = np.array([5.0, 3.0, np.pi / 4.0])  # (x, y, theta)
    # [docs-end:pose]
    out["pose"] = se2.distance(np.zeros(3), pose)

    # [docs-start:footprints]
    # A circular footprint is a single radius. Collision reduces to a point query
    # against an obstacle distance field inflated by this value.
    robot_radius = 0.3

    # Build a rectangular footprint with the factory.
    rect_fp = gc.PolygonFootprint.rectangle(half_length=0.35, half_width=0.25,
                                            samples_per_edge=6)

    # Build any convex polygon from body-frame vertices in counter-clockwise order.
    verts = [
        np.array([-0.35, -0.30]),  # rear right
        np.array([0.35, -0.20]),   # front right
        np.array([0.35, 0.20]),    # front left
        np.array([-0.35, 0.30]),   # rear left
    ]
    poly_fp = gc.PolygonFootprint(verts, samples_per_edge=4)
    # [docs-end:footprints]

    # [docs-start:metrics]
    # Holonomic. Every direction and turning cost the same.
    metric_holo = geodex.SE2LeftInvariantMetric(1.0, 1.0, 1.0)

    # Differential drive. A meter of sliding costs sqrt(10), about 3.2 meters of
    # driving.
    metric_diff = geodex.SE2LeftInvariantMetric(1.0, 10.0, 1.0)

    # Car-like. Turning trades against driving at a radius of 1.5 m.
    metric_car = geodex.SE2LeftInvariantMetric.car_like(turning_radius=1.5,
                                                        lateral_penalty=20.0)
    # [docs-end:metrics]
    v = np.array([0.3, 0.4, 0.5])
    out["metrics"] = [m.norm(pose, v) for m in (metric_holo, metric_diff, metric_car)]

    # [docs-start:grid]
    grid = gc.DistanceGrid()
    grid.load("willow_corridor_dist.txt")

    # World dimensions in meters.
    world_w = grid.width() * grid.resolution()
    world_h = grid.height() * grid.resolution()
    # [docs-end:grid]
    out["grid"] = [world_w, world_h]

    # [docs-start:parking-lot]
    car_hl, car_hw = 2.25, 0.9  # a 4.5 m x 1.8 m car

    obstacles = [
        gc.RectObstacle(5.0, 1.35, 0.0, car_hl, car_hw),   # parked car 1
        gc.RectObstacle(10.0, 1.35, 0.0, car_hl, car_hw),  # parked car 2
        gc.RectObstacle(21.0, 1.35, 0.0, car_hl, car_hw),  # parked car 3
        gc.RectObstacle(26.0, 1.35, 0.0, car_hl, car_hw),  # parked car 4
        gc.RectObstacle(15.0, -0.05, 0.0, 15.0, 0.05),     # curb
        gc.RectObstacle(15.0, 10.05, 0.0, 15.0, 0.05),     # sidewalk
    ]
    # [docs-end:parking-lot]

    # [docs-start:disc-validity]
    robot_radius = 0.3
    safety_margin = 0.10  # extra buffer beyond the robot radius

    def is_valid(q):
        return grid.distance_at(q[0], q[1]) > robot_radius + safety_margin
    # [docs-end:disc-validity]

    # [docs-start:holonomic-plan]
    # A manifold whose workspace bounds span the corridor map.
    se2 = geodex.SE2(wx=1.0, wy=1.0, wtheta=1.0,
                     x_lo=0.0, x_hi=world_w, y_lo=0.0, y_hi=world_h)

    start = np.array([2.0, 5.0, 0.0])
    goal = np.array([12.0, 6.0, -np.pi / 2.0])

    settings = geodex.PlanSettings(
        iterations=6000,
        planner=geodex.planners.GreedyRRTstar(range=4.5, greedy_ratio=0.9,
                                              rewire_factor=0.5),
        collision_check_resolution=grid.resolution(),
        seed=1,
    )
    result = geodex.plan(se2, start, goal, is_valid, settings=settings)
    print(f"holonomic: solved={result.solved} cost={result.cost:.3f}")
    # [docs-end:holonomic-plan]
    out["holonomic"] = {"solved": bool(result.solved), "cost": float(result.cost),
                        "path": result.path.tolist()}

    # [docs-start:time-budget]
    settings = geodex.PlanSettings(time=1.0)  # seconds, used when iterations is 0
    # [docs-end:time-budget]

    # [docs-start:footprint-checker]
    footprint = gc.PolygonFootprint.rectangle(half_length=0.35, half_width=0.25,
                                              samples_per_edge=6)
    checker = gc.FootprintGridChecker(grid, footprint, safety_margin=0.10)

    is_valid = checker.is_valid
    # [docs-end:footprint-checker]
    out["footprint_checker"] = [bool(checker.is_valid(start)), float(checker(start))]

    # [docs-start:diff-manifold]
    se2 = geodex.SE2(wx=1.0, wy=10.0, wtheta=1.0,
                     x_lo=0.0, x_hi=world_w, y_lo=0.0, y_hi=world_h)
    # [docs-end:diff-manifold]

    # [docs-start:directional]
    settings = geodex.PlanSettings(
        iterations=6000,
        planner=geodex.planners.GreedyRRTstar(range=7.0, greedy_ratio=0.9,
                                              rewire_factor=1.1),
        collision_check_resolution=grid.resolution(),
        seed=1,
    )
    result = geodex.plan(
        se2, start, goal, checker.is_valid, settings=settings,
        motion_validator=geodex.DirectionalMotionValidator(max_reverse_length=0.5),
    )
    print(f"differential drive: solved={result.solved} cost={result.cost:.3f}")
    # [docs-end:directional]
    out["directional"] = {"solved": bool(result.solved), "cost": float(result.cost),
                          "path": result.path.tolist()}

    # [docs-start:try-it]
    def reverse_travel(path):
        """Distance the robot drives backward, from the SE(2) logarithm of each edge."""
        return sum(max(0.0, -se2.log(a, b)[0]) for a, b in zip(path[:-1], path[1:]))

    free = geodex.plan(se2, start, goal, checker.is_valid, settings=settings)
    print(f"backward driving: {reverse_travel(result.path):.3f} m with the validator, "
          f"{reverse_travel(free.path):.3f} m without")
    # [docs-end:try-it]
    out["try_it"] = {"solved": bool(free.solved), "cost": float(free.cost),
                     "reverse": [reverse_travel(result.path), reverse_travel(free.path)]}

    # [docs-start:clearance]
    # Base metric, and an SDF inflated by the robot radius for a circular robot.
    base_metric = geodex.SE2LeftInvariantMetric(1.0, 1.0, 1.0)
    inflated_sdf = gc.InflatedSDF(gc.GridSDF(grid), 0.3)

    # Conformal clearance metric with kappa = 1.5 and beta = 3.0.
    clearance_metric = geodex.ClearanceMetric(base_metric, inflated_sdf, 1.5, 3.0)

    # SE(2) topology with the clearance geometry.
    se2 = geodex.SE2(x_lo=0.0, x_hi=world_w, y_lo=0.0, y_hi=world_h)
    cspace = geodex.ConfigurationSpace(se2, clearance_metric)
    # [docs-end:clearance]
    out["clearance"] = cspace.norm(start, np.array([1.0, 0.0, 0.0]))

    # [docs-start:diff-clearance]
    footprint = gc.PolygonFootprint.rectangle(0.35, 0.25, 6)
    checker = gc.FootprintGridChecker(grid, footprint, 0.01)

    # Binary validity for the planner.
    is_valid = checker.is_valid

    # Continuous signed distance for the clearance metric, from the same object.
    base_metric = geodex.SE2LeftInvariantMetric(1.0, 10.0, 1.0)
    clearance_metric = geodex.ClearanceMetric(base_metric, checker, 1.5, 3.0)

    se2 = geodex.SE2(wx=1.0, wy=10.0, wtheta=1.0,
                     x_lo=0.0, x_hi=world_w, y_lo=0.0, y_hi=world_h)
    cspace = geodex.ConfigurationSpace(se2, clearance_metric)
    # [docs-end:diff-clearance]
    out["diff_clearance"] = cspace.norm(start, np.array([1.0, 0.0, 0.0]))

    # [docs-start:car-metric]
    metric = geodex.SE2LeftInvariantMetric.car_like(1.5, 20.0)
    # Weights wx = 1.0, wy = 20.0, w_theta = 2.25 (= 1.5^2)
    # [docs-end:car-metric]

    # [docs-start:rect-sdf]
    sdf = gc.RectSmoothSDF(obstacles, 20.0, car_hw)
    # beta = 20 (smoothness), inflation = car_hw (half-width of the ego vehicle)
    # [docs-end:rect-sdf]
    out["rect_sdf"] = sdf(np.array([15.0, 5.0, 0.0]))

    # [docs-start:sat-validity]
    def is_valid(q):
        ego = gc.RectObstacle(q[0], q[1], q[2], car_hl, car_hw)
        return not any(gc.rects_overlap(ego, obs) for obs in obstacles)
    # [docs-end:sat-validity]
    out["sat_validity"] = [is_valid(np.array([15.0, 5.0, 0.0])),
                           is_valid(np.array([5.0, 1.35, 0.0]))]

    # [docs-start:parking]
    base_metric = geodex.SE2LeftInvariantMetric.car_like(1.5, 20.0)
    sdf = gc.RectSmoothSDF(obstacles, 20.0, car_hw)
    clearance_metric = geodex.ClearanceMetric(base_metric, sdf, 8.0, 3.0)

    se2 = geodex.SE2.car_like(1.5, 20.0, x_lo=0.0, x_hi=30.0, y_lo=0.0, y_hi=12.0)
    cspace = geodex.ConfigurationSpace(se2, clearance_metric)
    # [docs-end:parking]
    out["parking"] = cspace.norm(np.array([15.0, 5.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    return out


if __name__ == "__main__":
    results = main()
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(results, f)

#!/usr/bin/env python3
"""Plan the scenarios of the SE(2) planning tutorial for its figures and animations.

Each scenario plans in a space that examples/tutorials/se2_planning.py sets up with
geodex.plan and the collision module, then writes the densified raw and smoothed paths as
JSON for visualize_se2_tutorial.py and animate_se2_tutorial.py.

The --scenario flag selects one of five scenarios.
  holonomic:      circular robot, isotropic metric, grid distance field
  holo_clearance: circular robot, isotropic metric + clearance metric
  diff_drive:     rectangular robot, anisotropic metric, footprint checker
  diff_clearance: rectangular robot, anisotropic metric + clearance metric
  parking:        car-like robot, rectangle obstacles, parallel parking

The corridor scenarios read the distance grid of the tutorial example, and --dist-map
replaces it.

Usage:
  python docs/tools/figures/se2_tutorial.py --scenario=holonomic -o out.json
  python docs/tools/figures/se2_tutorial.py --scenario=parking -o parking.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import geodex
from geodex import collision as gc

DIST_MAP = (Path(__file__).resolve().parents[3] / "examples" / "tutorials" /
            "willow_corridor_dist.txt")

# Shared start and goal for the Willow-corridor scenarios (x, y, theta).
CORRIDOR_START = [2.0, 5.0, 0.0]
CORRIDOR_GOAL = [12.0, 6.0, -math.pi / 2.0]


def densify(space, path, step=0.3):
    """Resample a pose polyline along the manifold geodesic between consecutive waypoints.

    A straight-line plot of the result traces the SE(2) curve."""
    pts = [np.asarray(p, dtype=float) for p in path]
    if len(pts) < 2:
        return [p.tolist() for p in pts]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(space.distance(a, b) / step))
        for k in range(n):
            out.append(list(space.geodesic(a, b, k / n)))
    out.append(pts[-1].tolist())
    return out


def run_planner(space, label, metric_info, start, goal, is_valid, *, planner, seed,
                heuristic=None, iterations, collision_check_resolution=0.0,
                interp="base_geodesic",
                motion_validator=None):
    """Plan once and pack a densified run record in the visualizer's schema.

    Planning stops after a fixed iteration budget, and the seed fixes the low-discrepancy
    scramble and OMPL's RNG. Every scenario passes the SE(2) Loewner matrix lower bound
    from se2.matrix_lower_bound(). The clearance factor c(q) >= 1 only raises the base
    metric, and the free-space bound stays admissible. Tree edges follow the SE(2) group
    geodesic, as in plan()'s default, and the clearance cost still enters the search
    through distance()."""
    settings = geodex.PlanSettings(
        iterations=iterations, planner=planner,
        collision_check_resolution=collision_check_resolution, interp=interp, seed=seed,
    )
    result = geodex.plan(
        space, np.array(start), np.array(goal),
        collision=is_valid, settings=settings, heuristic=heuristic,
        motion_validator=motion_validator,
    )
    if result.solved:
        print(f"{label}: solved (cost {result.cost:.3f}, {len(result.path)} waypoints, "
              f"{result.time_ms:.0f} ms)")
    else:
        print(f"{label}: no exact solution found.", file=sys.stderr)
    return {
        "label": label,
        "metric_info": metric_info,
        "solved": result.solved,
        "planning_time_ms": result.time_ms,
        "raw_path": densify(space, result.raw_path),
        "smoothed_path": densify(space, result.path),
    }


def conformal_grid(sdf, kappa, beta, grid):
    """Sample the conformal factor c(q) = 1 + kappa * exp(-beta * sdf(q)) over the
    map cells at theta = 0. The clearance heatmap figure draws these values."""
    w, h, res = grid.width(), grid.height(), grid.resolution()
    values = []
    for row in range(h):
        y = (row + 0.5) * res
        for col in range(w):
            x = (col + 0.5) * res
            values.append(1.0 + kappa * math.exp(-beta * sdf(np.array([x, y, 0.0]))))
    return {"width": w, "height": h, "resolution": res, "values": values}


def map_info(grid, map_file):
    return {"width": grid.width(), "height": grid.height(),
            "resolution": grid.resolution(), "file": map_file}


def corridor_manifold(grid, wx, wy, wtheta):
    """An SE(2) manifold whose workspace bounds span the corridor map."""
    return geodex.SE2(wx=wx, wy=wy, wtheta=wtheta, x_lo=0.0, x_hi=grid.width() * grid.resolution(),
                      y_lo=0.0, y_hi=grid.height() * grid.resolution())


# ---------------------------------------------------------------------------
# Scenario: holonomic circular robot (baseline without a clearance metric)
# ---------------------------------------------------------------------------

def run_holonomic(grid, map_file, iterations, seed):
    robot_radius, safety = 0.3, 0.10
    se2 = corridor_manifold(grid, 1.0, 1.0, 1.0)

    def is_valid(q):
        return grid.distance_at(q[0], q[1]) > robot_radius + safety

    run = run_planner(
        se2, "Holonomic (isotropic)", "w=(1,1,1)", CORRIDOR_START, CORRIDOR_GOAL, is_valid,
        planner=geodex.planners.GreedyRRTstar(range=4.5, greedy_ratio=0.9, rewire_factor=0.5),
        seed=seed, heuristic=se2.matrix_lower_bound(), iterations=iterations,
        collision_check_resolution=grid.resolution(),
    )
    return {"scenario": "holonomic", "start": CORRIDOR_START, "goal": CORRIDOR_GOAL,
            "robot": {"type": "circle", "radius": robot_radius},
            "map": map_info(grid, map_file), "runs": [run]}


# ---------------------------------------------------------------------------
# Scenario: holonomic + clearance metric
# ---------------------------------------------------------------------------

def run_holo_clearance(grid, map_file, iterations, seed):
    robot_radius, safety = 0.3, 0.10
    kappa, beta = 1.5, 3.0

    se2 = corridor_manifold(grid, 1.0, 1.0, 1.0)
    inflated = gc.InflatedSDF(gc.GridSDF(grid), robot_radius)
    clearance = geodex.ClearanceMetric(geodex.SE2LeftInvariantMetric(1.0, 1.0, 1.0),
                                       inflated, kappa, beta)
    cspace = geodex.ConfigurationSpace(se2, clearance)

    def is_valid(q):
        return grid.distance_at(q[0], q[1]) > robot_radius + safety

    run = run_planner(
        cspace, "Holonomic + clearance", "k=1.5 b=3", CORRIDOR_START, CORRIDOR_GOAL, is_valid,
        planner=geodex.planners.GreedyRRTstar(range=2.5, greedy_ratio=0.9, rewire_factor=0.5),
        seed=seed, heuristic=se2.matrix_lower_bound(), iterations=iterations,
        collision_check_resolution=grid.resolution(), interp="base_geodesic",
    )
    return {"scenario": "holo_clearance", "start": CORRIDOR_START, "goal": CORRIDOR_GOAL,
            "robot": {"type": "circle", "radius": robot_radius},
            "map": map_info(grid, map_file),
            "conformal_grid": conformal_grid(inflated, kappa, beta, grid), "runs": [run]}


# ---------------------------------------------------------------------------
# Scenario: differential-drive rectangular robot
# ---------------------------------------------------------------------------

def run_diff_drive(grid, map_file, iterations, seed):
    robot_hl, robot_hw, safety = 0.35, 0.25, 0.10
    se2 = corridor_manifold(grid, 1.0, 10.0, 1.0)

    footprint = gc.PolygonFootprint.rectangle(robot_hl, robot_hw, 6)
    checker = gc.FootprintGridChecker(grid, footprint, safety)

    # The directional validator requires every edge to drive forward, with a small
    # reverse budget per edge.
    run = run_planner(
        se2, "Diff-drive (anisotropic)", "w=(1,10,1)", CORRIDOR_START, CORRIDOR_GOAL,
        checker.is_valid,
        planner=geodex.planners.GreedyRRTstar(range=7.0, greedy_ratio=0.9, rewire_factor=1.1),
        seed=seed, heuristic=se2.matrix_lower_bound(), iterations=iterations,
        collision_check_resolution=grid.resolution(), interp="base_geodesic",
        motion_validator=geodex.DirectionalMotionValidator(max_reverse_length=0.5),
    )
    return {"scenario": "diff_drive", "start": CORRIDOR_START, "goal": CORRIDOR_GOAL,
            "robot": {"type": "rectangle", "half_length": robot_hl, "half_width": robot_hw},
            "map": map_info(grid, map_file), "runs": [run]}


# ---------------------------------------------------------------------------
# Scenario: differential-drive + clearance metric
# ---------------------------------------------------------------------------

def run_diff_clearance(grid, map_file, iterations, seed):
    robot_hl, robot_hw, safety = 0.35, 0.25, 0.01
    kappa, beta = 1.5, 3.0

    se2 = corridor_manifold(grid, 1.0, 10.0, 1.0)
    footprint = gc.PolygonFootprint.rectangle(robot_hl, robot_hw, 6)
    checker = gc.FootprintGridChecker(grid, footprint, safety)
    # The footprint checker doubles as the continuous clearance SDF.
    clearance = geodex.ClearanceMetric(geodex.SE2LeftInvariantMetric(1.0, 10.0, 1.0),
                                       checker, kappa, beta)
    cspace = geodex.ConfigurationSpace(se2, clearance)

    run = run_planner(
        cspace, "Diff-drive + clearance", "w=(1,10,1) k=1.5 b=3", CORRIDOR_START, CORRIDOR_GOAL,
        checker.is_valid,
        planner=geodex.planners.GreedyRRTstar(range=2.5, greedy_ratio=0.9, rewire_factor=0.5),
        seed=seed, heuristic=se2.matrix_lower_bound(), iterations=iterations,
        collision_check_resolution=grid.resolution(), interp="base_geodesic",
        motion_validator=geodex.DirectionalMotionValidator(max_reverse_length=0.5),
    )
    return {"scenario": "diff_clearance", "start": CORRIDOR_START, "goal": CORRIDOR_GOAL,
            "robot": {"type": "rectangle", "half_length": robot_hl, "half_width": robot_hw},
            "map": map_info(grid, map_file), "runs": [run]}


# ---------------------------------------------------------------------------
# Scenario: car-like parallel parking
# ---------------------------------------------------------------------------

def run_parking(iterations, seed):
    car_hl, car_hw = 2.25, 0.9  # 4.5 x 1.8 m sedan
    world_w, world_h = 30.0, 12.0
    turning_radius, lateral_penalty = 1.5, 20.0
    kappa, beta = 8.0, 3.0

    # The planner range is a step length in the planning metric, not in meters. Next to the
    # parked cars the clearance factor c = 1 + kappa exp(-beta sdf) reaches 8.9 and
    # stretches a motion to about 3 times its open-space length.

    obstacles = [
        gc.RectObstacle(5.0, 1.35, 0.0, car_hl, car_hw),    # parked car 1
        gc.RectObstacle(10.0, 1.35, 0.0, car_hl, car_hw),   # parked car 2
        gc.RectObstacle(21.0, 1.35, 0.0, car_hl, car_hw),   # parked car 3
        gc.RectObstacle(26.0, 1.35, 0.0, car_hl, car_hw),   # parked car 4
        gc.RectObstacle(15.0, -0.05, 0.0, 15.0, 0.05),      # curb
        gc.RectObstacle(15.0, 10.05, 0.0, 15.0, 0.05),      # sidewalk
    ]
    start, goal = [25.0, 6.0, math.pi], [15.5, 1.35, 0.0]

    se2 = geodex.SE2.car_like(turning_radius, lateral_penalty,
                              x_lo=0.0, x_hi=world_w, y_lo=0.0, y_hi=world_h)
    base_metric = geodex.SE2LeftInvariantMetric.car_like(turning_radius, lateral_penalty)
    sdf = gc.RectSmoothSDF(obstacles, 20.0, car_hw)
    clearance = geodex.ClearanceMetric(base_metric, sdf, kappa, beta)
    cspace = geodex.ConfigurationSpace(se2, clearance)

    def is_valid(q):
        x, y, theta = q[0], q[1], q[2]
        if x - car_hl < 0.0 or x + car_hl > world_w or y - car_hw < 0.0 or y + car_hw > world_h:
            return False
        ego = gc.RectObstacle(x, y, theta, car_hl, car_hw)
        return not any(gc.rects_overlap(ego, obs) for obs in obstacles)

    run = run_planner(
        cspace, "Car-like parallel parking", "r=1.5 lp=20 k=8 b=3", start, goal, is_valid,
        planner=geodex.planners.GreedyRRTstar(range=40.0, greedy_ratio=0.9, rewire_factor=0.5),
        seed=seed, heuristic=se2.matrix_lower_bound(), iterations=iterations,
        collision_check_resolution=0.1,
        interp="base_geodesic",
    )
    return {"scenario": "parking", "start": start, "goal": goal,
            "robot": {"type": "rectangle", "half_length": car_hl, "half_width": car_hw},
            "rect_obstacles": [
                {"center": [o.cx, o.cy], "theta": o.theta,
                 "half_length": o.half_length, "half_width": o.half_width} for o in obstacles],
            "runs": [run]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist-map", type=Path, default=DIST_MAP,
                        help="Distance grid of the corridor scenarios")
    parser.add_argument("--scenario", default="holonomic",
                        choices=["holonomic", "holo_clearance", "diff_drive", "diff_clearance",
                                 "parking"])
    parser.add_argument("-o", "--output", default=None, help="Output JSON path")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Iteration budget; 0 uses the per-scenario default")
    parser.add_argument("--seed", type=int, default=1, help="Planner and low-discrepancy seed")
    args = parser.parse_args()

    output = args.output or f"se2_tutorial_{args.scenario}.json"
    # Per-scenario iteration budgets. A fixed iteration budget and the seed below make each
    # run reproducible.
    default_iterations = {"holonomic": 6000, "holo_clearance": 5000, "diff_drive": 6000,
                       "diff_clearance": 6000, "parking": 5000}
    iterations = args.iterations if args.iterations > 0 else default_iterations[args.scenario]

    # The plan seed fixes the planner's samplers. geodex.seed fixes the samplers of the
    # manifolds built below.
    geodex.seed(args.seed)

    if args.scenario == "parking":
        result = run_parking(iterations, seed=args.seed)
    else:
        grid = gc.DistanceGrid()
        if not grid.load(str(args.dist_map)):
            print(f"Error: could not load distance grid: {args.dist_map}", file=sys.stderr)
            return 1
        print(f"Loaded grid: {grid.width()}x{grid.height()} "
              f"({grid.width() * grid.resolution():.1f} x "
              f"{grid.height() * grid.resolution():.1f} m)")
        runners = {
            "holonomic": run_holonomic, "holo_clearance": run_holo_clearance,
            "diff_drive": run_diff_drive, "diff_clearance": run_diff_clearance,
        }
        result = runners[args.scenario](grid, str(args.dist_map), iterations, args.seed)

    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

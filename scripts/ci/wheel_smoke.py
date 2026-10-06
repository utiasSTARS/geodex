#!/usr/bin/env python3
"""Smoke test for an installed pygeodex wheel.

Plans on SE(2) under a clearance metric with a Python validity callable, and on the Panda
under its kinetic-energy metric against a VAMP collision scene. When the official ``ompl``
wheel is importable, it also plans with it and with geodex in one process, in both import
orders, and checks that the statically linked OMPL fork in geodex stays separate from it.

    python scripts/ci/wheel_smoke.py                 # the variant this CPU selects
    python scripts/ci/wheel_smoke.py --isa x86_64    # force an x86-64 variant of the wheel

Exits non-zero on any failure and prints one JSON line of timings.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import time

START = time.perf_counter()


def load_variant(isa):
    """Load one instruction-set variant as geodex._geodex_core before geodex imports."""
    package = importlib.util.find_spec("geodex").submodule_search_locations[0]
    directory = os.path.join(package, "_isa", isa)
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = os.path.join(directory, "_geodex_core" + suffix)
        if os.path.exists(path):
            break
    else:
        raise SystemExit(f"this wheel does not have an {isa} variant in {directory}")
    name = "geodex._geodex_core"
    loader = importlib.machinery.ExtensionFileLoader(name, path)
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(name, path, loader=loader)
    )
    sys.modules[name] = module
    loader.exec_module(module)


def plan_se2_clearance(geodex, np):
    """Plan on SE(2) around a disc with a clearance metric and a Python validity callable."""
    radius, margin = 1.5, 0.2

    def sdf(q):
        return float(np.hypot(q[0] - 5.0, q[1] - 5.0) - radius)

    se2 = geodex.SE2(x_lo=0.0, x_hi=10.0, y_lo=0.0, y_hi=10.0)
    metric = geodex.ClearanceMetric(
        geodex.SE2LeftInvariantMetric(1.0, 1.0, 1.0), sdf, kappa=1.5, beta=3.0
    )
    space = geodex.ConfigurationSpace(se2, metric)
    settings = geodex.PlanSettings(
        time=1.0,
        planner=geodex.planners.GreedyRRTstar(range=2.5, greedy_ratio=0.9, rewire_factor=0.5),
        seed=1,
    )
    result = geodex.plan(
        space,
        np.array([1.0, 1.0, 0.0]),
        np.array([9.0, 9.0, 0.0]),
        is_valid=lambda q: sdf(q) > margin,
        settings=settings,
        heuristic=se2.matrix_lower_bound(),
    )
    assert result.solved, "SE(2) clearance plan failed"
    assert all(sdf(q) > margin for q in result.path), "SE(2) path enters the obstacle margin"
    return result


def plan_panda_scene(geodex, np):
    """Plan for the Panda under its kinetic-energy metric in a VAMP box scene."""
    robot = geodex.robots.Panda()
    lo, hi = robot.joint_limits()
    start = np.asarray(lo + 0.3 * (hi - lo))
    goal = np.asarray(lo + 0.6 * (hi - lo))
    scene = geodex.Scene()
    scene.add_box(position=[0.55, 0.0, 0.4], size=[0.3, 0.6, 0.8], orientation=[0, 0, 0, 1])
    result = geodex.plan(
        robot, start, goal, collision=scene, settings=geodex.PlanSettings(time=2.0, seed=42)
    )
    assert result.solved, "Panda scene plan failed"
    assert np.allclose(result.path[0], start) and np.allclose(result.path[-1], goal)
    return result


def plan_ompl(ob, og):
    """Solve a 2-D problem with the official OMPL bindings."""
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0.0)
    bounds.setHigh(1.0)
    space.setBounds(bounds)
    setup = og.SimpleSetup(space)
    setup.setStateValidityChecker(lambda s: (s[0] - 0.5) ** 2 + (s[1] - 0.5) ** 2 > 0.04)
    start, goal = space.allocState(), space.allocState()
    start[0], start[1], goal[0], goal[1] = 0.1, 0.1, 0.9, 0.9
    setup.setStartAndGoalStates(start, goal)
    setup.setPlanner(og.RRTConnect(setup.getSpaceInformation()))
    setup.solve(1.0)
    assert setup.haveExactSolutionPath(), "official OMPL plan failed"


def coexist(order):
    """Plan with the official OMPL wheel and with geodex in one process."""
    import numpy as np

    if order == "ompl-first":
        from ompl import base as ob, geometric as og

        import geodex
    else:
        import geodex
        from ompl import base as ob, geometric as og
    plan_ompl(ob, og)
    plan_se2_clearance(geodex, np)
    if hasattr(geodex._geodex_core, "Scene"):
        plan_panda_scene(geodex, np)
    plan_ompl(ob, og)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--isa", choices=["x86_64", "x86_64_v3"])
    parser.add_argument("--coexist", choices=["ompl-first", "geodex-first"], help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.coexist:
        coexist(args.coexist)
        return 0
    if args.isa:
        load_variant(args.isa)

    import numpy as np

    import geodex

    report = {"version": geodex.__version__, "isa": args.isa or geodex._isa}
    report["import_s"] = round(time.perf_counter() - START, 3)

    result = plan_se2_clearance(geodex, np)
    report["first_plan_s"] = round(time.perf_counter() - START, 3)
    report["se2_cost"] = round(float(result.cost), 4)

    if hasattr(geodex._geodex_core, "Scene"):
        result = plan_panda_scene(geodex, np)
        report["panda_cost"] = round(float(result.cost), 4)
        report["panda_waypoints"] = len(result.path)
    else:
        try:
            geodex.Scene()
        except ImportError as error:
            report["vamp"] = f"unavailable: {error}"
        else:
            raise AssertionError("geodex.Scene should be unavailable without VAMP")

    if importlib.util.find_spec("ompl") is not None and not args.isa:
        for order in ("ompl-first", "geodex-first"):
            subprocess.run([sys.executable, __file__, "--coexist", order], check=True)
        report["ompl_coexistence"] = "ok"

    report["total_s"] = round(time.perf_counter() - START, 3)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

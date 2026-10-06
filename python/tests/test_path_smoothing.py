"""Tests for geodex.smooth_path, its settings and result, and plan() smoothing."""

import gc
import math
import pathlib
import weakref

import numpy as np
import pytest

import geodex


def _disc(q, center=(0.0, 0.0), radius=1.0):
    return math.hypot(q[0] - center[0], q[1] - center[1]) - radius


def _valid(q):
    return _disc(q) > 0.0


# The default corner tolerance. A returned path lies within it of a checked path and keeps
# clear of the disc shrunk by it.
_TOL = 1e-4


def _valid_within(q):
    return _disc(q, radius=1.0 - _TOL) > 0.0


def _detour():
    pts = [(-2.0, 0.0), (-1.5, 1.5), (-0.5, 1.6), (0.0, 2.0), (0.6, 1.4), (1.5, 1.5), (2.0, 0.0)]
    return [np.array(p) for p in pts]


def _distance_to_segment(q, a, b):
    d = b - a
    t = 0.0 if not d.any() else min(1.0, max(0.0, float(np.dot(q - a, d) / np.dot(d, d))))
    return float(np.linalg.norm(a + t * d - q))


def _certified(path, valid, res):
    """Check independently that every waypoint and every chord at spacing <= res is valid."""
    path = [np.asarray(q) for q in path]
    for k in range(len(path)):
        if not valid(path[k]):
            return False
        if k + 1 == len(path):
            break
        a, b = path[k], path[k + 1]
        n = max(1, math.ceil(np.linalg.norm(b - a) / res))
        for j in range(1, n):
            if not valid(a + (j / n) * (b - a)):
                return False
    return True


class TestPathSmoothingSettings:
    def test_defaults(self):
        s = geodex.PathSmoothingSettings()
        assert s.collision_check_resolution == pytest.approx(0.0)
        assert s.output_spacing == pytest.approx(0.0)
        assert s.seed == 42
        assert s.edge_provably_clear is None
        assert s.edge_validator is None
        assert s.edge_travel is None
        assert s.path_predicate is None
        assert s.round_corners is True
        assert s.corner_tolerance == pytest.approx(1e-4)
        assert s.corner_max_angle == pytest.approx(0.5 * math.pi)
        assert s.sharp_coordinates == 0

    def test_keywords_and_fields(self):
        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01, output_spacing=0.1,
                                         seed=7, round_corners=False, corner_tolerance=1e-3,
                                         corner_max_angle=1.0)
        assert s.collision_check_resolution == pytest.approx(0.01)
        assert s.output_spacing == pytest.approx(0.1)
        assert s.seed == 7
        assert s.round_corners is False
        assert s.corner_tolerance == pytest.approx(1e-3)
        assert s.corner_max_angle == pytest.approx(1.0)
        s.round_corners = True
        s.corner_tolerance = 2e-4
        assert s.round_corners is True and s.corner_tolerance == pytest.approx(2e-4)
        assert geodex.PathSmoothingSettings(sharp_coordinates=3).sharp_coordinates == 3
        s.sharp_coordinates = 2
        assert s.sharp_coordinates == 2
        s.edge_provably_clear = lambda a, b: False
        assert s.edge_provably_clear is not None
        assert s.edge_provably_clear(np.zeros(2), np.ones(2)) is False
        s.edge_provably_clear = None
        assert s.edge_provably_clear is None


class _Holder:
    pass


@pytest.mark.parametrize("hook", ["edge_provably_clear", "edge_validator", "edge_travel",
                                  "path_predicate"])
@pytest.mark.parametrize("through_init", [False, True])
def test_hook_that_reaches_its_settings_is_collected(hook, through_init):
    holder = _Holder()

    def fn(*args):
        return True

    # A function attribute, unlike a closure cell, survives `del holder` below.
    fn.holder = holder
    if through_init:
        holder.settings = geodex.PathSmoothingSettings(**{hook: fn})
    else:
        holder.settings = geodex.PathSmoothingSettings()
        setattr(holder.settings, hook, fn)
    probe = weakref.ref(holder)
    del holder, fn
    gc.collect()
    assert probe() is None


def test_hooks_survive_in_a_copy_of_the_settings():
    calls = []
    s = geodex.PathSmoothingSettings(edge_validator=lambda a, b: calls.append(1) or True)
    plan_settings = geodex.PlanSettings(smoothing=s)
    del s
    gc.collect()
    r = geodex.smooth_path(geodex.Euclidean(2), lambda q: True, _detour(), plan_settings.smoothing)
    assert r.collision_free
    assert calls


class TestSmoothPath:
    def test_shortens_around_an_obstacle_and_certifies(self):
        m = geodex.Euclidean(2)
        settings = geodex.PathSmoothingSettings(collision_check_resolution=0.01)
        r = geodex.smooth_path(m, _valid, _detour(), settings)
        assert isinstance(r, geodex.PathSmoothingResult)
        assert r.collision_free
        assert r.first_invalid_index is None
        optimum = 2.0 * math.sqrt(3.0) + math.pi / 3.0
        assert optimum - 1e-6 <= r.length < 1.03 * optimum
        assert _certified(r.path, _valid_within, 0.01)
        np.testing.assert_array_equal(r.path[0], _detour()[0])
        np.testing.assert_array_equal(r.path[-1], _detour()[-1])
        assert r.path.shape[1] == 2
        assert len(r.waypoints) == r.path.shape[0]

    def test_profile_counts_the_work(self):
        m = geodex.Euclidean(2)
        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01)
        r = geodex.smooth_path(m, _valid, _detour(), s)
        p = r.profile
        assert p.total_ms > 0.0
        assert p.point_checks > 0
        assert p.relax_moves > 0
        assert p.input_waypoints == 7
        assert p.output_waypoints == r.path.shape[0]
        assert p.fallback == 0

    def test_same_seed_gives_the_same_path(self):
        m = geodex.Euclidean(2)
        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01)
        a = geodex.smooth_path(m, _valid, _detour(), s)
        b = geodex.smooth_path(m, _valid, _detour(), s)
        np.testing.assert_array_equal(a.path, b.path)

    def test_scaling_the_problem_scales_the_path(self):
        m = geodex.Euclidean(2)
        scale = 128.0
        small = geodex.smooth_path(m, _valid, _detour())
        # corner_tolerance is the one absolute default. It scales with the problem here.
        s = geodex.PathSmoothingSettings(corner_tolerance=scale * 1e-4)
        big = geodex.smooth_path(m, lambda q: _valid(q / scale), [scale * q for q in _detour()],
                                 s)
        assert small.profile.rounded_corners > 0
        assert small.collision_free and big.collision_free
        np.testing.assert_allclose(big.path / scale, small.path, atol=1e-12)

    def test_uncertifiable_input_is_returned_with_the_truth(self):
        m = geodex.Euclidean(2)
        path = [np.array(p) for p in [(-2.0, 0.0), (-1.2, 0.1), (1.2, 0.1), (2.0, 0.0)]]
        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01)
        r = geodex.smooth_path(m, _valid, path, s)
        assert not r.collision_free
        assert r.first_invalid_index == 1
        np.testing.assert_array_equal(r.path, np.array(path))

    def test_clearance_metric_keeps_away_from_the_obstacle(self):
        m = geodex.Euclidean(2)
        metric = geodex.ClearanceMetric(geodex.ConstantSPDMetric(np.eye(2)), _disc, 4.0, 3.0)
        aware = geodex.ConfigurationSpace(m, metric)
        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01)
        plain = geodex.smooth_path(m, _valid, _detour(), s)
        cleared = geodex.smooth_path(aware, _valid, _detour(), s)
        assert plain.collision_free and cleared.collision_free

        def min_clearance(path):
            return min(
                _disc(path[k] + t * (path[k + 1] - path[k]))
                for k in range(len(path) - 1)
                for t in np.linspace(0.0, 1.0, 17)
            )

        assert min_clearance(plain.path) < 0.02
        assert min_clearance(cleared.path) > 0.15

    def test_output_spacing(self):
        m = geodex.Euclidean(2)
        # By default, the waypoints lie at one even step.
        r = geodex.smooth_path(m, _valid, _detour(),
                               geodex.PathSmoothingSettings(collision_check_resolution=0.01))
        assert r.collision_free and r.profile.rounded_corners > 0
        steps = np.linalg.norm(np.diff(r.path, axis=0), axis=1)
        assert steps.max() / steps.min() < 1.001
        assert _certified(r.path, _valid_within, 0.01)
        # With output_spacing set, the steps are at most that long. With rounded corners,
        # they are equal.
        for round_corners in (False, True):
            s = geodex.PathSmoothingSettings(collision_check_resolution=0.01, output_spacing=0.1,
                                             round_corners=round_corners)
            r = geodex.smooth_path(m, _valid, _detour(), s)
            assert r.collision_free
            steps = np.linalg.norm(np.diff(r.path, axis=0), axis=1)
            assert np.all(steps <= 0.1 + 1e-9)
            if round_corners:
                assert steps.max() / steps.min() < 1.001
            assert _certified(r.path, _valid_within, 0.01)

    def test_rejected_spacing_keeps_the_smoothers_waypoints(self):
        m = geodex.Euclidean(2)

        def free(q):
            return True

        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01)
        even = geodex.smooth_path(m, free, _detour(), s)
        s.path_predicate = lambda path, n=len(even.path): len(path) != n
        r = geodex.smooth_path(m, free, _detour(), s)
        assert r.collision_free
        assert len(r.path) != len(even.path)

    def test_hooks_are_called(self):
        m = geodex.Euclidean(2)
        calls = {"proof": 0, "predicate": 0}

        def proof(a, b):
            calls["proof"] += 1
            return _disc(a) + _disc(b) > np.linalg.norm(b - a)

        def predicate(path):
            calls["predicate"] += 1
            return all(_disc(q) >= 0.3 for q in path)

        s = geodex.PathSmoothingSettings(
            collision_check_resolution=0.01, edge_provably_clear=proof, path_predicate=predicate
        )
        r = geodex.smooth_path(m, _valid, _detour(), s)
        assert r.collision_free
        assert calls["proof"] > 0 and calls["predicate"] > 0
        assert r.profile.edge_proofs > 0
        assert all(_disc(q) >= 0.3 - 1e-12 for q in r.path)

    def test_edge_travel_spaces_the_checks(self):
        m = geodex.Euclidean(2)
        plain = geodex.smooth_path(m, _valid, _detour(),
                                   geodex.PathSmoothingSettings(collision_check_resolution=0.01))
        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01,
                                         edge_travel=lambda a, b: 0.5 * np.linalg.norm(b - a))
        sparse = geodex.smooth_path(m, _valid, _detour(), s)
        assert sparse.collision_free
        assert sparse.profile.point_checks < plain.profile.point_checks
        s.edge_travel = lambda a, b: -1.0
        with pytest.raises(ValueError):
            geodex.smooth_path(m, _valid, _detour(), s)

    def test_edge_validator_rejecting_everything_leaves_the_input(self):
        m = geodex.Euclidean(2)
        s = geodex.PathSmoothingSettings(edge_validator=lambda a, b: False)
        r = geodex.smooth_path(m, _valid, _detour(), s)
        assert not r.collision_free
        assert r.first_invalid_index == 0

    def test_se2_and_sphere(self):
        se2 = geodex.SE2(1.0, 20.0, 0.5)
        pts = [(0, 0, 0), (1, 1.2, 0.5), (2, 1.4, 0), (3, 1.2, -0.5), (4, 0, 0)]
        path = [np.array(p, dtype=float) for p in pts]
        r = geodex.smooth_path(se2, lambda q: _disc(q, (2.0, 0.0), 0.8) > 0.0, path)
        assert r.collision_free
        sphere = geodex.Sphere()
        ts = np.linspace(0.0, 0.9 * math.pi, 9)
        arc = [np.array([math.cos(t), 0.8 * math.sin(t), 0.6 * math.sin(t)]) for t in ts]
        arc = [q / np.linalg.norm(q) for q in arc]
        rs = geodex.smooth_path(sphere, lambda q: q[2] < 0.9, arc)
        assert rs.collision_free
        np.testing.assert_allclose(np.linalg.norm(rs.path, axis=1), 1.0, atol=1e-9)


def _turns(manifold, path):
    """Turning angle at every interior point between -log(c, previous) and log(c, next),
    measured with the metric at c."""
    out = []
    for a, c, b in zip(path[:-2], path[1:-1], path[2:]):
        p, q = manifold.log(c, a), manifold.log(c, b)
        cos = -manifold.inner(c, p, q) / math.sqrt(manifold.inner(c, p, p) * manifold.inner(c, q, q))
        out.append(math.acos(max(-1.0, min(1.0, cos))))
    return np.array(out)


class TestCornerRounding:
    def test_corners_become_smooth(self):
        m = geodex.Euclidean(2)
        off = geodex.smooth_path(m, _valid, _detour(),
                                 geodex.PathSmoothingSettings(collision_check_resolution=0.01,
                                                              round_corners=False))
        on = geodex.smooth_path(m, _valid, _detour(),
                                geodex.PathSmoothingSettings(collision_check_resolution=0.01))
        assert off.profile.rounded_corners == 0
        assert on.collision_free and on.profile.fallback == 0
        assert on.profile.rounded_corners > 0
        assert on.profile.kept_corners == 0 and on.profile.cusps == 0
        assert on.profile.rounding_retries > 0  # some curves shrink next to the disc
        # The rounded turn spreads over many waypoints, within about 1.5 steps over the
        # disc's radius at each.
        assert _turns(m, off.path).max() > 0.1
        assert _turns(m, on.path).max() < 0.035
        assert on.length <= off.length
        assert _certified(on.path, _valid_within, 0.01)
        np.testing.assert_array_equal(on.path[0], _detour()[0])
        np.testing.assert_array_equal(on.path[-1], _detour()[-1])

    def test_rounding_is_deterministic(self):
        m = geodex.Euclidean(2)
        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01, output_spacing=0.05)
        a = geodex.smooth_path(m, _valid, _detour(), s)
        b = geodex.smooth_path(m, _valid, _detour(), s)
        np.testing.assert_array_equal(a.path, b.path)

    def test_a_reversal_stays_a_corner(self):
        # Driving forward and backing up along one line turns by 180 degrees under the
        # metric. The path predicate makes the base reach x = 1.95.
        se2 = geodex.SE2(wx=1.0, wy=50.0, wtheta=2.0)
        path = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
        s = geodex.PathSmoothingSettings(collision_check_resolution=0.01,
                                         path_predicate=lambda p: any(q[0] >= 1.95 for q in p))
        r = geodex.smooth_path(se2, lambda q: True, path, s)
        assert r.collision_free
        assert r.profile.cusps == 1
        assert _turns(se2, r.path).max() > 0.99 * math.pi

    def test_invalid_corner_settings_raise(self):
        m = geodex.Euclidean(2)
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError):
                geodex.smooth_path(m, _valid, _detour(),
                                   geodex.PathSmoothingSettings(corner_tolerance=bad))
        with pytest.raises(ValueError):
            geodex.smooth_path(m, _valid, _detour(),
                               geodex.PathSmoothingSettings(corner_max_angle=4.0))
        # Without rounding, smooth_path reads corner_tolerance for the spacing and not
        # corner_max_angle.
        geodex.smooth_path(m, _valid, _detour(),
                           geodex.PathSmoothingSettings(round_corners=False, corner_max_angle=4.0))
        with pytest.raises(ValueError):
            geodex.smooth_path(m, _valid, _detour(),
                               geodex.PathSmoothingSettings(round_corners=False,
                                                            corner_tolerance=-1.0))

    def test_leading_coordinates_keep_the_corner(self):
        # The first two coordinates must turn sharply at (1, 0), and the path keeps within 0.1
        # of its corner. With sharp_coordinates=2, the last two turn along a curve.
        m = geodex.Euclidean(4)
        path = [np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 1.0, 0.0]),
                np.array([1.0, 1.0, 1.0, 1.0])]
        lead = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])]

        def near(q, poly):
            return min(_distance_to_segment(q, a, b) for a, b in zip(poly[:-1], poly[1:]))

        def valid(q):
            return near(q[:2], lead) < 1e-3 and near(q, path) < 0.1

        def max_turn(p, cols):
            d = np.diff(np.asarray(p)[:, cols], axis=0)
            n = np.linalg.norm(d, axis=1)
            ok = (n[:-1] > 1e-9) & (n[1:] > 1e-9)
            c = (d[:-1] * d[1:]).sum(axis=1)[ok] / (n[:-1] * n[1:])[ok]
            return float(np.arccos(np.clip(c, -1.0, 1.0)).max())

        s = geodex.PathSmoothingSettings(collision_check_resolution=0.005)
        together = geodex.smooth_path(m, valid, path, s)
        s.sharp_coordinates = 2
        split = geodex.smooth_path(m, valid, path, s)
        assert together.collision_free and split.collision_free
        assert together.profile.split_corners == 0
        assert split.profile.split_corners >= 1
        assert max_turn(split.path, [0, 1]) > 0.45 * math.pi
        assert max_turn(split.path, [2, 3]) < 0.25 * max_turn(together.path, [2, 3])
        with pytest.raises(ValueError):
            geodex.smooth_path(m, valid, path, geodex.PathSmoothingSettings(sharp_coordinates=-1))

    def test_plan_rounds_corners_by_default(self):
        m = geodex.Euclidean(2)
        m.set_sampling_bounds(np.array([-3.0, -3.0]), np.array([3.0, 3.0]))
        start, goal = np.array([-2.0, 0.1]), np.array([2.0, -0.1])
        # At the same fine spacing, a rounded corner turns a little at every waypoint.
        s = geodex.PlanSettings(iterations=800, seed=3, collision_check_resolution=0.01,
                                smoothing=geodex.PathSmoothingSettings(output_spacing=0.005))
        rounded = geodex.plan(m, start, goal, _valid, settings=s)
        s.smoothing = geodex.PathSmoothingSettings(round_corners=False, output_spacing=0.005)
        sharp = geodex.plan(m, start, goal, _valid, settings=s)
        assert rounded.smoothed and sharp.smoothed
        assert _turns(m, rounded.path).max() < 0.25 * _turns(m, sharp.path).max()
        assert _certified(rounded.path, _valid_within, 0.01)


class TestPlanSmoothing:
    def test_settings_round_trip(self):
        s = geodex.PlanSettings(
            smooth=False, smoothing=geodex.PathSmoothingSettings(output_spacing=0.2)
        )
        assert s.smooth is False
        assert s.smoothing.output_spacing == pytest.approx(0.2)
        s.smoothing.output_spacing = 0.3
        assert s.smoothing.output_spacing == pytest.approx(0.3)
        g = geodex.planners.GreedyRRTstar(max_neighbors=12)
        assert g.max_neighbors == 12

    def test_plan_reports_smoothing(self):
        m = geodex.Euclidean(2)
        m.set_sampling_bounds(np.array([-3.0, -3.0]), np.array([3.0, 3.0]))
        start, goal = np.array([-2.0, 0.1]), np.array([2.0, -0.1])
        s = geodex.PlanSettings(iterations=800, seed=3, collision_check_resolution=0.01,
                                smooth=False)
        raw = geodex.plan(m, start, goal, _valid, settings=s)
        s.smooth = True
        smooth = geodex.plan(m, start, goal, _valid, settings=s)
        assert raw.solved and smooth.solved
        assert not raw.smoothed and smooth.smoothed
        assert smooth.smooth_ms > 0.0
        assert smooth.cost < raw.cost
        assert _certified(smooth.path, _valid_within, 0.01)


_SCENE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "smoothing"
    / "shelf_post.scene.yaml"
)


@pytest.mark.skipif(
    not (hasattr(geodex._geodex_core, "robots") and hasattr(geodex._geodex_core, "vamp")
         and hasattr(geodex._geodex_core, "plan")),
    reason="needs the robots and VAMP build",
)
def test_kinetic_energy_arm_path_holds_along_chords():
    """With a callable validity, plan() interpolates the kinetic-energy arm along curved
    edges. The plan checks the scene grown by half the sphere travel between two checks and
    by the travel of the corner tolerance. Every waypoint and chord of the returned path then
    clears the scene itself."""
    robot = geodex.robots.Panda()
    env = geodex.vamp.load_scene(str(_SCENE))
    checker = geodex.vamp.make_vamp_checker("panda", env)
    speed = geodex.vamp.sphere_speed("panda", env)
    grown = geodex.vamp.make_vamp_checker(
        "panda", geodex.vamp.pad_scene(env, speed * (0.5 * 0.01 + 1e-4)))

    def valid(q):
        return bool(checker.is_valid(np.asarray(q)))

    def padded(q):
        return bool(grown.is_valid(np.asarray(q)))

    start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    goal = np.array([1.2, -0.4, -0.3, -1.9, 0.2, 1.9, 0.5])
    for seed in (1, 3):
        s = geodex.PlanSettings(iterations=1000, seed=seed, collision_check_resolution=0.01)
        r = geodex.plan(robot, start, goal, padded, settings=s)
        assert r.solved
        assert _certified(r.waypoints, valid, 0.001), f"seed {seed}"

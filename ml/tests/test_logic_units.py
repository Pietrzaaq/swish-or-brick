import json
import math
import unittest

from brickorswish.logic import HoopFrame, ShotLogicConfig
from brickorswish.logic.kalman import AxisKalman
from brickorswish.logic.tracker import BallTracker, Candidate
from brickorswish.paths import CONFIG_V1


def load_cfg() -> ShotLogicConfig:
    return ShotLogicConfig.load(CONFIG_V1)


class ConfigTest(unittest.TestCase):
    def test_loads_shared_config(self):
        cfg = load_cfg()
        self.assertEqual(cfg.version, 1)
        self.assertIsInstance(cfg.max_coast_frames, int)

    def test_rejects_unknown_and_missing_keys(self):
        with open(CONFIG_V1, encoding="utf-8") as fh:
            data = json.load(fh)
        with self.assertRaises(ValueError):
            ShotLogicConfig.from_dict({**data, "typo_key": 1})
        data.pop("gate_chi2")
        with self.assertRaises(ValueError):
            ShotLogicConfig.from_dict(data)


class GeometryTest(unittest.TestCase):
    def test_axis_aligned(self):
        hoop = HoopFrame([900, 400], [1020, 400], [960, 415])
        self.assertEqual(hoop.radius_px, 60.0)
        self.assertEqual(hoop.to_hoop(960, 400), (0.0, 0.0))
        self.assertEqual(hoop.to_hoop(1020, 400), (1.0, 0.0))
        self.assertEqual(hoop.to_hoop(960, 340), (0.0, -1.0))  # above the rim is negative y
        self.assertAlmostEqual(hoop.ellipse_b, 0.25)

    def test_tilted_rim_does_not_tilt_the_axes(self):
        # the clicked rim is tilted (perspective / phone roll), but "down" stays the image vertical
        hoop = HoopFrame([100, 200], [180, 260], [130, 245])
        self.assertEqual(hoop.radius_px, 50.0)
        x, y = hoop.to_hoop(140, 330)  # straight below the rim centre (140, 230)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(math.hypot(*hoop.to_hoop(180, 260)), 1.0)  # rim edge is still 1 radius away
        for px, py in [(0, 0), (512.5, -33.0), (140, 230)]:
            back = hoop.to_pixels(*hoop.to_hoop(px, py))
            self.assertAlmostEqual(back[0], px)
            self.assertAlmostEqual(back[1], py)

    def test_measured_down_sets_the_axes(self):
        a = math.radians(10.0)
        down = (-math.sin(a), math.cos(a))  # gravity leaning 10 degrees left in the image
        hoop = HoopFrame([900, 400], [1020, 400], [960, 415], down)
        x, y = hoop.to_hoop(960 + 120 * down[0], 400 + 120 * down[1])
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 2.0)
        back = hoop.to_pixels(0.7, -3.2)
        self.assertEqual(tuple(round(v, 9) for v in hoop.to_hoop(*back)), (0.7, -3.2))
        self.assertAlmostEqual(hoop.ellipse_b, 0.25)  # the drawn rim does not depend on gravity

    def test_degenerate_calibration(self):
        with self.assertRaises(ValueError):
            HoopFrame([1, 1], [1, 1], [1, 2])


class KalmanTest(unittest.TestCase):
    def test_tracks_a_falling_ball(self):
        g, dt, v0 = 42.9, 1.0 / 30.0, -10.0
        kf = AxisKalman(0.0, 0.04, 900.0)
        for i in range(1, 25):
            t = i * dt
            kf.predict(dt, g, 400.0)
            kf.update(v0 * t + 0.5 * g * t * t, 0.04)
        t = 24 * dt
        self.assertAlmostEqual(kf.vel, v0 + g * t, delta=0.5)


class TrackerTest(unittest.TestCase):
    def test_ignores_static_distractor_and_follows_ball(self):
        cfg = load_cfg()
        tracker = BallTracker(cfg)
        state = None
        for i in range(20):
            t = i / 30.0
            ball = Candidate(-1.0 + 2.0 * t, -8.0 + 0.5 * 42.9 * t * t, 0.8)
            distractor = Candidate(4.0, -3.0, 0.7)
            state = tracker.step(t, [distractor, ball])
            self.assertEqual(state.track_id, 1)
        self.assertAlmostEqual(state.est_x, ball.x, delta=0.1)
        self.assertGreater(state.vy, 0.0)

    def test_coasts_then_drops(self):
        cfg = load_cfg()
        tracker = BallTracker(cfg)
        tracker.step(0.0, [Candidate(0.0, -5.0, 0.9)])
        tracker.step(1 / 30, [Candidate(0.0, -4.9, 0.9)])
        for k in range(cfg.max_coast_frames):
            state = tracker.step((2 + k) / 30, [])
            self.assertFalse(state.observed)
        self.assertIsNone(tracker.step((2 + cfg.max_coast_frames) / 30, []))

    def test_never_starts_outside_watch_region(self):
        tracker = BallTracker(load_cfg())
        self.assertIsNone(tracker.step(0.0, [Candidate(20.0, 0.0, 0.99)]))
        self.assertTrue(math.isfinite(tracker.step(0.1, [Candidate(0.0, -2.0, 0.5)]).est_x))

    def test_weak_detection_continues_but_does_not_start(self):
        cfg = load_cfg()
        weak = (cfg.ball_track_min_conf + cfg.ball_min_conf) / 2
        tracker = BallTracker(cfg)
        self.assertIsNone(tracker.step(0.0, [Candidate(0.0, -5.0, weak)]))
        tracker.step(1 / 30, [Candidate(0.0, -5.0, 0.9)])
        state = tracker.step(2 / 30, [Candidate(0.0, -4.9, weak)])
        self.assertTrue(state.observed)
        self.assertEqual((state.seen_x, state.seen_y), (0.0, -4.9))
        # while a shot is under way the caller lowers the start threshold
        self.assertTrue(BallTracker(cfg).step(0.0, [Candidate(0.0, -5.0, weak)], weak).observed)

    def test_restarts_when_the_prediction_runs_away(self):
        # ball falling fast, then stopped by the net: the detection is far from the free-fall prediction
        cfg = load_cfg()
        tracker = BallTracker(cfg)
        for k in range(4):
            tracker.step(k / 30, [Candidate(0.0, -3.0 + k * 0.9, 0.9)])
        hidden = cfg.restart_after_coast_frames + 2
        for k in range(hidden):
            state = tracker.step((4 + k) / 30, [])
            self.assertFalse(state.observed)
        self.assertGreater(state.est_y - 0.5, cfg.reacquire_radius)  # the prediction is already far below
        state = tracker.step((4 + hidden) / 30, [Candidate(-0.4, 0.5, 0.4)])
        self.assertTrue(state.observed)
        self.assertEqual(state.track_id, 2)
        self.assertEqual((state.seen_x, state.seen_y), (-0.4, 0.5))


class StateMachineTest(unittest.TestCase):
    """Rim decisions come from measured positions only."""

    @staticmethod
    def _ball(t, x, y, vy=10.0, observed=True, fx=None, fy=None):
        from brickorswish.logic.tracker import BallState

        return BallState(t, x if fx is None else fx, y if fy is None else fy, 0.0, vy, observed, 1, 9, x, y)

    def _approach(self, sm):
        sm.step(0.0, self._ball(0.0, 0.0, -5.0))
        self.assertEqual(sm.state, "approach")

    def test_prediction_never_crosses_the_rim(self):
        from brickorswish.logic.state_machine import ShotStateMachine

        sm = ShotStateMachine(load_cfg())
        self._approach(sm)
        sm.step(0.1, self._ball(0.1, 0.0, -0.5))
        sm.step(0.13, self._ball(0.13, 0.0, 1.0, observed=False))  # predicted below the rim
        self.assertEqual(sm.state, "approach")

    def test_ball_resting_on_rim_is_not_a_crossing_even_if_the_filter_sinks(self):
        from brickorswish.logic.state_machine import ShotStateMachine

        sm = ShotStateMachine(load_cfg())
        self._approach(sm)
        sm.step(0.1, self._ball(0.1, 1.0, -0.4, vy=-1.0))  # bounced up onto the rim
        self.assertEqual(sm.state, "rattle")
        sm.step(0.13, self._ball(0.13, 0.9, -0.35, fy=0.05))  # filter thinks it is below; detection says above
        self.assertEqual(sm.state, "rattle")

    def test_crossing_bridges_a_hidden_stretch_from_sightings(self):
        from brickorswish.logic.state_machine import ShotStateMachine

        cfg = load_cfg()
        sm = ShotStateMachine(cfg)
        self._approach(sm)
        sm.step(0.1, self._ball(0.1, 0.24, -0.83))
        gap = cfg.cross_max_gap_s / 2
        sm.step(0.1 + gap, self._ball(0.1 + gap, -0.40, 1.25))  # next sighting is already in the net
        self.assertEqual(sm.state, "pending_make")
        ev = sm.step(0.15 + gap, self._ball(0.15 + gap, -0.77, 1.82))
        self.assertEqual((ev.outcome, ev.reason), ("make", "through_rim"))

    def test_ball_leaving_the_net_sideways_is_a_make(self):
        from brickorswish.logic.state_machine import ShotStateMachine

        sm = ShotStateMachine(load_cfg())
        self._approach(sm)
        sm.step(0.1, self._ball(0.1, -0.41, -1.26))
        sm.step(0.3, self._ball(0.3, 0.99, 1.95))  # hidden in the rim, next seen at the bottom of the net
        self.assertEqual(sm.state, "pending_make")
        ev = sm.step(0.33, self._ball(0.33, 1.07, 2.39))  # still moving sideways as it leaves the net
        self.assertEqual((ev.outcome, ev.reason), ("make", "through_rim"))

    def test_rolling_on_the_rim_edge_then_down_the_net_is_a_make(self):
        # real shot (9-minute session, #39): the ball's box sits off-centre while it rolls on the rim
        from brickorswish.logic.state_machine import ShotStateMachine

        sm = ShotStateMachine(load_cfg())
        self._approach(sm)
        sm.step(0.1, self._ball(0.1, -0.32, -0.16))
        sm.step(0.3, self._ball(0.3, -1.0, 0.52))  # crossed at x -0.48
        self.assertEqual(sm.state, "pending_make")
        self.assertIsNone(sm.step(0.33, self._ball(0.33, -1.19, 0.74)))
        ev = sm.step(0.6, self._ball(0.6, -0.28, 2.16))  # out of the bottom of the net
        self.assertEqual((ev.outcome, ev.reason), ("make", "through_rim"))

    def test_edge_crossing_then_out_of_the_net_is_a_make(self):
        # real shot (9-minute session, #26): hidden while it rolls in; the straight line says x -0.61
        from brickorswish.logic.state_machine import ShotStateMachine

        sm = ShotStateMachine(load_cfg())
        self._approach(sm)
        sm.step(0.1, self._ball(0.1, -0.99, -1.45))
        ev = sm.step(0.3, self._ball(0.3, 0.0, 2.31))
        self.assertEqual((ev.outcome, ev.reason), ("make", "through_net"))

    def test_edge_crossing_then_beside_the_net_is_a_miss(self):
        from brickorswish.logic.state_machine import ShotStateMachine

        sm = ShotStateMachine(load_cfg())
        self._approach(sm)
        sm.step(0.1, self._ball(0.1, 1.0, -0.25))
        sm.step(0.13, self._ball(0.13, 1.2, 0.5))  # crossed at x ~ 1.07: over the edge
        self.assertEqual(sm.state, "rattle")
        ev = sm.step(0.2, self._ball(0.2, 1.45, 1.4))
        self.assertEqual((ev.outcome, ev.reason), ("miss", "rim_out"))


class StaticSpotTest(unittest.TestCase):
    """The rim's red bracket is detected as a ball next to every shot; a ball that never moves is ignored."""

    HOOP = {"image_width": 1920, "image_height": 1080, "left": [900, 400], "right": [1020, 400], "front": [960, 415]}

    @staticmethod
    def _det(x, y, conf=0.3):  # rim units -> pixels
        return {"cls": "ball", "conf": conf, "x": 960 + 60 * x, "y": 400 + 60 * y, "w": 40.0, "h": 40.0}

    def _session(self, bracket_seconds):
        frames = []
        for i in range(30 * 30):  # 30 s at 30 fps
            t = i / 30.0
            dets = [self._det(1.22, 0.48)] if int(t) < bracket_seconds and i % 30 < 3 else []
            if 10.0 <= t < 10.5:  # a ball flying past once
                dets.append(self._det(-3.0 + 6.0 * (t - 10.0), -2.0))
            frames.append({"i": i, "t": t, "dets": dets})
        return {"fps": 30.0, "frames": frames}

    def test_bracket_is_ignored_and_the_ball_is_not(self):
        from brickorswish.logic.pipeline import ball_candidates, static_spots

        cfg = load_cfg()
        hf = HoopFrame.from_dict(self.HOOP)
        static = static_spots(self._session(cfg.static_min_seconds), hf, cfg)
        self.assertTrue(static)
        self.assertEqual(ball_candidates([self._det(1.25, 0.45)], hf, cfg, static), [])  # same spot, a bit off
        self.assertEqual(len(ball_candidates([self._det(1.22, -1.5)], hf, cfg, static)), 1)  # ball above it
        self.assertEqual(len(ball_candidates([self._det(-2.0, -2.0)], hf, cfg, static)), 1)

    def test_a_spot_seen_in_few_seconds_is_not_static(self):
        from brickorswish.logic.pipeline import static_spots

        cfg = load_cfg()
        hf = HoopFrame.from_dict(self.HOOP)
        self.assertEqual(static_spots(self._session(cfg.static_min_seconds - 1), hf, cfg), frozenset())

    def test_balls_dropping_through_the_rim_centre_are_not_static(self):
        from brickorswish.logic.pipeline import static_spots

        cfg = load_cfg()
        frames = []
        for i in range(30 * 30):  # a make every second, each in the rim centre for one frame only
            t = i / 30.0
            dets = [self._det(0.0, -2.0 + 0.7 * (i % 30))] if i % 30 < 6 else []
            frames.append({"i": i, "t": t, "dets": dets})
        static = static_spots({"fps": 30.0, "frames": frames}, HoopFrame.from_dict(self.HOOP), cfg)
        self.assertEqual(static, frozenset())


if __name__ == "__main__":
    unittest.main()

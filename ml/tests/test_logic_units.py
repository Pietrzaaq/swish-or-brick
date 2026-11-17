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

    def test_tilted_roundtrip(self):
        hoop = HoopFrame([100, 200], [180, 260], [130, 245])
        x, y = hoop.to_hoop(180, 260)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)
        for px, py in [(0, 0), (512.5, -33.0), (140, 230)]:
            back = hoop.to_pixels(*hoop.to_hoop(px, py))
            self.assertAlmostEqual(back[0], px)
            self.assertAlmostEqual(back[1], py)

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
        self.assertAlmostEqual(kf.v, v0 + g * t, delta=0.5)


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
        self.assertAlmostEqual(state.x, ball.x, delta=0.1)
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
        self.assertTrue(math.isfinite(tracker.step(0.1, [Candidate(0.0, -2.0, 0.5)]).x))


if __name__ == "__main__":
    unittest.main()

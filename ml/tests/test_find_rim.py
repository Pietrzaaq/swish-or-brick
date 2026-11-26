import math
import tempfile
import unittest
from pathlib import Path

from brickorswish.detect.find_rim import Ellipse
from brickorswish.logic import HoopFrame

try:
    import cv2
    import numpy as np
except ImportError:  # the stdlib-only test run
    cv2 = None

W, H = 360, 640
BACKBOARD = (70, 90, 60)   # BGR: dark green
RIM_BACK = (40, 40, 120)   # dark red: the shadowed back half
RIM_FRONT = (40, 50, 200)  # bright red: the lit front half


def scene(cx, cy, a, b, theta_deg=-4.0, player_x=None):
    """Rim with a dark thin back and a bright thick front, plus red distractors: a bracket touching it and shorts."""
    img = np.full((H, W, 3), (200, 200, 200), np.uint8)  # sky
    cv2.rectangle(img, (int(cx - 3 * a), int(cy - 4 * a)), (int(cx + 3 * a), int(cy + 0.6 * a)), BACKBOARD, -1)
    c, ax = (int(cx * 16), int(cy * 16)), (int(a * 16), int(b * 16))
    cv2.ellipse(img, c, ax, theta_deg, 180, 360, RIM_BACK, 2, cv2.LINE_AA, 4)
    cv2.ellipse(img, c, ax, theta_deg, 0, 180, RIM_FRONT, 4, cv2.LINE_AA, 4)
    cv2.rectangle(img, (int(cx + 0.85 * a), int(cy)), (int(cx + a), int(cy + 0.8 * a)), RIM_FRONT, -1)  # bracket
    if player_x is not None:
        cv2.rectangle(img, (player_x, 520), (player_x + 40, 560), (30, 30, 210), -1)  # shorts
    return img


class EllipseToTapsTest(unittest.TestCase):
    def test_taps_give_back_the_ellipse(self):
        e = Ellipse(500.0, 440.0, 20.0, 9.0, math.radians(-5))
        hf = HoopFrame.from_dict(e.hoop(720, 1280))
        self.assertAlmostEqual(hf.cx, 500.0, places=1)
        self.assertAlmostEqual(hf.cy, 440.0, places=1)
        self.assertAlmostEqual(hf.radius_px, 20.0, places=1)
        self.assertAlmostEqual(hf.ellipse_b, 9.0 / 20.0, places=2)
        self.assertAlmostEqual(math.degrees(hf.rim_angle), -5.0, places=1)
        taps = e.hoop(720, 1280)
        self.assertLess(taps["left"][0], taps["right"][0])
        self.assertGreater(taps["front"][1], 440.0)  # the front of the rim is below its centre


@unittest.skipIf(cv2 is None, "needs opencv-python-headless + numpy (ml/requirements-local.txt)")
class FitTest(unittest.TestCase):
    def test_fits_the_whole_ring_not_the_bright_front_or_the_bracket(self):
        from brickorswish.detect.find_rim import RimConfig, fit_ring, redness, seeds

        cfg = RimConfig()
        red = redness(scene(180.0, 300.0, 24.0, 11.0, player_x=40))
        fits = [f for f in (fit_ring(red, s, cfg) for s in seeds(red, cfg)) if f is not None]
        best = max(fits, key=lambda f: f.score)
        self.assertLess(math.hypot(best.cx - 180.0, best.cy - 300.0), 1.0)
        self.assertLess(abs(best.a / 24.0 - 1.0), 0.05)
        self.assertLess(abs(best.b / 11.0 - 1.0), 0.15)
        self.assertTrue(all(f.score < best.score for f in fits if abs(f.cx - 180) > 30))  # the shorts lose


@unittest.skipIf(cv2 is None, "needs opencv-python-headless + numpy (ml/requirements-local.txt)")
class FindRimTest(unittest.TestCase):
    def test_finds_the_rim_after_the_phone_settles(self):
        from brickorswish.detect.find_rim import RimConfig, find_rim

        video = str(Path(tempfile.mkdtemp()) / "clip.avi")
        fps, n = 10.0, 200  # the phone is set up during the first 3 s (rim elsewhere and bigger)
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        for i in range(n):
            if i < 30:
                img = scene(200.0 - i, 200.0 + 3 * i, 30.0 - 0.2 * i, 13.0, player_x=40 + (i * 7) % 250)
            else:
                img = scene(170.0, 290.0, 24.0, 11.0, player_x=40 + (i * 7) % 250)
            writer.write(img)
        writer.release()

        preview = Path(video).with_name("rim.png")
        result = find_rim(video, RimConfig(every_s=2.0), log=lambda _: None, preview_path=preview)
        self.assertTrue(preview.exists())
        self.assertGreaterEqual(result["hoop_frame"], 30)
        hf = HoopFrame.from_dict(result["hoop"])
        self.assertLess(math.hypot(hf.cx - 170.0, hf.cy - 290.0), 1.0)
        self.assertLess(abs(hf.radius_px / 24.0 - 1.0), 0.05)
        self.assertGreaterEqual(result["steady_from_s"], 3.0)

        # Motion is measured from the found (later) frame, so the bigger rim during setup must not leak in.
        from brickorswish.detect.motion import MotionConfig
        from brickorswish.detect.stabilize import measure_motion

        checks, *_ = measure_motion(video, result["hoop"], result["hoop_frame"], MotionConfig(), log=lambda _: None)
        settled = [c for c in checks if c.t >= 3.5]
        self.assertTrue(settled)
        for c in settled:
            self.assertAlmostEqual(c.scale, 1.0, delta=0.01, msg=c)
            self.assertLess(math.hypot(c.dx, c.dy), 1.5, msg=c)


if __name__ == "__main__":
    unittest.main()

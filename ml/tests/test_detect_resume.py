"""An interrupted detection run continues where it stopped. Needs opencv + numpy (ml/.venv); skipped otherwise."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # the stdlib-only test run
    cv2 = None

HOOP = {"image_width": 64, "image_height": 48, "left": [26, 20], "right": [38, 20], "front": [32, 22]}
N = 40


class FakeModel:
    """One 'sports ball' per image, centred; can crash after a number of images."""
    card = {}
    label = "fake"

    def __init__(self, crash_after=None):
        self.seen = 0
        self.crash_after = crash_after

    def predict(self, images, threshold):
        from brickorswish.detect.onnx_model import Box

        out = []
        for img in images:
            if self.crash_after is not None and self.seen >= self.crash_after:
                raise RuntimeError("Colab stopped the run")
            self.seen += 1
            h, w = img.shape[:2]
            out.append([Box("sports ball", 0.9, (w / 2 - 3, h / 2 - 3, w / 2 + 3, h / 2 + 3))])
        return out


@unittest.skipIf(cv2 is None, "needs opencv-python-headless + numpy (ml/requirements-local.txt)")
class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.video = str(self.dir / "clip.avi")
        writer = cv2.VideoWriter(self.video, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
        for k in range(N):
            writer.write(np.full((48, 64, 3), k * 5 % 255, dtype=np.uint8))
        writer.release()

    def _run(self, model, checkpoint, key):
        from brickorswish.detect.video import detect_video
        from brickorswish.logic import ShotLogicConfig
        from brickorswish.paths import CONFIG_V1

        return detect_video(self.video, HOOP, ShotLogicConfig.load(CONFIG_V1), model, batch_size=4,
                            log=lambda _: None, checkpoint=checkpoint, resume_key=key, checkpoint_every=10)

    def test_continues_after_a_crash_and_matches_an_uninterrupted_run(self):
        whole = self._run(FakeModel(), None, None)
        self.assertEqual(len(whole["frames"]), N)

        ckpt = self.dir / "progress.json"
        key = {"video": "clip.avi", "settings": 1}
        with self.assertRaises(RuntimeError):
            self._run(FakeModel(crash_after=25), ckpt, key)
        saved = json.loads(ckpt.read_text(encoding="utf-8"))
        self.assertFalse(saved["complete"])
        self.assertEqual(saved["done"], 24)  # saved at batch boundaries 12 and 24; the crash came at image 25

        second = FakeModel()
        resumed = self._run(second, ckpt, key)
        self.assertEqual(second.seen, N - 24)      # only the frames not done before
        self.assertEqual(resumed["frames"], whole["frames"])

        third = FakeModel()
        self.assertEqual(self._run(third, ckpt, key)["frames"], whole["frames"])
        self.assertEqual(third.seen, 0)            # finished earlier: nothing to detect

    def test_other_settings_start_from_scratch(self):
        ckpt = self.dir / "progress.json"
        self._run(FakeModel(), ckpt, {"threshold": 0.15})
        fresh = FakeModel()
        self._run(fresh, ckpt, {"threshold": 0.3})
        self.assertEqual(fresh.seen, N)


if __name__ == "__main__":
    unittest.main()

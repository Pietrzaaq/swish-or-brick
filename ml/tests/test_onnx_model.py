"""Pre/post-processing only; matching rfdetr is checked on real frames by `export_onnx export` (Colab)."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2  # noqa: F401
    import numpy as np
except ImportError:  # the stdlib-only test run
    np = None


@unittest.skipIf(np is None, "needs numpy + opencv-python-headless (ml/requirements-local.txt)")
class PostprocessTest(unittest.TestCase):
    NAMES = {1: "person", 37: "sports ball"}

    def _logit(self, p):
        return float(np.log(p / (1.0 - p)))

    def test_top_pairs_classes_boxes_and_threshold(self):
        from brickorswish.detect.onnx_model import postprocess

        q, c = 4, 91
        logits = np.full((q, c), -10.0, dtype=np.float32)
        logits[2, 37] = self._logit(0.9)   # query 2: a ball
        logits[0, 1] = self._logit(0.6)    # query 0: a person
        logits[2, 1] = self._logit(0.2)    # same query, second class: kept as its own box (rfdetr does too)
        logits[3, 37] = self._logit(0.1)   # below the threshold
        dets = np.array([[0.5, 0.5, 0.2, 0.4], [0, 0, 0, 0], [0.25, 0.75, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]],
                        dtype=np.float32)
        boxes = postprocess(dets, logits, image_w=400, image_h=200, threshold=0.15, num_select=300,
                            class_names=self.NAMES)
        self.assertEqual([b.cls for b in boxes], ["sports ball", "person", "person"])
        self.assertAlmostEqual(boxes[0].conf, 0.9, places=5)
        self.assertEqual(tuple(round(v, 3) for v in boxes[0].xyxy), (80.0, 140.0, 120.0, 160.0))
        self.assertEqual(tuple(round(v, 3) for v in boxes[1].xyxy), (160.0, 60.0, 240.0, 140.0))

    def test_boxes_are_clamped_to_the_image(self):
        from brickorswish.detect.onnx_model import postprocess

        logits = np.full((1, 91), -10.0, dtype=np.float32)
        logits[0, 37] = self._logit(0.8)
        dets = np.array([[0.02, 0.98, 0.2, 0.2]], dtype=np.float32)
        (box,) = postprocess(dets, logits, 100, 100, 0.15, 300, self.NAMES)
        self.assertEqual(tuple(round(v, 3) for v in box.xyxy), (0.0, 88.0, 12.0, 100.0))

    def test_num_select_limits_pairs(self):
        from brickorswish.detect.onnx_model import postprocess

        logits = np.full((5, 91), self._logit(0.5), dtype=np.float32)
        dets = np.tile(np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32), (5, 1))
        self.assertEqual(len(postprocess(dets, logits, 10, 10, 0.15, 7, self.NAMES)), 7)

    def test_preprocess_stretches_and_normalises(self):
        from brickorswish.detect.onnx_model import preprocess

        rgb = np.zeros((30, 60, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        x = preprocess(rgb, (384, 384), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.assertEqual(x.shape, (1, 3, 384, 384))
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(float(x[0, 0, 10, 10]), (1.0 - 0.485) / 0.229, places=4)
        self.assertAlmostEqual(float(x[0, 1, 10, 10]), (0.0 - 0.456) / 0.224, places=4)

    def test_refuses_a_model_whose_checksum_does_not_match_its_card(self):
        from brickorswish.detect.onnx_model import OnnxDetector

        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "model.onnx"
            model.write_bytes(b"not the exported model")
            (Path(d) / "model.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
            with self.assertRaises(ValueError):
                OnnxDetector(model)

    def test_refuses_a_model_and_card_swapped_together_if_not_pinned(self):
        from brickorswish.detect.onnx_model import OnnxDetector, read_pins, sha256_of
        from brickorswish.paths import MODEL_PINS

        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "model.onnx"
            model.write_bytes(b"some other model")
            (Path(d) / "model.json").write_text(json.dumps({"sha256": sha256_of(model)}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "models.sha256"):
                OnnxDetector(model, pinned=read_pins(MODEL_PINS))

    def test_pins_file_has_the_exported_nano_model(self):
        from brickorswish.detect.onnx_model import read_pins
        from brickorswish.paths import MODEL_PINS

        self.assertIn("5434fae86eb34585a033d0e161b7d71ec9f5bfdcc80bbf3e42b37ec4c45fbe19", read_pins(MODEL_PINS))


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from brickorswish.logic import HoopFrame, ShotLogicConfig
from brickorswish.logic.pipeline import _cell
from brickorswish.paths import CONFIG_V1
from brickorswish.train import CLASSES
from brickorswish.train.dataset import build, collect, original_name, split_of, warnings
from brickorswish.train.drafts import add_image, draft_boxes, drop_session, new_coco, pick_frames, rim_box
from brickorswish.train.fit import train_kwargs

HOOP = {"image_width": 1920, "image_height": 1080, "left": [900, 400], "right": [1020, 400], "front": [960, 415]}
SIZE = (1920, 1080)


def ball(x, y, conf=0.8, cls="ball"):  # rim units -> calibration pixels; 63 px = 0.53 rim diameters
    return {"cls": cls, "conf": conf, "x": 960 + 60 * x, "y": 400 + 60 * y, "w": 63.0, "h": 63.0}


class DraftsTest(unittest.TestCase):
    def setUp(self):
        self.cfg = ShotLogicConfig.load(CONFIG_V1)

    def test_rim_box_covers_the_ellipse(self):
        self.assertEqual(tuple(round(v, 6) for v in rim_box(HoopFrame.from_dict(HOOP))), (897.0, 382.0, 1023.0, 418.0))

    def test_frames_around_shots_and_now_and_then(self):
        transforms = [(0.0, 0.0, 1.0)] * 300
        transforms[60] = None  # hoop lost
        shot = {"shot": 1, "outcome": "make", "t_start": 2.0, "t_resolve": 3.0}
        picked = pick_frames(300, 30.0, [shot], transforms)
        self.assertIs(picked[(51, "crop")], shot)
        self.assertIs(picked[(108, "crop")], shot)
        self.assertNotIn((60, "crop"), picked)
        self.assertIs(picked[(90, "crop")], shot)  # a shot frame stays a shot frame
        self.assertIsNone(picked[(180, "crop")])
        self.assertIn((0, "full"), picked)
        self.assertNotIn((52, "crop"), picked)

    def test_ball_in_the_net_during_a_make(self):
        dets = [ball(0.1, 1.0)]
        names = lambda in_make: [n for n, _ in draft_boxes(dets, HOOP, self.cfg, frozenset(), (0.0, 0.0, 1.0),
                                                          in_make, (0, 0), SIZE)]
        self.assertEqual(names(True), ["ball_in_net", "rim"])
        self.assertEqual(names(False), ["ball", "rim"])

    def test_only_usable_balls_and_clear_people(self):
        dets = [ball(0, -3, conf=0.2), ball(2, -3), {"cls": "person", "conf": 0.6, "x": 500, "y": 600, "w": 80,
                "h": 200}, {"cls": "person", "conf": 0.3, "x": 700, "y": 600, "w": 80, "h": 200},
                {"cls": "chair", "conf": 0.9, "x": 100, "y": 100, "w": 50, "h": 50}]
        got = draft_boxes(dets, HOOP, self.cfg, frozenset(), (0.0, 0.0, 1.0), False, (0, 0), SIZE)
        self.assertEqual([n for n, _ in got], ["ball", "person", "rim"])

    def test_bracket_is_left_unlabelled(self):
        static = frozenset({_cell(1.22, 0.48, self.cfg)})
        got = draft_boxes([ball(1.22, 0.48)], HOOP, self.cfg, static, (0.0, 0.0, 1.0), False, (0, 0), SIZE)
        self.assertEqual([n for n, _ in got], ["rim"])

    def test_boxes_follow_the_camera_into_the_crop(self):
        got = dict(draft_boxes([ball(0, -3)], HOOP, self.cfg, frozenset(), (10.0, -5.0, 1.0), False, (400, 0), SIZE))
        x0, y0, x1, y1 = got["ball"]
        self.assertAlmostEqual((x0 + x1) / 2, 960 + 10 - 400)
        self.assertAlmostEqual((y0 + y1) / 2, 400 - 180 - 5)

    def test_rerun_replaces_a_session(self):
        coco, manifest = new_coco(), {}
        for name, session in (("a_000001.jpg", "a"), ("b_000001.jpg", "b")):
            add_image(coco, name, 100, 100, [("ball", (1, 1, 20, 20)), ("rim", (30, 30, 60, 40))])
            manifest[name] = {"session": session}
        self.assertEqual(drop_session(coco, manifest, "a"), {"a_000001.jpg"})
        self.assertEqual([im["file_name"] for im in coco["images"]], ["b_000001.jpg"])
        self.assertEqual(len(coco["annotations"]), 2)
        self.assertEqual(list(manifest), ["b_000001.jpg"])
        self.assertEqual([c["name"] for c in coco["categories"]], list(CLASSES))


class DatasetTest(unittest.TestCase):
    def test_roboflow_names_are_mapped_back(self):
        self.assertEqual(original_name("train/9min_000123_jpg.rf.0a1b2c3d.jpg"), "9min_000123.jpg")
        self.assertEqual(original_name("images/9min_000123.jpg"), "9min_000123.jpg")

    def test_same_20_seconds_same_split(self):
        self.assertEqual(split_of("9min", 40.0), split_of("9min", 59.9))
        splits = {split_of("9min", 20.0 * k) for k in range(200)}
        self.assertEqual(splits, {"train", "valid", "test"})

    def _export(self, cats):
        return {"categories": [{"id": i, "name": n} for i, n in cats],
                "images": [{"id": 1, "file_name": "s_000001.jpg"}, {"id": 2, "file_name": "s_000002.jpg"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": cats[0][0], "bbox": [1, 2, 10, 10]},
                                {"id": 2, "image_id": 1, "category_id": cats[0][0], "bbox": [5, 5, 0, 3]}]}

    def test_classes_by_name_and_empty_images_kept(self):
        labels = collect([self._export([(7, "ball_in_net"), (3, "ball")])])
        self.assertEqual(labels, {"s_000001.jpg": [("ball_in_net", [1.0, 2.0, 10.0, 10.0])], "s_000002.jpg": []})

    def test_unknown_class_is_refused(self):
        with self.assertRaises(ValueError):
            collect([self._export([(1, "basketball")])])

    def test_build_writes_rfdetr_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp, "images")
            images.mkdir()
            labels, manifest = {}, {}
            for k in range(40):
                name = f"s_{k:06d}.jpg"
                (images / name).write_bytes(b"jpg")
                labels[name] = [("ball", [1.0, 1.0, 5.0, 5.0])]
                manifest[name] = {"session": "s", "t": 20.0 * k, "crop": [0, 0, 640, 640]}
            counts = build(labels, manifest, images, Path(tmp, "dataset"))
            self.assertEqual(sum(c["images"] for c in counts.values()), 40)
            for split in ("train", "valid", "test"):
                coco = json.loads(Path(tmp, "dataset", split, "_annotations.coco.json").read_text())
                self.assertEqual([c["name"] for c in coco["categories"]], list(CLASSES))
                self.assertEqual(len(coco["images"]), counts[split]["images"])
                for im in coco["images"]:
                    self.assertTrue(Path(tmp, "dataset", split, im["file_name"]).exists())
            self.assertTrue(any("ball_in_net" in w for w in warnings(counts)))

    def test_labelled_image_missing_from_drafts_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build({"x.jpg": []}, {}, Path(tmp), Path(tmp, "dataset"))


class FitSettingsTest(unittest.TestCase):
    def test_unknown_settings_are_dropped_and_training_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "last.ckpt").write_bytes(b"")
            args = Namespace(dataset="d", out=tmp, epochs=5, batch=4, grad_accum=4, lr=1e-4, patience=10)
            kwargs, dropped = train_kwargs(args, {"dataset_dir", "output_dir", "epochs", "batch_size", "resume"})
            self.assertEqual(kwargs["resume"], str(Path(tmp, "last.ckpt")))
            self.assertIn("class_names", dropped)
            self.assertNotIn("class_names", kwargs)
            self.assertEqual(train_kwargs(args, None)[0]["class_names"], list(CLASSES))


if __name__ == "__main__":
    unittest.main()

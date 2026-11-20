import unittest

from brickorswish.detect.crop import box_to_det, frame_time, watch_crop
from brickorswish.logic import ShotLogicConfig
from brickorswish.paths import CONFIG_V1

HOOP = {"image_width": 1920, "image_height": 1080, "left": [900, 400], "right": [1020, 400], "front": [960, 415]}


class WatchCropTest(unittest.TestCase):
    def setUp(self):
        self.cfg = ShotLogicConfig.load(CONFIG_V1)

    def test_square_crop_shifted_inside_image(self):
        # rim radius 60 px: watch zone x 600..1320, y -320..760; squared to 1080, then shifted down to y=0.
        self.assertEqual(watch_crop(HOOP, self.cfg), (420, 0, 1500, 1080))

    def test_unsquared_crop_clamped(self):
        self.assertEqual(watch_crop(HOOP, self.cfg, square=False), (600, 0, 1320, 1080))

    def test_huge_hoop_falls_back_to_full_frame(self):
        hoop = {**HOOP, "left": [500, 500], "right": [1400, 500], "front": [950, 550]}
        self.assertEqual(watch_crop(hoop, self.cfg), (0, 0, 1920, 1080))


class BoxToDetTest(unittest.TestCase):
    def test_converts_corners_to_centre_and_offsets(self):
        d = box_to_det("sports ball", 0.87654, (10, 20, 40, 60), offset=(100, 200))
        self.assertEqual(d, {"cls": "ball", "conf": 0.8765, "x": 125.0, "y": 240.0, "w": 30.0, "h": 40.0})

    def test_other_classes_keep_coco_name(self):
        self.assertEqual(box_to_det("person", 0.5, (0, 0, 1, 1))["cls"], "person")

    def test_frame_time(self):
        self.assertEqual(frame_time(0, 30.0), 0.0)
        self.assertEqual(frame_time(1, 30.0), 0.033333)


if __name__ == "__main__":
    unittest.main()

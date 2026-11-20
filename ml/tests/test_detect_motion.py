import unittest

from brickorswish.detect.motion import (
    IDENTITY,
    Check,
    MotionConfig,
    describe,
    frame_transforms,
    lost_spans,
    plan_segments,
    rim_centre,
    shift_hoop,
    stabilize_dets,
    to_calibration,
    to_frame,
)

CFG = MotionConfig()
R = 20.0  # rim radius px
EVERY = 15  # frames between checks (0.5 s at 30 fps)
HOOP = {"image_width": 720, "image_height": 1280, "left": [500, 300], "right": [540, 300], "front": [520, 306]}


def checks(*poses):
    """One check per entry, EVERY frames apart. None = hoop not found (low score)."""
    out = []
    for k, p in enumerate(poses):
        i = k * EVERY
        if p is None:
            out.append(Check(i, i / 30, 0.2, 0.0, 0.0, 1.0))
        else:
            out.append(Check(i, i / 30, 0.9, p[0], p[1], p[2] if len(p) > 2 else 1.0))
    return out


class PlanSegmentsTest(unittest.TestCase):
    def test_still_camera_is_one_identity_segment(self):
        segs = plan_segments(checks(*[(0, 0)] * 10), R, CFG)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].transform, IDENTITY)
        self.assertEqual((segs[0].from_i, segs[0].to_i), (0, 9 * EVERY))

    def test_setup_then_moved_camera(self):
        # like the full-resolution video: marked, phone handled (hoop not found), then steady 120 px lower
        poses = [(0, 0)] * 4 + [None] * 10 + [(-12, 122)] * 20
        c = checks(*poses)
        segs = plan_segments(c, R, CFG)
        self.assertEqual([s.transform for s in segs], [IDENTITY, (-12.0, 122.0, 1.0)])
        n = len(poses) * EVERY
        tr = frame_transforms(segs, n, EVERY, CFG)
        self.assertEqual(tr[0], IDENTITY)
        self.assertEqual(tr[-1], (-12.0, 122.0, 1.0))
        (a, b), = lost_spans(tr)
        self.assertGreater(a, 3 * EVERY)
        self.assertLess(b, 14 * EVERY)
        lines = describe(segs, tr, 30.0)
        self.assertEqual(len(lines), 3)
        self.assertIn("NOT found", lines[1])
        self.assertIn("(-12, +122)", lines[2])

    def test_one_blocked_check_does_not_break_tracking(self):
        poses = [(0, 0)] * 5 + [None] + [(0, 0)] * 5
        segs = plan_segments(checks(*poses), R, CFG)
        self.assertEqual(len(segs), 1)
        tr = frame_transforms(segs, len(poses) * EVERY, EVERY, CFG)
        self.assertEqual(lost_spans(tr), [])

    def test_single_false_match_is_ignored(self):
        poses = [(0, 0)] * 5 + [(200, -50)] + [(0, 0)] * 5
        segs = plan_segments(checks(*poses), R, CFG)
        self.assertEqual([s.transform for s in segs], [IDENTITY])

    def test_jitter_within_tolerance_stays_one_pose(self):
        poses = [(0, 0), (1.5, -1), (-2, 0.5), (0.5, 1), (1, -1.5), (0, 0)]
        segs = plan_segments(checks(*poses), R, CFG)
        self.assertEqual(len(segs), 1)

    def test_bump_mid_session_switches_pose_at_midpoint(self):
        poses = [(0, 0)] * 6 + [(30, 0)] * 6
        segs = plan_segments(checks(*poses), R, CFG)
        tr = frame_transforms(segs, len(poses) * EVERY, EVERY, CFG)
        self.assertEqual(lost_spans(tr), [])
        switch = next(i for i, t in enumerate(tr) if t != IDENTITY)
        self.assertTrue(5 * EVERY < switch <= 6 * EVERY)

    def test_never_found_is_all_lost(self):
        poses = [None] * 8
        tr = frame_transforms(plan_segments(checks(*poses), R, CFG), 8 * EVERY, EVERY, CFG)
        self.assertEqual(lost_spans(tr), [(0, 8 * EVERY - 1)])


class TransformTest(unittest.TestCase):
    def test_round_trip_with_shift_and_scale(self):
        c0, tr = (520.0, 300.0), (10.0, -25.0, 1.2)
        x, y = to_frame(600, 100, c0, tr)
        self.assertAlmostEqual(to_calibration(x, y, c0, tr)[0], 600)
        self.assertAlmostEqual(to_calibration(x, y, c0, tr)[1], 100)

    def test_shift_hoop_moves_rim_centre(self):
        moved = shift_hoop(HOOP, (-12, 122, 1.0))
        self.assertEqual(rim_centre(moved), (508.0, 422.0))
        self.assertEqual(moved["image_width"], 720)

    def test_stabilize_dets_maps_back_to_calibration_frame(self):
        tr = (-12.0, 122.0, 1.0)
        c0 = rim_centre(HOOP)
        x, y = to_frame(530, 250, c0, tr)
        (d,) = stabilize_dets([{"cls": "ball", "conf": 0.9, "x": x, "y": y, "w": 24, "h": 24}], c0, tr)
        self.assertEqual((d["x"], d["y"], d["w"], d["cls"]), (530.0, 250.0, 24.0, "ball"))


if __name__ == "__main__":
    unittest.main()

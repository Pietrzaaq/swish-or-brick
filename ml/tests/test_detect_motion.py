import unittest

from brickorswish.detect.motion import (
    IDENTITY,
    Check,
    MotionConfig,
    Segment,
    ball_size_check,
    compose,
    describe,
    frame_transforms,
    lost_spans,
    pick_scale,
    plan_segments,
    rim_centre,
    rim_scale_from_ball,
    scale_hoop,
    should_reanchor,
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
    """None = hoop not found (low score)."""
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
        # from the full-resolution video: phone handled (hoop not found), then steady 120 px lower
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


class ReanchorTest(unittest.TestCase):
    def test_settled_elsewhere_reanchors(self):
        # from the full-resolution video: phone settled 177 px lower, weak 0.78 matches
        c = [Check(150 + k * EVERY, 0, 0.78, -25, 177, 1.0) for k in range(3)]
        self.assertTrue(should_reanchor(c, IDENTITY, R, CFG))

    def test_strong_match_at_anchor_pose_keeps_template(self):
        c = [Check(k * EVERY, 0, 0.96, 0.5, -0.5, 1.0) for k in range(5)]
        self.assertFalse(should_reanchor(c, IDENTITY, R, CFG))

    def test_weak_match_at_anchor_pose_reanchors(self):
        c = [Check(k * EVERY, 0, 0.8, 0, 0, 1.0) for k in range(3)]
        self.assertTrue(should_reanchor(c, IDENTITY, R, CFG))

    def test_not_while_moving_or_doubtful_or_too_soon(self):
        moving = [Check(k * EVERY, 0, 0.85, 0, 40 * k, 1.0) for k in range(3)]
        doubtful = [Check(k * EVERY, 0, 0.65, -25, 177, 1.0) for k in range(3)]
        too_soon = [Check(k * EVERY, 0, 0.85, -25, 177, 1.0) for k in range(2)]
        for c in (moving, doubtful, too_soon):
            self.assertFalse(should_reanchor(c, IDENTITY, R, CFG))

    def test_compose_adds_shifts_and_multiplies_scales(self):
        base, rel = (-25.0, 177.0, 1.1), (3.0, -2.0, 0.9)
        dx, dy, s = compose(base, rel)
        self.assertEqual((dx, dy), (-22.0, 175.0))
        self.assertAlmostEqual(s, 0.99)
        # same point as mapping through the anchor frame step by step
        c0 = rim_centre(HOOP)
        anchor_c = to_frame(*c0, c0, base)
        p_anchor = to_frame(600, 250, c0, base)
        step = to_frame(*p_anchor, anchor_c, rel)
        direct = to_frame(600, 250, c0, (dx, dy, s))
        self.assertAlmostEqual(step[0], direct[0])
        self.assertAlmostEqual(step[1], direct[1])


class PickScaleTest(unittest.TestCase):
    def test_keeps_current_zoom_unless_clearly_better(self):
        cands = [(0.82, 1.0, "a"), (0.85, 0.8, "b")]
        self.assertEqual(pick_scale(cands, 1.0, 0.05)[2], "a")

    def test_switches_when_clearly_better(self):
        cands = [(0.70, 1.0, "a"), (0.92, 1.1, "b")]
        self.assertEqual(pick_scale(cands, 1.0, 0.05)[2], "b")

    def test_best_when_current_zoom_not_searched(self):
        self.assertEqual(pick_scale([(0.7, 0.9, "a"), (0.8, 1.1, "b")], 1.0, 0.05)[2], "b")
        self.assertIsNone(pick_scale([], 1.0, 0.05))


class BallSizeCheckTest(unittest.TestCase):
    def _dets(self, spans):
        """spans: (first, last, ball width px)."""
        c = rim_centre(HOOP)
        frames = []
        for a, b, w in spans:
            for i in range(a, b + 1):
                frames.append({"i": i, "t": i / 30, "dets": [{"cls": "ball", "conf": 0.9, "x": c[0], "y": c[1] - 10,
                                                             "w": w, "h": w}]})
        return {"fps": 30, "frames": frames}

    def test_wrong_zoom_in_one_segment_is_flagged(self):
        # rim width 40 px: ball ratios 0.51 and 0.60, the second after a wrong 0.80 zoom
        segs = [Segment(0, 99, 0, 0, 1.0, 10), Segment(100, 199, 0, 0, 0.8, 10)]
        out = ball_size_check(self._dets([(0, 99, 20.4), (100, 199, 24.0)]), HOOP, segs, CFG)
        self.assertEqual([r["ratio"] for r in out["segments"]], [0.51, 0.6])
        self.assertFalse(out["segments_agree"])
        self.assertIn("wrong zoom", out["warnings"][0])
        self.assertIn("0.60 of the rim width", out["warnings"][1])

    def test_consistent_real_ball_is_quiet(self):
        segs = [Segment(0, 99, 0, 0, 1.0, 10), Segment(100, 199, 0, 0, 1.0, 10)]
        out = ball_size_check(self._dets([(0, 99, 21.0), (100, 199, 21.5)]), HOOP, segs, CFG)
        self.assertEqual(out["warnings"], [])

    def test_bad_taps_are_flagged(self):
        segs = [Segment(0, 99, 0, 0, 1.0, 10)]
        out = ball_size_check(self._dets([(0, 99, 36.0)]), HOOP, segs, CFG)  # ball 0.9 of the rim: taps too narrow
        self.assertEqual(len(out["warnings"]), 1)
        self.assertIn("check the hoop taps", out["warnings"][0])

    def test_rim_on_the_net_hooks_is_rescaled_to_the_ball(self):
        segs = [Segment(0, 99, 0, 0, 1.0, 10)]
        out = ball_size_check(self._dets([(0, 99, 24.44)]), HOOP, segs, CFG)  # 0.611: the 9-minute session
        k = rim_scale_from_ball(out, CFG)
        self.assertAlmostEqual(k, 0.611 / 0.53, places=3)
        again = ball_size_check(self._dets([(0, 99, 24.44)]), scale_hoop(HOOP, k), segs, CFG)
        self.assertAlmostEqual(again["ratio"], 0.53, places=2)
        self.assertEqual(again["warnings"], [])

    def test_rim_is_kept_when_the_ball_roughly_agrees_or_cannot_be_trusted(self):
        one = [Segment(0, 99, 0, 0, 1.0, 10)]
        close = ball_size_check(self._dets([(0, 99, 22.4)]), HOOP, one, CFG)  # 0.56: box a bit large
        self.assertIsNone(rim_scale_from_ball(close, CFG))
        few = ball_size_check(self._dets([(0, 29, 24.44)]), HOOP, one, CFG)
        self.assertIsNone(rim_scale_from_ball(few, CFG))
        far = ball_size_check(self._dets([(0, 99, 36.0)]), HOOP, one, CFG)  # 0.9: not the rim at all
        self.assertIsNone(rim_scale_from_ball(far, CFG))
        two = [Segment(0, 99, 0, 0, 1.0, 10), Segment(100, 199, 0, 0, 0.8, 10)]
        zoom = ball_size_check(self._dets([(0, 99, 24.44), (100, 199, 30.0)]), HOOP, two, CFG)
        self.assertIsNone(rim_scale_from_ball(zoom, CFG))

    def test_scale_hoop_keeps_the_centre(self):
        big = scale_hoop(HOOP, 1.5)
        self.assertEqual(rim_centre(big), rim_centre(HOOP))
        self.assertEqual(big["right"][0] - big["left"][0], 60.0)


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

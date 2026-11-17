"""Runs process() over every case in shared/fixtures/cases. The Dart port runs the same cases."""

import json
import unittest

from brickorswish.logic import ShotLogicConfig, process
from brickorswish.paths import CONFIG_V1, FIXTURE_CASES
from brickorswish.tools import gen_synthetic, update_golden

FRAME_TOL = 1.0 / 30.0 + 1e-9


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class FixtureCasesTest(unittest.TestCase):
    def setUp(self):
        self.cfg = ShotLogicConfig.load(CONFIG_V1)
        self.cases = sorted(p for p in FIXTURE_CASES.iterdir() if p.is_dir())
        self.assertTrue(self.cases, "no fixture cases found")

    def test_outcomes_match_expected(self):
        for case in self.cases:
            with self.subTest(case=case.name):
                got = process(_load(case / "detections.json"), _load(case / "hoop.json"), self.cfg)["shots"]
                expected = _load(case / "expected_events.json")["shots"]
                self.assertEqual([s["outcome"] for s in got], [s["outcome"] for s in expected])
                for g, e in zip(got, expected):
                    self.assertLessEqual(g["t_start"], g["t_resolve"])
                    for key in ("t_start", "t_resolve"):
                        if key in e:
                            self.assertAlmostEqual(g[key], e[key], delta=FRAME_TOL)
                    if "reason" in e:
                        self.assertEqual(g["reason"], e["reason"])

    def test_deterministic(self):
        for case in self.cases:
            with self.subTest(case=case.name):
                det, hoop = _load(case / "detections.json"), _load(case / "hoop.json")
                self.assertEqual(process(det, hoop, self.cfg), process(det, hoop, self.cfg))


class GoldenEventsUpToDateTest(unittest.TestCase):
    """golden_events.json is what the Dart port is checked against."""

    def test_golden_matches_current_logic(self):
        cfg = ShotLogicConfig.load(CONFIG_V1)
        for case in sorted(p for p in FIXTURE_CASES.iterdir() if p.is_dir()):
            with self.subTest(case=case.name):
                fresh = json.loads(json.dumps(update_golden.golden_for(case, cfg)))
                self.assertEqual(fresh, _load(case / "golden_events.json"),
                                 "run: py -m brickorswish.tools.update_golden")


class SyntheticFixturesUpToDateTest(unittest.TestCase):
    def test_committed_files_match_generator(self):
        for name, (builder, noise, seed) in gen_synthetic.CASES.items():
            with self.subTest(case=name):
                shots, _ = builder()
                duration = max(s.t0 for s in shots) + 3.0
                fresh = json.loads(json.dumps(gen_synthetic.render(shots, noise, seed, duration)))
                committed = _load(FIXTURE_CASES / name / "detections.json")
                self.assertEqual(fresh, committed, "run: py -m brickorswish.tools.gen_synthetic")


if __name__ == "__main__":
    unittest.main()

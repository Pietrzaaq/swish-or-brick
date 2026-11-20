import unittest

from brickorswish.eval.score import format_report, parse_verdict, score


def _events(*outcomes):
    return {"shots": [{"shot": i + 1, "outcome": o, "t_start": float(i), "t_resolve": i + 0.5, "reason": "r"}
                      for i, o in enumerate(outcomes)]}


class ParseVerdictTest(unittest.TestCase):
    def test_letters_spaces_and_case(self):
        self.assertEqual(parse_verdict("X o, -"), ["miss", "make", "not_a_shot"])

    def test_rejects_unknown_letters(self):
        with self.assertRaises(ValueError):
            parse_verdict("OXM")
        with self.assertRaises(ValueError):
            parse_verdict("-", allow_dash=False)


class ScoreTest(unittest.TestCase):
    def test_all_correct(self):
        r = score(_events("miss", "make", "make"), "XOO")
        self.assertEqual(r["metrics"], {"attempt_recall": 1.0, "attempt_precision": 1.0, "accuracy": 1.0,
                                        "uncertain_rate": 0.0})

    def test_wrong_uncertain_false_and_missed(self):
        # app: make, miss, uncertain, make(not a shot); you: make, make, miss, -; plus one undetected miss
        r = score(_events("make", "miss", "uncertain", "make"), "OOX-", missed="X")
        c = r["counts"]
        self.assertEqual((c["real"], c["false_attempts"], c["missed"], c["uncertain"], c["decided"], c["correct"]),
                         (3, 1, 1, 1, 2, 1))
        self.assertEqual(r["metrics"], {"attempt_recall": 0.75, "attempt_precision": 0.75, "accuracy": 0.5,
                                        "uncertain_rate": 0.3333})
        self.assertEqual(r["confusion"], {"make -> make": 1, "make -> miss": 1, "miss -> uncertain": 1})
        report = format_report(r)
        self.assertIn("WRONG", report)
        self.assertIn("FALSE ATTEMPT", report)
        self.assertIn("FAIL", report)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            score(_events("make", "make"), "O")

    def test_nothing_detected(self):
        r = score(_events(), "", missed="OO")
        self.assertEqual(r["metrics"]["attempt_recall"], 0.0)
        self.assertIsNone(r["metrics"]["accuracy"])


if __name__ == "__main__":
    unittest.main()

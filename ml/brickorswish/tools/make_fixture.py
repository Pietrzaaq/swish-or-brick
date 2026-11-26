"""Turn a Colab results folder into a real-video fixture case in shared/fixtures/cases/.

    python -m brickorswish.tools.make_fixture RESULTS_DIR real_<name> [--truth "XOOOOOO"]
        [--known-wrong 9,17 --why "falls in front of the rim"]

Refuses when the logic does not get every shot right: a fixture is a regression test, so it must pass when added.

--known-wrong lists exactly the shots the logic gets wrong: tests skip them against the truth, but
golden_events.json still pins what the logic says, so any change there still shows up.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from ..eval.score import load_json, score
from ..logic import ShotLogicConfig, process
from ..paths import CONFIG_V1, FIXTURE_CASES
from .update_golden import golden_for

NAME = re.compile(r"^real_[a-z0-9_]+$")


def make_fixture(results_dir: str | Path, name: str, truth: str | None = None, force: bool = False,
                 known_wrong: tuple[int, ...] = (), why: str = "") -> Path:
    if not NAME.match(name):
        raise ValueError("name must look like real_<lowercase_letters_digits_underscores>")
    src = Path(results_dir)
    dst = FIXTURE_CASES / name
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} exists (use --force to replace it)")
    labels = load_json(src / "labels.json") if truth is None else {"truth": truth, "missed": ""}

    cfg = ShotLogicConfig.load(CONFIG_V1)
    detections, hoop = load_json(src / "detections.json"), load_json(src / "hoop.json")
    events = process(detections, hoop, cfg)  # recompute: the folder's events.json may be from older logic
    result = score(events, labels["truth"], labels.get("missed", ""))
    c = result["counts"]
    wrong = tuple(r["shot"] for r in result["shots"] if r["predicted"] != r["truth"])
    if wrong != tuple(sorted(known_wrong)) or c["missed"]:
        raise ValueError("the current logic does not get every shot right on this video "
                         f"(wrong: {list(wrong)}, declared known wrong: {sorted(known_wrong)}, "
                         f"missed {c['missed']}); not a passing fixture")
    if known_wrong and not why:
        raise ValueError("say why the known-wrong shots are wrong (--why)")

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "detections.json", dst / "detections.json")
    shutil.copyfile(src / "hoop.json", dst / "hoop.json")
    expected = {"shots": [{"shot": r["shot"], "outcome": r["truth"],
                           **({"known_wrong": why} if r["shot"] in known_wrong else {})}
                          for r in result["shots"]]}
    for file, data in (("expected_events.json", expected), ("golden_events.json", golden_for(dst, cfg))):
        with open(dst / file, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=1)
            fh.write("\n")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("name", help="case name, e.g. real_session01_576p")
    ap.add_argument("--truth", help="verdict per detected shot (O/X); default: labels.json in results_dir")
    ap.add_argument("--force", action="store_true", help="replace an existing case")
    ap.add_argument("--known-wrong", default="", help="shots the logic is known to get wrong, e.g. 9,17")
    ap.add_argument("--why", default="", help="why those shots are wrong (stored with them)")
    args = ap.parse_args()
    known = tuple(int(s) for s in args.known_wrong.split(",") if s.strip())
    print("wrote", make_fixture(args.results_dir, args.name, args.truth, args.force, known, args.why))


if __name__ == "__main__":
    main()

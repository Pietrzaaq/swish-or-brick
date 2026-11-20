"""Turn a Colab results folder into a real-video fixture case in shared/fixtures/cases/.

    python -m brickorswish.tools.make_fixture RESULTS_DIR real_<name> [--truth "XOOOOOO"]

Copies detections.json and hoop.json, writes expected_events.json from your verdict (default:
labels.json in the folder) and golden_events.json from the current logic. Refuses when the logic
does not get every shot right: a fixture is a regression test, so it must pass when added.
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


def make_fixture(results_dir: str | Path, name: str, truth: str | None = None, force: bool = False) -> Path:
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
    if c["correct"] != c["detected"] or c["missed"]:
        raise ValueError("the current logic does not get every shot right on this video "
                         f"({result['confusion']}, missed {c['missed']}); not a passing fixture")

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "detections.json", dst / "detections.json")
    shutil.copyfile(src / "hoop.json", dst / "hoop.json")
    expected = {"shots": [{"shot": r["shot"], "outcome": r["truth"]} for r in result["shots"]]}
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
    args = ap.parse_args()
    print("wrote", make_fixture(args.results_dir, args.name, args.truth, args.force))


if __name__ == "__main__":
    main()

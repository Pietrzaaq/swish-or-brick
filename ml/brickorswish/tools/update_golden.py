"""Write golden_events.json (Python reference output) into every fixture case.

The Dart port must reproduce these within one frame. Re-run after any change to the
logic or the config, review the diff, and commit it together with the Dart change:

    py -m brickorswish.tools.update_golden          (from ml/)
"""

import json

from ..logic import ShotLogicConfig, process
from ..paths import CONFIG_V1, FIXTURE_CASES


def golden_for(case_dir, cfg: ShotLogicConfig) -> dict:
    with open(case_dir / "detections.json", encoding="utf-8") as fh:
        detections = json.load(fh)
    with open(case_dir / "hoop.json", encoding="utf-8") as fh:
        hoop = json.load(fh)
    return process(detections, hoop, cfg)


def main() -> None:
    cfg = ShotLogicConfig.load(CONFIG_V1)
    for case_dir in sorted(p for p in FIXTURE_CASES.iterdir() if p.is_dir()):
        with open(case_dir / "golden_events.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(golden_for(case_dir, cfg), fh, indent=1)
            fh.write("\n")
        print("wrote", case_dir.name)


if __name__ == "__main__":
    main()

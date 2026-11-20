"""Scores events.json against your verdict. Standard library only.

    python -m brickorswish.eval.score RESULTS_DIR --truth "XOOOOOO" [--missed "X"]

Verdict = one letter per DETECTED shot, in order: O = make, X = miss, - = not a real shot attempt.
Missed  = one letter (O/X) per real attempt the logic did not detect at all.
Writes labels.json (your verdict) and score.json (metrics) into RESULTS_DIR and prints a report.
Targets are the Milestone 1 exit criteria in docs/implementation-analysis.md §8.3.
"""

import argparse
import json
from pathlib import Path

LETTERS = {"O": "make", "X": "miss"}
NOT_A_SHOT = "not_a_shot"
TARGETS = {"attempt_recall": (">=", 0.95), "accuracy": (">=", 0.90), "uncertain_rate": ("<=", 0.10)}


def parse_verdict(text: str, allow_dash: bool = True) -> list[str]:
    """'X O o -' -> ['miss', 'make', 'make', 'not_a_shot']. Spaces and commas are ignored."""
    out = []
    for ch in text.upper():
        if ch.isspace() or ch == ",":
            continue
        if ch in LETTERS:
            out.append(LETTERS[ch])
        elif ch == "-" and allow_dash:
            out.append(NOT_A_SHOT)
        else:
            allowed = "O (make), X (miss)" + (", - (not a shot)" if allow_dash else "")
            raise ValueError(f"unexpected {ch!r} in {text!r}; use {allowed}")
    return out


def score(events: dict, truth: str, missed: str = "") -> dict:
    shots = events["shots"]
    verdicts = parse_verdict(truth)
    if len(verdicts) != len(shots):
        raise ValueError(f"{len(shots)} shots were detected but the verdict has {len(verdicts)} letters")
    counts = {"detected": len(shots), "real": 0, "false_attempts": 0, "missed": len(parse_verdict(missed, False)),
              "uncertain": 0, "decided": 0, "correct": 0}
    confusion: dict[str, int] = {}
    rows = []
    for shot, verdict in zip(shots, verdicts):
        rows.append({"shot": shot["shot"], "t_start": shot["t_start"], "predicted": shot["outcome"],
                     "reason": shot["reason"], "truth": verdict})
        if verdict == NOT_A_SHOT:
            counts["false_attempts"] += 1
            continue
        counts["real"] += 1
        if shot["outcome"] == "uncertain":
            counts["uncertain"] += 1
        else:
            counts["decided"] += 1
            counts["correct"] += shot["outcome"] == verdict
        key = f"{verdict} -> {shot['outcome']}"
        confusion[key] = confusion.get(key, 0) + 1
    return {"truth": truth, "missed": missed, "counts": counts, "metrics": metrics(counts),
            "confusion": dict(sorted(confusion.items())), "shots": rows}


def metrics(counts: dict) -> dict:
    """Rates from counts; None when there is nothing to divide by. Sum counts over sessions, then call this."""
    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    return {
        "attempt_recall": ratio(counts["real"], counts["real"] + counts["missed"]),
        "attempt_precision": ratio(counts["real"], counts["detected"]),
        "accuracy": ratio(counts["correct"], counts["decided"]),
        "uncertain_rate": ratio(counts["uncertain"], counts["real"]),
    }


def format_report(result: dict) -> str:
    lines = []
    for r in result["shots"]:
        if r["truth"] == NOT_A_SHOT:
            mark = "FALSE ATTEMPT"
        elif r["predicted"] == "uncertain":
            mark = "uncertain"
        else:
            mark = "ok" if r["predicted"] == r["truth"] else "WRONG"
        lines.append(f"  shot {r['shot']:>2} at {r['t_start']:6.1f}s  app: {r['predicted']:<9} "
                     f"you: {r['truth']:<10} {mark:<13} ({r['reason']})")
    c = result["counts"]
    if c["missed"]:
        lines.append(f"  + {c['missed']} real attempt(s) not detected: {result['missed']}")
    lines.append(f"{c['real'] + c['missed']} real attempts, {c['detected']} detected")
    for name, value in result["metrics"].items():
        target = TARGETS.get(name)
        verdict = ""
        if target and value is not None:
            op, limit = target
            ok = value >= limit if op == ">=" else value <= limit
            verdict = f"  target {op} {limit:.0%}: {'PASS' if ok else 'FAIL'}"
        shown = "n/a" if value is None else f"{value:.0%}"
        lines.append(f"  {name:<18} {shown:>5}{verdict}")
    if c["real"] + c["missed"] < 40:
        lines.append("  (small sample: the exit criteria are meant for sessions of 40+ shots)")
    return "\n".join(lines)


def save(results_dir: str | Path, result: dict) -> None:
    d = Path(results_dir)
    _dump(d / "labels.json", {"truth": result["truth"], "missed": result["missed"]})
    _dump(d / "score.json", result)


def _dump(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="folder with events.json (a Colab results folder)")
    ap.add_argument("--truth", help="verdict per detected shot; default: labels.json in the folder")
    ap.add_argument("--missed", default=None, help="outcomes of undetected real attempts, e.g. 'X'")
    args = ap.parse_args()
    d = Path(args.results_dir)
    labels = load_json(d / "labels.json") if args.truth is None else {}
    truth = args.truth if args.truth is not None else labels["truth"]
    missed = args.missed if args.missed is not None else labels.get("missed", "")
    result = score(load_json(d / "events.json"), truth, missed)
    save(d, result)
    print(format_report(result))


if __name__ == "__main__":
    main()

"""video + hoop.json -> detections.json, events.json, annotated.mp4 in one output folder.

    python -m brickorswish.detect.run_video VIDEO --hoop hoop.json --out OUT_DIR

Needs `opencv-python` and `rfdetr`. Cloud only (Colab); see ml/notebooks/detect_video.ipynb.
"""

import argparse
import json
from pathlib import Path

from ..logic import ShotLogicConfig, process
from ..paths import CONFIG_V1


def run(
    video: str | Path,
    hoop: dict,
    out_dir: str | Path,
    model=None,
    crop: str = "hoop",
    threshold: float = 0.15,
    batch_size: int = 8,
    annotate: bool = True,
    log=print,
) -> dict:
    from .annotate import annotate_video
    from .video import detect_video, load_model

    cfg = ShotLogicConfig.load(CONFIG_V1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "hoop.json", hoop)

    log("detecting...")
    detections, box = detect_video(video, hoop, cfg, model or load_model(), crop, threshold, batch_size, log)
    _dump(out / "detections.json", detections)

    events = process(detections, hoop, cfg)
    _dump(out / "events.json", events)
    for s in events["shots"]:
        log(f"  shot {s['shot']}: {s['outcome']:<9} {s['reason']:<14} t={s['t_start']:.2f}-{s['t_resolve']:.2f}s")
    log(f"{len(events['shots'])} shots")

    if annotate:
        log("annotating video...")
        annotate_video(video, out / "annotated.mp4", hoop, detections, cfg, box)
    log(f"written to {out}")
    return events


def _dump(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--hoop", required=True, help="hoop.json (3-click calibration)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--crop", choices=["hoop", "full"], default="hoop")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video")
    args = ap.parse_args()
    with open(args.hoop, encoding="utf-8") as fh:
        hoop = json.load(fh)
    run(args.video, hoop, args.out, crop=args.crop, threshold=args.threshold, batch_size=args.batch,
        annotate=not args.no_video)


if __name__ == "__main__":
    main()

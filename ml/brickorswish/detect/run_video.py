"""video + hoop.json -> camera_motion.json, detections.json, events.json, annotated.mp4 in one folder.

    python -m brickorswish.detect.run_video VIDEO --hoop hoop.json --out OUT_DIR [--hoop-frame N]

hoop.json holds your taps on frame N (--hoop-frame). The camera may move afterwards: the hoop is
re-found every 0.5 s (stabilize.py), detections are mapped back to frame N's pixels, and spans where
the hoop cannot be found are not counted. --no-stabilize assumes a still camera.

Needs `opencv-python` and `rfdetr`. Cloud only (Colab); see ml/notebooks/detect_video.ipynb.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..logic import HoopFrame, ShotLogicConfig, process
from ..paths import CONFIG_V1
from .motion import MotionConfig, Segment, ball_size_check, describe, frame_transforms, lost_spans, plan_segments


def run(
    video: str | Path,
    hoop: dict,
    out_dir: str | Path,
    model=None,
    crop: str = "hoop",
    threshold: float = 0.15,
    batch_size: int = 8,
    annotate: bool = True,
    hoop_frame: int = 0,
    stabilize: bool = True,
    motion_cfg: MotionConfig | None = None,
    log=print,
) -> dict:
    from .annotate import annotate_video
    from .video import detect_video, load_model

    cfg = ShotLogicConfig.load(CONFIG_V1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "hoop.json", hoop)

    transforms = None
    mcfg = motion_cfg or MotionConfig()
    if stabilize:
        transforms, motion = _camera_motion(video, hoop, hoop_frame, mcfg, out, log)

    log("detecting...")
    detections = detect_video(video, hoop, cfg, model or load_model(), crop, threshold, batch_size, transforms, log)
    _dump(out / "detections.json", detections)

    if stabilize:
        sizes = ball_size_check(detections, hoop, [Segment(**s) for s in motion["segments"]], mcfg)
        for row in sizes["segments"]:
            if row["ratio"] is not None:
                log(f"  ball / rim width {row['ratio']:.2f} (real ball {sizes['expected']}) in frames "
                    f"{row['from_i']}-{row['to_i']}, {row['n']} balls near the rim")
        for w in sizes["warnings"]:
            log("  WARNING: " + w)
        motion["ball_size"] = sizes
        _dump(out / "camera_motion.json", motion)

    events = process(detections, hoop, cfg)
    _dump(out / "events.json", events)
    for s in events["shots"]:
        log(f"  shot {s['shot']}: {s['outcome']:<9} {s['reason']:<14} t={s['t_start']:.2f}-{s['t_resolve']:.2f}s")
    log(f"{len(events['shots'])} shots")

    if annotate:
        log("annotating video...")
        annotate_video(video, out / "annotated.mp4", hoop, detections, cfg, crop, transforms)
    log(f"written to {out}")
    return events


def _camera_motion(video, hoop: dict, hoop_frame: int, mcfg: MotionConfig, out: Path, log) -> tuple[list, dict]:
    from .stabilize import measure_motion

    log("tracking the hoop (camera motion)...")
    checks, n_frames, fps, every, anchors = measure_motion(video, hoop, hoop_frame, mcfg, log)
    r = HoopFrame.from_dict(hoop).radius_px
    segments = plan_segments(checks, r, mcfg)
    transforms = frame_transforms(segments, n_frames, every, mcfg)
    timeline = describe(segments, transforms, fps)
    for line in timeline:
        log("  " + line)
    lost = sum(tr is None for tr in transforms)
    if lost:
        log(f"  WARNING: hoop not found in {lost / fps:.0f} s of {n_frames / fps:.0f} s; shots there are not counted. "
            "If that is most of the video, tap the hoop on a later frame (HOOP_AT_S).")
    motion = {
        "hoop_frame": hoop_frame,
        "check_every_frames": every,
        "timeline": timeline,
        "segments": [asdict(s) for s in segments],
        "lost": [{"from_i": a, "to_i": b, "from_t": round(a / fps, 3), "to_t": round((b + 1) / fps, 3)}
                 for a, b in lost_spans(transforms)],
        "anchors": anchors,
        "checks": [asdict(c) for c in checks],
    }
    _dump(out / "camera_motion.json", motion)
    return transforms, motion


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
    ap.add_argument("--hoop-frame", type=int, default=0, help="frame index the hoop taps were made on")
    ap.add_argument("--no-stabilize", action="store_true", help="assume the camera never moves")
    args = ap.parse_args()
    with open(args.hoop, encoding="utf-8") as fh:
        hoop = json.load(fh)
    run(args.video, hoop, args.out, crop=args.crop, threshold=args.threshold, batch_size=args.batch,
        annotate=not args.no_video, hoop_frame=args.hoop_frame, stabilize=not args.no_stabilize)


if __name__ == "__main__":
    main()

"""video + hoop.json -> camera_motion.json, detections.json, events.json, annotated.mp4 in one folder.

    python -m brickorswish.detect.run_video VIDEO --hoop hoop.json --out OUT_DIR [--hoop-frame N]

hoop.json holds your taps on frame N (--hoop-frame). The camera may move afterwards: the hoop is
re-found every 0.5 s (stabilize.py), detections are mapped back to frame N's pixels, and spans where
the hoop cannot be found are not counted. --no-stabilize assumes a still camera.

--resume-dir DIR (e.g. on Google Drive) keeps progress there: the camera-motion result and the detections
are saved while they are made, and a re-run with the same video, taps and settings continues where the last
one stopped (Colab can stop long runs). Delete DIR to start from scratch.

With --onnx model.onnx it runs the exported model locally (ml/requirements-local.txt: onnxruntime, numpy,
opencv-python-headless); without it, rfdetr itself (PyTorch, Colab: ml/notebooks/detect_video.ipynb).
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
    onnx: str | Path | None = None,
    crop: str = "hoop",
    threshold: float = 0.15,
    batch_size: int = 8,
    annotate: bool = True,
    hoop_frame: int = 0,
    stabilize: bool = True,
    motion_cfg: MotionConfig | None = None,
    resume_dir: str | Path | None = None,
    log=print,
) -> dict:
    from .annotate import annotate_video
    from .video import detect_video, load_model, model_label

    cfg = ShotLogicConfig.load(CONFIG_V1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "hoop.json", hoop)

    resume = Path(resume_dir) if resume_dir else None
    if resume:
        resume.mkdir(parents=True, exist_ok=True)
    transforms = None
    mcfg = motion_cfg or MotionConfig()
    if stabilize:
        transforms, motion = _camera_motion(video, hoop, hoop_frame, mcfg, out, log,
                                            resume / "camera_motion.json" if resume else None)

    log("detecting...")
    model = model or load_model(onnx)
    key = {"video": Path(video).name, "hoop": hoop, "hoop_frame": hoop_frame, "crop": crop, "threshold": threshold,
           "model": model_label(model), "stabilize": stabilize, "motion": asdict(mcfg)}
    detections = detect_video(video, hoop, cfg, model, crop, threshold, batch_size, transforms, log,
                              checkpoint=resume / "detections.progress.json" if resume else None, resume_key=key)
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


def _camera_motion(video, hoop: dict, hoop_frame: int, mcfg: MotionConfig, out: Path, log,
                   cache: Path | None = None) -> tuple[list, dict]:
    from .stabilize import measure_motion

    key = {"video": Path(video).name, "hoop": hoop, "hoop_frame": hoop_frame, "motion": asdict(mcfg)}
    if cache is not None and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("resume_key") == json.loads(json.dumps(key)) and "n_frames" in saved:
            segments = [Segment(**s) for s in saved["segments"]]
            transforms = frame_transforms(segments, saved["n_frames"], saved["check_every_frames"], mcfg)
            log(f"camera motion: reusing the result of an earlier run ({cache})")
            for line in saved["timeline"]:
                log("  " + line)
            _dump(out / "camera_motion.json", saved)
            return transforms, saved

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
        "resume_key": key,
        "n_frames": n_frames,
        "fps": fps,
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
    if cache is not None:
        _dump(cache, motion)
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
    ap.add_argument("--onnx", help="exported model (model.onnx + model.json card); default: rfdetr (PyTorch)")
    ap.add_argument("--crop", choices=["hoop", "full"], default="hoop")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video")
    ap.add_argument("--hoop-frame", type=int, default=0, help="frame index the hoop taps were made on")
    ap.add_argument("--no-stabilize", action="store_true", help="assume the camera never moves")
    ap.add_argument("--resume-dir", help="keep progress here and continue an interrupted run (e.g. on Google Drive)")
    args = ap.parse_args()
    with open(args.hoop, encoding="utf-8") as fh:
        hoop = json.load(fh)
    run(args.video, hoop, args.out, onnx=args.onnx, crop=args.crop, threshold=args.threshold, batch_size=args.batch,
        annotate=not args.no_video, hoop_frame=args.hoop_frame, stabilize=not args.no_stabilize,
        resume_dir=args.resume_dir)


if __name__ == "__main__":
    main()

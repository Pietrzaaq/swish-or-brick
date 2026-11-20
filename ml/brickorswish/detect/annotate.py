"""Draws detections, hoop, tracked ball and shot outcomes onto the video. Needs `opencv-python`; ffmpeg optional."""

import shutil
import subprocess
from pathlib import Path

from ..logic import HoopFrame, ShotLogicConfig
from ..logic.pipeline import ball_candidates
from ..logic.state_machine import MAKE, MISS, ShotStateMachine
from ..logic.tracker import BallTracker

# BGR
ORANGE = (0, 140, 255)
DIM_ORANGE = (0, 80, 150)
BLUE = (255, 120, 0)
GREY = (140, 140, 140)
YELLOW = (0, 220, 255)
GREEN = (60, 200, 60)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
OUTCOME_COLOUR = {MAKE: GREEN, MISS: RED}
BANNER_S = 1.5
TRAIL = 20


def annotate_video(
    video_path: str | Path,
    out_path: str | Path,
    hoop: dict,
    detections: dict,
    cfg: ShotLogicConfig,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Path:
    """Writes out_path (H.264 if ffmpeg is available, else mp4v). Re-runs tracker + state machine per frame
    (same steps as process()) so the overlay shows the live state."""
    import cv2

    out_path = Path(out_path)
    hf = HoopFrame.from_dict(hoop)
    tracker, machine = BallTracker(cfg), ShotStateMachine(cfg)
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(hoop["image_width"]), int(hoop["image_height"])
    raw = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), float(detections["fps"]), (w, h))
    s = max(0.5, min(w, h) / 1080.0)  # text/line scale
    counts = {"make": 0, "miss": 0, "uncertain": 0}
    trail: list[tuple[int, int]] = []
    banner = None  # (text, colour, until_t)
    try:
        for fr in detections["frames"]:
            ok, img = cap.read()
            if not ok:
                break
            t = float(fr["t"])
            ball = tracker.step(t, ball_candidates(fr["dets"], hf, cfg))
            event = machine.step(t, ball)
            if event is not None:
                counts[event.outcome] += 1
                banner = (f"{event.outcome.upper()} ({event.reason})", OUTCOME_COLOUR.get(event.outcome, YELLOW), t + BANNER_S)

            if crop_box is not None and tuple(crop_box) != (0, 0, w, h):
                cv2.rectangle(img, crop_box[:2], crop_box[2:], GREY, 1)
            _draw_zone(img, hf, cfg)
            _draw_hoop(img, hf, s)
            for d in fr["dets"]:
                _draw_det(img, d, hf, cfg, s)
            if ball is not None:
                p = tuple(int(round(v)) for v in hf.to_pixels(ball.x, ball.y))
                trail = (trail + [p])[-TRAIL:]
                cv2.circle(img, p, int(6 * s), GREEN, -1 if ball.observed else int(2 * s))
            else:
                trail = []
            if len(trail) > 1:
                for a, b in zip(trail, trail[1:]):
                    cv2.line(img, a, b, GREEN, max(1, int(2 * s)))

            hud = [
                f"t={t:.2f}s  frame {fr['i']}",
                f"state: {machine.state}",
                f"make {counts['make']}  miss {counts['miss']}  uncertain {counts['uncertain']}",
            ]
            for k, line in enumerate(hud):
                _text(img, line, (int(12 * s), int((30 + 30 * k) * s)), 0.7 * s, WHITE)
            if banner is not None and t <= banner[2]:
                _text(img, banner[0], (int(12 * s), int(150 * s)), 1.4 * s, banner[1], thick=3)
            writer.write(img)
    finally:
        cap.release()
        writer.release()

    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "23", "-movflags", "+faststart", str(out_path)],
            check=True,
        )
        raw.unlink()
    else:
        raw.replace(out_path)
    return out_path


def draw_hoop_preview(img, hoop: dict, cfg: ShotLogicConfig, crop_box=None):
    """Copy of img with the hoop ellipse, approach zone and crop box, to check a calibration."""
    import cv2

    out = img.copy()
    hf = HoopFrame.from_dict(hoop)
    s = max(0.5, min(out.shape[:2]) / 1080.0)
    if crop_box is not None:
        cv2.rectangle(out, crop_box[:2], crop_box[2:], GREY, max(1, int(2 * s)))
    _draw_zone(out, hf, cfg)
    _draw_hoop(out, hf, s)
    for key, colour in (("left", RED), ("right", GREEN), ("front", BLUE)):
        p = tuple(int(round(v)) for v in hoop[key])
        cv2.circle(out, p, int(5 * s), colour, -1)
    return out


def _draw_hoop(img, hf: HoopFrame, s: float) -> None:
    import math

    import cv2

    centre = (int(round(hf.cx)), int(round(hf.cy)))
    axes = (int(round(hf.radius_px)), max(1, int(round(hf.ellipse_b * hf.radius_px))))
    rx, ry = hf.to_pixels(1.0, 0.0)  # right rim edge
    angle = math.degrees(math.atan2(ry - hf.cy, rx - hf.cx))
    cv2.ellipse(img, centre, axes, angle, 0, 360, YELLOW, max(1, int(2 * s)))


def _draw_zone(img, hf: HoopFrame, cfg: ShotLogicConfig) -> None:
    """Approach zone: where a shot attempt starts."""
    import cv2
    import numpy as np

    hw, top = cfg.approach_half_width, cfg.approach_height
    pts = [hf.to_pixels(x, y) for x, y in ((-hw, -top), (hw, -top), (hw, 0.0), (-hw, 0.0))]
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, YELLOW, 1)


def _draw_det(img, d: dict, hf: HoopFrame, cfg: ShotLogicConfig, s: float) -> None:
    import cv2

    x0, y0 = int(round(d["x"] - d["w"] / 2)), int(round(d["y"] - d["h"] / 2))
    x1, y1 = int(round(d["x"] + d["w"] / 2)), int(round(d["y"] + d["h"] / 2))
    if d["cls"] == "ball":
        size = hf.size_in_rim_diameters(d["w"], d["h"])
        used = d["conf"] >= cfg.ball_min_conf and cfg.ball_min_size_rim_d <= size <= cfg.ball_max_size_rim_d
        colour, thick = (ORANGE, max(1, int(2 * s))) if used else (DIM_ORANGE, 1)
        label = f"{d['conf']:.2f}"
    elif d["cls"] == "person":
        colour, thick, label = BLUE, 1, None
    else:
        colour, thick, label = GREY, 1, f"{d['cls']} {d['conf']:.2f}"
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, thick)
    if label:
        _text(img, label, (x0, max(0, y0 - 4)), 0.45 * s, colour, thick=1)


def _text(img, text: str, org, scale: float, colour, thick: int = 2) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)

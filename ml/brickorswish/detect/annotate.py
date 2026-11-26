"""Draws detections, hoop, tracked ball and shot outcomes onto the video. Needs `opencv-python`; ffmpeg optional.

Browsers (and so viewer/frame_viewer.html) play H.264 or VP8, not OpenCV's own mp4v. With ffmpeg: H.264 .mp4.
Without it: VP8 .webm from OpenCV's bundled encoder, capped at WEBM_MAX_SIDE (full 1080p VP8 is ~2x slower and
larger, and the viewer zooms anyway)."""

import math
import shutil
import subprocess
from pathlib import Path

from ..logic import HoopFrame, ShotLogicConfig
from ..logic.pipeline import near_static, ball_candidates, static_spots
from ..logic.state_machine import MAKE, MISS, ShotStateMachine
from ..logic.tracker import BallTracker
from .crop import watch_crop
from .motion import IDENTITY, Transform, rim_centre, shift_hoop, to_frame

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
WEBM_MAX_SIDE = 1280


def annotate_video(
    video_path: str | Path,
    out_path: str | Path,
    hoop: dict,
    detections: dict,
    cfg: ShotLogicConfig,
    crop: str = "hoop",
    transforms: list[Transform | None] | None = None,
) -> Path:
    """Returns the path written (.webm if there is no ffmpeg). Re-runs the same steps as process() per frame so
    the overlay shows the live state. `detections` are in calibration-frame pixels; `transforms` maps them back."""
    import cv2

    out_path = Path(out_path)
    hf = HoopFrame.from_dict(hoop)
    static = static_spots(detections, hf, cfg)
    c0 = rim_centre(hoop)
    tracker, machine = BallTracker(cfg), ShotStateMachine(cfg)
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(hoop["image_width"]), int(hoop["image_height"])
    fps = float(detections["fps"])
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        raw, size = out_path.with_suffix(".raw.mp4"), (w, h)
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    else:
        k = min(1.0, WEBM_MAX_SIDE / max(w, h))
        raw, size = out_path.with_suffix(".webm"), (int(round(w * k / 2)) * 2, int(round(h * k / 2)) * 2)
        writer = cv2.VideoWriter(str(raw), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {raw}: no video encoder available")
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
            start_conf = cfg.ball_min_conf if machine.idle else cfg.ball_track_min_conf
            ball = tracker.step(t, ball_candidates(fr["dets"], hf, cfg, static), start_conf)
            event = machine.step(t, ball)
            if event is not None:
                counts[event.outcome] += 1
                banner = (f"{event.outcome.upper()} ({event.reason})", OUTCOME_COLOUR.get(event.outcome, YELLOW), t + BANNER_S)

            tr = IDENTITY if transforms is None else transforms[min(fr["i"], len(transforms) - 1)]
            hud_extra = []
            if tr is None:
                hud_extra.append(("HOOP NOT FOUND: not counting", RED))
                tr_draw = None
            else:
                tr_draw = tr
                moved = shift_hoop(hoop, tr)
                hf_now = HoopFrame.from_dict(moved, hf.down)
                if crop == "hoop":
                    box = watch_crop(moved, cfg)
                    cv2.rectangle(img, box[:2], box[2:], GREY, 1)
                _draw_zone(img, hf_now, cfg)
                _draw_hoop(img, hf_now, s)
                if tr != IDENTITY:
                    hud_extra.append((f"camera moved: hoop ({tr[0]:+.0f}, {tr[1]:+.0f}) px", YELLOW))
                for d in fr["dets"]:
                    _draw_det(img, d, hf, cfg, s, c0, tr, static)
            if ball is not None and tr_draw is not None:
                if ball.observed:  # trail joins real sightings only; a guess is a hollow circle
                    p = tuple(int(round(v)) for v in to_frame(*hf.to_pixels(ball.seen_x, ball.seen_y), c0, tr_draw))
                    trail = (trail + [p])[-TRAIL:]
                    cv2.circle(img, p, int(6 * s), GREEN, -1)
                else:
                    p = tuple(int(round(v)) for v in to_frame(*hf.to_pixels(ball.est_x, ball.est_y), c0, tr_draw))
                    cv2.circle(img, p, int(6 * s), GREEN, max(1, int(s)))
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
            for k, (line, colour) in enumerate(hud_extra):
                _text(img, line, (int(12 * s), int((120 + 30 * k) * s)), 0.7 * s, colour)
            if banner is not None and t <= banner[2]:
                _text(img, banner[0], (int(12 * s), int(230 * s)), 1.4 * s, banner[1], thick=3)
            writer.write(img if size == (w, h) else cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
        writer.release()

    if not ffmpeg:
        return raw
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", "-movflags", "+faststart", str(out_path)],
        check=True,
    )
    raw.unlink()
    return out_path


def draw_hoop_preview(img, hoop: dict, cfg: ShotLogicConfig, crop_box=None):
    """Copy of img with the hoop ellipse, approach zone and crop box."""
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
    """Tilt comes from the clicks, not from gravity."""
    import cv2

    centre = (int(round(hf.cx)), int(round(hf.cy)))
    axes = (int(round(hf.radius_px)), max(1, int(round(hf.ellipse_b * hf.radius_px))))
    angle = math.degrees(hf.rim_angle)
    cv2.ellipse(img, centre, axes, angle, 0, 360, YELLOW, max(1, int(2 * s)))


def _draw_zone(img, hf: HoopFrame, cfg: ShotLogicConfig) -> None:
    import cv2
    import numpy as np

    hw, top = cfg.approach_half_width, cfg.approach_height
    pts = [hf.to_pixels(x, y) for x, y in ((-hw, -top), (hw, -top), (hw, 0.0), (-hw, 0.0))]
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, YELLOW, 1)


def _draw_det(img, d: dict, hf: HoopFrame, cfg: ShotLogicConfig, s: float,
              c0: tuple[float, float], tr: Transform, static: frozenset = frozenset()) -> None:
    """d is in calibration-frame pixels. A 'ball' at a static spot (e.g. the rim's bracket) is drawn dim: the
    logic ignores it."""
    import cv2

    cx, cy = to_frame(d["x"], d["y"], c0, tr)
    bw, bh = d["w"] * tr[2], d["h"] * tr[2]
    x0, y0 = int(round(cx - bw / 2)), int(round(cy - bh / 2))
    x1, y1 = int(round(cx + bw / 2)), int(round(cy + bh / 2))
    if d["cls"] == "ball":
        size = hf.size_in_rim_diameters(d["w"], d["h"])
        is_static = bool(static) and near_static(*hf.to_hoop(d["x"], d["y"]), static, cfg)
        used = (d["conf"] >= cfg.ball_min_conf and cfg.ball_min_size_rim_d <= size <= cfg.ball_max_size_rim_d
                and not is_static)
        colour, thick = (ORANGE, max(1, int(2 * s))) if used else (DIM_ORANGE, 1)
        label = f"{d['conf']:.2f}" + (" static" if is_static else "")
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

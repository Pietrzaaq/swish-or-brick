"""Crop region around the hoop and detector-box -> detections.json conversion. Standard library only."""

import math

from ..logic import HoopFrame, ShotLogicConfig

# COCO names that get a project-wide name in detections.json. Everything else keeps its COCO name.
CLASS_RENAME = {"sports ball": "ball"}


def watch_crop(hoop: dict, cfg: ShotLogicConfig, square: bool = True) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1), end-exclusive, covering the tracker's watch zone.

    The detector resizes its input to a fixed square, so a square crop avoids distorting the ball.
    The box is clamped to the image; if it does not fit, it is shifted inside, then cut.
    """
    frame = HoopFrame.from_dict(hoop)
    w_img, h_img = int(hoop["image_width"]), int(hoop["image_height"])
    hw = cfg.watch_half_width
    corners = [frame.to_pixels(x, y) for x in (-hw, hw) for y in (-cfg.watch_top, cfg.watch_bottom)]
    x0 = min(p[0] for p in corners)
    x1 = max(p[0] for p in corners)
    y0 = min(p[1] for p in corners)
    y1 = max(p[1] for p in corners)
    if square:
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        x0, x1 = cx - side / 2.0, cx + side / 2.0
        y0, y1 = cy - side / 2.0, cy + side / 2.0
    x0, x1 = _fit(x0, x1, w_img)
    y0, y1 = _fit(y0, y1, h_img)
    return x0, y0, x1, y1


def _fit(lo: float, hi: float, size: int) -> tuple[int, int]:
    lo_i, hi_i = math.floor(lo), math.ceil(hi)
    span = hi_i - lo_i
    if span >= size:
        return 0, size
    if lo_i < 0:
        return 0, span
    if hi_i > size:
        return size - span, size
    return lo_i, hi_i


def box_to_det(cls_name: str, conf: float, xyxy, offset: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Detector box (corners, crop pixels) -> detections.json entry (centre + size, full-frame pixels)."""
    ox, oy = offset
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    return {
        "cls": CLASS_RENAME.get(cls_name, cls_name),
        "conf": round(float(conf), 4),
        "x": round((x0 + x1) / 2.0 + ox, 2),
        "y": round((y0 + y1) / 2.0 + oy, 2),
        "w": round(x1 - x0, 2),
        "h": round(y1 - y0, 2),
    }


def frame_time(i: int, fps: float) -> float:
    """Timestamp of frame i at a constant frame rate. Phone videos can be variable-rate; v1 ignores that."""
    return round(i / fps, 6)

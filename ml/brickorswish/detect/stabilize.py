"""Camera-motion measurement: where is the hoop now, relative to the calibration frame? Needs OpenCV.

A template of the backboard + rim is cut from the frame you tapped. Every check_every_s it is
searched for near its last position, then (if not found there) over the whole frame at a few scales.
Each search gives a motion.Check; motion.py decides what to trust. Cloud only (Colab).
"""

import math
from pathlib import Path

from ..logic import HoopFrame
from .motion import IDENTITY, Check, MotionConfig, Transform
from .video import read_frame


def measure_motion(video_path: str | Path, hoop: dict, calib_index: int, cfg: MotionConfig,
                   log=print) -> tuple[list[Check], int, float, int]:
    """Returns (checks, n_frames, fps, check_every_frames)."""
    import cv2

    hf = HoopFrame.from_dict(hoop)
    c0, r = (hf.cx, hf.cy), hf.radius_px
    tmpl, tl = _template(_prep(read_frame(video_path, calib_index)), c0, r, cfg)
    log(f"hoop template {tmpl.shape[1]}x{tmpl.shape[0]} px from frame {calib_index}")

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    every = max(1, round(cfg.check_every_s * fps))
    checks: list[Check] = []
    prev: Transform = IDENTITY
    i = 0
    try:
        while True:
            want = i % every == 0 or i == calib_index
            if want:
                ok, img = cap.read()
            else:
                ok = cap.grab()  # advance without converting the frame
            if not ok:
                break
            if want:
                score, tr = _locate(_prep(img), tmpl, tl, c0, r, prev, cfg)
                checks.append(Check(i, round(i / fps, 3), round(score, 3), round(tr[0], 2), round(tr[1], 2),
                                    round(tr[2], 4)))
                if score >= cfg.min_score:
                    prev = tr
            i += 1
    finally:
        cap.release()
    return checks, i, fps, every


def _prep(bgr):
    import cv2

    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def _template(gray, c0, r, cfg: MotionConfig):
    h, w = gray.shape[:2]
    x0 = max(0, int(c0[0] - cfg.template_half_width * r))
    x1 = min(w, int(c0[0] + cfg.template_half_width * r))
    y0 = max(0, int(c0[1] - cfg.template_above * r))
    y1 = min(h, int(c0[1] + cfg.template_below * r))
    return gray[y0:y1, x0:x1].copy(), (x0, y0)


def _locate(gray, tmpl, tl, c0, r, prev: Transform, cfg: MotionConfig) -> tuple[float, Transform]:
    """(score, transform). Local search around the previous pose first, then full frame, multi-scale."""
    import cv2

    ox, oy = c0[0] - tl[0], c0[1] - tl[1]  # rim centre inside the template

    def result(score, fx, fy, s):
        return score, (fx + s * ox - c0[0], fy + s * oy - c0[1], s)

    dx, dy, s = prev
    ts = _scaled(tmpl, s)
    ex, ey = c0[0] + dx - s * ox, c0[1] + dy - s * oy  # expected top-left
    m = cfg.local_search * r
    hit = _match(gray, ts, ex - m, ey - m, ex + ts.shape[1] + m, ey + ts.shape[0] + m)
    if hit and hit[0] >= cfg.min_score:
        return result(hit[0], hit[1], hit[2], s)

    g = min(1.0, 640.0 / max(gray.shape[:2]))
    small = cv2.resize(gray, None, fx=g, fy=g, interpolation=cv2.INTER_AREA) if g < 1 else gray
    best = None
    for s in cfg.scales:
        ts = _scaled(tmpl, s * g)
        if min(ts.shape[:2]) < 8 or ts.shape[0] >= small.shape[0] or ts.shape[1] >= small.shape[1]:
            continue
        hit = _match(small, ts, 0, 0, small.shape[1], small.shape[0])
        if hit and (best is None or hit[0] > best[0]):
            best = (hit[0], hit[1] / g, hit[2] / g, s)
    if best is None:
        return 0.0, prev
    score, fx, fy, s = best
    ts = _scaled(tmpl, s)
    pad = 2.0 / g + 2.0
    hit = _match(gray, ts, fx - pad, fy - pad, fx + ts.shape[1] + pad, fy + ts.shape[0] + pad)
    if hit:
        score, fx, fy = hit
    return result(score, fx, fy, s)


def _scaled(img, s: float):
    import cv2

    if abs(s - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def _match(gray, tmpl, x0, y0, x1, y1):
    """Best normalised correlation of tmpl inside the window (clamped to the image): (score, x, y) or None."""
    import cv2

    h, w = gray.shape[:2]
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    if x1 - x0 < tmpl.shape[1] or y1 - y0 < tmpl.shape[0]:
        return None
    res = cv2.matchTemplate(gray[y0:y1, x0:x1], tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if not math.isfinite(score):  # flat (textureless) window
        return None
    return min(1.0, float(score)), x0 + loc[0], y0 + loc[1]

"""Camera-motion measurement: where is the hoop now, relative to the calibration frame? Needs OpenCV.

A backboard + rim template is matched every check_every_s; motion.py decides what to trust.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from ..logic import HoopFrame
from .motion import IDENTITY, Check, MotionConfig, Transform, compose, pick_scale, should_reanchor, to_frame
from .video import read_frame


@dataclass
class _Anchor:
    tmpl: object
    tl: tuple[int, int]
    centre: tuple[float, float]
    radius: float  # px
    pose: Transform  # of the frame the template was cut from


def _cut(gray, c0, r, pose: Transform, cfg: MotionConfig) -> _Anchor:
    centre = to_frame(c0[0], c0[1], c0, pose)
    radius = r * pose[2]
    tmpl, tl = _template(gray, centre, radius, cfg)
    return _Anchor(tmpl, tl, centre, radius, pose)


def measure_motion(video_path: str | Path, hoop: dict, calib_index: int, cfg: MotionConfig,
                   log=print) -> tuple[list[Check], int, float, int, list[dict]]:
    """Returns (checks, n_frames, fps, check_every_frames, anchors). anchors: frames the template was cut from."""
    import cv2

    hf = HoopFrame.from_dict(hoop)
    c0, r = (hf.cx, hf.cy), hf.radius_px
    first = _cut(_prep(read_frame(video_path, calib_index)), c0, r, IDENTITY, cfg)
    log(f"hoop template {first.tmpl.shape[1]}x{first.tmpl.shape[0]} px from frame {calib_index}")

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    every = max(1, round(cfg.check_every_s * fps))
    checks: list[Check] = []
    anchors = [{"i": calib_index, "t": round(calib_index / fps, 3), "dx": 0.0, "dy": 0.0, "scale": 1.0}]

    def walker():
        """Each search starts where the last one found it, so the walk must start at the calibration frame."""
        anchor, prev, since_anchor = first, IDENTITY, []  # prev: relative to the current anchor

        def step(i: int, gray) -> None:
            nonlocal anchor, prev, since_anchor
            score, rel = _locate(gray, anchor, prev, cfg)
            tr = compose(anchor.pose, rel)
            check = Check(i, round(i / fps, 3), round(score, 3), round(tr[0], 2), round(tr[1], 2), round(tr[2], 4))
            checks.append(check)
            since_anchor.append(check)
            if score >= cfg.min_score:
                prev = rel
            if should_reanchor(since_anchor, anchor.pose, r, cfg):
                pose = (check.dx, check.dy, check.scale)
                anchor = _cut(gray, c0, r, pose, cfg)
                prev, since_anchor = IDENTITY, []
                anchors.append({"i": i, "t": check.t, "dx": pose[0], "dy": pose[1], "scale": pose[2]})
                log(f"  re-took the hoop template at {check.t:.1f} s (match was {check.score:.2f}, "
                    f"hoop at ({pose[0]:+.0f}, {pose[1]:+.0f}) px, scale {pose[2]:.2f})")

        return step

    # One sequential read, no seeking: frames decode exactly as detect_video decodes them. Frames before the
    # calibration frame are kept (lossless PNG) and walked backward afterwards.
    forward, before = walker(), []
    i = 0
    try:
        while True:
            want = i % every == 0 or i == calib_index
            if want:
                ok, img = cap.read()
            else:
                ok = cap.grab()
            if not ok:
                break
            if want:
                gray = _prep(img)
                if i < calib_index:
                    before.append((i, cv2.imencode(".png", gray)[1]))
                else:
                    forward(i, gray)
            i += 1
    finally:
        cap.release()
    backward = walker()
    for j, png in reversed(before):
        backward(j, cv2.imdecode(png, cv2.IMREAD_GRAYSCALE))
    checks.sort(key=lambda c: c.i)
    anchors.sort(key=lambda a: a["i"])
    return checks, i, fps, every, anchors


def _prep(bgr):
    import cv2

    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def _template(gray, c, r, cfg: MotionConfig):
    h, w = gray.shape[:2]
    x0 = max(0, int(c[0] - cfg.template_half_width * r))
    x1 = min(w, int(c[0] + cfg.template_half_width * r))
    y0 = max(0, int(c[1] - cfg.template_above * r))
    y1 = min(h, int(c[1] + cfg.template_below * r))
    return gray[y0:y1, x0:x1].copy(), (x0, y0)


def _locate(gray, anchor: _Anchor, prev: Transform, cfg: MotionConfig) -> tuple[float, Transform]:
    """(score, transform relative to the anchor frame). Local search first, then the full frame at a few scales."""
    import cv2

    tmpl, c, r = anchor.tmpl, anchor.centre, anchor.radius
    ox, oy = c[0] - anchor.tl[0], c[1] - anchor.tl[1]  # rim centre inside the template

    def result(score, fx, fy, s):
        return score, (fx + s * ox - c[0], fy + s * oy - c[1], s)

    dx, dy, s = prev
    ts = _scaled(tmpl, s)
    ex, ey = c[0] + dx - s * ox, c[1] + dy - s * oy  # expected top-left
    m = cfg.local_search * r
    hit = _match(gray, ts, ex - m, ey - m, ex + ts.shape[1] + m, ey + ts.shape[0] + m)
    if hit and hit[0] >= cfg.min_score:
        return result(hit[0], hit[1], hit[2], s)

    g = min(1.0, 640.0 / max(gray.shape[:2]))
    small = cv2.resize(gray, None, fx=g, fy=g, interpolation=cv2.INTER_AREA) if g < 1 else gray
    candidates = []
    for sc in sorted(set(cfg.scales) | {s}):
        ts = _scaled(tmpl, sc * g)
        if min(ts.shape[:2]) < 8 or ts.shape[0] >= small.shape[0] or ts.shape[1] >= small.shape[1]:
            continue
        coarse = _match(small, ts, 0, 0, small.shape[1], small.shape[0])
        if not coarse:
            continue
        # Refine at full resolution so scores of different scales are comparable.
        ts = _scaled(tmpl, sc)
        fx, fy = coarse[1] / g, coarse[2] / g
        pad = 2.0 / g + 2.0
        fine = _match(gray, ts, fx - pad, fy - pad, fx + ts.shape[1] + pad, fy + ts.shape[0] + pad)
        score, fx, fy = fine if fine else (coarse[0], fx, fy)
        candidates.append((score, sc, (fx, fy)))
    best = pick_scale(candidates, s, cfg.scale_switch_margin)
    if best is None:
        return 0.0, prev
    score, sc, (fx, fy) = best
    return result(score, fx, fy, sc)


def _scaled(img, s: float):
    import cv2

    if abs(s - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def _match(gray, tmpl, x0, y0, x1, y1):
    """(score, x, y) of the best normalised correlation inside the window, or None."""
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

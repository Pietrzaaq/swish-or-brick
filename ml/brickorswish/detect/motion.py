"""Camera-motion compensation, decision part. Standard library only.

stabilize.py (OpenCV) measures every few frames where the backboard/rim patch from the calibration
frame is now: a Check (dx, dy, scale, score). This module turns those noisy checks into segments of
constant camera pose plus "lost" spans where the hoop position cannot be trusted, and maps detections
into the calibration frame, so the shot logic (and its Dart port) never sees camera motion.

Transform (dx, dy, s): the rim centre moved by (dx, dy) px and the image scaled by s around it,
relative to the calibration frame. None = hoop lost for that frame.
"""

from dataclasses import dataclass
from statistics import median

Transform = tuple[float, float, float]
IDENTITY: Transform = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class MotionConfig:
    check_every_s: float = 0.5
    # Template around the rim, in rim radii: backboard above, a little net below.
    template_half_width: float = 4.0
    template_above: float = 6.0
    template_below: float = 1.5
    local_search: float = 3.0  # rim radii around the last known position before a full-frame search
    min_score: float = 0.6  # normalised cross-correlation needed to call the hoop found
    agree_tol: float = 0.5  # rim radii: checks closer than this describe the same camera pose
    agree_scale_tol: float = 0.05
    min_run: int = 3  # agreeing checks in a row before a pose is trusted (1.5 s at 0.5 s checks)
    max_bridge_checks: int = 2  # failed checks between two trusted poses that are bridged, not marked lost
    scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25)


@dataclass(frozen=True)
class Check:
    i: int
    t: float
    score: float
    dx: float
    dy: float
    scale: float


@dataclass(frozen=True)
class Segment:
    from_i: int
    to_i: int
    dx: float
    dy: float
    scale: float
    checks: int

    @property
    def transform(self) -> Transform:
        return (self.dx, self.dy, self.scale)


def plan_segments(checks: list[Check], rim_radius_px: float, cfg: MotionConfig) -> list[Segment]:
    """Runs of >= min_run found checks that agree with each other become segments; adjacent segments
    with the same pose and a bridgeable gap between them are merged (e.g. a player blocked one check)."""
    tol = cfg.agree_tol * rim_radius_px
    runs: list[list[Check]] = []
    cur: list[Check] = []

    def close():
        if len(cur) >= cfg.min_run:
            runs.append(list(cur))
        cur.clear()

    for c in sorted(checks, key=lambda c: c.i):
        if c.score < cfg.min_score:
            close()
            continue
        if cur and not _agree(_pose(cur), (c.dx, c.dy, c.scale), tol, cfg.agree_scale_tol):
            close()
        cur.append(c)
    close()

    segments = [_segment(r) for r in runs]
    merged: list[Segment] = []
    every = _check_spacing(checks)
    for seg in segments:
        prev = merged[-1] if merged else None
        if (prev and seg.from_i - prev.to_i <= (cfg.max_bridge_checks + 1) * every
                and _agree(prev.transform, seg.transform, tol, cfg.agree_scale_tol)):
            w1, w2 = prev.checks, seg.checks
            merged[-1] = Segment(prev.from_i, seg.to_i,
                                 round((prev.dx * w1 + seg.dx * w2) / (w1 + w2), 2),
                                 round((prev.dy * w1 + seg.dy * w2) / (w1 + w2), 2),
                                 round((prev.scale * w1 + seg.scale * w2) / (w1 + w2), 4), w1 + w2)
        else:
            merged.append(seg)
    return merged


def frame_transforms(segments: list[Segment], n_frames: int, check_every: int,
                     cfg: MotionConfig) -> list[Transform | None]:
    """Transform per frame. Segments cover their own span; short gaps between segments (and at the
    start/end of the video) are bridged, split at the midpoint; longer gaps stay None (hoop lost)."""
    out: list[Transform | None] = [None] * n_frames
    bridge = (cfg.max_bridge_checks + 1) * check_every
    segs = sorted(segments, key=lambda s: s.from_i)
    for k, seg in enumerate(segs):
        start, end = seg.from_i, seg.to_i
        if k == 0 and start <= bridge:
            start = 0
        if k == len(segs) - 1 and n_frames - 1 - end <= bridge:
            end = n_frames - 1
        if k > 0 and seg.from_i - segs[k - 1].to_i <= bridge:
            start = (segs[k - 1].to_i + seg.from_i) // 2 + 1
        if k < len(segs) - 1 and segs[k + 1].from_i - seg.to_i <= bridge:
            end = (seg.to_i + segs[k + 1].from_i) // 2
        for i in range(max(0, start), min(n_frames - 1, end) + 1):
            out[i] = seg.transform
    return out


def lost_spans(transforms: list[Transform | None]) -> list[tuple[int, int]]:
    """(first, last) frame index of every run of None."""
    spans, start = [], None
    for i, tr in enumerate(transforms):
        if tr is None and start is None:
            start = i
        elif tr is not None and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(transforms) - 1))
    return spans


def rim_centre(hoop: dict) -> tuple[float, float]:
    (lx, ly), (rx, ry) = hoop["left"], hoop["right"]
    return (lx + rx) / 2.0, (ly + ry) / 2.0


def to_frame(x: float, y: float, c0: tuple[float, float], tr: Transform) -> tuple[float, float]:
    """Calibration-frame pixel -> pixel in the moved frame."""
    dx, dy, s = tr
    return c0[0] + dx + s * (x - c0[0]), c0[1] + dy + s * (y - c0[1])


def to_calibration(x: float, y: float, c0: tuple[float, float], tr: Transform) -> tuple[float, float]:
    """Pixel in the moved frame -> calibration-frame pixel (inverse of to_frame)."""
    dx, dy, s = tr
    return c0[0] + (x - c0[0] - dx) / s, c0[1] + (y - c0[1] - dy) / s


def shift_hoop(hoop: dict, tr: Transform) -> dict:
    """hoop.json as it appears in the moved frame (same image size)."""
    c0 = rim_centre(hoop)
    out = dict(hoop)
    for key in ("left", "right", "front"):
        x, y = to_frame(hoop[key][0], hoop[key][1], c0, tr)
        out[key] = [x, y]
    return out


def stabilize_dets(dets: list[dict], c0: tuple[float, float], tr: Transform) -> list[dict]:
    """detections.json entries from the moved frame -> calibration-frame pixels."""
    s = tr[2]
    out = []
    for d in dets:
        x, y = to_calibration(d["x"], d["y"], c0, tr)
        out.append({**d, "x": round(x, 2), "y": round(y, 2), "w": round(d["w"] / s, 2), "h": round(d["h"] / s, 2)})
    return out


def describe(segments: list[Segment], transforms: list[Transform | None], fps: float) -> list[str]:
    """Human-readable timeline: where the camera was steady, where it moved, where the hoop was lost."""
    events = [(s.from_i, "pose", s) for s in segments] + [(a, "lost", (a, b)) for a, b in lost_spans(transforms)]
    lines = []
    for _, kind, item in sorted(events, key=lambda e: e[0]):
        if kind == "lost":
            a, b = item
            lines.append(f"{a / fps:6.1f}-{(b + 1) / fps:6.1f} s  hoop NOT found: shots here are not counted")
        else:
            a, b = item.from_i, item.to_i  # widen to the frames this segment was bridged over
            while a > 0 and transforms[a - 1] == item.transform:
                a -= 1
            while b < len(transforms) - 1 and transforms[b + 1] == item.transform:
                b += 1
            if abs(item.dx) < 1 and abs(item.dy) < 1 and abs(item.scale - 1) < 0.01:
                what = "hoop where you marked it"
            else:
                what = f"camera moved: hoop shifted ({item.dx:+.0f}, {item.dy:+.0f}) px, scale {item.scale:.2f}"
            lines.append(f"{a / fps:6.1f}-{(b + 1) / fps:6.1f} s  {what}")
    return lines


def _pose(run: list[Check]) -> Transform:
    return (median(c.dx for c in run), median(c.dy for c in run), median(c.scale for c in run))


def _segment(run: list[Check]) -> Segment:
    dx, dy, s = _pose(run)
    return Segment(run[0].i, run[-1].i, round(dx, 2), round(dy, 2), round(s, 4), len(run))


def _agree(a: Transform, b: Transform, tol_px: float, scale_tol: float) -> bool:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= tol_px and abs(a[2] - b[2]) <= scale_tol


def _check_spacing(checks: list[Check]) -> int:
    idx = sorted(c.i for c in checks)
    gaps = [b - a for a, b in zip(idx, idx[1:]) if b > a]
    return median(gaps) if gaps else 1

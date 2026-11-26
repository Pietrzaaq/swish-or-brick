"""Finds and fits the rim automatically: video -> hoop.json (left, right, front) + the frame it was fitted on.

    python -m brickorswish.detect.find_rim VIDEO --out hoop.json [--preview rim.png]

Stills are medians of a few frames, so a passing ball disappears. The fit maximises how much redder the
ellipse outline is than just inside and outside it, so a thin ring beats a filled red blob (shorts, shoes).
The calibration frame is the middle of the longest steady stretch: the phone has settled there.

Works for red and orange rims only (Lab a*). Anything else -> taps (`--hoop hoop.json`).
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RimConfig:
    every_s: float = 3.0
    max_stills: int = 60            # long videos: stills further apart
    still_frames: int = 5           # frames per still (median), `still_step` apart
    still_step: int = 2
    min_redness: float = 8.0        # Lab a* (0 = grey) for a seed blob
    min_width_frac: float = 0.012   # seed blob width vs image width
    max_width_frac: float = 0.25
    min_score: float = 5.0          # ring contrast (a* units); the test rim scored 6.4-6.8, other things < 4.5
    min_axis_ratio: float = 0.12    # b / a
    max_axis_ratio: float = 0.75
    max_tilt_deg: float = 25.0
    same_rim: float = 0.6           # rim radii
    steady_centre: float = 0.1      # rim radii
    steady_scale: float = 0.08      # relative size change
    min_stills: int = 3             # in a row
    discover_stills: int = 10       # stills searched blob by blob; the rest only follow what was found
    max_candidates: int = 3
    blur: float = 0.08              # rim radii


@dataclass
class Ellipse:
    cx: float
    cy: float
    a: float      # semi-major, px
    b: float      # semi-minor, px
    theta: float  # major axis angle, radians (image coordinates, y down)
    score: float = 0.0

    def hoop(self, width: int, height: int) -> dict:
        """The 3 'taps': ends of the major axis and the point nearest the camera."""
        ux, uy = math.cos(self.theta), math.sin(self.theta)
        vx, vy = -uy, ux  # minor axis; points down when the major axis points right
        if vy < 0:
            vx, vy = -vx, -vy
        left, right = (self.cx - self.a * ux, self.cy - self.a * uy), (self.cx + self.a * ux, self.cy + self.a * uy)
        if left[0] > right[0]:
            left, right = right, left
        front = (self.cx + self.b * vx, self.cy + self.b * vy)
        r2 = lambda p: [round(p[0], 2), round(p[1], 2)]  # noqa: E731
        return {"image_width": width, "image_height": height, "left": r2(left), "right": r2(right), "front": r2(front)}


def redness(bgr):
    """Lab a* (0 = grey), negatives set to 0."""
    import cv2
    import numpy as np

    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32) - 128.0
    return np.maximum(a, 0.0)


def _sample(img, xs, ys):
    """Bilinear sample; outside the image counts as 0."""
    import numpy as np

    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx, fy = xs - x0, ys - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c, y0c = np.minimum(np.maximum(x0, 0), w - 2), np.minimum(np.maximum(y0, 0), h - 2)
    v = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
         + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    return np.where(inside, v, 0.0)


_N = 96
_COS_SIN = None


def ring_score(red, cx, cy, a, b, theta, cap, side: float = 0.3) -> float:
    """Mean of (redness on the outline - the redder of just inside / just outside), each sample capped."""
    import numpy as np

    global _COS_SIN
    if _COS_SIN is None:
        t = np.linspace(0.0, 2.0 * np.pi, _N, endpoint=False)
        _COS_SIN = (np.cos(t), np.sin(t))
    c, s = _COS_SIN
    ct, st = math.cos(theta), math.sin(theta)
    ex, ey = a * c, b * s
    px, py = cx + ex * ct - ey * st, cy + ex * st + ey * ct
    nx, ny = b * c, a * s  # outward normal (ellipse frame)
    nn = np.hypot(nx, ny)
    nx, ny = nx / nn, ny / nn
    d = max(2.5, side * a)
    wx, wy = d * (nx * ct - ny * st), d * (nx * st + ny * ct)
    v = np.minimum(_sample(red, np.concatenate((px, px + wx, px - wx)), np.concatenate((py, py + wy, py - wy))), cap)
    on, out, inn = v[:_N], v[_N:2 * _N], v[2 * _N:]
    return float(np.mean(on - np.maximum(out, inn)))


def _valid(p, cfg: RimConfig) -> bool:
    cx, cy, a, b, th = p
    return a >= 5.0 and cfg.min_axis_ratio <= b / a <= cfg.max_axis_ratio and abs(th) <= math.radians(cfg.max_tilt_deg)


def _search(f, p, steps, rounds):
    s = f(p)
    for _ in range(rounds):
        improved = True
        while improved:
            improved = False
            for k in range(len(p)):
                for sign in (1, -1):
                    q = list(p)
                    q[k] += sign * steps[k]
                    sq = f(q)
                    if sq > s + 1e-6:
                        p, s, improved = q, sq, True
        steps = [v / 2 for v in steps]
    return p, s


def _scorer(red, cx, cy, a, cfg: RimConfig):
    """Blurred so the middle of the rim's thick lit front band scores best; unblurred, every position in it ties."""
    import cv2
    import numpy as np

    x0, x1 = int(max(0, cx - 3 * a)), int(min(red.shape[1], cx + 3 * a))
    y0, y1 = int(max(0, cy - 3 * a)), int(min(red.shape[0], cy + 3 * a))
    roi = np.ascontiguousarray(red[y0:y1, x0:x1], dtype=np.float32)
    if roi.size == 0:
        return lambda p: -1e9
    roi = cv2.GaussianBlur(roi, (0, 0), max(0.8, cfg.blur * a))
    cap = max(float(np.percentile(roi, 99.5)), 1.0)

    def f(p):
        return ring_score(roi, p[0] - x0, p[1] - y0, p[2], p[3], p[4], cap) if _valid(p, cfg) else -1e9

    return f


def fit_ring(red, guess: tuple[float, float, float], cfg: RimConfig) -> Ellipse | None:
    """The blob is often only the bright front of the ring, so the search also starts higher up and wider."""
    gx, gy, ga = guess
    f = _scorer(red, gx, gy, ga, cfg)
    best_p, best_s, best_steps = None, -1e9, None
    for sa in (0.8, 1.0, 1.25):
        for dy in (-0.3, 0.0, 0.3):
            a = ga * sa
            steps = [0.25 * a, 0.25 * a, 0.12 * a, 0.12 * a, math.radians(4)]
            p, s = _search(f, [gx, gy + dy * a, a, 0.45 * a, 0.0], steps, 3)
            if s > best_s:
                best_p, best_s, best_steps = p, s, [v / 8 for v in steps]
    best_p, best_s = _search(f, best_p, best_steps, 4)
    if best_s < cfg.min_score:
        return None
    return Ellipse(*map(float, best_p), score=float(best_s))


def refine(red, e: Ellipse, cfg: RimConfig) -> Ellipse | None:
    f = _scorer(red, e.cx, e.cy, e.a, cfg)
    p, s = _search(f, [e.cx, e.cy, e.a, e.b, e.theta], [0.1 * e.a, 0.1 * e.a, 0.05 * e.a, 0.05 * e.a,
                                                         math.radians(2)], 7)
    return Ellipse(*map(float, p), score=float(s)) if s >= cfg.min_score else None


def seeds(red, cfg: RimConfig, limit: int = 10) -> list[tuple[float, float, float]]:
    """(cx, cy, half-width) of red blobs of a plausible rim size, largest first."""
    import cv2
    import numpy as np

    h, w = red.shape
    mask = (red >= cfg.min_redness).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    found = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if not (cfg.min_width_frac * w <= bw <= cfg.max_width_frac * w) or bh > 1.5 * bw:
            continue
        found.append((float(area), (x + bw / 2.0, y + bh / 2.0, bw / 2.0)))
    found.sort(key=lambda s: -s[0])
    return [s for _, s in found[:limit]]


def _stills(video, cfg: RimConfig, info: dict):
    """Yields (first frame index, median image)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every = max(1, round(cfg.every_s * fps), total // max(1, cfg.max_stills) + 1)
    info["fps"] = fps
    span = (cfg.still_frames - 1) * cfg.still_step
    buf, start, i = [], None, 0
    try:
        while True:
            if i % every == 0:
                start, buf = i, []
            want = start is not None and i - start <= span and (i - start) % cfg.still_step == 0
            ok = cap.grab()
            if not ok:
                break
            if want:
                ok, img = cap.retrieve()
                buf.append(img)
                if len(buf) == cfg.still_frames:
                    yield start, np.median(np.stack(buf), axis=0).astype(np.uint8)
                    start = None
            i += 1
    finally:
        cap.release()
    info["frames"] = i


def _same_rim(e: Ellipse, ref: Ellipse, cfg: RimConfig) -> bool:
    return math.hypot(e.cx - ref.cx, e.cy - ref.cy) <= cfg.same_rim * ref.a and 0.75 <= e.a / ref.a <= 1.33


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def find_rim(video, cfg: RimConfig | None = None, log=print, preview_path=None) -> dict:
    """{'hoop': {...}, 'hoop_frame': i, ...diagnostics}; raises ValueError if no rim is found."""
    import cv2
    import numpy as np

    cfg = cfg or RimConfig()
    info: dict = {}
    stills = []  # (first frame index, redness as uint8, JPEG for the preview)
    for start, img in _stills(video, cfg, info):
        h, w = img.shape[:2]
        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1] if preview_path else None
        stills.append((start, np.minimum(redness(img), 255).astype(np.uint8), jpg))
    if not stills:
        raise ValueError(f"no frames decoded from {video}")
    fps = info["fps"]
    red_of = lambda k: stills[k][1].astype(np.float32)  # noqa: E731
    t_of = lambda k: stills[k][0] / fps  # noqa: E731

    look = sorted({round(x) for x in np.linspace(0, len(stills) - 1, min(cfg.discover_stills, len(stills)))})
    groups: list[list[Ellipse]] = []
    for k in look:
        red = red_of(k)
        for s in seeds(red, cfg):
            e = fit_ring(red, s, cfg)
            if e is None:
                continue
            for g in groups:
                if _same_rim(e, g[0], cfg):
                    g.append(e)
                    break
            else:
                groups.append([e])
    groups.sort(key=lambda g: sum(e.score for e in g), reverse=True)
    candidates = [max(g, key=lambda e: e.score) for g in groups[:cfg.max_candidates]]
    if not candidates:
        raise ValueError("no red/orange rim found: tap the hoop instead (--hoop hoop.json)")

    track: list[tuple[int, Ellipse]] = []
    for k in range(len(stills)):
        red = red_of(k)
        for c in candidates:
            r = refine(red, c, cfg)
            if r is not None:
                track.append((k, r))
                break

    # Longest steady stretch by total score, so a red thing found in a few stills cannot beat the rim.
    runs: list[list[tuple[int, Ellipse]]] = []
    for k, e in track:
        last = runs[-1][-1] if runs else None
        if last and k == last[0] + 1 and math.hypot(e.cx - last[1].cx, e.cy - last[1].cy) <= cfg.steady_centre * e.a \
                and abs(e.a / last[1].a - 1.0) <= cfg.steady_scale:
            runs[-1].append((k, e))
        else:
            runs.append([(k, e)])
    if not runs:
        raise ValueError("no red/orange rim found: tap the hoop instead (--hoop hoop.json)")
    run = max(runs, key=lambda r: sum(e.score for _, e in r))
    if len(run) < cfg.min_stills:
        raise ValueError(f"rim steady in only {len(run)} stills in a row: tap the hoop instead (--hoop hoop.json)")
    fits = [e for _, e in run]
    e = Ellipse(*(_median([getattr(f, n) for f in fits]) for n in ("cx", "cy", "a", "b", "theta", "score")))
    middle = run[len(run) // 2][0]
    frame = stills[middle][0]
    spread = max(math.hypot(f.cx - e.cx, f.cy - e.cy) for f in fits)
    log(f"rim: {len(stills)} stills; found in {len(track)}; steady from {t_of(run[0][0]):.0f} s to "
        f"{t_of(run[-1][0]):.0f} s ({len(run)} stills, centres within {spread:.1f} px); calibration frame {frame} "
        f"({frame / fps:.1f} s): centre ({e.cx:.1f}, {e.cy:.1f}), radius {e.a:.1f} px, b/a {e.b / e.a:.2f}, "
        f"tilt {math.degrees(e.theta):.1f} deg, score {e.score:.1f}")
    if e.a < 12:
        log("  the rim is small in the picture (< 12 px radius): film closer for better calls")
    result = {
        "hoop": e.hoop(w, h),
        "hoop_frame": frame,
        "fps": round(fps, 3),
        "ellipse": {k2: round(v, 3) for k2, v in asdict(e).items()},
        "steady_from_s": round(t_of(run[0][0]), 2),
        "steady_to_s": round(t_of(run[-1][0]), 2),
        "stills": len(stills),
        "track": [{"t": round(t_of(k), 2), "cx": round(f.cx, 2), "cy": round(f.cy, 2), "a": round(f.a, 2),
                   "b": round(f.b, 2), "score": round(f.score, 2)} for k, f in track],
        "config": asdict(cfg),
    }
    if preview_path:
        preview(cv2.imdecode(stills[middle][2], cv2.IMREAD_COLOR), result, preview_path)
    return result


def preview(img, result: dict, path) -> None:
    """Fitted rim (yellow; dots: left blue, right red, front green), plus a 6x zoom."""
    import cv2
    import numpy as np

    e = result["ellipse"]
    zoom_r = int(max(20, 2.2 * e["a"]))
    x0, y0 = int(max(0, e["cx"] - zoom_r)), int(max(0, e["cy"] - zoom_r))
    crop = img[y0:y0 + 2 * zoom_r, x0:x0 + 2 * zoom_r]
    k = 6
    z = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)

    def draw(canvas, s, ox, oy):
        sh = 4
        c = (int(round((e["cx"] - ox) * s * (1 << sh))), int(round((e["cy"] - oy) * s * (1 << sh))))
        axes = (int(round(e["a"] * s * (1 << sh))), int(round(e["b"] * s * (1 << sh))))
        cv2.ellipse(canvas, c, axes, math.degrees(e["theta"]), 0, 360, (0, 255, 255), 1, cv2.LINE_AA, sh)
        for name, col in (("left", (255, 0, 0)), ("right", (0, 0, 255)), ("front", (0, 255, 0))):
            px, py = result["hoop"][name]
            cv2.circle(canvas, (int(round((px - ox) * s)), int(round((py - oy) * s))), max(2, s), col, -1)

    draw(z, k, x0, y0)
    full = img.copy()
    draw(full, 1, 0, 0)
    scale = z.shape[0] / full.shape[0]
    full = cv2.resize(full, (int(full.shape[1] * scale), z.shape[0]))
    cv2.imwrite(str(path), np.hstack([full, z]))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="rim.json to write: 'hoop' (the 3 points), 'hoop_frame', the fit")
    ap.add_argument("--preview", help="PNG: the frame with the fitted rim and a zoom")
    args = ap.parse_args()
    result = find_rim(args.video, preview_path=args.preview)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

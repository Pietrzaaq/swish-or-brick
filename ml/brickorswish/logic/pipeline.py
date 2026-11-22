"""process(): detections + hoop calibration + config -> shot events. Pure function."""

import math

from .config import ShotLogicConfig
from .geometry import HoopFrame
from .state_machine import ShotStateMachine
from .tracker import BallTracker, Candidate


def ball_candidates(dets: list[dict], hoop: HoopFrame, cfg: ShotLogicConfig) -> list[Candidate]:
    out = []
    for d in dets:
        if d["cls"] != "ball" or d["conf"] < cfg.ball_track_min_conf:
            continue
        size = hoop.size_in_rim_diameters(d["w"], d["h"])
        if not (cfg.ball_min_size_rim_d <= size <= cfg.ball_max_size_rim_d):
            continue
        x, y = hoop.to_hoop(d["x"], d["y"])
        out.append(Candidate(x, y, float(d["conf"])))
    return out


def in_watch_zone(c: Candidate, cfg: ShotLogicConfig) -> bool:
    return abs(c.x) <= cfg.watch_half_width and -cfg.watch_top <= c.y <= cfg.watch_bottom


def fit_accel(ts: list[float], vs: list[float]) -> tuple[float, float]:
    """Least-squares v = a + b*t + c*t^2 -> (acceleration 2c, RMS residual)."""
    n = len(ts)
    t0 = sum(ts) / n
    s = [0.0] * 5  # sums of t^k, k = 0..4 (t centred for conditioning)
    r = [0.0] * 3  # sums of v*t^k, k = 0..2
    for t, v in zip(ts, vs):
        u = t - t0
        p = 1.0
        for k in range(5):
            s[k] += p
            if k < 3:
                r[k] += v * p
            p *= u
    m = [[s[0], s[1], s[2], r[0]], [s[1], s[2], s[3], r[1]], [s[2], s[3], s[4], r[2]]]
    for i in range(3):  # Gauss-Jordan; the matrix is positive definite for >= 3 distinct times
        piv = m[i][i]
        if abs(piv) < 1e-12:
            return 0.0, math.inf
        for j in range(i, 4):
            m[i][j] /= piv
        for k in range(3):
            if k != i:
                f = m[k][i]
                for j in range(i, 4):
                    m[k][j] -= f * m[i][j]
    a, b, c = m[0][3], m[1][3], m[2][3]
    sq = 0.0
    for t, v in zip(ts, vs):
        u = t - t0
        e = v - (a + b * u + c * u * u)
        sq += e * e
    return 2.0 * c, math.sqrt(sq / n)


def _median(values: list[float]) -> float:
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def estimate_gravity(detections: dict, hoop: dict, cfg: ShotLogicConfig) -> tuple[float, float] | None:
    """The ball's acceleration in flight as seen in this video, (x, y) in rim radii / s^2 on image axes, measured
    from its own free flights, or None if there are too few clean ones.

    Free flight = a window of consecutive frames with exactly one confident ball whose path fits a parabola
    closely (rim and ground bounces don't fit); the median over all such windows is the result. It differs from
    the config's textbook value because of the ball's distance from the camera, and it is NOT straight down: a
    ball flying away from the camera seems to slow down in the picture (perspective), which adds a sideways part
    (about 10 degrees in our videos while the hoop's posts are vertical). So it drives the tracker's prediction,
    while the decision axes stay on the true vertical."""
    frame = HoopFrame.from_dict(hoop)  # image axes: zone and size filtering only
    n = cfg.gravity_fit_window
    windows: list[tuple[float, float]] = []
    run: list[tuple[float, float, float]] = []

    def flush() -> None:
        for s in range(0, len(run) - n + 1, max(1, n // 2)):
            w = run[s:s + n]
            ts = [p[0] for p in w]
            ax, ex = fit_accel(ts, [p[1] for p in w])
            ay, ey = fit_accel(ts, [p[2] for p in w])
            if max(ex, ey) <= cfg.gravity_fit_max_rms:
                windows.append((ax, ay))

    for fr in detections["frames"]:
        balls = [c for c in ball_candidates(fr["dets"], frame, cfg)
                 if c.conf >= cfg.gravity_fit_min_conf and in_watch_zone(c, cfg)]
        if len(balls) == 1:
            run.append((float(fr["t"]), balls[0].x, balls[0].y))
        else:
            flush()
            run = []
    flush()

    g0 = cfg.gravity_rim_r_per_s2
    plausible = [w for w in windows if 0.3 * g0 <= math.hypot(w[0], w[1]) <= 2.0 * g0]
    if len(plausible) < cfg.gravity_fit_min_windows:
        return None
    ax = _median([w[0] for w in plausible])
    ay = _median([w[1] for w in plausible])
    mag = math.hypot(ax, ay)
    if mag <= 0.0 or ay / mag < math.cos(math.radians(cfg.gravity_max_tilt_deg)):
        return None
    return ax, ay


def prepare(detections: dict, hoop: dict, cfg: ShotLogicConfig) -> tuple[HoopFrame, tuple[float, float]]:
    """Hoop frame (true vertical) and the ball's flight acceleration in it, for the tracker."""
    frame = HoopFrame.from_dict(hoop)
    accel = estimate_gravity(detections, hoop, cfg)
    return frame, (0.0, cfg.gravity_rim_r_per_s2) if accel is None else accel


def process(detections: dict, hoop: dict, cfg: ShotLogicConfig) -> dict:
    """Returns {"shots": [...]} matching shared/schema/events.schema.json."""
    frame, accel = prepare(detections, hoop, cfg)
    tracker = BallTracker(cfg, accel)
    machine = ShotStateMachine(cfg)
    shots = []
    last_t = None
    for fr in detections["frames"]:
        t = float(fr["t"])
        if last_t is not None and t <= last_t:
            raise ValueError(f"frame times must increase strictly (frame {fr['i']})")
        last_t = t
        # While a shot is under way a weak detection may (re)start the track; otherwise only a confident one.
        start_conf = cfg.ball_min_conf if machine.idle else cfg.ball_track_min_conf
        ball = tracker.step(t, ball_candidates(fr["dets"], frame, cfg), start_conf)
        event = machine.step(t, ball)
        if event is not None:
            shots.append(event.to_dict())
    if last_t is not None:
        event = machine.finish(last_t)
        if event is not None:
            shots.append(event.to_dict())
    return {"shots": shots}

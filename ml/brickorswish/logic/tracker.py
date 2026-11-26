"""Single-ball tracker: picks the detection nearest to where the Kalman filter expects the ball, coasts through
short gaps (net, rim) on the prediction, and restarts when the prediction has run away."""

import math
from dataclasses import dataclass

from .config import ShotLogicConfig
from .kalman import AxisKalman


@dataclass(frozen=True)
class Candidate:
    x: float  # rim units
    y: float
    conf: float


@dataclass(frozen=True)
class BallState:
    """est_*: the Kalman estimate, smooth but it assumes free fall, so it sinks below a ball resting on the rim.
    seen_*: the detection itself (the prediction while coasting). Rim decisions use seen_*."""
    t: float
    est_x: float
    est_y: float
    vx: float
    vy: float
    observed: bool  # False while coasting on the prediction
    track_id: int
    hits: int
    seen_x: float = 0.0
    seen_y: float = 0.0


class BallTracker:
    def __init__(self, cfg: ShotLogicConfig):
        self.cfg = cfg
        self._kx: AxisKalman | None = None
        self._ky: AxisKalman | None = None
        self._last_t = 0.0
        self._misses = 0
        self._hits = 0
        self._track_id = 0

    def step(self, t: float, candidates: list[Candidate], start_min_conf: float | None = None) -> BallState | None:
        """Any candidate can continue the track; only one >= start_min_conf (default ball_min_conf) starts one."""
        cfg = self.cfg
        if self._kx is not None:
            dt = t - self._last_t
            self._last_t = t
            self._kx.predict(dt, 0.0, cfg.kalman_accel_noise)
            self._ky.predict(dt, cfg.gravity_rim_r_per_s2, cfg.kalman_accel_noise)
            match = self._match(candidates)
            if match is not None:
                cand, manoeuvre = match
                if manoeuvre:
                    self._kx.inflate_velocity(cfg.kalman_init_vel_var)
                    self._ky.inflate_velocity(cfg.kalman_init_vel_var)
                self._kx.update(cand.x, cfg.kalman_meas_var)
                self._ky.update(cand.y, cfg.kalman_meas_var)
                self._misses = 0
                self._hits += 1
                return self._state(t, observed=True, meas=cand)
            self._misses += 1
            min_conf = cfg.ball_min_conf if start_min_conf is None else start_min_conf
            fresh = self._pick_start([c for c in candidates if c.conf >= min_conf])
            # The prediction has run away from the ball (rim / net stopped it): start over on the detection.
            runaway = fresh is not None and self._misses > cfg.restart_after_coast_frames
            if not runaway and self._misses <= cfg.max_coast_frames:
                return self._state(t, observed=False)
            self._kx = self._ky = None

        min_conf = cfg.ball_min_conf if start_min_conf is None else start_min_conf
        start = self._pick_start([c for c in candidates if c.conf >= min_conf])
        if start is None:
            return None
        self._track_id += 1
        self._kx = AxisKalman(start.x, cfg.kalman_meas_var, cfg.kalman_init_vel_var)
        self._ky = AxisKalman(start.y, cfg.kalman_meas_var, cfg.kalman_init_vel_var)
        self._last_t = t
        self._misses = 0
        self._hits = 1
        return self._state(t, observed=True, meas=start)

    def _match(self, candidates: list[Candidate]) -> tuple[Candidate, bool] | None:
        """Best candidate inside the gate; else the nearest within reacquire_radius, flagged as a bounce.
        The gate: distance from the prediction in standard deviations, squared (gate_chi2 9.21 = 99% of real balls)."""
        cfg = self.cfg
        sx = self._kx.innovation_var(cfg.kalman_meas_var)
        sy = self._ky.innovation_var(cfg.kalman_meas_var)
        best, best_d2 = None, math.inf
        near, near_d = None, math.inf
        for c in candidates:
            dx = c.x - self._kx.pos
            dy = c.y - self._ky.pos
            d2 = dx * dx / sx + dy * dy / sy
            if d2 < best_d2:
                best, best_d2 = c, d2
            d = math.hypot(dx, dy)
            if d < near_d:
                near, near_d = c, d
        if best is not None and best_d2 <= cfg.gate_chi2:
            return best, False
        if near is not None and near_d <= cfg.reacquire_radius:
            return near, True
        return None

    def _pick_start(self, candidates: list[Candidate]) -> Candidate | None:
        cfg = self.cfg
        best = None
        for c in candidates:
            if abs(c.x) > cfg.watch_half_width:
                continue
            if c.y < -cfg.watch_top or c.y > cfg.watch_bottom:
                continue
            if best is None or c.conf > best.conf:
                best = c
        return best

    def _state(self, t: float, observed: bool, meas: Candidate | None = None) -> BallState:
        return BallState(
            t=t,
            est_x=self._kx.pos,
            est_y=self._ky.pos,
            vx=self._kx.vel,
            vy=self._ky.vel,
            observed=observed,
            track_id=self._track_id,
            hits=self._hits,
            seen_x=meas.x if meas is not None else self._kx.pos,
            seen_y=meas.y if meas is not None else self._ky.pos,
        )

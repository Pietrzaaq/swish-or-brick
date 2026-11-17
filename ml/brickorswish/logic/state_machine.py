"""Make/miss state machine in hoop units (see docs/implementation-analysis.md §3.4).

IDLE -> APPROACH        ball in approach zone moving down
APPROACH -> PENDING_MAKE  crosses rim plane (y=0) downward with |x| <= make_max_abs_x
         -> RATTLE        crosses near the rim edge, or reverses upward near the rim
         -> MISS          crosses wide, or leaves the zone sideways
PENDING_MAKE -> MAKE      seen below make_confirm_depth, still inside net_exit_max_abs_x
             -> RATTLE    pops back up
             -> MISS      rolls out sideways below the rim plane
RATTLE -> PENDING_MAKE    comes back down through the rim
       -> MISS            comes down outside, or leaves the zone
       -> UNCERTAIN       rattle_timeout_s without resolution
any active -> UNCERTAIN   ball not observed for lost_timeout_s (pending_make_lost_timeout_s
                          in PENDING_MAKE, where the net hides the ball)
"""

from dataclasses import dataclass

from .config import ShotLogicConfig
from .tracker import BallState

IDLE = "idle"
APPROACH = "approach"
PENDING_MAKE = "pending_make"
RATTLE = "rattle"

MAKE = "make"
MISS = "miss"
UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ShotEvent:
    shot: int
    outcome: str
    t_start: float
    t_resolve: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "shot": self.shot,
            "outcome": self.outcome,
            "t_start": round(self.t_start, 4),
            "t_resolve": round(self.t_resolve, 4),
            "reason": self.reason,
        }


class ShotStateMachine:
    def __init__(self, cfg: ShotLogicConfig):
        self.cfg = cfg
        self.state = IDLE
        self._shots = 0
        self._t_start = 0.0
        self._t_state = 0.0
        self._t_last_seen = 0.0
        self._cooldown_until = -1.0
        self._prev: BallState | None = None

    def step(self, t: float, ball: BallState | None) -> ShotEvent | None:
        if ball is not None and ball.hits < self.cfg.min_hits:
            ball = None
        prev = self._prev
        if prev is not None and (ball is None or ball.track_id != prev.track_id):
            prev = None
        self._prev = ball
        if ball is not None and ball.observed:
            self._t_last_seen = t

        if self.state == IDLE:
            self._idle(t, ball)
            return None
        return self._active(t, ball, prev)

    def finish(self, t: float) -> ShotEvent | None:
        """End of stream: an unresolved attempt is uncertain."""
        if self.state == IDLE:
            return None
        return self._resolve(t, UNCERTAIN, "stream_ended")

    # --- states -------------------------------------------------------------

    def _idle(self, t: float, ball: BallState | None) -> None:
        cfg = self.cfg
        if ball is None or t < self._cooldown_until:
            return
        in_zone = abs(ball.x) <= cfg.approach_half_width and -cfg.approach_height <= ball.y < 0.0
        if in_zone and ball.vy > 0.0:
            self._t_start = t
            self._t_last_seen = t
            self._enter(APPROACH, t)

    def _active(self, t: float, ball: BallState | None, prev: BallState | None) -> ShotEvent | None:
        cfg = self.cfg
        lost_limit = cfg.pending_make_lost_timeout_s if self.state == PENDING_MAKE else cfg.lost_timeout_s
        if t - self._t_last_seen > lost_limit:
            return self._resolve(t, UNCERTAIN, "lost_in_net" if self.state == PENDING_MAKE else "lost")
        if self.state == RATTLE and t - self._t_state > cfg.rattle_timeout_s:
            return self._resolve(t, UNCERTAIN, "rattle_timeout")
        if ball is None:
            return None

        if self.state == APPROACH:
            return self._approach(t, ball, prev)
        if self.state == PENDING_MAKE:
            return self._pending_make(t, ball)
        return self._rattle(t, ball, prev)

    def _approach(self, t: float, ball: BallState, prev: BallState | None) -> ShotEvent | None:
        cfg = self.cfg
        x_cross = _crossing_x(prev, ball)
        if x_cross is not None:
            if abs(x_cross) <= cfg.make_max_abs_x:
                self._enter(PENDING_MAKE, t)
            elif abs(x_cross) <= cfg.rim_contact_abs_x:
                self._enter(RATTLE, t)
            else:
                return self._resolve(t, MISS, "wide")
            return None
        if ball.observed and ball.vy < 0.0 and ball.y > -cfg.near_rim_height and abs(ball.x) <= cfg.rim_contact_abs_x:
            self._enter(RATTLE, t)
            return None
        if abs(ball.x) > cfg.approach_half_width:
            return self._resolve(t, MISS, "left_zone")
        return None

    def _pending_make(self, t: float, ball: BallState) -> ShotEvent | None:
        cfg = self.cfg
        if not ball.observed:
            return None  # the net hides the ball; never decide on a prediction here
        if ball.vy < 0.0 and ball.y < 0.0:
            self._enter(RATTLE, t)
            return None
        if ball.y > 0.0 and abs(ball.x) > cfg.net_exit_max_abs_x:
            return self._resolve(t, MISS, "rolled_out")
        if ball.y >= cfg.make_confirm_depth:
            return self._resolve(t, MAKE, "through_rim")
        return None

    def _rattle(self, t: float, ball: BallState, prev: BallState | None) -> ShotEvent | None:
        cfg = self.cfg
        x_cross = _crossing_x(prev, ball)
        if x_cross is not None:
            if abs(x_cross) <= cfg.make_max_abs_x:
                self._enter(PENDING_MAKE, t)
                return None
            return self._resolve(t, MISS, "rim_out")
        if ball.observed and abs(ball.x) > cfg.approach_half_width:
            return self._resolve(t, MISS, "rim_out")
        return None

    # --- helpers ------------------------------------------------------------

    def _enter(self, state: str, t: float) -> None:
        self.state = state
        self._t_state = t

    def _resolve(self, t: float, outcome: str, reason: str) -> ShotEvent:
        self._shots += 1
        event = ShotEvent(self._shots, outcome, self._t_start, t, reason)
        self.state = IDLE
        self._cooldown_until = t + self.cfg.debounce_s
        return event


def _crossing_x(prev: BallState | None, cur: BallState) -> float | None:
    """x where the ball crossed the rim plane (y=0) moving down, linearly interpolated."""
    if prev is None or not (prev.y < 0.0 <= cur.y):
        return None
    f = (0.0 - prev.y) / (cur.y - prev.y)
    return prev.x + f * (cur.x - prev.x)

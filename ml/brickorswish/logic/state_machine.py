"""Make/miss state machine in hoop units (see docs/implementation-analysis.md §3.4).

IDLE -> APPROACH        ball in approach zone moving down
APPROACH -> PENDING_MAKE  crosses rim plane (y=0) downward with |x| <= make_max_abs_x
         -> RATTLE        crosses near the rim edge, or reverses upward near the rim
         -> MISS          crosses wide, or leaves the zone sideways
PENDING_MAKE -> MAKE      seen below make_confirm_depth, still inside net_exit_max_abs_x
             -> RATTLE    pops back up
             -> MISS      rolls out sideways below the rim plane
RATTLE -> PENDING_MAKE    comes back down through the rim
       -> MISS            comes down outside, leaves the zone, or is seen deep below the rim after
                          crossing it at the edge (it went down outside the make zone)
       -> UNCERTAIN       rattle_timeout_s without resolution
any active -> UNCERTAIN   ball not observed for lost_timeout_s (pending_make_lost_timeout_s
                          in PENDING_MAKE, where the net hides the ball)

Only SEEN balls decide anything. A rim crossing is placed by interpolating between the last ball seen
above the rim plane and the first one seen below it (at most cross_max_gap_s apart, across track restarts):
the detector loses the ball exactly where it overlaps rim and net, and the tracker's free-fall prediction
knows nothing about rims and nets, so a crossing computed from predictions is a guess.
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
        self._above: BallState | None = None  # last ball SEEN above the rim plane during this shot
        self._cross_x: float | None = None  # where this shot last crossed the rim plane downward

    @property
    def idle(self) -> bool:
        return self.state == IDLE

    def step(self, t: float, ball: BallState | None) -> ShotEvent | None:
        if self.state == IDLE:
            if ball is not None and ball.hits < self.cfg.min_hits:
                ball = None  # a shot only starts on a confirmed track
            if ball is not None and ball.observed:
                self._t_last_seen = t
            self._idle(t, ball)
            return None

        if ball is not None and ball.observed:
            self._t_last_seen = t
        event = self._active(t, ball)
        if event is None and ball is not None and ball.observed and ball.my < 0.0:
            self._above = ball
        return event

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
            self._above = ball if ball.observed else None
            self._cross_x = None
            self._enter(APPROACH, t)

    def _active(self, t: float, ball: BallState | None) -> ShotEvent | None:
        cfg = self.cfg
        lost_limit = cfg.pending_make_lost_timeout_s if self.state == PENDING_MAKE else cfg.lost_timeout_s
        if t - self._t_last_seen > lost_limit:
            return self._resolve(t, UNCERTAIN, "lost_in_net" if self.state == PENDING_MAKE else "lost")
        if self.state == RATTLE and t - self._t_state > cfg.rattle_timeout_s:
            return self._resolve(t, UNCERTAIN, "rattle_timeout")
        if ball is None or not ball.observed:
            return None  # never decide on a prediction

        if self.state == APPROACH:
            return self._approach(t, ball)
        if self.state == PENDING_MAKE:
            return self._pending_make(t, ball)
        return self._rattle(t, ball)

    def _crossing(self, ball: BallState) -> float | None:
        """x where the ball crossed the rim plane downward, between the last ball seen above it and this
        one (seen at or below it), or None."""
        above = self._above
        if above is None or ball.my < 0.0 or ball.t - above.t > self.cfg.cross_max_gap_s:
            return None
        self._above = None
        f = (0.0 - above.my) / (ball.my - above.my)
        self._cross_x = above.mx + f * (ball.mx - above.mx)
        return self._cross_x

    def _approach(self, t: float, ball: BallState) -> ShotEvent | None:
        cfg = self.cfg
        x_cross = self._crossing(ball)
        if x_cross is not None:
            if abs(x_cross) <= cfg.make_max_abs_x:
                self._enter(PENDING_MAKE, t)
            elif abs(x_cross) <= cfg.rim_contact_abs_x:
                self._enter(RATTLE, t)
            else:
                return self._resolve(t, MISS, "wide")
            return None
        if ball.vy < 0.0 and ball.my > -cfg.near_rim_height and abs(ball.mx) <= cfg.rim_contact_abs_x:
            self._enter(RATTLE, t)
            return None
        if abs(ball.mx) > cfg.approach_half_width:
            return self._resolve(t, MISS, "left_zone")
        return None

    def _pending_make(self, t: float, ball: BallState) -> ShotEvent | None:
        cfg = self.cfg
        if ball.vy < 0.0 and ball.my < 0.0:
            self._enter(RATTLE, t)
            return None
        if ball.my > 0.0 and abs(ball.mx) > cfg.net_exit_max_abs_x:
            return self._resolve(t, MISS, "rolled_out")
        if ball.my >= cfg.make_confirm_depth:
            return self._resolve(t, MAKE, "through_rim")
        return None

    def _rattle(self, t: float, ball: BallState) -> ShotEvent | None:
        cfg = self.cfg
        x_cross = self._crossing(ball)
        if x_cross is not None:
            if abs(x_cross) <= cfg.make_max_abs_x:
                self._enter(PENDING_MAKE, t)
                return None
            return self._resolve(t, MISS, "rim_out")
        if abs(ball.mx) > cfg.approach_half_width:
            return self._resolve(t, MISS, "rim_out")
        if (ball.my >= cfg.make_confirm_depth and self._cross_x is not None
                and abs(self._cross_x) > cfg.make_max_abs_x):
            return self._resolve(t, MISS, "rim_out")  # went down at the edge and never came back through
        return None

    # --- helpers ------------------------------------------------------------

    def _enter(self, state: str, t: float) -> None:
        self.state = state
        self._t_state = t

    def _resolve(self, t: float, outcome: str, reason: str) -> ShotEvent:
        self._shots += 1
        event = ShotEvent(self._shots, outcome, self._t_start, t, reason)
        self.state = IDLE
        self._above = None
        self._cross_x = None
        self._cooldown_until = t + self.cfg.debounce_s
        return event

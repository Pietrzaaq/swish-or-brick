"""Make/miss state machine (see docs/implementation-analysis.md §3.4).

Positions are in rim units (geometry.py): (0, 0) is the rim centre, 1 = rim radius, y grows downward, so y < 0 is
above the rim and |x| < 1 is inside it.

IDLE -> APPROACH        ball in approach zone moving down
APPROACH -> PENDING_MAKE  crosses rim plane (y=0) downward with |x| <= make_max_abs_x
         -> RATTLE        crosses near the rim edge, or reverses upward near the rim
         -> MISS          crosses wide, or leaves the zone sideways
PENDING_MAKE -> MAKE      seen below make_confirm_depth (it may already be leaving the net sideways)
             -> RATTLE    pops back up
RATTLE -> PENDING_MAKE    comes back down through the rim
       -> MAKE / MISS     first seen >= exit_depth below the rim: within exit_max_abs_x make, else miss
       -> MISS            crosses wide (|x| > rim_contact_abs_x), or leaves the zone
       -> UNCERTAIN       rattle_timeout_s without resolution
any active -> UNCERTAIN   not seen for lost_timeout_s (pending_make_lost_timeout_s in PENDING_MAKE: the net hides it)

Only SEEN balls decide. A rim crossing is interpolated between the last ball seen above the rim plane and the
first seen below it, even across track restarts: the detector loses the ball where it overlaps rim and net, and
the free-fall prediction knows nothing about them. Fitting the incoming arc is worse: any touch of rim or board
changes the path. A ball rolling around the rim does not follow the straight line, so an edge crossing is
decided by where the ball comes out below.

make_max_abs_x: the ball's centre must cross within rim radius - ball radius ~ 0.47, plus measurement noise.
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
        if event is None and ball is not None and ball.observed and ball.seen_y < 0.0:
            self._above = ball
        return event

    def finish(self, t: float) -> ShotEvent | None:
        """End of stream: an unresolved attempt is uncertain."""
        if self.state == IDLE:
            return None
        return self._resolve(t, UNCERTAIN, "stream_ended")

    def _idle(self, t: float, ball: BallState | None) -> None:
        cfg = self.cfg
        if ball is None or t < self._cooldown_until:
            return
        in_zone = abs(ball.est_x) <= cfg.approach_half_width and -cfg.approach_height <= ball.est_y < 0.0
        if in_zone and ball.vy > 0.0:
            self._t_start = t
            self._t_last_seen = t
            self._above = ball if ball.observed else None
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
        """x where the ball crossed the rim plane downward since the last sighting above it, or None."""
        above = self._above
        if above is None or ball.seen_y < 0.0 or ball.t - above.t > self.cfg.cross_max_gap_s:
            return None
        self._above = None
        f = (0.0 - above.seen_y) / (ball.seen_y - above.seen_y)
        return above.seen_x + f * (ball.seen_x - above.seen_x)

    def _exit(self, t: float, ball: BallState) -> ShotEvent | None:
        """A ball seen at least exit_depth below the rim plane after touching the rim: out of the net, or beside it."""
        cfg = self.cfg
        if ball.seen_y < cfg.exit_depth:
            return None
        if abs(ball.seen_x) <= cfg.exit_max_abs_x:
            return self._resolve(t, MAKE, "through_net")
        return self._resolve(t, MISS, "rim_out")

    def _approach(self, t: float, ball: BallState) -> ShotEvent | None:
        cfg = self.cfg
        x_cross = self._crossing(ball)
        if x_cross is not None:
            if abs(x_cross) <= cfg.make_max_abs_x:
                self._enter(PENDING_MAKE, t)
            elif abs(x_cross) <= cfg.rim_contact_abs_x:
                self._enter(RATTLE, t)
                return self._exit(t, ball)  # this sighting may already be below the net
            else:
                return self._resolve(t, MISS, "wide")
            return None
        if ball.vy < 0.0 and ball.seen_y > -cfg.near_rim_height and abs(ball.seen_x) <= cfg.rim_contact_abs_x:
            self._enter(RATTLE, t)
            return None
        if abs(ball.seen_x) > cfg.approach_half_width:
            return self._resolve(t, MISS, "left_zone")
        return None

    def _pending_make(self, t: float, ball: BallState) -> ShotEvent | None:
        cfg = self.cfg
        if ball.vy < 0.0 and ball.seen_y < 0.0:
            self._enter(RATTLE, t)
            return None
        if ball.seen_y >= cfg.make_confirm_depth:
            return self._resolve(t, MAKE, "through_rim")
        return None

    def _rattle(self, t: float, ball: BallState) -> ShotEvent | None:
        cfg = self.cfg
        x_cross = self._crossing(ball)
        if x_cross is not None and abs(x_cross) <= cfg.make_max_abs_x:
            self._enter(PENDING_MAKE, t)
            return None
        if x_cross is not None and abs(x_cross) > cfg.rim_contact_abs_x:
            return self._resolve(t, MISS, "rim_out")
        if abs(ball.seen_x) > cfg.approach_half_width:
            return self._resolve(t, MISS, "rim_out")
        return self._exit(t, ball)

    def _enter(self, state: str, t: float) -> None:
        self.state = state
        self._t_state = t

    def _resolve(self, t: float, outcome: str, reason: str) -> ShotEvent:
        self._shots += 1
        event = ShotEvent(self._shots, outcome, self._t_start, t, reason)
        self.state = IDLE
        self._above = None
        self._cooldown_until = t + self.cfg.debounce_s
        return event

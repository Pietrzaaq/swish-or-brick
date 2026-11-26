"""process(): detections + hoop calibration + config -> shot events. Pure function."""

import math

from .config import ShotLogicConfig
from .geometry import HoopFrame
from .state_machine import ShotStateMachine
from .tracker import BallTracker, Candidate


def ball_candidates(dets: list[dict], hoop: HoopFrame, cfg: ShotLogicConfig,
                    static: frozenset = frozenset()) -> list[Candidate]:
    out = []
    for d in dets:
        if d["cls"] != "ball" or d["conf"] < cfg.ball_track_min_conf:
            continue
        size = hoop.size_in_rim_diameters(d["w"], d["h"])
        if not (cfg.ball_min_size_rim_d <= size <= cfg.ball_max_size_rim_d):
            continue
        x, y = hoop.to_hoop(d["x"], d["y"])
        if static and near_static(x, y, static, cfg):
            continue
        out.append(Candidate(x, y, float(d["conf"])))
    return out


def _cell(x: float, y: float, cfg: ShotLogicConfig) -> tuple[int, int]:
    # floor(v + 0.5), not round(): Python rounds halves to even, Dart away from zero.
    return math.floor(x / cfg.static_cell + 0.5), math.floor(y / cfg.static_cell + 0.5)


def _near(cell: tuple[int, int], cells) -> bool:
    return any((cell[0] + i, cell[1] + j) in cells for i in (-1, 0, 1) for j in (-1, 0, 1))


def near_static(x: float, y: float, static: frozenset, cfg: ShotLogicConfig) -> bool:
    return _near(_cell(x, y, cfg), static)


def static_spots(detections: dict, hoop: HoopFrame, cfg: ShotLogicConfig) -> frozenset:
    """Cells where a ball candidate stayed (also there the frame before) in >= static_min_seconds different seconds:
    a ball-like thing that never moves, like the rim's bracket. Counting only stays keeps out the rim centre, which
    real balls pass through on every make. Needs the whole session (a live run would build it up as it goes)."""
    seconds: dict[tuple[int, int], set[int]] = {}
    prev: set[tuple[int, int]] = set()
    for fr in detections["frames"]:
        sec = math.floor(float(fr["t"]))
        cells = {_cell(c.x, c.y, cfg) for c in ball_candidates(fr["dets"], hoop, cfg)}
        for cell in cells:
            if _near(cell, prev):
                seconds.setdefault(cell, set()).add(sec)
        prev = cells
    return frozenset(cell for cell, secs in seconds.items() if len(secs) >= cfg.static_min_seconds)


def process(detections: dict, hoop: dict, cfg: ShotLogicConfig) -> dict:
    """Returns {"shots": [...]} matching shared/schema/events.schema.json."""
    frame = HoopFrame.from_dict(hoop)
    static = static_spots(detections, frame, cfg)
    tracker = BallTracker(cfg)
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
        ball = tracker.step(t, ball_candidates(fr["dets"], frame, cfg, static), start_conf)
        event = machine.step(t, ball)
        if event is not None:
            shots.append(event.to_dict())
    if last_t is not None:
        event = machine.finish(last_t)
        if event is not None:
            shots.append(event.to_dict())
    return {"shots": shots}

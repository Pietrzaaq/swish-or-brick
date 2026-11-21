"""process(): detections + hoop calibration + config -> shot events. Pure function."""

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


def process(detections: dict, hoop: dict, cfg: ShotLogicConfig) -> dict:
    """Returns {"shots": [...]} matching shared/schema/events.schema.json."""
    frame = HoopFrame.from_dict(hoop)
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
        ball = tracker.step(t, ball_candidates(fr["dets"], frame, cfg), start_conf)
        event = machine.step(t, ball)
        if event is not None:
            shots.append(event.to_dict())
    if last_t is not None:
        event = machine.finish(last_t)
        if event is not None:
            shots.append(event.to_dict())
    return {"shots": shots}

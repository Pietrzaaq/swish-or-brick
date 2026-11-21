"""Typed view of shared/config/shot_logic.v<N>.json."""

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ShotLogicConfig:
    version: int

    # Candidate filtering. A track starts only on ball_min_conf; once a shot is under way, detections down to
    # ball_track_min_conf are used too (the ball fades to 15-30% confidence where it overlaps rim, net, board).
    ball_min_conf: float
    ball_track_min_conf: float
    ball_min_size_rim_d: float
    ball_max_size_rim_d: float

    # Tracker (hoop units, seconds)
    gravity_rim_r_per_s2: float
    kalman_accel_noise: float
    kalman_meas_var: float
    kalman_init_vel_var: float
    gate_chi2: float
    reacquire_radius: float
    max_coast_frames: int
    # After this many frames without a sighting, a credible detection outside the gate restarts the track
    # (the free-fall prediction has run away from a ball the rim or net stopped).
    restart_after_coast_frames: int
    min_hits: int

    # Region where a new track may start
    watch_half_width: float
    watch_top: float
    watch_bottom: float

    # Approach zone (above the rim)
    approach_half_width: float
    approach_height: float

    # Make / miss geometry
    make_max_abs_x: float
    rim_contact_abs_x: float
    near_rim_height: float
    net_exit_max_abs_x: float
    make_confirm_depth: float
    # Rim crossings are judged between the last ball SEEN above the rim plane and the first one seen below it,
    # never from predictions; if those two sightings are further apart than this, the crossing is not judged.
    cross_max_gap_s: float

    # Timing
    rattle_timeout_s: float
    lost_timeout_s: float
    pending_make_lost_timeout_s: float
    debounce_s: float

    @classmethod
    def from_dict(cls, data: dict) -> "ShotLogicConfig":
        names = [f.name for f in fields(cls)]
        missing = [n for n in names if n not in data]
        unknown = [k for k in data if k not in names]
        if missing or unknown:
            raise ValueError(f"config mismatch: missing={missing} unknown={unknown}")
        values = {}
        for f in fields(cls):
            raw = data[f.name]
            values[f.name] = int(raw) if f.type is int or f.type == "int" else float(raw)
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "ShotLogicConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

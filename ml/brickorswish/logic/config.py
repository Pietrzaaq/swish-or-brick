"""Typed view of shared/config/shot_logic.v<N>.json.

Distances are in rim units (1 = rim radius, see geometry.py), times in seconds.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ShotLogicConfig:
    version: int

    # --- 1. Which detections count as the ball ---
    # A track starts only on ball_min_conf; during a shot ball_track_min_conf is enough
    # (the ball fades to 15-30% confidence where it overlaps rim, net, board).
    ball_min_conf: float
    ball_track_min_conf: float
    ball_min_size_rim_d: float  # box size, in rim diameters
    ball_max_size_rim_d: float
    # Ball-like things that never move (the rim's bracket): a static_cell-sized cell where a ball candidate stayed
    # in >= static_min_seconds different seconds is ignored, with its neighbours.
    static_cell: float
    static_min_seconds: int

    # --- 2. Following the ball (tracker + Kalman) ---
    gravity_rim_r_per_s2: float  # 9.81 m/s^2 / 0.2286 m rim radius
    kalman_accel_noise: float
    kalman_meas_var: float
    kalman_init_vel_var: float
    gate_chi2: float
    reacquire_radius: float  # nearest detection this close continues the track after a bounce
    max_coast_frames: int  # frames followed on the prediction alone before the track is dropped
    # After this many frames without a sighting, a detection outside the gate restarts the track
    # (the prediction has run away from a ball the rim or net stopped).
    restart_after_coast_frames: int
    min_hits: int  # sightings before a track may start a shot
    # Region where a new track may start
    watch_half_width: float
    watch_top: float
    watch_bottom: float

    # --- 3. Make / miss decisions (state_machine.py) ---
    # Approach zone (above the rim)
    approach_half_width: float
    approach_height: float
    make_max_abs_x: float
    rim_contact_abs_x: float
    near_rim_height: float
    make_confirm_depth: float
    # Max gap between the last sighting above the rim plane and the first below it for a crossing to count.
    cross_max_gap_s: float
    # After an edge crossing or a rattle, the first sighting >= exit_depth below the rim plane decides: within
    # exit_max_abs_x it came out of the net (make), else it fell beside it.
    # exit_max_abs_x: a ball's centre inside the net is within rim radius - ball radius ~ 0.47.
    exit_depth: float
    exit_max_abs_x: float

    # --- 4. Timeouts ---
    rattle_timeout_s: float
    lost_timeout_s: float
    pending_make_lost_timeout_s: float
    debounce_s: float  # no new shot this soon after the last one

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

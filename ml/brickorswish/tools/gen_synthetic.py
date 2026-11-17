"""Generate synthetic fixture cases into shared/fixtures/cases/synthetic_*.

Ball paths are simulated in hoop units with simple physics (gravity, bounces as
velocity changes, net drag), then projected to pixels with a fixed calibration and
written as detector output. The ground-truth outcome is known by construction.

Deterministic: a fixed seed per case. Re-run after changing this file:

    py -m brickorswish.tools.gen_synthetic          (from ml/)
"""

import json
import math
import random
import shutil
from dataclasses import dataclass, field

from ..logic.geometry import HoopFrame
from ..paths import FIXTURE_CASES

FPS = 30.0
G = 42.9  # rim radii / s^2
BALL_D_RIM_D = 0.52  # ball diameter / rim diameter

HOOP = {
    "image_width": 1920,
    "image_height": 1080,
    "left": [900.0, 400.0],
    "right": [1020.0, 400.0],
    "front": [960.0, 415.0],
}


@dataclass
class Phase:
    """Constant-acceleration flight. At phase start, velocity is optionally replaced
    (set_v) or scaled (scale_v) to model a bounce or the net catching the ball."""

    duration: float
    accel_y: float = G
    set_v: tuple[float, float] | None = None
    scale_v: tuple[float, float] | None = None


@dataclass
class Shot:
    t0: float
    start: tuple[float, float]
    v0: tuple[float, float]
    phases: list[Phase]
    hidden: list[tuple[float, float]] = field(default_factory=list)  # absolute time windows


def simulate(shot: Shot, t: float) -> tuple[float, float] | None:
    """Ball position (hoop units) at absolute time t, or None outside the shot."""
    if t < shot.t0:
        return None
    x, y = shot.start
    vx, vy = shot.v0
    local = t - shot.t0
    for ph in shot.phases:
        if ph.set_v is not None:
            vx, vy = ph.set_v
        if ph.scale_v is not None:
            vx, vy = vx * ph.scale_v[0], vy * ph.scale_v[1]
        step = min(local, ph.duration)
        x += vx * step
        y += vy * step + 0.5 * ph.accel_y * step * step
        vy += ph.accel_y * step
        if local <= ph.duration:
            return x, y
        local -= ph.duration
    return None


def fall_time(dy: float) -> float:
    """Time to fall dy units from rest."""
    return math.sqrt(2.0 * dy / G)


def drop_shot(t0: float, x_start: float, x_at_rim: float, rim_y: float = 0.0, top: float = -9.0) -> tuple[Shot, float]:
    """A ball released from rest at (x_start, top) reaching (x_at_rim, rim_y). Returns the
    shot (first phase only) and the time to reach rim_y."""
    tf = fall_time(rim_y - top)
    vx = (x_at_rim - x_start) / tf
    return Shot(t0, (x_start, top), (vx, 0.0), [Phase(tf)]), tf


def make_through_net(t0: float, x_start: float, x_cross: float, hide_net: bool = True, lost_forever: bool = False) -> Shot:
    shot, tf = drop_shot(t0, x_start, x_cross)
    shot.phases += [
        Phase(0.25, accel_y=0.0, scale_v=(0.2, 0.35)),  # net drag
        Phase(0.5),  # free fall below the net
    ]
    t_cross = t0 + tf
    if lost_forever:
        shot.hidden.append((t_cross + 0.03, t_cross + 10.0))
    elif hide_net:
        shot.hidden.append((t_cross + 0.03, t_cross + 0.13))
    return shot


def case_swish() -> tuple[list[Shot], list[str]]:
    return [make_through_net(0.5, -1.2, 0.1)], ["make"]


def case_wide_miss() -> tuple[list[Shot], list[str]]:
    shot, _ = drop_shot(0.5, -1.0, 2.6)
    shot.phases.append(Phase(0.4))
    return [shot], ["miss"]


def case_front_rim_out() -> tuple[list[Shot], list[str]]:
    shot, tf = drop_shot(0.5, -1.5, -0.95, rim_y=-0.3)
    vy_impact = G * tf
    shot.phases.append(Phase(1.0, set_v=(-3.0, -0.45 * vy_impact)))
    return [shot], ["miss"]


def case_rattle_in() -> tuple[list[Shot], list[str]]:
    shot, tf = drop_shot(0.5, -0.4, 0.85, rim_y=-0.3)
    vy_impact = G * tf
    up = -0.35 * vy_impact
    t_air = 2.0 * (-up) / G  # back to the same height
    shot.phases += [
        Phase(t_air, set_v=(-1.2, up)),
        Phase(0.4 / 30.0 * 3),  # short fall to the rim plane and a bit below
        Phase(0.25, accel_y=0.0, scale_v=(0.2, 0.35)),
        Phase(0.5),
    ]
    return [shot], ["make"]


def case_lost_in_net() -> tuple[list[Shot], list[str]]:
    return [make_through_net(0.5, -1.2, 0.1, lost_forever=True)], ["uncertain"]


def case_two_shots() -> tuple[list[Shot], list[str]]:
    miss, _ = drop_shot(3.5, -1.0, 2.6)
    miss.phases.append(Phase(0.4))
    return [make_through_net(0.5, -1.2, 0.1), miss], ["make", "miss"]


def case_gap_at_rim() -> tuple[list[Shot], list[str]]:
    """Detector misses the ball for 4 frames around the rim plane (motion blur)."""
    shot = make_through_net(0.5, -1.2, 0.1, hide_net=False)
    t_cross = 0.5 + fall_time(9.0)
    shot.hidden.append((t_cross - 0.08, t_cross + 0.05))
    return [shot], ["make"]


@dataclass
class Noise:
    jitter: float = 0.0  # std dev, hoop units
    drop_rate: float = 0.0
    distractor: tuple[float, float] | None = None  # static false positive, hoop units


CASES = {
    "synthetic_swish": (case_swish, Noise(), 1),
    "synthetic_wide_miss": (case_wide_miss, Noise(), 2),
    "synthetic_front_rim_out": (case_front_rim_out, Noise(), 3),
    "synthetic_rattle_in": (case_rattle_in, Noise(), 4),
    "synthetic_lost_in_net": (case_lost_in_net, Noise(), 5),
    "synthetic_two_shots": (case_two_shots, Noise(), 6),
    "synthetic_gap_at_rim": (case_gap_at_rim, Noise(), 7),
    "synthetic_swish_noisy": (case_swish, Noise(jitter=0.05, drop_rate=0.15, distractor=(8.0, -3.0)), 8),
    "synthetic_rattle_in_noisy": (case_rattle_in, Noise(jitter=0.05, drop_rate=0.1, distractor=(-7.5, -6.0)), 9),
}


def render(shots: list[Shot], noise: Noise, seed: int, duration: float) -> dict:
    rng = random.Random(seed)
    hoop = HoopFrame.from_dict(HOOP)
    size_px = BALL_D_RIM_D * 2.0 * hoop.radius_px
    frames = []
    n = int(round(duration * FPS))
    for i in range(n):
        t = i / FPS
        dets = []
        for shot in shots:
            pos = simulate(shot, t)
            if pos is None or any(a <= t <= b for a, b in shot.hidden):
                continue
            if noise.drop_rate and rng.random() < noise.drop_rate:
                continue
            x = pos[0] + (rng.gauss(0.0, noise.jitter) if noise.jitter else 0.0)
            y = pos[1] + (rng.gauss(0.0, noise.jitter) if noise.jitter else 0.0)
            dets.append(_det(hoop, x, y, size_px, 0.8))
        if noise.distractor is not None:
            dets.append(_det(hoop, noise.distractor[0], noise.distractor[1], size_px, 0.6))
        frames.append({"i": i, "t": round(t, 6), "dets": dets})
    return {"fps": FPS, "source": "synthetic", "frames": frames}


def _det(hoop: HoopFrame, x: float, y: float, size_px: float, conf: float) -> dict:
    px, py = hoop.to_pixels(x, y)
    return {"cls": "ball", "conf": conf, "x": round(px, 2), "y": round(py, 2), "w": round(size_px, 2), "h": round(size_px, 2)}


def generate() -> list[str]:
    for old in FIXTURE_CASES.glob("synthetic_*"):
        shutil.rmtree(old)
    names = []
    for name, (builder, noise, seed) in CASES.items():
        shots, outcomes = builder()
        duration = max(s.t0 for s in shots) + 3.0
        case_dir = FIXTURE_CASES / name
        case_dir.mkdir(parents=True)
        _write(case_dir / "hoop.json", HOOP)
        _write(case_dir / "detections.json", render(shots, noise, seed, duration))
        expected = {"shots": [{"shot": k + 1, "outcome": o} for k, o in enumerate(outcomes)]}
        _write(case_dir / "expected_events.json", expected)
        names.append(name)
    return names


def _write(path, data) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    for case in generate():
        print("wrote", case)

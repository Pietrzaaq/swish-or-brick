# ml/: Python reference implementation

`brickorswish.logic` (tracker + make/miss state machine), `brickorswish.tools` and the tests use the **Python standard library only**. Keep it that way: this code must run on any machine without installing third-party packages. Detector, video and training code will live in separate modules with their own optional dependencies, run in the cloud (Codespaces, Colab), not here.

## Run (from `ml/`)

```sh
py -m unittest discover -s tests -t .          # all tests
py -m brickorswish.tools.gen_synthetic          # regenerate shared/fixtures/cases/synthetic_*
```

(`py` is the Windows launcher; use `python3` elsewhere. Python 3.12+.)

## Layout

| Path | What |
|---|---|
| `brickorswish/logic/geometry.py` | Hoop frame from the 3-click calibration (pixels ⇄ rim units) |
| `brickorswish/logic/kalman.py` | Per-axis 2-state Kalman filter with gravity as control input |
| `brickorswish/logic/tracker.py` | Single-ball tracker: chi-square gating, re-acquire after bounces, coasting |
| `brickorswish/logic/state_machine.py` | IDLE → APPROACH → PENDING_MAKE / RATTLE → make / miss / uncertain |
| `brickorswish/logic/pipeline.py` | `process(detections, hoop, cfg)`: the pure function the Dart port must match |
| `brickorswish/tools/gen_synthetic.py` | Physics-based synthetic cases with known outcomes |

Thresholds live in `../shared/config/shot_logic.v1.json`; units are explained in `../shared/README.md`.

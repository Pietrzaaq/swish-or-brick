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
| `brickorswish/tools/deps_audit.py` | Checks installed packages against OSV.dev (CVEs, advisories, malicious packages) and writes exact pins |
| `brickorswish/detect/` | Video → `detections.json` (RF-DETR Nano, COCO) → `events.json` + annotated video. Cloud only, except `crop.py` |
| `notebooks/detect_video.ipynb` | Colab notebook running `brickorswish.detect` on an uploaded video |

## Real video (Colab, not on this machine)

Push the branch, then open
`https://colab.research.google.com/github/Pietrzaaq/swish-or-brick/blob/<branch>/ml/notebooks/detect_video.ipynb`,
pick a T4 GPU runtime and run the cells. Set `BRANCH` in the first cell to the same branch.

The notebook stops if PyTorch is older than 2.10 (CVE-2026-24747) or if any installed dependency has a known
vulnerability. The first run's zip contains `requirements-colab.txt`; commit it here as `ml/requirements-colab.txt`
and later runs install exactly those versions.

Thresholds live in `../shared/config/shot_logic.v1.json`; units are explained in `../shared/README.md`.

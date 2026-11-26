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
| `brickorswish/detect/motion.py` | Camera-motion decisions (stdlib): hoop checks → steady segments / lost spans; maps detections back to the tap frame |
| `brickorswish/detect/stabilize.py` | Finds the backboard/rim patch every 0.5 s (OpenCV template matching). Cloud only |
| `brickorswish/eval/score.py` | Your verdict per detected shot (O/X/-) → accuracy vs the Milestone 1 targets (`labels.json`, `score.json`) |
| `brickorswish/tools/make_fixture.py` | Colab results folder + verdict → `shared/fixtures/cases/real_*` (only if the logic gets every shot right, or the wrong ones are declared with `--known-wrong N,N --why "..."`) |
| `brickorswish/tools/deps_audit.py` | Checks installed packages against OSV.dev (CVEs, advisories, malicious packages) and writes exact pins |
| `brickorswish/detect/` | Video → `detections.json` (RF-DETR Nano, COCO) → `events.json` + annotated video |
| `brickorswish/detect/find_rim.py` | Finds and fits the rim automatically (red/orange rims): no taps; picks a frame where the phone has settled. `rim_preview.png` shows the fit |
| `brickorswish/detect/onnx_model.py` | The same RF-DETR network exported to ONNX, run with onnxruntime: local detection without PyTorch |
| `brickorswish/detect/export_onnx.py` | Colab: export RF-DETR → `model.onnx` + `model.json`, compare its boxes with RF-DETR's, check it runs with only the 3 local packages |
| `models.sha256` | Models allowed for local runs (SHA-256 in git); any other `model.onnx` is refused, even with a matching `model.json` |
| `requirements-local.txt` | The only packages for local detection: onnxruntime, numpy, opencv-python-headless (exact files, checksums) |
| `notebooks/detect_video.ipynb` | Colab notebook running `brickorswish.detect` on an uploaded video |
| `brickorswish/train/` | Our own detector (ball, ball_in_net, rim, person): `drafts` (frames + draft labels from a run), `dataset` (labelling-tool export → RF-DETR folders), `fit` (training, cloud GPU only) |
| `notebooks/train_detector.ipynb` | Colab notebook for the whole training round, from frames to a checked `model.onnx` |
| `viewer/frame_viewer.html` | open in Chrome/Edge: step `annotated.mp4` (Colab) or `annotated.webm` (local run) frame by frame, jump to shots, zoom, save a frame (local only, no libraries) |

## Real video (Colab, not on this machine)

Push the branch, then open
`https://colab.research.google.com/github/Pietrzaaq/swish-or-brick/blob/<branch>/ml/notebooks/detect_video.ipynb`,
pick a T4 GPU runtime and run the cells. Set `BRANCH` in the first cell to the same branch.

The notebook stops if PyTorch is older than 2.10 (CVE-2026-24747) or if any installed dependency has a known
vulnerability. The first run's zip contains `requirements-colab.txt`; commit it here as `ml/requirements-colab.txt`
and later runs install exactly those versions.

After a run, type your verdict in the notebook's verdict cell (or locally:
`py -m brickorswish.eval.score results/<id> --truth "XOOOOOO"`). Download the zip into `ml/results/`
(git-ignored: personal footage). To keep a run as a regression test for Python and Dart:
`py -m brickorswish.tools.make_fixture results/<id> real_<name>`.

Thresholds live in `../shared/config/shot_logic.v1.json`; units are explained in `../shared/README.md`.

## Local detection (no Colab)

1. In Colab, run the notebook's export cell once (`EXPORT_ONNX = True`). Put `model.onnx` + `model.json` in
   `ml/models/rfdetr-nano-coco/` (git-ignored).
2. Once, in `ml/`: `py -3.14 -m venv .venv`, then
   `.venv\Scripts\python -m pip install --no-deps --only-binary=:all: --require-hashes -r requirements-local.txt`
3. Per video (keep your recordings in `ml/videos/`, git-ignored):
   `.venv\Scripts\python -m brickorswish.detect.run_video videos\2min.MP4 --out results\NAME --onnx models\rfdetr-nano-coco\model.onnx`
   The rim is found automatically; check `results\NAME\rim_preview.png`. The fit can lock onto the ring of
   net hooks just inside the rim: when the ball says the rim is more than 8% off, it is rescaled after detection
   (logged; `hoop.json` has the result, `rim.json` the raw fit). The annotated video is
   `annotated.webm` (VP8, at most 1280 px tall: browsers can't play OpenCV's mp4 without ffmpeg); `--no-video`
   skips it (the slowest step after detection). To use your own taps instead:
   `--hoop hoop.json --hoop-frame N`. Only the rim: `.venv\Scripts\python -m brickorswish.detect.find_rim VIDEO --out rim.json --preview rim.png`

## Training our own detector (cloud only)

Open `ml/notebooks/train_detector.ipynb` in Colab the same way as the detection notebook (same branch). It makes
frames with draft labels from your earlier runs, you correct them on the CVAT website, then it builds the dataset,
trains RF-DETR Nano on a T4 and exports a checked `model.onnx`. The notebook explains each step. Nothing of it runs
on this machine except the result: add its `ml/models.sha256` line, then `run_video ... --onnx models\brickorswish-v1\model.onnx`.

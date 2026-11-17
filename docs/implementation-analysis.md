# BrickOrSwish: Implementation Analysis

*Status: draft for discussion. Research date: 2026-11-15. No application code yet.*

How to read the verification markers:

- **[verified]**: confirmed by a web search on the research date. Sources are listed at the end.
- **[unverified]**: from memory or conflicting sources. Check before relying on it.
- **[estimate]**: an engineering guess that the first milestones must measure.

---

## 0. TL;DR

| Area | Recommendation |
|---|---|
| Detector | **RF-DETR Nano** (Apache-2.0, actively maintained, official TFLite and Core ML export). **YOLOX-Tiny/Nano** (Apache-2.0, plain CNN, very mobile-friendly but unmaintained) is the fallback. Avoid Ultralytics (AGPL). |
| Hoop | Not in COCO. **Manual marking** (tap the rim once per session) for v1. Train a hoop class later only for auto-setup. |
| Ball | COCO "sports ball" is a usable baseline but not good enough. Plan to fine-tune on your own footage plus CC BY 4.0 datasets. |
| Tracking | No generic multi-object tracker. Use a **single-ball tracker** (gated nearest neighbour plus a small Kalman filter with gravity), hand-written so it ports 1:1 to Dart. |
| Make/miss | A **state machine in a hoop-relative coordinate system**: rim-plane crossing inside the rim ellipse, plus post-rim exit direction, plus optional net-motion evidence. Uncertain calls are flagged instead of guessed. |
| Inference runtime | **One `.tflite` (FP16) model for both platforms** via LiteRT: GPU delegate on Android, Core ML delegate (Metal fallback) on iOS. The Flutter binding goes behind a `Detector` interface; candidates are `flutter_litert` and `tflite_flutter`, and a spike decides. |
| Processing mode | Design for **live**, but keep a **record, then analyze** mode working from day one. It removes the FPS risk and doubles as the video-file input. |
| iOS without Mac | GitHub Actions `macos-26` runs `flutter build ios --release --no-codesign`, the output is zipped into an unsigned `.ipa`, and Sideloadly (Windows) signs it with your free Apple ID. Re-sign every 7 days. |
| Biggest risk | **Depth ambiguity from behind the shooter**: a short or long shot can overlap the rim in 2D. Use camera elevation, ellipse geometry, post-rim behaviour and an honest "uncertain" class. |

---

## 1. Detection model

### 1.1 Requirements

- Three classes: `ball`, `hoop` (rim, optionally backboard), and `person`.
- The ball is **small** (often 10–25 px wide in a 1080p frame from behind the free-throw line) and **fast**, with motion blur at the release and near the rim.
- Must run on mobile at 15 inferences per second or more ([estimate] of what shot logic needs, see §3) on mid-range Android and iPhone.
- Licence for code **and** pretrained weights must be compatible with Apache-2.0 or MIT.
- Export to TFLite (LiteRT) and ideally Core ML, without heroic conversion work.

### 1.2 Candidates

| Model | Licence (code / weights) | Maintenance | Small objects | Mobile speed | Fine-tuning ease | Export TFLite / Core ML |
|---|---|---|---|---|---|---|
| **RF-DETR** N/S/M (Roboflow) | Apache-2.0 for N, S and M [verified]. **XL/2XL are under the Roboflow PML 1.0 licence, avoid them** [verified] | Active. Docs at 1.8.x; a company backs it [verified] | Good on COCO-small for its size (vendor claim) | Unknown on phones. A community LiteRT build of Nano claims it runs fully on the phone GPU [verified the claim exists; no figures] | **Easiest**: `pip install rfdetr`, then `model.train(dataset_dir=...)` on COCO-JSON | **Official** TFLite (FP32/FP16/INT8) and **native Core ML `.mlpackage`** export [verified]. The TFLite path goes through onnx2tf; docs called it "experimental" in an earlier version [verified] |
| **YOLOX** Nano/Tiny/S (Megvii) | Apache-2.0 [verified] | **Dormant**: no evidence of activity since about 2022–2023 [verified absence; check repo] | Decent; anchor-free CNN | Very good. Plain convs, delegate-friendly. Long history on ncnn/TFLite | Medium: custom `Exp` Python file, COCO-JSON, older PyTorch pins may break | ONNX, then `onnx2tf` to TFLite, then `coremltools`. Community path, works but nothing is official |
| **YOLO (MIT) by MultimediaTechLab** (YOLOv9/v7 rewrite) | MIT [verified]. The *original* WongKinYiu/yolov9 is **GPL-3.0, avoid** [verified] | Moderate; last push about March 2026 per one aggregator; parts still WIP [verified, sources conflict] | Good (v9-T/S) | Good (CNN) | Medium; Hydra configs | ONNX/TFLite/Core ML paths exist in repo docs [unverified] |
| **DEIM / DEIMv2** (Atto/Femto/Pico) | Apache-2.0 [verified via mirrors, not the LICENSE file] | Research code, 2025–2026 | Weak for Atto (23.8 AP at 320px); Pico 38.5 AP [verified, paper] | Tiny (0.5–1.5M params) but DETR decoder ops may not be delegate-friendly | Medium (research configs) | Only ONNX/TensorRT documented; **no TFLite path found** [verified absence] |
| **D-FINE / RT-DETRv2** | Apache-2.0 [verified, aggregators] | Research code | Good | RT-DETRv2-S needed rework for the LiteRT GPU (`grid_sample`) [verified]. Qualcomm publishes D-FINE TFLite builds | Medium | Community LiteRT/Qualcomm builds exist |
| **MediaPipe Model Maker** object detector | Apache-2.0 | **Stale**: `mediapipe-model-maker` 0.2.1.4 (Apr 2024); install issues reported on newer Python [verified] | **Poor**: MobileNetV2 at 256 px or MobileNet-MultiHW at 384 px inputs | Excellent | Easiest *if it installs* | TFLite with metadata only; no Core ML (MediaPipe runs TFLite on iOS) |
| **PP-PicoDet** (PaddleDetection) | Apache-2.0 [verified] | Unclear; aggregators disagree [verified conflict] | Okay | Excellent (about 1M params) | Hard: PaddlePaddle ecosystem, then paddle2onnx, then onnx2tf | Indirect |
| **Ultralytics YOLO11/YOLO26** | **AGPL-3.0, code and trained weights** [verified] | Very active | Very good | Very good | Easiest of all | Official everything |

### 1.3 About Ultralytics and AGPL

You asked for a clean way around AGPL. **There isn't a practical one.** Ultralytics states that trained models are AGPL-covered too [verified]. Your app *would* be open source, but AGPL is not "Apache-compatible": linking or bundling AGPL weights makes the combined distributed app AGPL. That also clashes with App Store distribution terms, a long-standing (A)GPL debate [unverified legal opinion]. The alternatives are almost as good. **Do not use Ultralytics, not even for prototyping**, because prototype weights tend to leak into the product. Many open-source basketball repos (e.g. `avishah3/AI-Basketball-Shot-Detection-Tracker`) use YOLOv8. Read their *logic* for ideas, but don't copy code without checking their licence.

### 1.4 Recommendation

1. **Primary: RF-DETR Nano.** It has the best maintenance, the best fine-tuning ergonomics for someone new to CV, and the only official Core ML *and* TFLite export among the permissive options. Roboflow published an RF-DETR-based basketball shot tracker in August 2026 [verified], which shows the model handles this domain.
2. **Fallback: YOLOX-Tiny.** Switch if RF-DETR Nano misses the on-device latency budget in the Milestone 2 spike. As a pure CNN it carries the lowest risk on GPU, Core ML and NNAPI delegates. Pin an old PyTorch in a separate venv for training if needed.
3. **Gate the decision with measurement, not opinion.** Before any custom training, export both COCO-pretrained models to TFLite and time them on your iPhone and one mid-range Android (see §4.6).

### 1.5 Architectural trick that matters more than the model choice

The camera and hoop are **static**. Make/miss only needs the ball **near the rim**. So:

- Run detection on a **hoop-centred crop** (e.g. 320×320 or 416×416 around the rim, covering about 4 rim-widths). The ball then becomes medium-sized instead of tiny, inference gets faster, and small-object accuracy stops being the bottleneck.
- Optionally run a second, cheaper pass on the full frame at low rate (5 fps) to see the release and the shooter. Shot *attempts* can be inferred from the ball entering the crop from above, so this pass is optional for v1.

This means v1 does not need `person` detection at all. Keep `person` for later features (two-player mode, heatmap).

---

## 2. Pretrained starting point and datasets

### 2.1 What works out of the box

| Class | COCO-pretrained? | Expected quality for this use case |
|---|---|---|
| `person` (COCO id 0 / name `person`) | Yes | Good. Usable immediately. |
| `ball` via COCO `sports ball` (id 37 in the 91-id scheme, index 32 in the 80-class list) | Yes | **Mediocre** [estimate]. COCO sports balls are mostly mid-size and sharp. Expect poor recall on a blurred 15 px ball and false positives on round objects (lights, heads). Good enough for the first prototype on the hoop crop. |
| `hoop` | **No COCO class** | Needs manual marking (v1) or custom training (later). |

### 2.2 Datasets

Candidates found on Roboflow Universe that list **CC BY 4.0** [verified listing; check each page]:

| Dataset | Classes | Size | Notes |
|---|---|---|---|
| `hooper-ibdsr/basketball-detection-v6` | ball, holder, hoop | about 48k images | Largest. Likely mostly broadcast/gym footage. |
| `luca-av0mw/bball-evty4` | Basketball, Basketball Hoop | ? | Exactly your classes. |
| `toothless-zone/basketball-mo7dc` | basketball | ? | |
| `sharangs/basket-ball-hoop` | hoop | ? | Hoop-only. |
| University of Arizona "Basketball Shooting Robot" | ball, rim | about 9.8k | Used by the Roboflow RF-DETR shot tracker. **Licence not verified.** |

Licence cautions:

- **CC BY 4.0 lets you redistribute weights trained on it, with attribution.** Add a `NOTICE`/`docs/ATTRIBUTION.md` and an "Open-source licences" screen in the app.
- Universe datasets are often **forks of forks** with frames grabbed from YouTube or broadcasts. The uploader's "CC BY 4.0" does not prove they owned the images. Treat them as **bootstrap data**, and make **your own recorded and labelled footage** the core dataset that you can fully vouch for.
- **COCO**: annotations are CC BY 4.0, while the images carry mixed Flickr licences [verified]. Publishing COCO-pretrained weights under Apache-2.0 is universal industry practice (RF-DETR, YOLOX and others do it), but it is not legally settled [verified that no source settles it]. Acceptable risk for a free OSS app. Mention it in the attribution doc.

### 2.3 Labelling your own data

- **Tool: CVAT** (MIT licence [unverified], self-host via Docker or cvat.ai free tier) or **Roboflow** (convenient; on the free plan, workspaces/datasets are public [unverified]. That suits an OSS project, but it means footage of you is public).
- Label **frames from your own sessions**: about 1,000–2,000 frames, oversampling frames where the ball is near the rim, blurred, or partly behind the net. With pre-labelling by the COCO model plus correction, this takes 1–2 weekends [estimate].
- Store datasets and weights on the **Hugging Face Hub** (datasets plus model repos, with versioned revisions). Reference them by revision hash from `ml/` configs and from the app's model-download or bundling step.

---

## 3. Tracking and shot logic

### 3.1 Coordinate system

From the manual hoop marking (2 clicks on the left and right rim edges, plus a third on the front or back rim edge to get the ellipse height), build a **hoop frame**:

- origin = rim centre; x-unit = rim radius in pixels; y down.
- The rim is an **ellipse**, because the camera is elevated: the rim's apparent height is much smaller than its width.
- All thresholds are expressed in rim units, never pixels. That makes them resolution-independent and identical in Python and Dart.

A real rim is 18 in (45.7 cm) and a ball is 9.4–9.5 in (about 24 cm), so the ball is about 0.52 rim diameters wide. The detected ball size gives a sanity check and a weak depth cue.

### 3.2 Ball tracking

Solo shootaround means **at most one ball in play** (two players later is a different problem). A generic MOT library (ByteTrack via Roboflow `trackers`, Apache-2.0 [verified]) is overkill and hard to port. Instead:

1. **Candidate filtering**: per frame, keep `ball` detections above a confidence threshold, with size in [0.3, 0.9] rim diameters.
2. **Kalman filter** with state `(x, y, vx, vy)` and a **known constant acceleration in y** (gravity in pixel units, estimated from the data or fitted per shot). About 80 lines of plain arithmetic, so it ports trivially to Dart.
3. **Gating**: take the candidate closest to the prediction (Mahalanobis or simple normalised distance). If none falls within the gate, coast for up to N frames (`N≈6` at 30 fps [estimate]), then drop the track.
4. **Trajectory fitting**: for each airborne segment, fit a **parabola `y = a·t² + b·t + c`, `x = d·t + e`** (least squares). This fills occlusion gaps and gives the predicted rim-plane crossing point.

### 3.3 Shot attempt detection

The simplest robust v1 definition, given only a hoop crop:

- **Attempt** = a ball track that **enters the "approach zone"** (a box about 3 rim-widths wide and 4 rim-heights above the rim) **while moving downward**, or whose fitted parabola has its apex above the rim plane and passes within ±2 rim radii horizontally.
- **Airballs** that never enter the zone are not counted in v1. That is an accepted limitation, revisited once the full-frame pass exists. Alternative later: release detection (the ball separates from the `person` box going upward).
- **Debounce**: one attempt cannot start within about 1.0 s of the previous attempt's resolution.

### 3.4 Make vs miss: state machine

```
IDLE ──ball enters approach zone moving down──▶ APPROACH
APPROACH ──ball reaches rim plane──▶ AT_RIM
AT_RIM ──(a) crossing point inside rim ellipse AND ball continues down below rim
         with low |vx| and appears below the rim/in the net region            ──▶ MAKE
       ──(b) ball reverses upward (rim bounce)                                  ──▶ RATTLE
       ──(c) ball exits sideways outside ellipse x-range or below rim outside ──▶ MISS
RATTLE ──ball comes back down through the ellipse (a)──▶ MAKE ("rattled in")
       ──ball leaves zone sideways / upward away──▶ MISS
       ──timeout (≈2.0 s) without resolution──▶ UNCERTAIN
Any state ──track lost > T without evidence──▶ UNCERTAIN
```

Key rules:

- **Rim-plane crossing with interpolation.** At 30 fps a ball moving at about 7 m/s covers about 23 cm per frame, which is half a rim diameter. Never require a frame *exactly* at the rim. Interpolate the crossing point between the last frame above and the first frame below, or use the fitted parabola.
- **Net occlusion.** After a make, the ball disappears or is partially hidden in the net for several frames. Treat "lost inside the rim ellipse just below the rim plane" as **pending make**, confirmed when the ball **reappears below the net** within about 0.5 s with a mostly vertical velocity. The Roboflow tracker uses the ball's drop speed after reappearing (a make slows to 10–90% of free-fall speed because the net drags it) [verified as their heuristic]. That is a good second feature.
- **Rim bounces.** Never resolve on the first rim contact. Only resolve when the ball **leaves the hoop zone**. This also prevents one rattled shot from being counted as miss plus make, a bug the Roboflow author hit [verified].
- **Optional net-motion signal.** Classic CV, no ML: in a small ROI covering the net, compute frame differencing (OpenCV `absdiff` + threshold). A swish moves the net strongly; a miss off the backboard moves it weakly or not at all. Use it as **tie-breaking evidence**, not a primary signal: wind and outdoor nets make it noisy.
- **"Uncertain" is a first-class outcome.** Per the Roboflow post, some front-rim bounce-outs look identical to rattle-ins from one point of view [verified]. Count uncertain shots separately and let the user correct them in the UI with a single tap. Corrections are labelled data.

### 3.5 Portability rules (Python ⇄ Dart)

- The shot logic is a **pure function**: `events = process(detections_sequence, hoop_calibration, config)`. No I/O, no clocks, no randomness.
- Use only `float64` arithmetic. No numpy-specific behaviour such as broadcasting or `polyfit`. Write the 3×3 least-squares solve by hand in both languages.
- Put every threshold in one `config.json` shared by both languages and versioned with the fixtures.
- Compare events with **tolerances** (time ±1 frame, positions ±0.05 rim units), not bitwise equality.

---

## 4. Mobile runtime in Flutter (Android and iOS equally)

### 4.1 Versions (as of 2026-11-15)

| Component | Version | Status |
|---|---|---|
| Flutter stable | 3.47 (Aug 2026); 3.50 scheduled for November 2026 | [verified schedule; check `flutter --version`] |
| `camera` | 0.12.0+x (min Flutter 3.35) | [verified]. Android uses the **CameraX** implementation (`camera_android_camerax`) by default [verified] |
| `flutter_litert` | 3.9.4 | [verified]. LiteRT binding with GPU, Metal, Core ML and XNNPACK delegates, Interpreter and CompiledModel APIs. **Publisher not verified.** |
| `tflite_flutter` | 0.12.1 | [verified]. Official TF-team package, slow maintenance. Forks: `tflite_core` (0.12.5, Oct 2026), `tflite_flutter_gdx_plus` (0.12.3, Jul 2026) [verified] |
| ONNX Runtime for Flutter | fragmented (`onnxruntime` unmaintained, `onnxruntime_v2`, `flutter_ort_plugin` 0.0.1, …) | [verified]. **Not recommended**: no clear standard package |
| `litert-torch` (PyTorch → `.tflite`) | 0.9.x, Beta, Linux-only | [verified] |

### 4.2 Inference plugin

**Recommendation: LiteRT (TFLite) through a thin Dart interface.**

```dart
abstract interface class Detector {
  Future<List<Detection>> detect(FrameCrop crop); // runs off the UI isolate
  String get backend;                            // "gpu" | "coreml" | "metal" | "xnnpack" | "cpu"
}
```

- **Spike both** `flutter_litert` and `tflite_flutter` (or its most active fork) in Milestone 2. Pick the one that **actually applies the delegate on both platforms**. `flutter_litert`'s changelog shows earlier releases *silently fell back to CPU* on iOS, so **always log and display the active backend** [verified].
- **FP16 precision on GPU** changes outputs per model on Adreno, Xclipse and Apple Metal [verified, `flutter_litert` docs]. Validate detections against the Python reference on-device before trusting fp16.
- Don't build a custom platform-channel inference layer unless the plugins fail. Keep that option (Swift + Core ML, Kotlin + LiteRT via Pigeon) behind the same `Detector` interface.

### 4.3 One model format or two?

**Start with one `.tflite` (FP16 weights) for both platforms.**

- Android: **GPU delegate** first, then XNNPACK CPU. **NNAPI is deprecated as of Android 15** [unverified, from memory]; don't rely on it.
- iOS: **Core ML delegate** (uses the Neural Engine on A12+; supports float models only) [verified], with a fallback to the **Metal GPU delegate**, then XNNPACK.
- **Only add a native Core ML `.mlpackage` path** (RF-DETR exports it officially [verified]) if the iOS spike shows the TFLite Core ML delegate partitions the graph badly, e.g. many CPU fallbacks for DETR ops. That costs about 150 lines of Swift plus a Pigeon interface. Doable, but debugging it realistically needs the work Mac.
- **Avoid INT8 at first.** Core ML delegate is float-only, and quantisation adds an accuracy-debugging dimension you don't need yet.
- Input layout: `litert-torch` emits **NCHW** while `onnx2tf` emits **NHWC** [verified]. Pin the export toolchain and assert the input shape in a test.

### 4.4 Camera frame access

- `camera` → `CameraController.startImageStream(onImage)`.
  - Android delivers **YUV_420_888** (3 planes, row strides with padding); iOS delivers **BGRA8888** (set `imageFormatGroup: ImageFormatGroup.bgra8888`).
  - Request `ResolutionPreset.high` (720p) or `veryHigh` (1080p), whichever keeps the ball at 10 px or more in the hoop crop. Check whether you can get **60 fps**: `camera` has historically had limited fps control [unverified for 0.12]. 60 fps halves the per-frame motion, which is a big help for rim logic.
- **Crop first, then convert.** Converting a full 1080p YUV frame to RGB in Dart is the classic performance trap. Copy only the hoop-crop rows out of the planes, convert just those (about 400×400) to RGB float, then resize to the model input.
- **Threading**:
  - The UI isolate receives `CameraImage`, copies the crop bytes into a `TransferableTypedData`, and sends it to a **long-lived worker isolate** that owns the interpreter.
  - **Drop frames while the worker is busy** (a latest-frame-wins mailbox). Never queue frames.
  - Timestamps: use `CameraImage` sensor timestamps where available, not arrival time. The tracker needs correct Δt when frames drop.
- If Dart-side crop or convert is still too slow, move it native (Kotlin/Swift) via a plugin. That is a known, bounded fallback.

### 4.5 Video-file input mode

Needed for (a) analysing a recorded session, (b) "record then analyze" as the FPS-risk fallback, and (c) running the **same app pipeline on your eval videos** on a device.

- Flutter has **no good cross-platform frame-by-frame decoder**. `video_player` is for playback only.
- `ffmpeg_kit_flutter` was **retired by its author in 2025** [unverified, from memory] and has LGPL/GPL build-variant licensing. Avoid it.
- **Recommendation: a small native plugin.**
  - iOS: `AVAssetReader` + `AVAssetReaderTrackOutput` (BGRA).
  - Android: `MediaExtractor` + `MediaCodec` to an `ImageReader` (YUV).
  - Both expose the same `Stream<Frame>` as the camera path, so the downstream pipeline stays identical. About 200 lines per platform.
- For the live mode, also **record the video** (optional, user setting) so a disputed session can be re-analysed with a better model later.

### 4.6 Realistic FPS

No trustworthy published phone benchmarks exist for RF-DETR Nano or YOLOX-Tiny. The one concrete data point: YOLO26n on a flagship Xiaomi 17 Adreno GPU runs at about 8.5 ms [verified]. Estimates for **a 320–416 px hoop crop** [estimate]:

| Device class | Small CNN (YOLOX-Tiny) | RF-DETR Nano |
|---|---|---|
| iPhone (A15+) via Core ML/ANE | 3–8 ms | 8–20 ms (depends on op support) |
| Mid-range Android (e.g. Snapdragon 7-series / Dimensity 7000) GPU delegate | 10–25 ms | 20–60 ms |
| Mid-range Android CPU (XNNPACK, 4 threads) | 25–50 ms | 60–150 ms |

Pipeline target: **camera at 30 fps (60 if possible), inference at 15 fps or more on the hoop crop**. Below about 10 fps, the rim-interaction window (≈0.3 s) gets only 3 samples, so live make/miss becomes unreliable and the app should switch to record-then-analyze for that device.

**The first device spike must produce real numbers for this table.**

---

## 5. Testing strategy

### 5.1 Ground truth

Per video, store `ml/data/<session>/labels.csv` (small, so it lives in git) next to the video on the HF Hub:

```csv
shot_id,t_release_s,t_resolve_s,outcome,notes
1,12.40,13.55,make,swish
2,21.05,22.30,miss,front rim
3,30.10,31.90,make,rattled
```

Outcomes: `make | miss | airball | uncertain_gt` (when even you can't tell from the video). A tiny OpenCV labelling tool (space = pause, `m` = make, `x` = miss, arrows = step) makes this about 1.5× real-time.

### 5.2 Eval harness (`ml/eval/`)

- **Matching**: greedily match predicted events to ground-truth shots by `t_resolve` within ±1.0 s.
- **Metrics per session and overall**:
  - attempt recall and precision
  - make/miss accuracy on matched shots
  - confusion matrix
  - "uncertain" rate
  - **absolute count error**: |pred makes − true makes| and the same for attempts. This is what the user feels.
  - ball detection recall inside the hoop crop (needs some frame-level box labels)
- **Debug output**: an annotated MP4 (ball track, rim ellipse, state labels) and an HTML or Markdown report. The single most useful tool for tuning heuristics.
- **Data split by session, never by frame.** Keep at least 2 sessions (different lighting or location) as a held-out test set you don't tune on.

### 5.3 Shared fixtures (`shared/fixtures/`)

```
shared/fixtures/
  schema/            detections.schema.json, events.schema.json, hoop.schema.json
  config/shot_logic.v1.json
  cases/
    swish_001/       detections.json, hoop.json, expected_events.json
    rattle_in_002/
    front_rim_out_003/
    net_occlusion_004/
    synthetic_parabola_*/  (generated: clean parabolas, dropped frames, noise)
```

- `detections.json` comes from real videos (Python detector output) **and** from synthetic generators, which are cheap and give exact expected answers.
- `expected_events.json` is reviewed by a human once, then frozen.
- **Python**: `pytest` parametrised over `cases/*`.
- **Dart**: shot logic lives in a **pure Dart package** (`app/packages/shot_logic`, no Flutter dependency), so `dart test` runs it headless and fast with the same parametrisation.
- A fixture change requires updating both implementations in the same PR, which CI enforces.

### 5.4 CI (GitHub Actions)

| Workflow | Runner | Trigger | Does |
|---|---|---|---|
| `ml.yml` | `ubuntu-latest` | changes in `ml/**`, `shared/**` | `uv sync`, `ruff`, `pytest` (logic plus fixtures; no GPU, no videos) |
| `dart-logic.yml` | `ubuntu-latest` | `app/packages/**`, `shared/**` | `dart analyze`, `dart test` on fixtures |
| `app.yml` | `ubuntu-latest` | `app/**` | `flutter analyze`, `flutter test`, `flutter build apk --debug` |
| `ios.yml` | `macos-26` | manual dispatch and tags | unsigned `.ipa` (see §6) |
| `eval.yml` (later) | `ubuntu-latest` or self-hosted | manual | downloads eval videos from HF Hub, runs the full eval, posts a metrics table on the PR |

GitHub-hosted runners, including macOS, are **free for public repositories** [unverified current terms]. Make the repo public early, or budget for macOS minutes.

---

## 6. iOS build and test without a Mac

### 6.1 Build: unsigned `.ipa` on GitHub Actions

`macos-26` (Apple Silicon) is GA with Xcode 26.x [verified].

```yaml
# .github/workflows/ios.yml
name: ios-unsigned
on:
  workflow_dispatch:
  push:
    tags: ['v*']
jobs:
  build:
    runs-on: macos-26
    defaults: { run: { working-directory: app } }
    steps:
      - uses: actions/checkout@v5            # [unverified: check latest major]
      - uses: subosito/flutter-action@v2     # [unverified: check latest major]
        with: { channel: stable, cache: true }
      - run: flutter pub get
      - run: flutter build ios --release --no-codesign
      - name: Package unsigned ipa
        run: |
          mkdir -p Payload
          cp -R build/ios/iphoneos/Runner.app Payload/
          zip -qr ../BrickOrSwish-unsigned.ipa Payload
      - uses: actions/upload-artifact@v4     # [unverified: check latest major]
        with: { name: ios-unsigned-ipa, path: BrickOrSwish-unsigned.ipa }
```

Notes:

- `--release` (or `--profile`), not `--debug`: Flutter **debug** builds on iOS need an attached debugger or tooling to launch [unverified for current Flutter], so they won't start from the home screen.
- Set `NSCameraUsageDescription` (and `NSPhotoLibraryUsageDescription` for importing videos) in `ios/Runner/Info.plist`. Without them the app **crashes** on first camera access.
- Set `IPHONEOS_DEPLOYMENT_TARGET` deliberately (e.g. iOS 15 or 16) and keep the Podfile in sync.

### 6.2 Install: sideloading to your iPhone from Windows

1. Install **Sideloadly** (Windows). It needs the non-Microsoft-Store iTunes and iCloud installers [unverified current requirement].
2. On the iPhone, enable **Developer Mode** (Settings → Privacy & Security; required since iOS 16) [verified].
3. Download the `.ipa` artifact from GitHub Actions, drag it into Sideloadly, sign in with your free Apple ID, and install. Sideloadly supports iOS 26 [verified].
4. On the device, trust the developer profile (Settings → General → VPN & Device Management).
5. **Every 7 days**: re-sign (Sideloadly can auto-refresh over Wi-Fi when the PC is on [verified]). Alternative: **SideStore** (open-source AltStore fork) refreshes on-device via a local VPN [verified].

### 6.3 What a free Apple ID cannot do

| Limitation | Impact on BrickOrSwish |
|---|---|
| Provisioning profile **expires after 7 days** [verified] | Re-sign weekly. Local data survives re-signing with the same bundle ID [unverified per tool]. Keep stats exportable anyway. |
| **3 sideloaded apps** active at once; about **10 App IDs per 7 days** [verified, sources disagree on details] | Fine for one app. Don't churn bundle IDs. |
| Can't reuse a bundle ID that exists on the App Store [verified] | Choose a unique ID now, e.g. `io.github.<you>.brickorswish`. |
| **No Push Notifications, iCloud, Sign in with Apple, Associated Domains, In-App Purchase** [verified] | Irrelevant for v1 (local only). **Relevant for the backend phase**: Sign in with Apple and universal links need the paid account. |
| **Camera, Photos, local storage, Core ML, Metal**: no entitlement needed | **Everything v1 needs works.** |
| No TestFlight, no App Store, can't hand the build to friends | Testers each need their own sideloading setup. |
| No `flutter run` / hot reload / debugger on the iPhone from Windows | iOS gets **release-build testing only**. Do day-to-day development on Android (hot reload, debugger). Pipe iOS logs to an in-app log screen and an export-log button. |

### 6.4 When the work Mac or a paid account becomes necessary

- **Work Mac (occasional)** for:
  - native iOS crashes without a usable stack trace (Xcode device logs, symbolication)
  - **profiling** Core ML or Metal delegate performance in **Instruments**
  - debugging the native video-decoder or Core ML plugin code
  - CocoaPods or Swift Package Manager breakages that are faster to fix interactively

  Plan one Mac session around the Milestone 2 device spike.
- **Paid Apple Developer Program (99 USD/year)** when:
  - you want **external testers** (TestFlight)
  - the weekly re-sign becomes annoying (paid gives 1-year profiles)
  - backend features need Sign in with Apple or push
  - you **publish**. In the EU, alternative marketplaces and web distribution still require a paid account and Apple notarisation [unverified details].

---

## 7. Risks and unknowns (ranked)

| # | Risk | Likelihood × impact | Mitigation |
|---|---|---|---|
| 1 | **Depth ambiguity from behind the shooter**: short or long misses overlap the rim in 2D and look like makes | High × High | Elevate the camera 2.0–2.5 m or more so the rim is a clearly open ellipse. Require the crossing *inside the ellipse* **and** post-rim behaviour (straight down, net slow-down). Use ball size vs rim size as a weak depth cue, net motion as a tie-breaker, and an explicit "uncertain" class with one-tap correction. **Validate in Milestone 1 before anything else.** Also write a setup guide with a screen overlay ("put the rim in this box"). |
| 2 | **Ball detection recall near the rim** (blur, net occlusion, backboard clutter, sun or gym lights) | High × High | Hoop-crop inference, parabola fill-in, fine-tuning on your own footage, 60 fps capture if possible. |
| 3 | **On-device FPS on mid-range Android** (and DETR ops on delegates) | Medium × High | Early device spike, crop-first, YOLOX fallback, record-then-analyze mode as a permanent fallback. |
| 4 | **Python ⇄ Dart logic drift** | Medium × Medium | Pure-function design, shared config, fixtures, CI on both. |
| 5 | **Flutter LiteRT plugin quality** (silent CPU fallback, fp16 differences, abandoned packages) | Medium × Medium | `Detector` interface, show the active backend, on-device golden test comparing outputs to the Python reference. Native-plugin fallback. |
| 6 | **Dataset licence provenance** (Universe datasets of unclear origin) | Medium × Medium | Own footage as the core; attribution file; keep a dataset manifest (source, licence, URL, revision). |
| 7 | **iOS feedback loop without a Mac** (slow: CI build, then sideload, then test) | High × Low-Medium | Develop on Android; batch iOS tests; one planned work-Mac session; in-app logs. |
| 8 | **Environment variation** (outdoor glare, chain or no nets, different rims, other players in frame) | Medium × Medium | Record diverse eval sessions early. State supported conditions in v1 ("indoor or outdoor, nylon net, solo"). |
| 9 | **Camera shake or tripod movement** invalidates the hoop calibration | Low-Medium × Medium | Detect large global motion and prompt re-calibration. Later: automatic hoop re-detection. |
| 10 | **Tooling churn** (onnx2tf, litert-torch Beta, LiteRT 2.x API changes) | Medium × Low | Pin everything in `uv.lock` and the export scripts; store exported models with the exact toolchain versions in the model card. |

---

## 8. Recommendation

### 8.1 Stack

| Layer | Choice |
|---|---|
| Licence | **MIT** for app and code (decided 2026-11-17). Third-party attributions (datasets, pretrained weights) go in `docs/ATTRIBUTION.md` and the in-app licences screen. |
| Python env | Python 3.12, **uv** (lockfile), ruff, pytest |
| CV basics | `opencv-python` (Apache-2.0 since 4.5), `numpy` |
| Detector | `rfdetr` (Nano), fallback YOLOX-Tiny |
| Labelling | CVAT or Roboflow (see Q6), plus a custom OpenCV event-labelling script |
| Data and weights | Hugging Face Hub (dataset repo plus model repo, pinned revisions) |
| App | Flutter stable, `camera`, LiteRT plugin (spike decides), pure-Dart `shot_logic` package, native video-decode plugin |
| CI | GitHub Actions (ubuntu for everything, macos-26 for the iOS `.ipa`) |
| iOS install | Sideloadly (or SideStore) with a free Apple ID |

### 8.2 Repo layout (your proposal, slightly adjusted)

```
app/                       Flutter app
  packages/shot_logic/     pure Dart (no Flutter): tracker + state machine (port of ml/brickorswish/logic)
  packages/video_frames/   native plugin: AVAssetReader / MediaCodec (later)
ml/
  pyproject.toml, uv.lock
  brickorswish/            importable package
    detect/                model wrappers -> detections.json
    logic/                 tracker + state machine (the reference implementation)
    eval/                  matching, metrics, report, annotated video
    tools/                 hoop_marker.py, label_events.py, export_tflite.py
  configs/                 dataset/model manifests (HF repo + revision)
  data/<session>/          labels.csv, hoop.json (videos NOT in git)
shared/
  fixtures/                as in §5.3
  config/                  shot_logic.v1.json (single source of thresholds)
backend/                   later
docs/
```

Changes from your proposal: the `shot_logic` Dart package is split out so it's testable without Flutter, and the thresholds config lives in `shared/` so neither language owns it. Everything else holds up.

### 8.3 Milestone 1: Python prototype that counts makes and misses on a recorded video, with manual hoop marking

**Exit criteria** (proposed): on **2 held-out sessions** (at least 40 shots each), attempt recall of 95% or more, make/miss accuracy of 90% or more on detected attempts, uncertain rate of 10% or less, all fully offline, and a deterministic re-run gives identical events.

| Step | Work | Output |
|---|---|---|
| 1 | **Record data first.** 6–8 sessions × about 50 shots, phone on tripod behind the shooter, elevated about 2 m or more, 1080p and **60 fps if your phone allows**. Vary distance, time of day and indoor/outdoor. Include deliberate rattles, airballs and bank shots. | Raw videos on HF Hub (private dataset repo until you decide on publishing your face) |
| 2 | **Label events** with a tiny OpenCV tool (`ml/brickorswish/tools/label_events.py`). | `labels.csv` per session, in git |
| 3 | **Scaffold `ml/`**: `uv init`, `uv add opencv-python numpy rfdetr`, `uv add --dev pytest ruff`, plus a CI workflow running ruff and pytest. | Green CI |
| 4 | **Hoop marker**: show the first frame, click the left rim edge, right rim edge, and front edge, fit the ellipse, save `hoop.json` (pixel coordinates plus video resolution). | `hoop.json` per session |
| 5 | **Detection dump**: run COCO-pretrained RF-DETR Nano on the **hoop crop** of every frame, keep `sports ball` (and log everything else for analysis). Write `detections.json` in the fixture schema. Measure the **ball recall near the rim** on about 200 hand-checked frames. | Detection files, recall number |
| 6 | **Synthetic fixtures + tracker**: generate clean or noisy parabolas (make, short, long, rattle, dropped frames). Implement the Kalman filter, gating and parabola fit. Unit-test it. | Tracker passing synthetic cases |
| 7 | **State machine** (§3.4), driven by `shared/config/shot_logic.v1.json`. | `process()` → `events.json` |
| 8 | **Eval harness + annotated video** (§5.2). Tune on the training sessions only. | Metrics report |
| 9 | **Decision point.** If ball recall (step 5) is the bottleneck, label about 1,000 frames from your sessions and fine-tune RF-DETR Nano (`model.train(...)`, Colab or Kaggle GPU if you have no local NVIDIA GPU). If the logic is the bottleneck (depth ambiguity), adjust the camera setup guidance and rules. | Metrics meeting exit criteria |
| 10 | **Freeze fixtures** from 4–6 real-video cases plus the synthetic ones, as the contract for the Dart port. | `shared/fixtures/cases/*` |

**Milestone 2** (preview, not detailed here): a device spike exporting COCO-RF-DETR-Nano and YOLOX-Tiny to `.tflite`, timed on your iPhone (via a sideloaded benchmark build) and one mid-range Android. In parallel, the Dart `shot_logic` port goes green on the fixtures. **Milestone 3**: the live camera pipeline with hoop marking by tap, and the record-then-analyze mode.

---

## 9. Questions for you

1. **Licence**: Apache-2.0 (my recommendation, explicit patent grant) or MIT?
2. **Live vs post-session**: is "record the session, get results 30–60 s after you stop" acceptable for v1 on slower phones, or is live counting essential from day one? This is the biggest de-risking lever.
3. **Test devices**: which iPhone model do you have? Which Android phone(s) can you test on, and what counts as "mid-range" for you (e.g. a Samsung A5x / Pixel 7a class)? I'll set FPS targets from those.
4. **Where do you shoot?** Indoor gym, outdoor court, or both? Nylon net, chain net, or sometimes no net? This decides how much the net-motion signal and outdoor-light robustness matter.
5. **Camera position**: is "behind the shooter, elevated about 2 m or more" (tall tripod) realistic for you, or will it often be lower? Lower placement makes risk #1 much worse. A side or 45° placement is far easier for make/miss, if that's acceptable for the product.
6. **Data publishing**: are you OK with your recorded sessions (you visible) being **public** on HF Hub or Roboflow's free tier? That affects the labelling tool choice and whether the eval set can run in public CI.
7. **Labelling effort**: are you willing to spend about 1–2 weekends labelling ball boxes if COCO "sports ball" isn't good enough? (Likely needed.)
8. **Compute**: do you have a local NVIDIA GPU, or will training run on Colab/Kaggle?
9. **Paid Apple account timing**: any planned date or trigger (e.g. "when 5 friends want to test")?
10. **Minimum OS versions**: OK with iOS 16+ and Android 10+ (API 29) to keep camera and ML paths modern?
11. **Repo visibility**: public from day one (free macOS CI minutes, community) or private until Milestone 1?

---

## Sources

Detection models and licences:
- [YOLOX LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) · [Roboflow YOLOX page](https://playground.roboflow.com/models/megvii/yolox)
- [RF-DETR licence and sizes](https://playground.roboflow.com/models/roboflow/rf-detr) · [RF-DETR N/S/M blog](https://blog.roboflow.com/rf-detr-nano-small-medium/) · [RF-DETR export docs](https://rfdetr.roboflow.com/latest/learn/export/) · [RF-DETR export 1.8.2](https://rfdetr.roboflow.com/1.8.2/learn/export/) · [RF-DETR-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT) · [Roboflow mobile detectors](https://blog.roboflow.com/mobile-object-detection-models/)
- [MultimediaTechLab/YOLO (MIT)](https://awesome.ecosyste.ms/projects/github.com%2FMultimediaTechLab%2FYOLO) · [Article on YOLO MIT vs GPL](https://www.issoh.co.jp/md/tech/2571)
- [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) · [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) · [DEIMv2 paper](https://arxiv.org/html/2509.20787v1) · [D-FINE (Roboflow)](https://playground.roboflow.com/models/ustc/d-fine) · [D-FINE on Qualcomm](https://huggingface.co/qualcomm/D-FINE) · [RT-DETRv2-S LiteRT](https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT)
- [MediaPipe Model Maker future (forum)](https://discuss.ai.google.dev/t/future-of-tflite-model-maker-and-mediapipe-model-maker/31527) · [mediapipe-model-maker (Snyk)](https://snyk.io/advisor/python/mediapipe-model-maker) · [Model Maker object detector](https://developers.google.com/edge/mediapipe/solutions/customization/object_detector)
- [PaddleDetection README](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/README_en.md) · [NanoDet-Plus mirror](https://sourceforge.net/mirror/nanodet-plus/profile/)
- [Ultralytics licence](https://ultralytics.com/license) · [Ultralytics AGPL forum answer](https://community.ultralytics.com/t/yolov11-agpl-license-for-on-premises-use/1529) · [Roboflow on AGPL](https://blog.roboflow.com/why-agpl-3-0-is-a-risk-for-computer-vision-teams/) · [Ultralytics LiteRT export notes](https://www.ultralytics.com/blog/export-ultralytics-yolo-with-litert-for-edge-and-browser-deployment)

Datasets:
- [basketball-detection-v6](https://universe.roboflow.com/hooper-ibdsr/basketball-detection-v6) · [bball (ball + hoop)](https://universe.roboflow.com/luca-av0mw/bball-evty4) · [Basketball (toothless-zone)](https://universe.roboflow.com/toothless-zone/basketball-mo7dc/dataset/) · [Basket Ball Hoop](https://universe.roboflow.com/sharangs/basket-ball-hoop)
- [COCO licensing summary](https://huggingface.co/datasets/shunk031/MSCOCO/raw/main/README.md)

Shot logic references:
- [Roboflow: AI basketball shot evaluator (RF-DETR, Aug 2026)](https://blog.roboflow.com/how-to-build-an-ai-basketball-shot-evaluator/) · [avishah3 tracker](https://github.forks.workers.dev/avishah3/AI-Basketball-Shot-Detection-Tracker) · [chonyy/basketball-shot-detection](https://www.github.com/chonyy/basketball-shot-detection)
- [Supervision FAQ (MIT)](https://supervision.roboflow.com/develop/faq/) · [roboflow/trackers](https://pypi.org/project/trackers/2.2.0/)

Mobile runtime:
- [flutter_litert](https://pub.dev/packages/flutter_litert) · [flutter_litert docs](https://pub.dev/documentation/flutter_litert/latest/) · [tflite_flutter changelog](https://pub.dev/packages/tflite_flutter/changelog) · [tflite_core](https://pub.dev/packages/tflite_core) · [tflite_flutter_gdx_plus](https://pub.dev/packages/tflite_flutter_gdx_plus/changelog)
- [LiteRT Core ML delegate](https://developers.google.com/edge/litert/ios/coreml) · [LiteRT iOS GPU delegate](https://developers.google.com/edge/litert/ios/gpu) · [TFLite is now LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/) · [LiteRT universal framework (2026)](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/) · [litert-torch](https://pypi.org/project/litert-torch/)
- [camera changelog](https://pub.dev/packages/camera/changelog) · [camera_android_camerax](https://pub.dev/documentation/camera_android_camerax/latest/)
- [onnxruntime_v2](https://pub.dev/documentation/onnxruntime_v2/latest) · [flutter_ort_plugin](https://pub.dev/packages/flutter_ort_plugin)
- [Flutter what's new](https://docs.flutter.dev/whats-new) · [Flutter 3.41](https://flutter.dev/blog/whats-new-in-flutter-3-41)

iOS without a Mac:
- [GitHub: macOS 26 runners GA](https://github.blog/changelog/2026-02-26-macos-26-is-now-generally-available-for-github-hosted-runners/) · [runner image notes](https://newreleases.io/project/github/actions/runner-images/release/macos-26%2F20260422.0018)
- [Unsigned IPA for Sideloadly](https://dev.to/oivoodoo/build-unsigned-ios-ipa-to-install-via-sideloadly-236f) · [Flutter IPA without paid account](https://marc-ko.github.io/2024/12/10/flutter_ipa) · [Debugging Flutter iOS from Windows](https://startdebugging.net/2026/01/debugging-flutter-ios-from-windows-a-real-device-workflow-flutter-3-x/)
- [Sideloadly FAQ](https://sideloadly.io/faq) · [Corellium Sideloadly guide](https://support.corellium.com/features/apps/sideloadly) · [SideStore on iOS 26](https://fr0stb1rd.is-a.dev/posts/ios-26-unlimited-sideload-sidestore-livecontainer/)
- [Expo: Apple Developer free vs paid](https://expo.dev/blog/apple-developer-program-explained-free-vs-paid) · [Apple forums: personal team capabilities](https://developer.apple.com/forums/thread/803362) · [Apple membership comparison](https://developer.apple.com/support/compare-memberships)

### Things I could not verify (check before relying on them)

- Current maintenance activity of the YOLOX and PaddleDetection repos (only indirect signals).
- The publisher and maintenance model of `flutter_litert`. Which of `tflite_flutter`, `tflite_core` and `flutter_litert` is the healthiest long-term choice.
- Real phone latency of RF-DETR Nano and YOLOX-Tiny (none published). The §4.6 numbers are estimates.
- NNAPI deprecation in Android 15, and the `ffmpeg_kit_flutter` retirement (both from memory).
- The exact free-Apple-ID limits (3 apps vs 10 App IDs / 4 or 7 days; sources disagree).
- Latest major versions of `actions/checkout`, `subosito/flutter-action` and `actions/upload-artifact`.
- Licence of the University of Arizona "Basketball Shooting Robot" dataset; whether Roboflow's free tier still forces public datasets; the CVAT licence.
- Whether `camera` 0.12 allows requesting 60 fps.
- Flutter 3.50 release status (scheduled for November 2026).

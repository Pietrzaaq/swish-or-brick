"""Colab: export RF-DETR Nano (COCO) to ONNX for local runs, and prove the ONNX path gives rfdetr's own boxes.

    python -m brickorswish.detect.export_onnx export --out DIR --video VIDEO --hoop hoop.json [--hoop-frame N]
    python -m brickorswish.detect.export_onnx check-minimal DIR/model.onnx --video VIDEO --hoop hoop.json

`export` (needs rfdetr + onnx + onnxruntime) writes DIR/model.onnx, DIR/model.json (the card OnnxDetector
reads: input, normalisation, classes, SHA-256) and DIR/parity.json, after running rfdetr.predict() and
OnnxDetector on the same crops of the video and comparing every box. Exit code 1 if they disagree.

`check-minimal` runs in an environment with ONLY onnxruntime + numpy + opencv-python-headless (what the laptop
has) and fails if the model needs anything else or loads protobuf / flatbuffers / packaging / torch.
"""

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

PARITY_MIN_IOU = 0.9      # every rfdetr box >= COMPARE_CONF must have an ONNX box of the same class this close
PARITY_MAX_DCONF = 0.05   # ...with a confidence at most this far off
COMPARE_CONF = 0.3        # = ball_min_conf: the boxes that matter
LIST_CONF = 0.15          # = detection threshold: a partner may sit just below COMPARE_CONF


def sample_crops(video: str, hoop: dict, hoop_frame: int, n: int) -> list[tuple[int, object]]:
    """n RGB crops of the hoop region (same box detection uses), evenly spread from the tap frame on."""
    import cv2

    from ..logic import ShotLogicConfig
    from ..paths import CONFIG_V1
    from .crop import watch_crop

    x0, y0, x1, y1 = watch_crop(hoop, ShotLogicConfig.load(CONFIG_V1))
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, (total - hoop_frame) // n)
    wanted = set(range(hoop_frame, total, step))
    crops, i = [], 0
    while len(crops) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in wanted:
            crops.append((i, cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if not crops:
        raise ValueError(f"no frames read from {video}")
    return crops


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(reference: list, candidate: list) -> list[dict]:
    """Per reference box (>= COMPARE_CONF): best same-class candidate's IoU and confidence difference."""
    rows = []
    for cls, conf, box in reference:
        if conf < COMPARE_CONF:
            continue
        best = max(((iou(box, b), abs(conf - c)) for n, c, b in candidate if n == cls), default=(0.0, 1.0))
        rows.append({"cls": cls, "conf": round(conf, 4), "iou": round(best[0], 4), "dconf": round(best[1], 4)})
    return rows


def export(out: Path, video: str, hoop: dict, hoop_frame: int, n: int, opset: int) -> int:
    from importlib.metadata import version

    import onnxruntime as ort
    from rfdetr import RFDETRNano
    from rfdetr.assets.coco_classes import COCO_CLASSES

    from .onnx_model import OnnxDetector, sha256_of
    from .video import predict_boxes

    out.mkdir(parents=True, exist_ok=True)
    model = RFDETRNano()
    written = model.export(output_dir=str(out / "raw"), opset_version=opset)
    raw = Path(written) if written else next((out / "raw").glob("*.onnx"))
    onnx_path = out / "model.onnx"
    shutil.move(str(raw), onnx_path)
    shutil.rmtree(out / "raw", ignore_errors=True)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    outputs = [o.name for o in sess.get_outputs()]
    print(f"input {inp.name} {inp.shape}; outputs " + ", ".join(f"{o.name} {o.shape}" for o in sess.get_outputs()))
    if outputs[:2] != ["dets", "labels"]:
        print(f"unexpected outputs {outputs}: expected dets, labels")
        return 1
    num_select = 300
    for holder in (getattr(model, "model_config", None), getattr(getattr(model, "model", None), "args", None)):
        num_select = int(getattr(holder, "num_select", num_select) or num_select)
    card = {
        "label": f"rfdetr-nano-coco rfdetr {version('rfdetr')} opset {opset}",
        "exported": date.today().isoformat(),
        "sha256": sha256_of(onnx_path),
        "input_name": inp.name,
        "input_shape": [int(v) for v in inp.shape],
        "output_names": outputs[:2],
        "means": [float(v) for v in model.means],
        "stds": [float(v) for v in model.stds],
        "resize": "stretch, bilinear, no antialias (cv2.INTER_LINEAR)",
        "num_select": num_select,
        "classes": {str(k): v for k, v in sorted(COCO_CLASSES.items())},
    }
    (out / "model.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(f"model.onnx {onnx_path.stat().st_size / 1e6:.1f} MB, sha256 {card['sha256']}")

    onnx_model = OnnxDetector(onnx_path)
    crops = sample_crops(video, hoop, hoop_frame, n)
    rows = []
    for i, rgb in crops:
        ref = predict_boxes(model, [rgb], LIST_CONF)[0]
        got = predict_boxes(onnx_model, [rgb], LIST_CONF)[0]
        rows += [{"frame": i, **r} for r in compare(ref, got)]
    worst_iou = min((r["iou"] for r in rows), default=1.0)
    worst_dconf = max((r["dconf"] for r in rows), default=0.0)
    ok = worst_iou >= PARITY_MIN_IOU and worst_dconf <= PARITY_MAX_DCONF
    summary = {"frames": len(crops), "boxes_compared": len(rows), "balls_compared": sum(r["cls"] == "sports ball" for r in rows),
               "worst_iou": worst_iou, "worst_conf_diff": worst_dconf, "ok": ok}
    (out / "parity.json").write_text(json.dumps({"summary": summary, "boxes": rows}, indent=1), encoding="utf-8")
    print(f"parity on {len(crops)} frames: {len(rows)} boxes >= {COMPARE_CONF} compared "
          f"({summary['balls_compared']} balls), worst IoU {worst_iou:.3f}, worst confidence difference {worst_dconf:.3f}")
    print("ONNX matches rfdetr" if ok else "MISMATCH: do not use this model (see parity.json)")
    return 0 if ok and rows else 1


def check_minimal(onnx_path: Path, video: str, hoop: dict, hoop_frame: int) -> int:
    from .onnx_model import OnnxDetector

    det = OnnxDetector(onnx_path)
    crops = sample_crops(video, hoop, hoop_frame, 5)
    found = sum(len(boxes) for boxes in det.predict([rgb for _, rgb in crops], LIST_CONF))
    loaded = sorted({m.split(".")[0] for m in sys.modules})
    extra = [m for m in ("google", "flatbuffers", "packaging", "torch", "rfdetr") if m in loaded]
    print(f"ran on {len(crops)} frames with only onnxruntime + numpy + opencv: {found} boxes; "
          f"extra modules loaded: {extra or 'none'}")
    return 0 if not extra else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--opset", type=int, default=17)
    e.add_argument("--frames", type=int, default=60, help="frames to compare rfdetr and ONNX on")
    m = sub.add_parser("check-minimal")
    m.add_argument("model")
    for p in (e, m):
        p.add_argument("--video", required=True)
        p.add_argument("--hoop", required=True)
        p.add_argument("--hoop-frame", type=int, default=0)
    args = ap.parse_args()
    hoop = json.loads(Path(args.hoop).read_text(encoding="utf-8"))
    if args.cmd == "export":
        return export(Path(args.out), args.video, hoop, args.hoop_frame, args.frames, args.opset)
    return check_minimal(Path(args.model), args.video, hoop, args.hoop_frame)


if __name__ == "__main__":
    sys.exit(main())

"""Runs RF-DETR Nano over a video and returns detections in shared/schema/detections.schema.json.

The model is either rfdetr itself (PyTorch; Colab, see ml/notebooks/detect_video.ipynb) or the same network
exported to ONNX (onnx_model.OnnxDetector; runs locally with onnxruntime + numpy + opencv-python-headless).
"""

import time
from collections.abc import Callable
from pathlib import Path

from ..logic import ShotLogicConfig
from .crop import box_to_det, frame_time, watch_crop
from .motion import IDENTITY, Transform, rim_centre, shift_hoop, stabilize_dets


def load_model(onnx_path: str | Path | None = None):
    """RF-DETR Nano: the exported ONNX model at `onnx_path` (checked against its model.json card), or else
    COCO-pretrained rfdetr (PyTorch; downloads the weights on first use)."""
    if onnx_path is not None:
        from .onnx_model import OnnxDetector

        return OnnxDetector(onnx_path)
    from rfdetr import RFDETRNano

    return RFDETRNano()


def predict_boxes(model, images, threshold: float) -> list[list[tuple[str, float, tuple]]]:
    """(class name, confidence, xyxy in image pixels) per box, per image, from either kind of model."""
    from .onnx_model import Box

    results = (model.predict(images, threshold) if hasattr(model, "card")
               else model.predict(images, threshold=threshold, include_source_image=False))
    if not isinstance(results, list):
        results = [results]
    out = []
    for dets in results:
        if isinstance(dets, list) and (not dets or isinstance(dets[0], Box)):
            out.append([(b.cls, b.conf, b.xyxy) for b in dets])
            continue
        names = dets.data.get("class_name") if dets.data else None
        if names is None:
            from rfdetr.assets.coco_classes import COCO_CLASSES

            names = [COCO_CLASSES.get(int(c), str(int(c))) for c in dets.class_id]
        out.append([(str(n), float(c), tuple(float(v) for v in xyxy))
                    for xyxy, c, n in zip(dets.xyxy, dets.confidence, names)])
    return out


def model_label(model) -> str:
    if hasattr(model, "card"):
        return f"onnx {model.label}"
    try:
        from importlib.metadata import version

        return f"rfdetr {version('rfdetr')} nano coco"
    except Exception:  # noqa: BLE001 - informational only
        return "rfdetr ? nano coco"


def read_frame(video_path: str | Path, index: int = 0):
    """One BGR frame, decoded exactly as detect_video() decodes it (same rotation handling)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        for _ in range(index + 1):
            ok, img = cap.read()
            if not ok:
                raise ValueError(f"video has fewer than {index + 1} frames")
        return img
    finally:
        cap.release()


def detect_video(
    video_path: str | Path,
    hoop: dict,
    cfg: ShotLogicConfig,
    model,
    crop: str = "hoop",
    threshold: float = 0.15,
    batch_size: int = 8,
    transforms: list[Transform | None] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Detections in the CALIBRATION frame's pixel coordinates (the frame hoop.json was tapped on).

    crop="hoop" runs on the watch zone around the rim, "full" on the whole frame. `transforms` (from
    stabilize/motion) gives the camera pose per frame: the crop follows the hoop, boxes are mapped back
    to the calibration frame, and frames with None (hoop lost) get no detections, so nothing is counted
    there. Without transforms the camera is assumed still.

    `threshold` is deliberately below cfg.ball_min_conf: the logic does its own filtering, and the
    low-confidence boxes are useful when judging recall.
    """
    import cv2

    if crop not in ("hoop", "full"):
        raise ValueError("crop must be 'hoop' or 'full'")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    c0 = rim_centre(hoop)
    frames: dict[int, dict] = {}
    pending: list[tuple[int, object, tuple[int, int, int, int], Transform]] = []
    started = time.monotonic()
    size_checked = False

    def flush():
        results = predict_boxes(model, [img for _, img, _, _ in pending], threshold)
        for (i, _, box, tr), dets in zip(pending, results):
            frames[i] = {"i": i, "t": frame_time(i, fps), "dets": stabilize_dets(_to_dets(dets, box[:2]), c0, tr)}
        pending.clear()

    try:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if not size_checked:
                if (w, h) != (hoop["image_width"], hoop["image_height"]):
                    raise ValueError(f"video is {w}x{h} but hoop.json is {hoop['image_width']}x{hoop['image_height']}")
                log(f"{w}x{h} @ {fps:.2f} fps, ~{total} frames")
                size_checked = True
            if transforms is None:
                tr = IDENTITY
            else:
                tr = transforms[min(i, len(transforms) - 1)]
            if tr is None:
                frames[i] = {"i": i, "t": frame_time(i, fps), "dets": []}  # hoop lost: detect nothing
            else:
                box = watch_crop(shift_hoop(hoop, tr), cfg) if crop == "hoop" else (0, 0, w, h)
                x0, y0, x1, y1 = box
                pending.append((i, cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), box, tr))
                if len(pending) == batch_size:
                    flush()
            i += 1
            if total and i % max(1, total // 10) == 0:
                log(f"  {i}/{total} frames, {time.monotonic() - started:.0f}s")
        if pending:
            flush()
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")

    moving = "stabilized" if transforms is not None else "still camera"
    source = f"{Path(video_path).name} | {model_label(model)} | crop={crop} | thr={threshold} | {moving}"
    return {"fps": round(fps, 3), "source": source, "frames": [frames[k] for k in sorted(frames)]}


def _to_dets(boxes, offset: tuple[int, int]) -> list[dict]:
    """[(class name, confidence, xyxy in crop pixels)] -> detections.json entries in full-frame pixels."""
    out = [box_to_det(n, c, xyxy, offset) for n, c, xyxy in boxes]
    out.sort(key=lambda d: (d["cls"] != "ball", -d["conf"]))
    return out

"""Runs COCO-pretrained RF-DETR Nano over a video and returns detections in shared/schema/detections.schema.json.

Needs `opencv-python` and `rfdetr` (PyTorch). Cloud only; see ml/notebooks/detect_video.ipynb.
"""

import time
from collections.abc import Callable
from pathlib import Path

from ..logic import ShotLogicConfig
from .crop import box_to_det, frame_time, watch_crop


def load_model():
    """COCO-pretrained RF-DETR Nano. Downloads the weights on first use."""
    from rfdetr import RFDETRNano

    return RFDETRNano()


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
    log: Callable[[str], None] = print,
) -> tuple[dict, tuple[int, int, int, int]]:
    """Returns (detections, crop_box). crop="hoop" runs on the watch zone around the rim, "full" on the whole frame.

    `threshold` is deliberately below cfg.ball_min_conf: the logic does its own filtering, and the low-confidence
    boxes are useful when judging recall.
    """
    import cv2

    if crop not in ("hoop", "full"):
        raise ValueError("crop must be 'hoop' or 'full'")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[dict] = []
    pending: list[tuple[int, object]] = []
    box = None
    started = time.monotonic()

    def flush():
        results = model.predict([img for _, img in pending], threshold=threshold, include_source_image=False)
        for (i, _), dets in zip(pending, results):
            frames.append({"i": i, "t": frame_time(i, fps), "dets": _to_dets(dets, (box[0], box[1]))})
        pending.clear()

    try:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if box is None:
                h, w = bgr.shape[:2]
                if (w, h) != (hoop["image_width"], hoop["image_height"]):
                    raise ValueError(f"video is {w}x{h} but hoop.json is {hoop['image_width']}x{hoop['image_height']}")
                box = watch_crop(hoop, cfg) if crop == "hoop" else (0, 0, w, h)
                log(f"{w}x{h} @ {fps:.2f} fps, ~{total} frames, detector crop {box}")
            x0, y0, x1, y1 = box
            pending.append((i, cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
            if len(pending) == batch_size:
                flush()
            i += 1
            if total and i % max(1, total // 10) == 0:
                log(f"  {i}/{total} frames, {time.monotonic() - started:.0f}s")
        if pending:
            flush()
    finally:
        cap.release()
    if box is None:
        raise ValueError(f"no frames decoded from {video_path}")

    try:
        from importlib.metadata import version

        rf_version = version("rfdetr")
    except Exception:  # noqa: BLE001 - informational only
        rf_version = "?"
    source = f"{Path(video_path).name} | rfdetr {rf_version} nano coco | crop={crop} {list(box)} | thr={threshold}"
    return {"fps": round(fps, 3), "source": source, "frames": frames}, box


def _to_dets(dets, offset: tuple[int, int]) -> list[dict]:
    """supervision.Detections -> list of detections.json entries in full-frame pixels."""
    names = dets.data.get("class_name") if dets.data else None
    if names is None:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        names = [COCO_CLASSES.get(int(c), str(int(c))) for c in dets.class_id]
    out = [box_to_det(str(n), c, xyxy, offset) for xyxy, c, n in zip(dets.xyxy, dets.confidence, names)]
    out.sort(key=lambda d: (d["cls"] != "ball", -d["conf"]))
    return out

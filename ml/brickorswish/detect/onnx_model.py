"""RF-DETR exported to ONNX, run with ONNX Runtime: the same network and weights as rfdetr.RFDETRNano, without
PyTorch. Needs only `onnxruntime`, `numpy` and `opencv-python-headless` (ml/requirements-local.txt).

The network sits between two thin layers that rfdetr's predict() does in Python; they are reproduced here and
checked against rfdetr itself by `brickorswish.detect.export_onnx` (in Colab):
- before: RGB image -> stretched to the model's square input (bilinear, no antialias), /255, ImageNet mean/std;
- after: sigmoid on the class logits, top `num_select` over all (query, class) pairs, boxes from normalised
  (cx, cy, w, h) to pixel corners of the ORIGINAL image, clamped, then `confidence > threshold`.

The model card (model.json next to model.onnx) holds the input size, normalisation, class names and the
file's SHA-256; the file is refused if the checksum does not match.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]  # pixels of the image passed in


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(rgb, size: tuple[int, int], means, stds):
    """HxWx3 uint8 RGB -> 1x3xHxW float32, as rfdetr predict() prepares it."""
    import cv2
    import numpy as np

    h, w = size
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.asarray(means, dtype=np.float32)) / np.asarray(stds, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def postprocess(dets, logits, image_w: int, image_h: int, threshold: float, num_select: int,
                class_names: dict[int, str]) -> list[Box]:
    """One image: dets (Q, 4) normalised cx, cy, w, h; logits (Q, C) raw. Mirrors rfdetr's PostProcess."""
    import numpy as np

    q, c = logits.shape
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    flat = prob.reshape(-1)
    k = min(num_select, flat.shape[0])
    order = np.argsort(-flat, kind="stable")[:k]
    out = []
    for idx in order:
        conf = float(flat[idx])
        if not conf > threshold:
            break  # sorted: the rest are lower
        query, cls = divmod(int(idx), c)
        cx, cy, bw, bh = (float(v) for v in dets[query])
        x0 = min(max((cx - bw / 2.0) * image_w, 0.0), image_w)
        y0 = min(max((cy - bh / 2.0) * image_h, 0.0), image_h)
        x1 = min(max((cx + bw / 2.0) * image_w, 0.0), image_w)
        y1 = min(max((cy + bh / 2.0) * image_h, 0.0), image_h)
        out.append(Box(class_names.get(cls, str(cls)), conf, (x0, y0, x1, y1)))
    return out


class OnnxDetector:
    """predict(list of RGB uint8 images, threshold) -> list of list[Box], one list per image."""

    def __init__(self, model_path: str | Path, card_path: str | Path | None = None, threads: int = 0):
        import onnxruntime as ort

        model_path = Path(model_path)
        card_path = Path(card_path) if card_path else model_path.with_suffix(".json")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        digest = sha256_of(model_path)
        if digest != card["sha256"]:
            raise ValueError(f"{model_path.name}: checksum {digest[:12]}... does not match its card "
                             f"({card['sha256'][:12]}...); refusing to load it")
        ort.disable_telemetry_events()
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        # CPU only, explicitly: no other provider (e.g. the Azure one, which can call a cloud endpoint) is used.
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.card = card
        self.input_name = card["input_name"]
        _, _, h, w = card["input_shape"]
        self.size = (int(h), int(w))
        self.means, self.stds = card["means"], card["stds"]
        self.num_select = int(card["num_select"])
        self.class_names = {int(k): v for k, v in card["classes"].items()}
        self.outputs = card["output_names"]  # [boxes, logits]
        self.label = card.get("label", model_path.stem)

    def predict(self, images, threshold: float) -> list[list[Box]]:
        results = []
        for rgb in images:
            h, w = rgb.shape[:2]
            dets, logits = self.session.run(self.outputs, {self.input_name: preprocess(rgb, self.size, self.means, self.stds)})
            results.append(postprocess(dets[0], logits[0], w, h, threshold, self.num_select, self.class_names))
        return results

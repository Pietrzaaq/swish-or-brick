"""Corrected labels -> RF-DETR dataset: train/, valid/, test/, each with _annotations.coco.json and its images.

    python -m brickorswish.train.dataset LABELS --drafts DRAFTS_DIR --out DATASET_DIR

LABELS is what the labelling tool exports: CVAT "COCO 1.0" (zip or json) or Roboflow "COCO" (zip). Classes are
matched by name, images by file name (Roboflow's renamed copies too); the images themselves come from DRAFTS_DIR.

Split by time, not by frame: neighbouring frames are near-copies, so all frames from the same 20 s of a session go
to the same split. Otherwise validation would score the model on copies of its training images.
"""

import argparse
import json
import re
import shutil
import zipfile
import zlib
from pathlib import Path

from . import CLASSES

BLOCK_S = 20.0
# Roboflow renames "a.jpg" to "a_jpg.rf.<hash>.jpg".
_ROBOFLOW_NAME = re.compile(r"^(.*)_(jpe?g|png)\.rf\.[0-9a-fA-F]+\.(jpe?g|png)$")


def split_of(session: str, t: float) -> str:
    k = zlib.crc32(f"{session}:{int(t // BLOCK_S)}".encode()) % 10
    return "valid" if k == 8 else "test" if k == 9 else "train"


def original_name(file_name: str) -> str:
    base = file_name.replace("\\", "/").rsplit("/", 1)[-1]
    m = _ROBOFLOW_NAME.match(base)
    return f"{m.group(1)}.{m.group(2)}" if m else base


def read_exports(path: Path) -> list[dict]:
    """Every COCO json in a json file or a zip (Roboflow puts one in each split folder)."""
    if path.suffix.lower() != ".zip":
        return [json.loads(path.read_text(encoding="utf-8"))]
    with zipfile.ZipFile(path) as z:
        found = [json.loads(z.read(n)) for n in z.namelist() if n.lower().endswith(".json")]
    return [c for c in found if "images" in c and "annotations" in c]


def collect(exports: list[dict]) -> dict[str, list[tuple[str, list[float]]]]:
    """file name -> [(class, [x, y, w, h])]; images without boxes are kept (they teach "nothing here")."""
    labels: dict[str, list[tuple[str, list[float]]]] = {}
    unknown = set()
    for coco in exports:
        used = {a["category_id"] for a in coco["annotations"]}
        names = {c["id"]: c["name"] for c in coco["categories"]}
        unknown |= {names[i] for i in used if names.get(i) not in CLASSES}
        files = {im["id"]: original_name(im["file_name"]) for im in coco["images"]}
        for name in files.values():
            labels.setdefault(name, [])
        for a in coco["annotations"]:
            cls = names.get(a["category_id"])
            if cls in CLASSES and a["bbox"][2] > 0 and a["bbox"][3] > 0:
                labels[files[a["image_id"]]].append((cls, [float(v) for v in a["bbox"]]))
    if unknown:
        raise ValueError(f"unknown classes {sorted(unknown)}: use exactly {list(CLASSES)} in the labelling tool")
    return labels


def build(labels: dict, manifest: dict, images: Path, out: Path) -> dict:
    missing = [n for n in labels if n not in manifest or not (images / n).exists()]
    if missing:
        raise ValueError(f"{len(missing)} labelled images are not in the drafts folder, e.g. {missing[:3]}")
    if out.exists():
        shutil.rmtree(out)
    cats = [{"id": k + 1, "name": n, "supercategory": "none"} for k, n in enumerate(CLASSES)]
    splits = {s: {"images": [], "annotations": [], "categories": cats} for s in ("train", "valid", "test")}
    counts = {s: {"images": 0, **{n: 0 for n in CLASSES}} for s in splits}
    for name in sorted(labels):
        m = manifest[name]
        split = split_of(m["session"], m["t"])
        coco = splits[split]
        (out / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(images / name, out / split / name)
        image_id = len(coco["images"]) + 1
        w, h = m["crop"][2] - m["crop"][0], m["crop"][3] - m["crop"][1]
        coco["images"].append({"id": image_id, "file_name": name, "width": w, "height": h})
        counts[split]["images"] += 1
        for cls, (x, y, bw, bh) in labels[name]:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                                        "category_id": CLASSES.index(cls) + 1, "bbox": [x, y, bw, bh],
                                        "area": round(bw * bh, 1), "iscrowd": 0})
            counts[split][cls] += 1
    for split, coco in splits.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        (out / split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return counts


def warnings(counts: dict) -> list[str]:
    out = []
    for split in ("valid", "test"):
        absent = [n for n in CLASSES if counts[split][n] == 0]
        if absent:
            out.append(f"{split} has no {', '.join(absent)}: the score for them means nothing (label more sessions)")
    if counts["train"]["ball_in_net"] < 150:
        out.append(f"only {counts['train']['ball_in_net']} ball_in_net boxes to train on; aim for 300+")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="the labelling tool's COCO export (.zip or .json)")
    ap.add_argument("--drafts", required=True, help="folder written by brickorswish.train.drafts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    drafts = Path(args.drafts)
    manifest = json.loads((drafts / "manifest.json").read_text(encoding="utf-8"))
    labels = collect(read_exports(Path(args.labels)))
    counts = build(labels, manifest, drafts / "images", Path(args.out))
    for split, row in counts.items():
        print(f"{split:<5} " + "  ".join(f"{k} {v}" for k, v in row.items()))
    for w in warnings(counts):
        print("WARNING: " + w)


if __name__ == "__main__":
    main()

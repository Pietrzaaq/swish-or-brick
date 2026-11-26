"""Training frames with draft labels from a finished run: video + its results folder -> images + COCO json.

    python -m brickorswish.train.drafts VIDEO RESULTS_DIR --out DRAFTS_DIR --session NAME

Run it once per session into the same DRAFTS_DIR (a re-run replaces that session's images). Drafts come from the
run itself: ball and person boxes from the COCO detector, the rim from hoop.json, and ball_in_net for a ball inside
the net during a shot the logic called a make. They are only a head start: every image still needs checking.

Images are the square crop around the rim that detection uses (crop.watch_crop), plus a full frame now and then so
the model also learns the rim and players at full-frame scale. Frames come mostly from around shots.
"""

import argparse
import json
import math
from pathlib import Path

from ..detect.crop import watch_crop
from ..detect.motion import IDENTITY, MotionConfig, Segment, frame_transforms, rim_centre, shift_hoop, to_frame
from ..logic import HoopFrame, ShotLogicConfig
from ..logic.pipeline import near_static, static_spots
from ..paths import CONFIG_V1
from . import CLASSES

PERSON_MIN_CONF = 0.5
# A ball centre in the net, in rim units: below the rim plane, within the net's width.
NET_Y = (0.2, 2.3)
NET_ABS_X = 1.1
JPEG_QUALITY = 92


def rim_box(hf: HoopFrame, pad: float = 0.05) -> tuple[float, float, float, float]:
    """Pixel box around the rim ellipse (a = radius, b = ellipse_b * radius, tilted by rim_angle)."""
    a = hf.radius_px * (1.0 + pad)
    b = hf.ellipse_b * hf.radius_px + pad * hf.radius_px
    c, s = math.cos(hf.rim_angle), math.sin(hf.rim_angle)
    hw = math.hypot(a * c, b * s)
    hh = math.hypot(a * s, b * c)
    return hf.cx - hw, hf.cy - hh, hf.cx + hw, hf.cy + hh


def pick_frames(n_frames: int, fps: float, shots: list[dict], transforms: list, shot_step_s: float = 0.1,
                idle_every_s: float = 3.0, full_every_s: float = 15.0, before_s: float = 0.3,
                after_s: float = 0.6) -> dict[tuple[int, str], dict | None]:
    """(frame, "crop" | "full") -> the shot it belongs to (None between shots). Frames where the hoop was lost
    are skipped: their boxes cannot be placed."""
    out: dict[tuple[int, str], dict | None] = {}

    def add(i: int, kind: str, shot: dict | None) -> None:
        if 0 <= i < n_frames and transforms[i] is not None:
            out.setdefault((i, kind), shot)

    step = max(1, round(shot_step_s * fps))
    for s in shots:
        first = math.floor((s["t_start"] - before_s) * fps)
        last = math.ceil((s["t_resolve"] + after_s) * fps)
        for i in range(first, last + 1, step):
            add(i, "crop", s)
    for every, kind in ((idle_every_s, "crop"), (full_every_s, "full")):
        if every > 0:
            for i in range(0, n_frames, max(1, round(every * fps))):
                add(i, kind, None)
    return out


def draft_boxes(dets: list[dict], hoop: dict, cfg: ShotLogicConfig, static: frozenset, tr, in_make: bool,
                origin: tuple[int, int], size: tuple[int, int]) -> list[tuple[str, tuple[float, float, float, float]]]:
    """(class, (x0, y0, x1, y1)) in the saved image's pixels. dets are in calibration-frame pixels (detections.json);
    tr moves them back to where they were in this frame."""
    hf = HoopFrame.from_dict(hoop)
    c0 = rim_centre(hoop)
    ox, oy = origin
    out = []
    for d in dets:
        if d["cls"] == "ball":
            if d["conf"] < cfg.ball_min_conf:
                continue
            if not cfg.ball_min_size_rim_d <= hf.size_in_rim_diameters(d["w"], d["h"]) <= cfg.ball_max_size_rim_d:
                continue
            x, y = hf.to_hoop(d["x"], d["y"])
            if static and near_static(x, y, static, cfg):
                continue  # the rim's bracket: left unlabelled, so the model learns it is not a ball
            name = "ball_in_net" if in_make and NET_Y[0] <= y <= NET_Y[1] and abs(x) <= NET_ABS_X else "ball"
        elif d["cls"] == "person" and d["conf"] >= PERSON_MIN_CONF:
            name = "person"
        else:
            continue
        cx, cy = to_frame(d["x"], d["y"], c0, tr)
        w, h = d["w"] * tr[2], d["h"] * tr[2]
        box = _clip((cx - w / 2 - ox, cy - h / 2 - oy, cx + w / 2 - ox, cy + h / 2 - oy), size)
        if box:
            out.append((name, box))
    x0, y0, x1, y1 = rim_box(HoopFrame.from_dict(shift_hoop(hoop, tr)))
    box = _clip((x0 - ox, y0 - oy, x1 - ox, y1 - oy), size)
    if box:
        out.append(("rim", box))
    return out


def _clip(box, size: tuple[int, int]):
    w, h = size
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    x1, y1 = min(float(w), box[2]), min(float(h), box[3])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)


def new_coco() -> dict:
    return {"images": [], "annotations": [],
            "categories": [{"id": k + 1, "name": n, "supercategory": "none"} for k, n in enumerate(CLASSES)]}


def drop_session(coco: dict, manifest: dict, session: str) -> set[str]:
    """Removes an earlier export of `session`, so a re-run does not duplicate it. Returns its file names."""
    gone = {name for name, m in manifest.items() if m["session"] == session}
    ids = {im["id"] for im in coco["images"] if im["file_name"] in gone}
    coco["images"] = [im for im in coco["images"] if im["id"] not in ids]
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] not in ids]
    for name in gone:
        del manifest[name]
    return gone


def add_image(coco: dict, file_name: str, width: int, height: int, boxes) -> None:
    image_id = max((im["id"] for im in coco["images"]), default=0) + 1
    coco["images"].append({"id": image_id, "file_name": file_name, "width": width, "height": height})
    next_id = max((a["id"] for a in coco["annotations"]), default=0) + 1
    for k, (name, (x0, y0, x1, y1)) in enumerate(boxes):
        w, h = round(x1 - x0, 1), round(y1 - y0, 1)
        coco["annotations"].append({"id": next_id + k, "image_id": image_id, "category_id": CLASSES.index(name) + 1,
                                    "bbox": [x0, y0, w, h], "area": round(w * h, 1), "iscrowd": 0})


def _transforms(results: Path, n_frames: int) -> list:
    path = results / "camera_motion.json"
    if not path.exists():
        return [IDENTITY] * n_frames
    motion = json.loads(path.read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in motion["segments"]]
    trs = frame_transforms(segments, motion["n_frames"], motion["check_every_frames"], MotionConfig())
    return trs + [None] * max(0, n_frames - len(trs))


def export(video: Path, results: Path, out: Path, session: str, **pick) -> dict:
    import cv2

    cfg = ShotLogicConfig.load(CONFIG_V1)
    hoop = json.loads((results / "hoop.json").read_text(encoding="utf-8"))
    detections = json.loads((results / "detections.json").read_text(encoding="utf-8"))
    shots = json.loads((results / "events.json").read_text(encoding="utf-8"))["shots"]
    by_frame = {f["i"]: f for f in detections["frames"]}
    static = static_spots(detections, HoopFrame.from_dict(hoop), cfg)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(detections["fps"])
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    transforms = _transforms(results, n_frames)
    wanted = pick_frames(n_frames, fps, shots, transforms, **pick)
    by_index: dict[int, list[tuple[str, dict | None]]] = {}
    for (i, kind), shot in wanted.items():
        by_index.setdefault(i, []).append((kind, shot))

    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    coco_path, manifest_path = out / "annotations.coco.json", out / "manifest.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8")) if coco_path.exists() else new_coco()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for old in drop_session(coco, manifest, session):
        (images / old).unlink(missing_ok=True)

    counts = {n: 0 for n in CLASSES}
    hidden_in_net = 0
    i = 0
    try:
        while by_index:
            ok, bgr = cap.read()
            if not ok:
                break
            for kind, shot in by_index.pop(i, []):
                h_img, w_img = bgr.shape[:2]
                if (w_img, h_img) != (hoop["image_width"], hoop["image_height"]):
                    raise ValueError(f"video is {w_img}x{h_img} but hoop.json is "
                                     f"{hoop['image_width']}x{hoop['image_height']}")
                tr = transforms[i]
                x0, y0, x1, y1 = watch_crop(shift_hoop(hoop, tr), cfg) if kind == "crop" else (0, 0, w_img, h_img)
                t = i / fps
                in_make = shot is not None and shot["outcome"] == "make" and shot["t_start"] <= t <= shot["t_resolve"] + 0.5
                dets = by_frame.get(i, {"dets": []})["dets"]
                boxes = draft_boxes(dets, hoop, cfg, static, tr, in_make, (x0, y0), (x1 - x0, y1 - y0))
                name = f"{session}_{i:06d}{'_full' if kind == 'full' else ''}.jpg"
                cv2.imwrite(str(images / name), bgr[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                add_image(coco, name, x1 - x0, y1 - y0, boxes)
                for cls, _ in boxes:
                    counts[cls] += 1
                hidden = in_make and not any(c in ("ball", "ball_in_net") for c, _ in boxes)
                hidden_in_net += hidden
                manifest[name] = {"session": session, "frame": i, "t": round(t, 3), "kind": kind,
                                  "crop": [x0, y0, x1, y1], "shot": shot["shot"] if shot else None,
                                  "outcome": shot["outcome"] if shot else None, "make_without_ball": hidden}
            i += 1
    finally:
        cap.release()

    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=0), encoding="utf-8")
    (out / "check_these.txt").write_text(
        "".join(f"{name}\n" for name, m in sorted(manifest.items()) if m["make_without_ball"]), encoding="utf-8")
    n = sum(1 for m in manifest.values() if m["session"] == session)
    return {"session": session, "images": n, "boxes": counts, "make_without_ball": hidden_in_net}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("results", help="the run's folder: hoop.json, detections.json, events.json, camera_motion.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", required=True, help="short name, used in the image file names")
    ap.add_argument("--shot-step-s", type=float, default=0.1, help="one frame every this many seconds around shots")
    ap.add_argument("--idle-every-s", type=float, default=3.0, help="one crop every this many seconds anyway")
    ap.add_argument("--full-every-s", type=float, default=15.0, help="one full frame every this many seconds")
    args = ap.parse_args()
    summary = export(Path(args.video), Path(args.results), Path(args.out), args.session,
                     shot_step_s=args.shot_step_s, idle_every_s=args.idle_every_s, full_every_s=args.full_every_s)
    print(f"{summary['session']}: {summary['images']} images, draft boxes {summary['boxes']}")
    print(f"  {summary['make_without_ball']} frames during makes with no ball box (listed in check_these.txt): "
          "the ball is probably hidden in the net; draw ball_in_net there if you can see it")


if __name__ == "__main__":
    main()

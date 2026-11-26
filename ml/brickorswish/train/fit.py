"""Fine-tune RF-DETR Nano on a dataset from brickorswish.train.dataset. Cloud GPU only (needs PyTorch).

    python -m brickorswish.train.fit DATASET_DIR --out RUN_DIR [--epochs 50]

Starts from the COCO-pretrained Nano. Continues from RUN_DIR's last checkpoint if there is one (Colab and Kaggle
stop long runs), so keep RUN_DIR on Drive. Best weights: RUN_DIR/checkpoint_best_total.pth; turn them into
model.onnx with brickorswish.detect.export_onnx export --weights.
"""

import argparse
import sys
from pathlib import Path

from . import CLASSES

RESUME_FROM = ("last.ckpt", "checkpoint.pth")  # newer rfdetr (full state) / older
BEST = "checkpoint_best_total.pth"


def train_kwargs(args, allowed: set[str] | None) -> tuple[dict, list[str]]:
    """rfdetr refuses unknown settings, and they differ between its versions: only the ones it knows are passed."""
    out = Path(args.out)
    kwargs = {
        "dataset_dir": str(Path(args.dataset)),
        "output_dir": str(out),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "grad_accum_steps": args.grad_accum,
        "lr": args.lr,
        "class_names": list(CLASSES),
        "early_stopping": True,
        "early_stopping_patience": args.patience,
        "checkpoint_interval": 5,
        "tensorboard": False,
        "wandb": False,
    }
    resume = next((out / n for n in RESUME_FROM if (out / n).exists()), None)
    if resume is not None:
        kwargs["resume"] = str(resume)
    if allowed is None:
        return kwargs, []
    dropped = sorted(k for k in kwargs if k not in allowed and k not in ("device", "resolution"))
    return {k: v for k, v in kwargs.items() if k not in dropped}, dropped


def _allowed_settings() -> set[str] | None:
    try:
        from rfdetr.config import TrainConfig
    except ImportError:
        return None
    fields = getattr(TrainConfig, "model_fields", None) or getattr(TrainConfig, "__fields__", None)
    return set(fields) if fields else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="batch x grad-accum = images per update (16)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10, help="stop after this many epochs without improvement")
    args = ap.parse_args()

    import torch
    from rfdetr import RFDETRNano

    if not torch.cuda.is_available():
        print("No GPU: training on a CPU would take days. Runtime > Change runtime type > T4 GPU.")
        return 1
    Path(args.out).mkdir(parents=True, exist_ok=True)
    kwargs, dropped = train_kwargs(args, _allowed_settings())
    if dropped:
        print(f"this rfdetr version does not know {dropped}: training without them")
    print("continuing from " + kwargs["resume"] if "resume" in kwargs else "starting from COCO-pretrained Nano")
    RFDETRNano().train(**kwargs)

    best = Path(args.out) / BEST
    if not best.exists():
        print(f"no {BEST} in {args.out}: training did not finish an evaluation")
        return 1
    names = list(RFDETRNano(pretrain_weights=str(best)).class_names)[:len(CLASSES)]
    print(f"best weights: {best}; class names in it: {names}")
    if names != list(CLASSES):
        print(f"WARNING: expected {list(CLASSES)}; export uses our order, check the dataset categories")
    return 0


if __name__ == "__main__":
    sys.exit(main())

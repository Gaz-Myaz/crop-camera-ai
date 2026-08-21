"""
train.py

Train a YOLOv8 segmentation model on the dataset prepared by
prepare_dataset.py, then report what it actually learned.

Segmentation rather than plain detection is deliberate: a bounding box round
a plant also contains soil, mulch and neighbouring leaves, and averaging NDVI
inside a box therefore measures the background as much as the plant. A mask
follows the actual foliage, which is what makes per_plant_ndvi.py possible.

--------------------------------------------------------------------
    uv pip install ultralytics

    uv run train.py --data dataset/dataset.yaml
    uv run train.py --data dataset/dataset.yaml --model yolov8s-seg.pt --epochs 150
    uv run train.py --data dataset/dataset.yaml --validate-only --weights runs/.../best.pt

Model size: yolov8n-seg is the default because the rovers are Raspberry Pis
and the nano model is the only one that runs at a sensible rate there. Train
's' or 'm' as well if you have the GPU time -- the accuracy gap is worth
knowing before you commit to the small one, and you can always deploy nano
while using a larger model to judge how much you gave up.

Reading the result: mAP50 is the headline, but per-class numbers are what
matter. A model averaging 0.90 while scoring 0.4 on the one disease you
actually care about is a bad model wearing a good number, so this prints
the per-class breakdown and flags any class that lags badly.
--------------------------------------------------------------------
"""

import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def report(metrics, names) -> None:
    """Print per-class results and call out the weak classes explicitly."""
    print("\n" + "=" * 66)
    print("RESULTS")
    print("=" * 66)
    try:
        box_map50 = float(metrics.box.map50)
        seg_map50 = float(metrics.seg.map50)
        print(f"  overall  box mAP50 {box_map50:.3f}   mask mAP50 {seg_map50:.3f}")
        print(f"           box mAP50-95 {float(metrics.box.map):.3f}   "
              f"mask mAP50-95 {float(metrics.seg.map):.3f}")
    except Exception:
        print("  (could not read summary metrics from this ultralytics version)")
        return

    try:
        per_class = list(metrics.seg.ap50)
        idx = list(metrics.ap_class_index)
        print(f"\n  per-class mask mAP50:")
        weak = []
        for i, ap in zip(idx, per_class):
            name = names.get(i, str(i)) if isinstance(names, dict) else str(i)
            flag = ""
            if ap < 0.5:
                flag = "   <-- WEAK"
                weak.append(name)
            elif ap < 0.7:
                flag = "   <-- marginal"
            print(f"    {name:<28} {ap:.3f}{flag}")
        if weak:
            print(f"\n  {len(weak)} class(es) below 0.50: {', '.join(weak)}")
            print("  Usually too few training examples of that class, or it looks")
            print("  genuinely like another class. Check the class counts printed")
            print("  by prepare_dataset.py --inspect before assuming it's the model.")
    except Exception:
        pass

    print("\n  Remember these numbers describe the dataset you trained on.")
    print("  Accuracy on someone else's greenhouse photos is NOT a")
    print("  prediction of accuracy on your field. Treat it as a starting")
    print("  point to fine-tune once you have your own labelled images.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=str(_HERE / "dataset" / "dataset.yaml"))
    ap.add_argument("--model", type=str, default="yolov8n-seg.pt",
                    help="Starting weights. Pretrained -seg checkpoints transfer "
                         "far better than training from scratch on a small set.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto-fit to memory")
    ap.add_argument("--device", type=str, default=None,
                    help="'0' for first GPU, 'cpu' to force CPU. Default: auto.")
    ap.add_argument("--project", type=str, default=str(_HERE / "runs"))
    ap.add_argument("--name", type=str, default="strawberry-seg")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--weights", type=str, default=None,
                    help="Weights to validate with --validate-only.")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        print("Run prepare_dataset.py --convert <downloaded_folder> first.")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.  uv pip install ultralytics")
        return 1

    if args.validate_only:
        if not args.weights:
            print("ERROR: --validate-only needs --weights <path to best.pt>")
            return 1
        model = YOLO(args.weights)
        metrics = model.val(data=str(data_path), imgsz=args.imgsz, split="test")
        report(metrics, getattr(model, "names", {}))
        return 0

    print(f"Training {args.model} on {data_path}")
    print("First run downloads the pretrained checkpoint. On CPU this will be "
          "slow -- hours, not minutes. A GPU is strongly advised for 100 epochs.")
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=25,        # stop if validation stalls; avoids overfitting a small set
        seed=0,             # reproducible runs
    )

    metrics = model.val(data=str(data_path), imgsz=args.imgsz)
    report(metrics, getattr(model, "names", {}))

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print(f"Try it:   uv run per_plant_ndvi.py --weights {best} --demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

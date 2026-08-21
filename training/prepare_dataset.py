"""
prepare_dataset.py

Turn a downloaded strawberry dataset into the layout YOLO training expects.

Written against the Kaggle "Strawberry Disease Detection Dataset" (2,500
images, seven disease classes, polygon segmentation masks in LabelMe-style
JSON), but it inspects what it finds rather than assuming, because dataset
layouts drift and a converter that assumes wrongly produces a training set
that looks fine and teaches the model nonsense.

--------------------------------------------------------------------
Step 1 -- LOOK before converting. This never writes anything:

    uv run prepare_dataset.py --inspect <downloaded_folder>

It reports the directory tree, how many images and annotations it found,
which class names appear and how often, and prints one annotation file so
you can confirm the structure matches what the converter expects.

Step 2 -- convert:

    uv run prepare_dataset.py --convert <downloaded_folder> --out dataset

Produces:

    dataset/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt      one line per instance:
                                         class_id x1 y1 x2 y2 ... (normalised)
      dataset.yaml                       what train.py reads

If the source already has train/val/test folders those splits are kept --
re-splitting someone else's benchmark makes your numbers incomparable to
theirs. Otherwise it splits by --val-frac / --test-frac, grouping by
filename stem so augmented copies of one photo can't land on both sides of
the split and inflate your score.
--------------------------------------------------------------------
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = ("train", "val", "test")


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT)


def find_annotations(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.json"))


def load_shapes(ann_path: Path):
    """Extract [(label, [(x, y), ...]), ...] from a LabelMe-style JSON.

    Returns [] for anything that doesn't parse as expected, so one odd file
    doesn't abort a 2,500-image conversion.
    """
    try:
        data = json.loads(ann_path.read_text())
    except Exception:
        return []
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return []
    out = []
    for sh in shapes:
        label = sh.get("label")
        pts = sh.get("points")
        if not label or not isinstance(pts, list) or len(pts) < 3:
            continue  # fewer than 3 points is not a polygon
        try:
            out.append((str(label), [(float(x), float(y)) for x, y in pts]))
        except Exception:
            continue
    return out


def image_size(data: dict, img_path: Path):
    """Prefer the size recorded in the annotation; fall back to the image."""
    h, w = data.get("imageHeight"), data.get("imageWidth")
    if isinstance(h, (int, float)) and isinstance(w, (int, float)) and h and w:
        return int(w), int(h)
    try:
        import cv2
        im = cv2.imread(str(img_path))
        if im is not None:
            return im.shape[1], im.shape[0]
    except Exception:
        pass
    return None


def inspect(root: Path) -> None:
    print(f"Inspecting {root}\n" + "=" * 64)
    if not root.exists():
        print("ERROR: that path does not exist.")
        return

    subdirs = sorted(p for p in root.rglob("*") if p.is_dir())
    print(f"Directories ({len(subdirs)}):")
    for d in subdirs[:25]:
        n_img = len([p for p in d.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXT])
        n_json = len([p for p in d.iterdir() if p.is_file() and p.suffix == ".json"])
        if n_img or n_json:
            print(f"  {d.relative_to(root)}/   {n_img} images, {n_json} json")
    if len(subdirs) > 25:
        print(f"  ... and {len(subdirs)-25} more")

    images, anns = find_images(root), find_annotations(root)
    print(f"\nTotals: {len(images)} images, {len(anns)} json annotation files")

    if not images:
        print("\nNo images found. Is this the right folder, or is it still zipped?")
        return

    existing = [s for s in SPLIT_NAMES
                if any(p.name.lower() == s for p in root.rglob("*") if p.is_dir())]
    if existing:
        print(f"Existing splits detected: {existing}  (these will be preserved)")

    labels = Counter()
    polys = 0
    parsed = 0
    for a in anns[:400]:                       # sample; enough to characterise
        shapes = load_shapes(a)
        if shapes:
            parsed += 1
        for label, pts in shapes:
            labels[label] += 1
            polys += 1
    print(f"\nParsed {parsed}/{min(len(anns),400)} sampled annotations, "
          f"{polys} polygons")
    if labels:
        print("\nClasses found:")
        for name, count in labels.most_common():
            print(f"  {count:>6}  {name}")
    else:
        print("\nNo polygons parsed. The annotation format may differ from")
        print("LabelMe. Sample file below -- send it to me and I'll adapt the")
        print("converter.")

    if anns:
        print(f"\nSample annotation ({anns[0].name}):")
        try:
            data = json.loads(anns[0].read_text())
            for k, v in data.items():
                if k == "shapes" and isinstance(v, list) and v:
                    print(f"  shapes: [{len(v)} items], first = "
                          f"{ {kk: (vv[:2] if kk=='points' else vv) for kk, vv in v[0].items()} }")
                elif k == "imageData":
                    print("  imageData: <base64 omitted>")
                else:
                    print(f"  {k}: {str(v)[:80]}")
        except Exception as e:
            print(f"  could not parse: {e}")


def convert(root: Path, out: Path, val_frac: float, test_frac: float,
            seed: int) -> int:
    anns = find_annotations(root)
    if not anns:
        print("ERROR: no .json annotations found. Run --inspect first.")
        return 1

    # Pair each annotation with its image.
    images_by_stem = {}
    for p in find_images(root):
        images_by_stem.setdefault(p.stem, p)

    records = []
    class_names = set()
    unmatched = 0
    for a in anns:
        shapes = load_shapes(a)
        if not shapes:
            continue
        img = images_by_stem.get(a.stem)
        if img is None:
            unmatched += 1
            continue
        try:
            data = json.loads(a.read_text())
        except Exception:
            continue
        size = image_size(data, img)
        if size is None:
            continue
        records.append((img, shapes, size, a))
        class_names.update(lbl for lbl, _ in shapes)

    if not records:
        print("ERROR: parsed no usable image+annotation pairs. Run --inspect.")
        return 1
    if unmatched:
        print(f"NOTE: {unmatched} annotations had no matching image; skipped.")

    names = sorted(class_names)
    class_id = {n: i for i, n in enumerate(names)}
    print(f"{len(records)} annotated images, {len(names)} classes: {names}")

    # Respect the dataset's own split when it has one.
    def split_of(img_path: Path):
        parts = {q.lower() for q in img_path.relative_to(root).parts}
        for s in SPLIT_NAMES:
            if s in parts:
                return s
        if "valid" in parts or "validation" in parts:
            return "val"
        return None

    preset = {split_of(r[0]) for r in records} - {None}
    if preset:
        print(f"Using the dataset's own splits: {sorted(preset)}")
        assign = {r[0]: (split_of(r[0]) or "train") for r in records}
    else:
        # Group by stem prefix so augmented variants of one photo stay together.
        rng = random.Random(seed)
        stems = sorted({r[0].stem.split("_aug")[0] for r in records})
        rng.shuffle(stems)
        n = len(stems)
        n_val, n_test = int(n * val_frac), int(n * test_frac)
        bucket = {}
        for i, st in enumerate(stems):
            bucket[st] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
        assign = {r[0]: bucket[r[0].stem.split("_aug")[0]] for r in records}
        print(f"Split {n} image groups -> "
              f"train {sum(v=='train' for v in bucket.values())}, "
              f"val {sum(v=='val' for v in bucket.values())}, "
              f"test {sum(v=='test' for v in bucket.values())}")

    for s in SPLIT_NAMES:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    written = Counter()
    for img, shapes, (w, h), _ in records:
        s = assign[img]
        shutil.copy2(img, out / "images" / s / img.name)
        lines = []
        for label, pts in shapes:
            # YOLO segmentation: class then normalised polygon vertices.
            coords = []
            for x, y in pts:
                coords.append(f"{min(max(x / w, 0.0), 1.0):.6f}")
                coords.append(f"{min(max(y / h, 0.0), 1.0):.6f}")
            lines.append(f"{class_id[label]} " + " ".join(coords))
        (out / "labels" / s / f"{img.stem}.txt").write_text("\n".join(lines))
        written[s] += 1

    yaml = out / "dataset.yaml"
    yaml.write_text(
        "# Generated by prepare_dataset.py\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(names)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )

    print(f"\nWrote {dict(written)} to {out}")
    print(f"dataset.yaml -> {yaml}")
    print(f"\nNext:  uv run train.py --data {yaml}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect", type=str, help="Report a dataset's structure; writes nothing.")
    ap.add_argument("--convert", type=str, help="Source folder to convert.")
    ap.add_argument("--out", type=str, default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.inspect:
        inspect(Path(args.inspect).expanduser())
        return 0
    if args.convert:
        return convert(Path(args.convert).expanduser(), Path(args.out).expanduser(),
                       args.val_frac, args.test_frac, args.seed)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# training

Train a plant-segmentation model, and combine its masks with NDVI to get a health figure **per plant** rather than per frame.

## Why segmentation, and why combine it with NDVI

YOLO and NDVI answer different questions and neither replaces the other. NDVI *measures* — every pixel gets a health number derived from physics. YOLO *identifies* — it finds discrete objects and outlines them. Asking a detector to judge plant stress from appearance alone means asking it to learn the thing NDVI measures directly, and it will be worse at it, because early stress is invisible in ordinary colour images. That is the entire reason for the infrared camera.

Put them together and you get something neither gives alone:

```
YOLO mask        "these pixels are plant #7"
NDVI array       "this pixel reads 0.38"
        ↓
"plant #7 is stressed, worst tenth at 0.38, while its neighbours are at 0.71"
```

A frame-level average says the bed looks fine. A per-plant figure sends someone to a specific plant — and once positions are logged, it becomes a record you can follow for that individual plant across a season.

Masks rather than boxes, because a box around a plant also contains soil, mulch and neighbouring leaves. Averaging NDVI inside a box substantially measures the background.

## The catch: run YOLO on the colour camera

Detection models are trained on ordinary colour photographs. An NDVI-converted camera outputs something close to monochrome red (measured on ours: R=150, G=0, B=1.5), and YOLO does poorly on it.

So each sensor does what it is good at: **stereo camera → distance, NIR pair → health measurement, colour camera → identity.** On a two-camera NDVI rig the colour camera is an ordinary RGB sensor and is the right input.

## Workflow

### 1. Get a dataset

**See [DATASETS.md](DATASETS.md)** for the full survey — datasets organised by problem, licences flagged, plus the libraries worth adding and which public data bears on the mud-versus-necrosis question.

The best public starting point is the [Kaggle Strawberry Disease Detection Dataset](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset) — 2,500 images, seven disease classes, real polygon masks, shot in greenhouses under natural light. Published YOLOv8 results on it reach roughly 92–93% mAP50.

Check its licence before commercial use, and treat the accuracy figures as describing *that* dataset. Korean greenhouse conditions are not your field.

### 2. Look at it before converting

```bash
uv run prepare_dataset.py --inspect ~/Downloads/strawberry_disease
```

Writes nothing. Reports the directory tree, image and annotation counts, every class name with its frequency, and one full annotation so you can confirm the format. **Read the class counts** — a class with 40 examples will train badly, and knowing that now saves you diagnosing it as a model problem later.

### 3. Convert to YOLO format

```bash
uv run prepare_dataset.py --convert ~/Downloads/strawberry_disease --out dataset
```

Existing train/val/test folders are preserved — re-splitting someone else's benchmark makes your numbers incomparable to theirs. Where no split exists, it groups by filename stem so augmented copies of one photo cannot land on both sides and inflate your score.

### 4. Train

```bash
uv run train.py --data dataset/dataset.yaml
uv run train.py --data dataset/dataset.yaml --model yolov8s-seg.pt --epochs 150
```

`yolov8n-seg` is the default because the rovers are Raspberry Pis and nano is what runs there at a sensible rate. Training a larger model too is worth it to learn how much accuracy the small one costs you.

Expect hours on CPU. A GPU is strongly advised for 100 epochs.

The report prints **per-class** mAP and flags weak classes, because a model averaging 0.90 while scoring 0.4 on the one disease you care about is a bad model wearing a good number.

### 5. Per-plant NDVI

```bash
uv run per_plant_ndvi.py --demo          # no camera, model or dataset needed

uv run per_plant_ndvi.py \
    --weights runs/strawberry-seg/weights/best.pt \
    --image frame.png --ndvi ndvi.npy
```

`--demo` builds a synthetic bed with four plants of known health plus two berries and checks the pipeline recovers exactly those classes. Useful as a regression test after any change to the statistics.

Output is one record per plant plus a compact text summary — the form the LLM agent consumes:

```
4 plants measured; mean plant NDVI +0.50; worst plant is critical.
2 needing attention: plant 3 critical (NDVI +0.13), plant 2 stressed (NDVI +0.37).
2 fruit/flower detections (excluded from health).
```

## Three decisions worth understanding

These come from the companion two-camera NDVI project, which established them by measurement rather than argument.

**Statistics use the eroded interior of each mask.** Boundary pixels are leaf/background mixtures landing mid-scale. This is not a subtle correction: in `--demo`, a genuinely vigorous plant (median NDVI +0.797) was reported as *Severe stress* before erosion depth was raised, because a 3px mixture ring is about 10% of a 55px-radius plant's area — landing exactly on the worst-decile statistic. `DEFAULT_ERODE_PX` must exceed the mixture band at your image scale; `--demo` prints how much area erosion removes so you can see it working.

**Status comes from the worst tenth, not the mean.** A mean lets a healthy majority hide a dying edge. The worst-decile rule is also monotonic — more damage can never produce a better status — which most natural-looking alternatives fail to guarantee.

**Fruit and flowers are excluded from health statistics but still reported.** Neither photosynthesises, so both read as stressed foliage on a thriving plant. Leaving them in produces a false alarm that peaks exactly at harvest. Classes are recognised by name via `NON_FOLIAGE_KEYWORDS` — check that against your own class names.

## Preconditions and limits

**The RGB image and NDVI array must be registered to each other.** Pixel (y, x) must be the same ground in both. Feed it an unaligned pair and it will measure one plant's outline against another patch's NDVI, and report it confidently. Shapes are checked; identical shapes do not prove alignment.

**The health thresholds in `HEALTH_CLASSES` are reasoned starting points, not validated constants.** They are the first thing to calibrate against real flagged frames.

**A model trained on public data is a starting point, not a finished detector.** Different crop variety, lighting, camera angle and growth stage all cost accuracy. Plan to fine-tune on your own labelled images.

## Files

| File | Purpose |
|---|---|
| `prepare_dataset.py` | Inspect a downloaded dataset; convert LabelMe-style polygons to YOLO segmentation format. |
| `train.py` | Train and validate a YOLOv8-seg model; per-class reporting. |
| `per_plant_ndvi.py` | Combine masks with NDVI into per-plant health. `--demo` runs with no dependencies on hardware or data. |

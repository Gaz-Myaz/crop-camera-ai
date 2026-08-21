"""
per_plant_ndvi.py

The bridge between the two halves of this project: YOLO segmentation says
*where each plant is*, NDVI says *how healthy each pixel is*. Combine them
and you get a health figure per plant instead of per frame.

That distinction is the point. A frame-level number tells you "this bed
averages 0.61". A per-plant number tells you "plant 7 is at 0.38 while its
neighbours are at 0.71" -- which is an actionable instruction to go and look
at one plant, and, once positions are logged, something you can track for
that individual plant across weeks.

--------------------------------------------------------------------
PRECONDITION: the RGB image and the NDVI array must be ALIGNED -- pixel
(y, x) must be the same patch of ground in both. On a two-camera rig that
means registration is already solved. Feed it an unregistered pair and it
will compute one plant's outline over another patch's NDVI and report the
result with complete confidence. This is checked as far as shape allows,
but identical shapes do not prove alignment.

--------------------------------------------------------------------
    uv run per_plant_ndvi.py --demo
        Synthetic scene, no camera, no model, no dataset. Verifies the
        statistics and the health logic end to end.

    uv run per_plant_ndvi.py --weights runs/strawberry-seg/weights/best.pt \\
                             --image frame.png --ndvi ndvi.npy

--------------------------------------------------------------------
Three decisions here are borrowed from the companion NDVI project, because it
established them by measurement rather than argument:

1. Statistics come from the ERODED interior of each mask. Boundary pixels
   are leaf/background mixtures that land mid-scale; counting them puts a
   permanent ring of false stress around every plant.
2. The plant's status comes from its WORST TENTH (p10), not its mean. A
   mean lets a healthy majority hide a dying edge, and the worst-decile rule
   is monotonic -- more damage can never yield a better status.
3. Fruit and flowers are EXCLUDED from health statistics but still
   reported. Neither photosynthesises, so both read as stressed foliage on a
   thriving plant; leaving them in produced a false alarm that peaked exactly
   at harvest.
--------------------------------------------------------------------
"""

import argparse
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

# NDVI band edges, lowest bound first. Starting points from the companion
# NDVI project -- reasoned, NOT field-validated. These are the first numbers to
# calibrate against real flagged frames.
HEALTH_CLASSES = [
    (0.72, "Vigorous"),
    (0.58, "Healthy"),
    (0.46, "Needs attention"),
    (0.34, "Stressed"),
    (0.22, "Severe stress"),
    (0.10, "Critical"),
    (-1.01, "Dead / necrotic"),
]

# Detected classes whose names contain any of these are structures that do
# not photosynthesise. They are reported, but kept out of health statistics.
NON_FOLIAGE_KEYWORDS = ("fruit", "berry", "strawberry", "flower", "blossom", "bloom")


def classify_ndvi(value: float) -> str:
    for lower, label in HEALTH_CLASSES:
        if value >= lower:
            return label
    return HEALTH_CLASSES[-1][1]


def is_non_foliage(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(k in lowered for k in NON_FOLIAGE_KEYWORDS)


# Pixels to strip from every mask boundary before measuring. This MUST exceed
# the width of the leaf/background mixture band at your image scale, or the
# worst-decile statistic ends up measuring the band instead of the plant.
# A 55px-radius plant with a 3px mixture ring carries ~10% of its area in that
# ring -- landing exactly on p10 and reporting a vigorous plant as severely
# stressed. Raise it for blurrier optics or heavier downscaling; check the
# effect with --demo, which prints how much area erosion removes.
DEFAULT_ERODE_PX = 3


def erode_mask(mask: np.ndarray, iterations: int = DEFAULT_ERODE_PX) -> np.ndarray:
    """Shrink a mask away from its boundary. Falls back to the original when
    erosion would erase a small instance entirely -- a small plant should
    still be measured, just with the caveat that its edges are included."""
    if mask.sum() == 0:
        return mask
    try:
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
        if eroded.sum() >= 25:
            return eroded.astype(bool)
    except Exception:
        pass
    return mask


def per_plant_stats(masks, class_ids, names, ndvi, valid=None,
                    min_pixels: int = 40,
                    erode_px: int = DEFAULT_ERODE_PX) -> list[dict]:
    """One record per detected instance.

    masks       (N, H, W) boolean array of instance masks
    class_ids   length-N ints indexing `names`
    names       {id: class_name}
    ndvi        (H, W) float array, -1..1, ALIGNED with the masks
    valid       optional (H, W) boolean; False where NDVI is unmeasurable
    """
    ndvi = np.asarray(ndvi, dtype=np.float32)
    if valid is None:
        valid = np.ones_like(ndvi, dtype=bool)

    records = []
    for i, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != ndvi.shape:
            raise ValueError(
                f"mask {i} is {mask.shape} but the NDVI array is {ndvi.shape}. "
                "They must describe the same pixels.")
        cid = int(class_ids[i])
        cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)

        interior = erode_mask(mask, erode_px) & valid
        n = int(interior.sum())
        rec = {
            "id": i,
            "class": cname,
            "counts_toward_health": not is_non_foliage(cname),
            "pixels": int(mask.sum()),
            "measured_pixels": n,
            "edge_pixels_dropped": int(mask.sum()) - n,
        }
        if n < min_pixels:
            # Too little measurable area to say anything honest about.
            rec.update({"mean_ndvi": None, "p10_ndvi": None, "median_ndvi": None,
                        "status": "unmeasured"})
            records.append(rec)
            continue

        vals = ndvi[interior]
        p10 = float(np.percentile(vals, 10))
        rec.update({
            "mean_ndvi": round(float(vals.mean()), 4),
            "median_ndvi": round(float(np.median(vals)), 4),
            "p10_ndvi": round(p10, 4),
            "p90_ndvi": round(float(np.percentile(vals, 90)), 4),
            # Status from the worst tenth, so a dying edge cannot be averaged away.
            "status": classify_ndvi(p10) if rec["counts_toward_health"] else "n/a",
        })
        records.append(rec)
    return records


def frame_summary(records: list[dict]) -> dict:
    """Roll the per-plant records into one frame-level view."""
    foliage = [r for r in records
               if r["counts_toward_health"] and r["status"] != "unmeasured"]
    other = [r for r in records if not r["counts_toward_health"]]

    if not foliage:
        return {"plants": 0, "worst_status": None, "mean_of_plant_medians": None,
                "non_foliage_detections": len(other), "unhealthy_plants": []}

    order = [label for _, label in HEALTH_CLASSES]
    rank = {label: i for i, label in enumerate(order)}   # 0 = best
    worst = max(foliage, key=lambda r: rank.get(r["status"], 0))
    unhealthy = [r for r in foliage if rank.get(r["status"], 0) >= rank["Needs attention"]]

    return {
        "plants": len(foliage),
        "worst_status": worst["status"],
        "worst_plant_id": worst["id"],
        "mean_of_plant_medians": round(
            float(np.mean([r["median_ndvi"] for r in foliage])), 4),
        "non_foliage_detections": len(other),
        "unhealthy_plants": [
            {"id": r["id"], "status": r["status"], "p10_ndvi": r["p10_ndvi"]}
            for r in sorted(unhealthy, key=lambda r: r["p10_ndvi"])
        ],
    }


def summarise_for_agent(records: list[dict], summary: dict) -> str:
    """One short paragraph of plain text -- the form the LLM agent consumes.
    Deliberately compact: the model only needs what it must decide on."""
    if summary["plants"] == 0:
        return "No measurable plants in view."
    parts = [f"{summary['plants']} plants measured; "
             f"mean plant NDVI {summary['mean_of_plant_medians']:+.2f}; "
             f"worst plant is {summary['worst_status'].lower()}"]
    if summary["unhealthy_plants"]:
        worst_few = ", ".join(
            f"plant {p['id']} {p['status'].lower()} (NDVI {p['p10_ndvi']:+.2f})"
            for p in summary["unhealthy_plants"][:4])
        parts.append(f"{len(summary['unhealthy_plants'])} needing attention: {worst_few}")
    else:
        parts.append("no plants below the attention threshold")
    if summary["non_foliage_detections"]:
        parts.append(f"{summary['non_foliage_detections']} fruit/flower detections "
                     f"(excluded from health)")
    return ". ".join(parts) + "."


# ---------------------------------------------------------------- demo ----

def build_demo(h=360, w=640):
    """A synthetic bed: four plants of decreasing health plus two berries.
    Lets the statistics and health logic be verified with no camera, no
    trained model and no dataset."""
    rng = np.random.default_rng(0)
    ndvi = np.full((h, w), 0.02, np.float32)       # bare mulch
    masks, class_ids = [], []
    names = {0: "leaf", 1: "strawberry_fruit"}

    truth = [("Vigorous", 0.80), ("Healthy", 0.64),
             ("Stressed", 0.40), ("Critical", 0.16)]
    for i, (_, level) in enumerate(truth):
        cy, cx, r = 180, 90 + i * 150, 55
        yy, xx = np.ogrid[:h, :w]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        ndvi[m] = level + rng.normal(0, 0.02, m.sum()).astype(np.float32)
        # A ring of leaf/mulch mixture around every real plant -- the exact
        # artefact the erosion step exists to remove.
        edge = ((yy - cy) ** 2 + (xx - cx) ** 2 <= (r + 3) ** 2) & ~m
        ndvi[edge] = 0.25
        masks.append(m | edge)
        class_ids.append(0)

    for i, cx in enumerate((120, 300)):            # berries: low NDVI, healthy plant
        yy, xx = np.ogrid[:h, :w]
        m = (yy - 175) ** 2 + (xx - cx) ** 2 <= 16 ** 2
        ndvi[m] = 0.12
        masks.append(m)
        class_ids.append(1)

    return np.array(masks), class_ids, names, ndvi, [t[0] for t in truth]


def run_demo(erode_px: int = DEFAULT_ERODE_PX) -> int:
    masks, class_ids, names, ndvi, truth = build_demo()
    records = per_plant_stats(masks, class_ids, names, ndvi, erode_px=erode_px)
    summary = frame_summary(records)

    print("Synthetic bed: 4 plants of known health + 2 berries\n")
    print(f"{'id':>3} {'class':<18}{'px':>7}{'edge-':>7}{'median':>9}{'p10':>8}  "
          f"{'status':<16}{'expected'}")
    ok = True
    ti = 0
    for r in records:
        exp = ""
        if r["counts_toward_health"]:
            exp = truth[ti]; ti += 1
            if r["status"] != exp:
                ok = False
        med = f"{r['median_ndvi']:+.3f}" if r["median_ndvi"] is not None else "  --  "
        p10 = f"{r['p10_ndvi']:+.3f}" if r["p10_ndvi"] is not None else "  --  "
        print(f"{r['id']:>3} {r['class']:<18}{r['measured_pixels']:>7}"
              f"{r['edge_pixels_dropped']:>7}{med:>9}{p10:>8}  "
              f"{r['status']:<16}{exp}")

    print(f"\nrecovered every planted health class: {'YES' if ok else 'NO'}")
    print(f"berries kept out of health stats: "
          f"{'YES' if summary['non_foliage_detections'] == 2 else 'NO'}")
    print(f"frame worst status: {summary['worst_status']} "
          f"(plant {summary['worst_plant_id']})")
    print(f"\nwhat the AI agent would receive:\n  {summarise_for_agent(records, summary)}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--erode", type=int, default=DEFAULT_ERODE_PX,
                    help="Pixels stripped from each mask boundary before measuring.")
    ap.add_argument("--weights", type=str)
    ap.add_argument("--image", type=str, help="RGB frame for the segmentation model")
    ap.add_argument("--ndvi", type=str, help=".npy NDVI array, aligned with --image")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--json-out", type=str)
    args = ap.parse_args()

    if args.demo:
        return run_demo(args.erode)

    if not (args.weights and args.image and args.ndvi):
        ap.print_help()
        print("\n(--weights, --image and --ndvi are all required, or use --demo)")
        return 1

    import cv2
    from ultralytics import YOLO

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: could not read {args.image}")
        return 1
    ndvi = np.load(args.ndvi)
    if ndvi.shape != img.shape[:2]:
        print(f"ERROR: NDVI is {ndvi.shape} but the image is {img.shape[:2]}.")
        print("They must be registered to each other -- same pixel, same ground.")
        return 1

    model = YOLO(args.weights)
    result = model.predict(img, conf=args.conf, verbose=False)[0]
    if result.masks is None:
        print("No instances detected.")
        return 0

    masks = result.masks.data.cpu().numpy() > 0.5
    if masks.shape[1:] != ndvi.shape:      # YOLO may return masks at model scale
        masks = np.array([cv2.resize(m.astype(np.uint8), (ndvi.shape[1], ndvi.shape[0]),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                          for m in masks])
    class_ids = [int(c) for c in result.boxes.cls.cpu().numpy()]

    records = per_plant_stats(masks, class_ids, result.names, ndvi,
                              erode_px=args.erode)
    summary = frame_summary(records)
    for r in records:
        print(r)
    print("\n" + summarise_for_agent(records, summary))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"plants": records, "summary": summary}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

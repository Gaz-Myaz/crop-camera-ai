# Datasets and libraries

Reference for training the vision side of this project. Organised by **the problem you're solving**, because "best strawberry dataset" has no answer until you say what you want it to do.

Licences are flagged throughout. This is a commercial farm, and several of the best-known agricultural datasets are non-commercial — worth knowing before you build on one.

---

## Start here: which problem?

| You want to… | Use | Sensor |
|---|---|---|
| Find diseased tissue on leaves/fruit | Kaggle Strawberry Disease | RGB |
| Outline individual strawberry plants | StrawDI_Db1 | RGB |
| Count fruit / estimate yield & ripeness | Roboflow strawberries, Strawberry Object Detection | RGB |
| **Tell mud from dead leaf** | **WE3DS** (see below) | **RGB-D** |
| Exercise the NDVI path offline | WeedsGalore | Multispectral (NIR) |
| Detect weeds between rows | CropAndWeed, WeedsGalore | RGB / multispectral |

---

## The mud problem

Worth its own section, because it's the hardest open issue in the companion two-camera NDVI project, and it has a specific data answer.

**The problem.** Rain and irrigation throw soil out of the planting hole onto the plastic mulch, right around each crown — exactly where the "is this attached to a plant?" test says *yes, this is part of the plant*. Mud-splashed mulch reads hue 31; dead strawberry leaf reads hue 30. Colour cannot separate them. Measured impact: a heavy mud apron reports **17–35% of a healthy bed as dead crop**.

**What was already ruled out.** A shape discriminator (lesion = compact blob, apron = thin shell) was tried and separates the two *backwards* — real necrosis put 52–57% of its area just outside the canopy, mud aprons only 22–47%, because a thick apron extends well past that band. Don't re-derive this.

**The remaining signal is height.** Mud lies flat on the mulch plane; necrotic leaf sits 10–25cm up in the canopy. With registration solved at bed distance, the canopy carries residual disparity between two cameras and the mulch doesn't. That's exactly what the stereo pipeline in `decxin-sm-2930v1/` measures.

**The dataset that matches this.** [**WE3DS**](https://zenodo.org/records/7457983) — 2,568 RGB-D stereo image pairs, 17 plant species, and crucially **an explicit soil class**, CC BY 4.0. It is the only public set I found combining depth, ground-level agricultural imagery, and labelled soil. Wrong crop, but the right *structure*: it lets you develop and test "separate ground-plane material from canopy-height material using depth" before your own rig exists.

Also relevant if you go further into 3D: [Crops3D](https://www.nature.com/articles/s41597-024-04290-0) (1,180 point clouds, 8 crop types, CC BY 4.0) and [Pheno4D](https://www.ipb.uni-bonn.de/data/pheno4d/) (temporally consistent point-cloud segmentation).

---

## Strawberry datasets

**[Kaggle Strawberry Disease Detection](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)** — 2,500 images, 419×419, **seven disease classes**, polygon segmentation masks, shot in South Korean greenhouses under natural light. Published YOLOv8 results reach ~92–93% mAP50 ([BrunoKreiner/strawberry_diseases](https://github.com/BrunoKreiner/strawberry_diseases), about 10 points above the Mask R-CNN baseline). **The best starting point for disease work.** Check Kaggle's terms; the GitHub repo states no licence.

**[StrawDI_Db1](https://strawdi.github.io/)** — 3,100 images at 1008×756, 17,938 annotations, instance segmentation, gathered across 20 plantations over 150 hectares in Huelva, Spain. **Real open-field conditions at scale**, which matters more than raw count. Best source for *plant/fruit instance outlines* — the masks `per_plant_ndvi.py` needs. Licence not clearly stated; check before commercial use.

**[Roboflow dyafars/strawberries](https://universe.roboflow.com/dyafars/strawberries-yw5el)** — 1,143 images, three classes (immature / ripe / rotten), **bounding boxes only**, CC BY 4.0, outdoors. Good for ripeness and yield counting. No disease classes, no masks, no published metrics.

**Strawberry Dataset for Object Detection** (Zenodo) — 813 images, 4,568 objects, three classes including *peduncle*. Small, but peduncle labels matter if robotic picking is ever on the roadmap.

---

## Multispectral — the only ones that exercise NDVI

Almost every agricultural dataset is plain RGB, which trains the *colour* camera path only. These carry real near-infrared:

**[WeedsGalore](https://github.com/GFZ/weedsgalore)** — UAV, five bands (R/G/B/**NIR**/RedEdge), 156 tiles, maize plus four weed species, **CC BY 4.0**. Small, but it's genuine NIR: you can compute real NDVI and test the whole measurement path offline. The most useful multispectral set here for your purposes.

**[Multispectral UAV dataset of wheat, soybean and barley](https://www.mdpi.com/2306-5729/8/5/88)** — wrong crop, but a continental-climate open field rather than a greenhouse, which may sit closer to your growing conditions than the indoor sets do.

**[Fields of the World](https://fieldsofthe.world/)** — 70,484 Sentinel-2 chips for field-boundary segmentation. Satellite scale, not rover scale — useful for block-level mapping later, not for plant health.

**GrowliFlower** — UAV RGB + multispectral, cauliflower, with growth-trait phenotyping.

---

## Weeds, soil and background

**[WE3DS](https://zenodo.org/records/7457983)** — see the mud section. RGB-D, soil class, CC BY 4.0.

**[PhenoBench](https://www.phenobench.org/)** — 2,872 UAV images of sugar beet with semantic, instance *and* panoptic labels down to leaf level. CC BY-SA 4.0 (share-alike — derivatives must carry the same licence).

**CropAndWeed** — 8,034 images, 74 species, with stem keypoints. ⚠️ **Custom non-commercial licence.**

**VegAnn** — 3,775 images across 26+ species, simple vegetation-vs-background segmentation. Useful for a robust canopy mask independent of NDVI.

---

## General plant disease

**PlantWild** — 30,030 in-the-wild images, 146 disease/healthy classes. ⚠️ **CC BY-NC-ND 4.0** — non-commercial *and* no derivatives, so a model trained on it is legally awkward for a commercial farm. Fine for benchmarking, not for production.

**PlantVillage** — the classic disease benchmark, but almost entirely single leaves on plain backgrounds. Models trained on it collapse on field imagery. Useful as a sanity check, misleading as a foundation.

---

## Licence summary

| Dataset | Licence | Commercial use |
|---|---|---|
| WeedsGalore, WE3DS, Crops3D, Roboflow strawberries | CC BY 4.0 | Yes, with attribution |
| PhenoBench | CC BY-SA 4.0 | Yes, but derivatives inherit share-alike |
| CropAndWeed | Custom non-commercial | **No** |
| PlantWild | CC BY-NC-ND 4.0 | **No** |
| Kaggle Strawberry Disease, StrawDI_Db1 | Not clearly stated | **Check before relying on it** |

For a commercial deployment, "licence not stated" is not the same as "free to use". If a dataset ends up load-bearing, get that in writing from the authors.

---

## Libraries

### Already installed

**[Ultralytics](https://docs.ultralytics.com/)** — YOLO training, validation, export. `train.py` uses it.

**[SAM 2](https://docs.ultralytics.com/models/sam-2)** — Segment Anything 2, **available through Ultralytics you already have**. This is the highest-leverage tool on this page: point at an object, get a pixel-accurate mask. Turns "draw 500 polygons by hand" into "click 500 times and correct the misses". When you start labelling your own farm images, start here.

### Worth adding

**[FiftyOne](https://docs.voxel51.com/)** (`pip install fiftyone`) — visually browse and query a dataset before training on it. Find mislabelled images, spot class imbalance, compare model predictions against ground truth side by side. Most people discover their dataset is the problem *after* three failed training runs; this is how you find out first. Has SAM 2 built in for annotation.

**[Albumentations](https://albumentations.ai/)** — augmentation that transforms masks along with images. Important with small datasets, which yours will be.

**[supervision](https://supervision.roboflow.com/)** (Roboflow) — utilities for annotating frames, filtering detections, counting objects across a video. Saves writing the same bookkeeping code again.

### If you go deeper into multispectral

**[TorchGeo](https://pytorch.org/blog/geospatial-deep-learning-with-torchgeo/)** (Microsoft) — PyTorch datasets, samplers and pretrained models for geospatial and multispectral data. Handles arbitrary band counts, which ordinary vision libraries assume is three. The right tool if NDVI work grows past `numpy` arrays.

**rasterio / spectral** — reading GeoTIFF and multi-band imagery.

### Annotation

**[Label Studio](https://labelstud.io/)** or **[CVAT](https://www.cvat.ai/)** — self-hosted, your data stays yours. **[Roboflow](https://roboflow.com/)** is faster to start with and exports straight to YOLO format, but images go to their cloud — worth a thought before uploading a commercial farm's imagery.

### Alternative frameworks

**MMDetection / MMSegmentation** and **Detectron2** are more flexible than Ultralytics and considerably more work. Only worth it if you outgrow YOLO — which, for this, you probably won't.

---

## Suggested sequence

**Now, offline.** Download the Kaggle disease dataset, run `prepare_dataset.py --inspect`, then train. You get a working model and, more importantly, a validated pipeline. In parallel, grab WE3DS and start on the depth-versus-mud question — that's the problem with no workaround.

**First farm visit.** Collect a few hundred images with the real rover cameras, in daylight, covering healthy plants, stressed plants, mud-splashed beds, and varied light. That single collection validates the NDVI channel mapping against real chlorophyll, calibrates the NDVI project's class thresholds, *and* becomes your fine-tuning set.

**After.** Label those images with SAM 2 assistance, fine-tune the public-data model on them, and compare against the rule-based classifier on the same frames rather than replacing it.

One thing worth internalising: every accuracy figure quoted on this page describes someone else's crop, camera and light. Treat them as evidence the *approach* works, never as a prediction for your field.

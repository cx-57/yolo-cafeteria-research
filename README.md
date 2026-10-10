# YOLO Cafeteria Research

Detect cafeteria objects with YOLO11, then apply **separate school-specific disposal rules** (recycle / compost / trash). Object detection does not itself determine disposal.

## Files to understand

| File | Purpose |
|---|---|
| `config.yaml` | Final class IDs and image targets |
| `src/prepare.py` | Convert downloaded Roboflow datasets into review candidates |
| `src/lvis.py` | Download real LVIS paper-plate photos and matching YOLO labels |
| `src/audit.py` | Check reviewed image/label pairs, duplicate risks and provenance |
| `src/train.py` | Train Ultralytics YOLO11 |

`requirements.txt` has just **Ultralytics, PyYAML, and Pillow**. Everything else uses standard Python. The two files in `tests/` are small automated regression tests. Older dated audit reports are preserved as research evidence, not scripts you need to run.

## The four steps

**1. Get candidate images.** Download a Roboflow YOLO export into `data/raw/<source>/`, then run:

```bash
python3 src/prepare.py data/raw/school_lunch school_lunch
```

For LVIS disposable paper plates, **one command** downloads the official annotations and the COCO **JPEG photos** referenced by them, validates images, and converts plate boxes (class `1`):

```bash
python3 src/lvis.py
```

If you already have the verified \`LVIS_Paper_Plate_Candidate.zip\` archive, you can directly integrate its real images into the *local* project (rather than download them again):

\`\`\`bash
python3 src/lvis.py --import-archive ~/Downloads/LVIS_Paper_Plate_Candidate.zip
\`\`\`

This imports **344 provisional plate photos / 834 boxes** to \`data/processed/lvis_plate/\` and holds **44 flagged photos / 293 boxes** in \`data/candidates/lvis_plate/\`. Images in \`data/processed\` are **not yet fully visually reviewed or complete for the other ten classes**; inspect/reannotate before final 11-class training. Large JPEG files remain local and are not tracked in GitHub. A separate plate-addition ZIP is available from the ChatGPT deliverable.

The LVIS files are placed at `data/candidates/lvis_plate/images/`, `labels/`, and `review.csv`. The script can reuse archives in `data/raw/lvis/` and existing COCO photos with `--coco-dir /path/to/coco`. Run it again to retry missing downloads. The last verified LVIS candidate count was **388 images and 1,127 plate boxes** (328 train, 60 validation). **44 images have potentially incomplete plate annotations**; see `review.csv` for flags. These candidate images are NOT automatically accepted for model training.

**2. Review labels and photos.** Only keep NHS-like objects. Inspect *every* candidate for missing plates and other visible target classes. Correct boxes manually before moving an image and its label into the matching `data/processed/<source>/images/` and `labels/` folders. Maintain provenance in `data/manifests/`. Do not mix source train/validation images or nearly identical backgrounds across different evaluation splits. Keep the natural NHS test set completely untouched.

**3. Check reviewed data.**

```bash
python3 -m unittest discover -s tests -v
python3 src/audit.py --manifests data/manifests --output data/audits/current
```

The audit checks data integrity; it **cannot** prove that every visible target object was annotated. It never silently deletes images. Raw downloads, candidates, and trained weights are excluded from GitHub in `.gitignore`.

**4. Train only after review and a proper train/val split.**

```bash
python3 src/train.py path/to/data.yaml
```

The final class IDs are: `0 food`, `1 plate`, `2 paper_boat`, `3 utensil`, `4 compartment_container`, `5 hinged_container`, `6 napkin`, `7 drink_carton`, `8 condiment_packet`, `9 wrapper`, `10 bottle`. The model is not ready for a credible 11-class evaluation until missing classes and incomplete annotations are addressed. Use new, separate NHS cafeteria scenes for final testing.

## Current verified status (October 9–10, 2026)

The **accepted** collection has **1,616 images / 3,824 annotations**; this includes the 44 reviewed condiment photos. The verified LVIS batch is a **separate 388-image candidate set** and does not change the accepted total. Missing accepted classes: **plate, paper_boat, compartment_container, napkin**. See `data/audits/current/summary.json` and `data/audits/lvis_2026-10-10/README.md` for details.

Data origins, licenses, and review scope are listed in `data/datasets.csv`. LVIS annotations are CC BY 4.0; COCO image licenses/rights belong to their respective source images. Source photos are only included in the user's local candidate folder after running the LVIS command.

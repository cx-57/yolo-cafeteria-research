# Additional evaluation images: missing-class screening (November 1, 2026)

## Current state and rule

The previously assembled 52-image external ZIP is a **provisional food/plate review set**, not an independently verified test set. Its source/capture provenance is unresolved and its photos have no independent human disposal-bin labels. Do not overwrite or relabel it. The frozen active checkpoints and sorting policy must remain unchanged.

The next intake should **prioritize absent classes**, not add more food/plate images. The objective is a *visually reviewed candidate pool*, not a manufactured total of exactly 100.

## Investigate these independently sourced collections

| Source | Published source details | Screening hints | Limits |
|---|---|---|---|
| [CoxPWD v3, Mendeley Data](https://data.mendeley.com/datasets/bdzg4tjy63/3), DOI 10.17632/bdzg4tjy63.3 | 4,195 beach-plastic images, COCO annotations, CC BY 4.0, published February 2026 | `bottle` -> bottle candidate; `spoon` -> utensil candidate; `packet` -> wrapper candidate | Beach litter is not cafeteria waste. `packet` need not match the project's wrapper definition; `bottle` may include out-of-scope vessels. Visually inspect. |
| [PackWISE v2, Fraunhofer](https://fordatis.fraunhofer.de/handle/fordatis/463.2), DOI 10.24406/fordatis/420.2 | 586 conveyor-belt postconsumer packaging images, 23 COCO object categories | Screen packaging for drink cartons and hinged food containers | Conveyors differ from cafeterias. A general box or container is **not** automatically a drink carton or hinged container. Check precise license and original categories before selection. |
| [Recyclable Waste Image Dataset](https://data.mendeley.com/datasets/h5pxbsdz4m/1), DOI 10.17632/h5pxbsdz4m.1 | 1,000 author-collected plastic/paper/cardboard/metal/glass photos, CC BY 4.0, published June 2026 | Screen for suitable bottles, cartons and negative examples | Material labels are **not** our seven object classes or local disposal-bin truth. No bounding boxes should be assumed. |
| [DWSD](https://data.mendeley.com/datasets/gr99ny6b8p/1), DOI 10.17632/gr99ny6b8p.1 | 784 waste photos, 14 object categories, CC BY 4.0 | Screen plastic bottles and food-container candidates | Older source could overlap historical datasets; do not assume its `plastic container` class means hinged container. |

Avoid TACO-derived mirrors without proving they were absent from historical training and model selection; the repository previously used trash datasets that might overlap.

## Stage images from a COCO dataset (never run inference at this step)

Download the public original dataset on the Mac, extract it *outside* the Git repository, and locate one COCO `_annotations.coco.json` and the corresponding image directory. From the repository root:

```sh
./scripts/run_specialists.sh --help
python3 scripts/stage_holdout_candidates.py \
  --coco "/path/to/extracted/CoxPWD/train/_annotations.coco.json" \
  --images-root "/path/to/extracted/CoxPWD/train" \
  --source "CoxPWD v3" \
  --source-url "https://data.mendeley.com/datasets/bdzg4tjy63/3" \
  --license "CC BY 4.0; confirm individual terms" \
  --total 48 --per-class 14 \
  --out "../yolo-cafeteria-research-assets/independent-evaluation-20261101/coxpwd-review-01"
```

These paths are illustrative; use the actual downloaded file structure. The script **does not download data**, perform human visual review, run YOLO, label bins, or verify independence. It only copies deterministic COCO-category-screened candidates, produces a `review.csv` and, if Pillow is installed, a contact sheet. The output folder must not already exist. Only exact COCO class-name aliases become *hints*. For missing class aliases, report the dataset's actual COCO categories rather than silently inventing mappings.

For collections without COCO labels, inspect source images manually instead; do not fake a COCO file. Keep every review batch and large photo archive external.

## Individual-review and provenance gate

1. Open **every** staged image at native resolution. Record KEEP / REJECT / UNCERTAIN, visible project classes, reason, background suitability, occlusion, ambiguity and whether it is a single-item product shot versus actual discarded waste. A model-generated contact sheet is *not* the individual review.
2. Preserve original image hashes, source version/DOI, author, license, original filename and where available capture/day/group IDs.
3. Compare exact SHA-256, decoded-pixel hashes, perceptual-near matches and recorded source/capture groups against the **full Mac historical image/data/experiment inventory**, including prior test and selected model experiments. Near-hash hits need visual adjudication; no hash hit alone proves independence.
4. Mark source/capture or lifetime model history gaps **UNRESOLVED**, not independent. Different source URLs and recent publication dates are insufficient.
5. Treat `packet` -> wrapper, `spoon` -> utensil, and bottle/packaging mappings as *review candidates* only. Do not claim milk/juice drink cartons or hinged containers exist until seen.
6. Do **not** infer actual compost/recycling/landfill labels from object category, package appearance or a model's predictions. Ground truth needs documented school acceptance rules and independent human adjudication.
7. Freeze verified membership, annotation conventions and human labels before the first model inference. Do not revise selection based on outcomes. Preserve an independent final holdout on a separate capture day where feasible.

**Required reporting:** number visually inspected, KEEP/REJECT/UNCERTAIN, per-class visible coverage, exact/near overlaps, independently certified (currently **zero**), human bin labels (currently **zero**), and a path to external contact sheets and ZIP. Do not promote an unresolved set to independent evaluation.

## Network-enabled acquisition (November 1, 2026)

A dedicated [GitHub Actions acquisition workflow](../.github/workflows/acquire-coxpwd-candidates.yml) is configured for **CoxPWD v3**. It downloads the public ZIP to the transient runner, selects at most 48 COCO-source-hint images (bottle, spoon, packet), creates a review CSV/contact sheet, and uploads a 7-day artifact named `coxpwd-provisional-candidates-not-independent`. The source photos are **not committed into Git**. The workflow triggers when its own file is pushed, and can be triggered manually from the Actions tab. **Do not assume a successful run merely because the workflow file exists**; inspect its run logs and artifact count.

It cannot certify original-photo independence, human-reviewed class identities, cafeteria relevance, or disposal-bin ground truth. Every image remains `PENDING` review and `independent=NO`. Cartons and hinged containers are not sourced by CoxPWD category names and still require a distinct intake. The workflow does not run YOLO, label bins, retrain, or modify frozen checkpoints.

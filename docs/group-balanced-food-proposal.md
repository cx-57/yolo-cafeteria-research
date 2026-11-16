# Group-balanced food candidate — user chose audit only

The user explicitly declined further training and cleanup and chose audit-only work. The prepared command below is historical proposal material, not permission to run. See [current audit-only results](small-food-audit-only.md). No new training or cleanup has run. The current live settings, frozen original weights, and dataset annotations/splits remain unchanged.

## New evidence

The existing 2372-image training pool has 2501 food boxes. Only 45 are below 32² pixels at 640, in 24 images and 10 recorded capture groups. Fifteen images from one group contain clipped remnants of the same larger meal rather than 15 independent crumb scenes. The remaining nine images mostly show food in distant meal scenes. These labels were inspected, not changed.

Across all food-positive training images, the largest recorded group accounts for 189 of 1000 images. This is concentration of a recorded group, not proof that every image is an exact duplicate. Group-balanced sampling is a defensible short experiment; it is not a guarantee of improved small-food accuracy.

[Audit and source records](../runs/sorter/small-food-generalization-20261115/existing-small-food-audit.json) · [Visual review page1](../runs/sorter/small-food-generalization-20261115/existing-small-food-page-1.jpg) · [page2](../runs/sorter/small-food-generalization-20261115/existing-small-food-page-2.jpg)

## Fixed candidate

- One epoch on 256 existing training images from 256 recorded groups: 128 food-positive and 128 food-negative. No group repeats within this subset.
- Stable SHA256 selection independent of model errors/scores; prefer original images over augmentation variants within a group. Food sources are sampled round-robin, with other-class rehearsal support.
- All seven classes retain support: food 128 images, plate 87, utensil 46, hinged_container 3, drink_carton 84, wrapper 36, bottle 37. The rare container class still has only three recorded training groups.
- Same original seven-class YOLO11n initialization; CPU, 640 pixels, batch 1, seed 42, AdamW 0.00005, first 10 layers gradient-frozen, existing mild scale augmentation 0.15, no mosaic/mixup/cutmix/copy-paste.
- Use only the last checkpoint after that one epoch. No resume, retries, extra epochs or best-checkpoint selection.
- Original 186-image validation membership and 248-image test membership remain intact. Test images are copied for provenance but are not scored for selection. New subset manifests leave original splits and labels untouched.
- No current user-camera frame, outside 34-photo diagnostic image or newly downloaded source-review image is included in training.

This is an exploratory data-rebalancing trial. The validation cohort has no genuinely tiny food boxes, so a passing development result cannot certify real crumb accuracy.

[Selection manifest](../runs/sorter/small-food-generalization-20261115/group-balanced-prepared/selection-manifest.json) · [Native preflight](../runs/sorter/small-food-generalization-20261115/group-balanced-prepared/native-preflight.json) · [Profile](../runs/sorter/small-food-generalization-20261115/group-balanced-prepared/profile.yaml)

## Evaluation and preservation gates

Keep the original unified model fixed. The new candidate would be evaluated only as a food-only fallback: confident original food is preserved, other-six output stays on the original model, and the candidate supplies food only when primary food is missing at 60%.

Compare actual candidate routing against the current fallback on the fixed 186-image development cohort with unchanged 640 input, 0.001 AP confidence, 0.7 NMS IoU, 300 max detections and 0.60/0.50 operating confidence/matching IoU. Require food precision/recall and food AP50/AP50–95 not to decline from the current fallback, and require all other-six predictions/AP to remain identical. The current fallback has 295 correct food boxes, 6 false food boxes and 48 misses.

Review all 34 exposed outside photos, keeping confident original food outputs. Require zero newly visible food boxes on the five reviewed clean plate/paper negatives, and inspect any new boxes manually; class presence is not localization correctness. User bread/platform and room/spoon observations remain development diagnostics only and never become training labels. Do not evaluate the original held-out test to choose the candidate. No promotion happens automatically, even if these checks pass.

## Disk-space condition and requested cleanup scope

The native runner requires 123,754,936 bytes for protected input snapshots. Require an additional 512 MiB free reserve before starting. Low free space currently blocks a safe launch.

A read-only hash verification confirmed the original 5612 image/label files and both generated experimental copies still match their recorded SHA256 values. The preferred cleanup, only if authorized and needed, is the **stopped trial’s generated duplicate input directory**:

`/Users/Charles1/Desktop/yolo-cafeteria-research-assets/artifacts/small-food-scale-candidate-20261115/input`

It holds 327,723,709 logical bytes (about 328 MB), including generated caches. Preserve the original dataset under `cumulative20-100views-20261031/train/input`, the completed control input copy, all trained weights, status/recipe manifests, logs, source-review photos and validation results. Re-verify the exact directory and original hashes before any approved cleanup. If cleanup leaves insufficient reserve, refuse training and report the shortfall; do not remove any additional files.

[Cleanup preflight](../runs/sorter/small-food-generalization-20261115/generated-copy-cleanup-preflight.json) · [Exact proposed run state](../runs/sorter/small-food-generalization-20261115/group-balanced-prepared/proposed-run.json)

## Prepared command — requires new training permission

```sh
python3 scripts/training.py train \
  --data runs/sorter/small-food-generalization-20261115/group-balanced-prepared/data.yaml \
  --profile runs/sorter/small-food-generalization-20261115/group-balanced-prepared/profile.yaml \
  --model ../yolo-cafeteria-research-assets/artifacts/cumulative20-100views-20261031/train/weights/best.pt \
  --output ../yolo-cafeteria-research-assets/artifacts/group-balanced-food256-20261115 \
  --device cpu --epochs 1
```

Estimated training time is about 1–2 minutes at previously measured CPU throughput, plus startup and validation; this is an estimate, not a runtime guarantee. If the camera is active before an approved run, pause it for training and restore its same detection-only full-view command afterward. If it is already closed, leave it closed. Keep hardware off.

## New source review

The [AlphaTrash publisher](https://github.com/Patipol-BKK/alphatrash-dataset) provides food-scrap classification photos captured inside a sorting compartment. A revision-pinned 18-image sample of its original training split was selected before inference, with one image per half-hour capture window where timestamps were available. Visual review found 9 of 12 organic scenes dominated by greens and a common collection-box background. Image-level categories are not complete seven-class object boxes. All 18 images are quarantined and excluded from this trial. Original publisher splits were not edited. MIT attribution/license text is retained locally.

The [Voxel51 food-waste dataset card](https://huggingface.co/datasets/Voxel51/food-waste-dataset) describes 375 institutional meal images and machine-generated YOLO-E ingredient masks. Only its pinned README and metadata were inspected. Those masks were not accepted as independent human ground truth; no images from that source entered training or evaluation. The [RealWaste publisher](https://archive.ics.uci.edu/dataset/908/realwaste) describes a landfill classification dataset, not complete seven-class object boxes; its full archive was not downloaded.

[Source review](../runs/sorter/small-food-generalization-20261115/new-source-review/visual-review.json) · [Pinned source manifest](../runs/sorter/small-food-generalization-20261115/new-source-review/source-manifest.json)

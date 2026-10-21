# Active scope changed: plate and wrapper only

The user removed bottle from the cafeteria experiment. Bottle training is cancelled. Use [the active plate/wrapper experiment](../plate-wrapper-focus/README.md), which separately validates at plate80.64% and wrapper86.35% AP50 with bottle predictions excluded. The checkpoint retains its original head internally and is used through the class-filtered inference configuration. The three-class results below are archived history.

---

# Current status: epoch10 evaluated; plate and wrapper above80%, bottle below target

Target remains **AP50 >=80% separately for plate, wrapper and plastic beverage bottle**. It has not been achieved. All performance comparisons below the historical marker are development results on the old splits, which contained confirmed related-capture leakage.

All 1,302 image files and the current manifest are now local. All 23 focused tests passed. The full165-test suite also passed with `TMPDIR=/private/tmp`; the default temporary directory exposes a pre-existing `/var` versus `/private/var` string-comparison failure in one unrelated subset test. No original pipeline changes were made. The separate `capture-grouped-clean` dataset built successfully and passed image decoding, JPEG checks, bounding-box validation, verified-negative checks, recorded original/byte/pixel/capture/scene split checks, and fingerprint verification. A separate preflight confirmed all prior image hashes and every label file byte are preserved, and all original 226 validation anchors remain in validation.

| Split | Images | Verified negatives | Plate objects | Wrapper objects | Bottle objects |
|---|---:|---:|---:|---:|---:|
| Train | 1056 | 84 | 650 | 394 | 356 |
| Validation | 246 | 3 | 237 | 94 | 67 |

Twenty images moved from training to validation: nine members of the visually reviewed counter/river capture groups plus eleven additional members linked through existing original-photo groups. These links are conservatively preserved; filename prefixes do not assign splits. No images or boxes were dropped. `source-label-reconciliation/capture_regroup_preflight.json` records exact moves, checks, fingerprint and validation protocol. No overlap remains in the recorded groups; unrecognized related captures remain a limitation of review, not something hashes can rule out.

The fresh initialization is the original 80-class COCO `model/weights/yolo11n.pt`, confirmed model scale `n`, SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`. MPS is available. Old fine-tuned checkpoint initialization is rejected for this new protocol. Continuation preserves the same dataset fingerprint and validation protocol.

The completed short startup run used `capture-clean80-smoke-recipe.json`: standard Ultralytics augmentation, batch4, imgsz640, MPS, fixed80-epoch schedule, wall budget1 minute checked only at an epoch boundary. It completed one epoch in116.86 seconds of trainer time; the optimizer-bearing resume checkpoint was verified to retain the80-epoch horizon. This is a startup check, not an80%-accuracy claim. Runtime log: `/tmp/focus-capture-clean80-start.log`.

## Latest separately evaluated checkpoint: epoch10

After the user-run replay epochs8–9, the agent completed one short local epoch10 continuation. Trainer time207.499 seconds (~3.5 minutes), followed by final validation and post-training dataset integrity checks. Saving and resume bookkeeping completed cleanly in this run. The latest epoch10 checkpoint, on all246 unchanged validation images at640, gives:

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 94.57% | 58.65% | 80.64% | 63.89% |
| wrapper | 74.59% | 82.98% | 86.35% | 59.55% |
| bottle | 61.11% | 46.92% | 46.88% | 32.04% |
| Overall | 76.76% | 62.85% | 71.29% | 51.83% |

**Plate and wrapper pass80% AP50 in this same epoch10 checkpoint. Bottle does not; the all-class target remains unmet.** These are AP50 scores, not a claim that every precision/recall value exceeds80%. Current results: `capture-bottle-replay-e10/evaluation-epoch10/metrics.json`; weights: `capture-bottle-replay-e10/train/weights/last.pt` (SHA256 `ed78743eeecffdfb8d6aa69aac7d528199500ecc4d637e0ac82113607dd0bd09`).

The retained best weakest-class checkpoint is still epoch9: plate73.29%,wrapper88.33%,bottle47.98%. That model has slightly higher bottle AP50 and lower plate AP50. Its `best_min_ap50.pt` copy is preserved in the epoch10 experiment. Do not combine epoch10 plate/wrapper scores with epoch9 bottle scores into a fictional model result.

Epoch9 showed +8.91 bottle points and−5.94 plate points relative to retained epoch6; epoch10 recovered plate above80% while bottle stayed~47%. These are observed model results. Replay is not isolated from extra training/exposure because the standard control did not complete.

Higher-resolution inference at960 on epoch9 weights reduced AP50 to plate65.96%,wrapper75.08%,bottle28.44%. The official comparison remains640. This rejects the tested inference-only change; higher-resolution training was not tested.

Saving records after the user run were stale: native epoch9 `last.pt`/`best.pt` were valid, but resume was at8 and selection at6. The old epoch8 resume was preserved and pointers synchronized to independently evaluated epoch9, including optimizer and80-epoch horizon. Exact stop/error cause remains unconfirmed; final terminal output was requested. Recovery evidence: `capture-bottle-replay-e8/epoch9_checkpoint_recovery.json`. The local epoch10 job saved normally. The script now rejects a latest-resume copy behind its results CSV, preventing an unnoticed rewind. The new regression and full168-test suite passed under `TMPDIR=/private/tmp`.

## Source-specific diagnosis and correction

| Bottle subset, epoch6 weights | Objects | AP50 |
|---|---:|---:|
| Training: close-up/video sources |178|96.58%|
| Training: field/context sources |178|39.34%|
| Validation: all bottle examples |67|39.06%|

The training rows are fit diagnostics, not validation/target results. The current epoch10 field/context training fit is57.45% AP50 on the same124 training images/178 bottle objects, versus46.88% on held-out validation bottles. Both improved from the epoch6 measurements, with a remaining~10.6-point gap. This is evidence of broader learning alongside a generalization gap; it is not a training-only80% claim. Field-fit evidence: `capture-bottle-replay-e10/training_field_fit_diagnostic.json`. Aggregate training bottle AP50 of67.47% hid the source mix: difficult field/context examples were weak even within training. Replaying113 reviewed trash-scene/external bottle images3 times per epoch retains every1056 unique training image and the same246 validation images. It produces1282 training views, weighted boxes plate650/wrapper414/bottle664. No original image, box or verified negative was removed, and no pending candidate was admitted. Replay preflight: `source-label-reconciliation/bottle_replay_preflight.json`.

The earlier epoch7 truncated resume copy was preserved and recovered from its intact source. Resume and selected-weight copies now use temporary files with atomic replacement. The interrupted-copy regression and full168-test suite passed under `TMPDIR=/private/tmp`. Only the isolated experiment script/tests changed; the original11-class pipeline remains untouched.

## Main remaining data problems

Historical legacy models (before capture regrouping) measured roughly95–99% training AP50, including~96% bottle, while old bottle development validation stayed~61–63%. Those are different models/splits; use the current source-specific diagnostic above for the fresh run. More short fine-tuning attempts did not solve this generalization gap. The old split's best bottle result was63.08%, and the latest context run selected71.18%/58.65%/61.21% on the historical126-image set. Those results are not independent-capture evidence after the leakage discovery.

The revised training pool has140/356 bottle boxes below4% image area; validation has51/67. Multiple-bottle scenes number23 training and10 validation. More independent small/cluttered bottle scenes need complete three-class annotation. Forty-six recovered source photos are staged candidates, not approved training data. The suspected missing bus bottle was checked against the native original and is malt vinegar: exclusion is correct. Keep the plastic beverage bottle scope, preserve one box per plate stack, and do not treat ambiguous scenes as negatives.

A final independent capture holdout is still needed for a generalization claim: repeatedly selecting checkpoints against this public validation set is development, even though the fresh model has not trained on its validation photos.

## Commands

Build already completed; do not rerun into the existing directory. Its exact command was:

```bash
.venv/bin/python scripts/focus_three_class.py regroup --frozen-data data/focus-three-class-improvement/bottle-context-reviewed --data data/focus-three-class-improvement/capture-grouped-clean --review-files model/focus-three-class-improvement/source-label-reconciliation/confirmed_capture_reviews.json
```

The startup run command (already launched; do not launch twice):

```bash
.venv/bin/python scripts/focus_three_class.py train --data data/focus-three-class-improvement/capture-grouped-clean --model model/focus-three-class-improvement/capture-clean80-start --recipe model/focus-three-class-improvement/capture-clean80-smoke-recipe.json
```

Separate evaluation of the selected startup checkpoint (completed):

```bash
.venv/bin/python scripts/focus_three_class.py evaluate --data data/focus-three-class-improvement/capture-grouped-clean --model model/focus-three-class-improvement/capture-clean80-start --eval-weights model/focus-three-class-improvement/capture-clean80-start/train/weights/best_min_ap50.pt --eval-name evaluation-capture-grouped
```

The next user-run experiment continues the verified epoch10 optimizer with the same field/context replay configuration and80-epoch horizon. It preserves the retained epoch9 checkpoint. The6-minute budget is checked after complete epochs; actual time can be~7–10 minutes and varies with machine load. Reaching80% is not guaranteed:

```bash
.venv/bin/python scripts/focus_three_class.py continue_train --data data/focus-three-class-improvement/capture-grouped-clean --continue-from model/focus-three-class-improvement/capture-bottle-replay-e10/train/weights/last_resume.pt --model model/focus-three-class-improvement/capture-bottle-replay-chunk2 --training-config data/focus-three-class-improvement/capture-grouped-clean/data-bottle-replay.yaml --epochs 80 --minutes 6
```

Evaluate that saved checkpoint separately:

```bash
.venv/bin/python scripts/focus_three_class.py evaluate --data data/focus-three-class-improvement/capture-grouped-clean --model model/focus-three-class-improvement/capture-bottle-replay-chunk2 --eval-weights model/focus-three-class-improvement/capture-bottle-replay-chunk2/train/weights/best_min_ap50.pt --eval-name evaluation-capture-grouped
```

Evaluation outputs include per-class precision, recall, AP50, AP50–95, overall metrics, checkpoint SHA, dataset fingerprint and protocol. There is no claim that the next chunk will reach80%.

---

## Historical notes below (superseded where they conflict with the current finding)


# Plate / wrapper / plastic beverage bottle experiment

Current target: **80% AP50 independently for each of the three classes**, measured by separate saved-checkpoint evaluation. This is detection AP50, not ordinary classification accuracy. The target has not been achieved. Precision, recall and AP50–95 are reported alongside AP50.

## Current status: accumulation test completed; next dataset prepared

Both one-epoch stability probes completed and all four separate saved-checkpoint evaluations passed dataset integrity checks. Gradient accumulation did not improve bottle AP50 in this short paired comparison. Neither probe replaces the preserved epoch3 checkpoint (`818161d62c00ffc580c6b5ce0c6261b58fbccd699b5227e3471f4dce3a5b5693`, fixed126 AP50 plate71.22%,wrapper64.86%,bottle63.08%). This does not establish that accumulation can never help over longer training.

### control, fixed126, separately evaluated latest weights

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 93.01% | 51.82% | 75.86% | 55.18% |
| wrapper | 85.76% | 52.94% | 62.26% | 50.25% |
| bottle | 85.81% | 57.78% | 58.59% | 45.67% |
| Overall | 88.19% | 54.18% | 65.57% | 50.37% |

### control, expanded226, separately evaluated latest weights

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 95.98% | 73.84% | 88.36% | 74.88% |
| wrapper | 94.30% | 87.64% | 93.10% | 68.58% |
| bottle | 84.75% | 57.78% | 58.52% | 45.65% |
| Overall | 91.68% | 73.09% | 79.99% | 63.04% |

### accum64, fixed126, separately evaluated latest weights

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 92.02% | 50.47% | 75.18% | 57.10% |
| wrapper | 74.37% | 52.94% | 63.12% | 50.69% |
| bottle | 88.71% | 48.89% | 57.72% | 44.29% |
| Overall | 85.03% | 50.77% | 65.34% | 50.69% |

### accum64, expanded226, separately evaluated latest weights

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 95.63% | 73.93% | 88.24% | 77.33% |
| wrapper | 95.39% | 85.39% | 93.14% | 68.25% |
| bottle | 77.02% | 52.17% | 57.65% | 44.28% |
| Overall | 89.35% | 70.50% | 79.67% | 63.29% |

There are tradeoffs: control improved fixed126 bottle precision to85.81% and AP50–95 to45.67%, while bottle AP50 fell to58.59%. Accum64 bottle AP50 is57.72%. These are not all-class improvements or80%-per-class results. Do not replace AP50 with precision to claim success.

### Additional data: reviewed context scenes

Using existing `lvis_food.archive_array`, the official train archive checksum was verified and streamed without loading the1.1GB JSON into memory. It contains630 water-bottle image IDs/1449 annotated objects;538 candidate IDs were not represented among existing LVIS-tagged V2 filenames or earlier staged IDs and had small-bottle annotations. This is a candidate inventory, not538 approved new training photos.40 links were attempted and33 additional photographs downloaded;7 links failed. Eight photographs passed native-resolution review of all three target classes, adding28 plastic beverage bottle boxes. Other candidates remain unapproved, including uncertain bottle material, non-beverage containers and ambiguous/incomplete plate or wrapper coverage. Source water_bottle labels were never accepted as complete three-class labels automatically.

The new separate **`data/focus-three-class-improvement/bottle-context-reviewed`** dataset has **1302 images,1076train/226val**. It retains every one of the1294 previous images, annotations and split assignments. All retained label bytes were compared and match exactly; image/pixel hashes, rows and splits match the original manifest, with full build verification before and after copying. All eight additions are positive training-only examples. No validation image, annotation or membership changed; original11-class data and models remain untouched.

| Split | Images | Verified negatives | Plate objects | Wrapper objects | Bottle objects |
|---|---:|---:|---:|---:|---:|
| Train |1076|84|650|399|378|
| Validation |226|3|237|89|45|

Dataset fingerprint: `3986872df5850b7acf862df91fa3329e1bc9408d9d1646d8a07151bcf1b08089`. Coverage: `{"train": {"bottle_images": 321, "bottle_objects": 378, "objects_below4pct": 155, "multiple_bottle_images": 26}, "val": {"bottle_images": 36, "bottle_objects": 45, "objects_below4pct": 36, "multiple_bottle_images": 7}}`. The remaining source/size gap is documented; adding eight photos does not prove it is solved.

Leakage checks include source-ID screening, exact image and decoded-pixel checks, preserved original/capture split groups, and a new bidirectional mutual SIFT/RANSAC check against all226 validation images for all20 admitted external photos. No duplicate was flagged; a resized/re-encoded positive control produced246 inliers. Earlier one-way feature matching produced false matches through repeated correspondences; those scores are not accepted as a robust duplicate signal. The stronger screen remains a heuristic and does not prove complete capture independence. Hash-bound new sign-off: `bottle-candidates/approved_training_reviews_context.json`.

One lecture photo is a larger Flickr rendition than the official annotation dimensions; proportional source bounds were explicitly checked on the native image. A baseball photo with a one-pixel width discrepancy received manually checked native bottle bounds. The strict JPEG/trailer check passed all images; no loader normalization is assumed.

### Next user-run experiment (prepared, not trained)

The recipe retains the same earlier fine-tuning/replay settings, starts from the preserved63.08% checkpoint and adds the eight reviewed photos at four views each:1076 unique training images,1788 exposures per epoch. These repeated exposures are not independent new images. A new optimizer phase is explicit because the dataset changed; any gain cannot be attributed to data alone. All three classes remain in training and evaluation. The80-epoch schedule is retained, with a seven-minute epoch-complete stopping budget to normally finish aroundthree epochs/~9minutes; setup and a slow final epoch can add time. No model trained on the new1302-image dataset has yet been evaluated.

```sh
.venv/bin/python scripts/focus_three_class.py train \
  --data data/focus-three-class-improvement/bottle-context-reviewed \
  --model model/focus-three-class-improvement/bottle-context80-start \
  --recipe model/focus-three-class-improvement/bottle-context80-recipe.json
```

Dataset build already completed and passed box/negative/leakage integrity checks. For reconstruction only, choose a fresh destination (existing outputs are protected) and regenerate the recipe/replay paths for that destination:

```sh
.venv/bin/python scripts/focus_three_class.py build_external \
  --frozen-data data/focus-three-class-improvement/bottle-enriched-clean \
  --data data/focus-three-class-improvement/bottle-context-reviewed \
  --review-files data/focus-three-class-improvement/bottle-candidates/approved_training_reviews_context.json
```

After training, evaluate both selected and latest checkpoints on both unchanged sets:

```sh
for checkpoint in best_min_ap50 last; do
  .venv/bin/python scripts/focus_three_class.py evaluate \
    --data data/focus-three-class-improvement/all-reviewed-stack \
    --model model/focus-three-class-improvement/bottle-context80-start \
    --eval-name "evaluation-${checkpoint}-fixed126" \
    --eval-weights "model/focus-three-class-improvement/bottle-context80-start/train/weights/${checkpoint}.pt" || break
  .venv/bin/python scripts/focus_three_class.py evaluate \
    --data data/focus-three-class-improvement/bottle-context-reviewed \
    --model model/focus-three-class-improvement/bottle-context80-start \
    --eval-name "evaluation-${checkpoint}-expanded226" \
    --eval-weights "model/focus-three-class-improvement/bottle-context80-start/train/weights/${checkpoint}.pt" || break
done
```

Keep the earlier63.08% checkpoint if the new model regresses. Passing the training callback on expanded validation alone does not meet the requirement; require80% AP50 separately for every class on both sets, and distinguish these reused development sets from an independent deployment test.

Everything below is historical.

## Current next step: paired stability diagnosis after epoch7 regression

The completed continuation reached epoch7. Separate latest-checkpoint evaluation passed integrity checks on both unchanged datasets. The latest model regressed; the inherited epoch3 selected checkpoint remains preserved with verified fixed126 bottle AP50 of63.08%. It is not an80%-per-class model.

### Epoch7 fixed126

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 76.88% | 65.55% | 73.09% | 50.45% |
| wrapper | 43.71% | 47.06% | 44.00% | 34.65% |
| bottle | 56.60% | 60.00% | 52.10% | 40.18% |
| Overall | 59.06% | 57.54% | 56.40% | 41.76% |

### Epoch7 expanded226

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 86.72% | 79.87% | 87.64% | 74.43% |
| wrapper | 86.66% | 83.15% | 85.98% | 61.29% |
| bottle | 57.03% | 60.00% | 52.00% | 40.14% |
| Overall | 76.80% | 74.34% | 75.21% | 58.62% |

Bottle precision fell76.71%→56.60% while recall remained near60%, which supports investigating false positives rather than claiming a confidence-threshold adjustment fixes AP. The galleries show examples involving bags, foil packaging and glass containers. Training-only mining of258 already-audited no-bottle images found six with bottle predictions at confidence≥0.25. No labels were changed and no validation images were added to training. These findings do not prove the sole cause of regression.

Saved diagnostics live in `bottle-recovery80-chunk9`: `bottle_failure_gallery.jpg`, `bottle_false_positive_gallery.jpg`, and `hard_negative_training_candidates.json/.jpg`. The initial low-confidence failure diagnostic emitted two NMS time-limit warnings; its matching counts are approximate and must not be treated as official performance metrics. The separate official evaluations and training-only mining did not emit those warnings.

Several remaining external candidates appeared to reuse existing litter scenes and remain unapproved. An additional SIFT/RANSAC check compared12 admitted photos against all226 validation images. Visual inspection of each candidate's strongest flagged match found different scenes; no convincing overlap was found. This extra heuristic is not proof of complete capture/source independence. Details: `external_validation_visual_duplicate_audit.json`.

Next, test optimizer instability with a paired experiment. Both probes use the exact same epoch3 selected weights, dataset, replay, AdamW, learning rate, warmup, augmentation and seed. Only `nbs` differs:4 versus64. This tests averaging gradients over more images per optimizer update while retaining physical batch4 and MPS. It does not increase the independent image count or guarantee better generalization. Both probes start identical new optimizer phases, so their paired comparison controls for that reset. Each stops after a completed epoch once one minute has elapsed (normally one ~3-minute epoch), preserving an80-epoch horizon. Compare actual completed epoch counts; different counts invalidate a matched one-epoch interpretation.

Recipes and their training configurations passed preflight, including exact checkpoint hash, audited train membership, frozen validation and fresh output checks. No probe training has been launched. The pair is expected to take roughly6–8minutes plus integrity/setup time; the user runs it:

```sh
for variant in control accum64; do
  .venv/bin/python scripts/focus_three_class.py train \
    --data data/focus-three-class-improvement/bottle-enriched-clean \
    --model "model/focus-three-class-improvement/bottle-stability-${variant}" \
    --recipe "model/focus-three-class-improvement/bottle-stability-${variant}-recipe.json" || break
done
```

Evaluate both probes' saved `last.pt` on fixed126 and expanded226 before choosing further training. Compare against epoch3 plate71.22%/wrapper64.86%/bottle63.08% on fixed126 and86.53%/91.95%/62.99% on expanded226. Do not promote a model merely for higher expanded overall AP. Both validation sets are reused development sets; independent deployment validation remains outstanding. Further evidence may call for more independent training scenes rather than more epochs.

## Latest completed result: bottle recovery, three epochs

The user completed `bottle-recovery80-start`: three additional fine-tuning epochs,552.85seconds training. All four separate saved-checkpoint evaluations completed successfully: selected and latest weights, each on fixed126 and expanded226. Selected/latest numeric metrics are identical, although stripped checkpoint serialization produces different file hashes.

Bottle fixed126 AP50 improved55.12%→63.08% (+7.96percentage points); plate74.39%→71.22%, wrapper67.60%→64.86%. This is a bottle gain with regressions in the other classes, not an all-class win or target attainment. Expanded226 is86.53%plate/91.95%wrapper/62.99%bottle, overall80.49%; overall80% does not meet the per-class80% requirement. These are reused development sets, not an independent deployment test. The trial does not isolate external-data effects from the new optimizer phase.

### Fixed126: separately evaluated selected checkpoint

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 77.98% | 62.04% | 71.22% | 53.54% |
| wrapper | 59.53% | 58.82% | 64.86% | 50.76% |
| bottle | 76.71% | 58.56% | 63.08% | 42.80% |
| Overall | 71.41% | 59.81% | 66.39% | 49.03% |

### Expanded226: separately evaluated selected checkpoint

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 86.24% | 79.75% | 86.53% | 74.96% |
| wrapper | 91.31% | 87.64% | 91.95% | 63.89% |
| bottle | 76.89% | 59.15% | 62.99% | 42.77% |
| Overall | 84.81% | 75.51% | 80.49% | 60.54% |

Selected checkpoint SHA256: `818161d62c00ffc580c6b5ce0c6261b58fbccd699b5227e3471f4dce3a5b5693`. Latest checkpoint SHA256: `f174c5aec0b69c6680694ef8d756de8654b37c60b8e6acf7835998b915ca0d36`. Full object support, data fingerprints and metrics are in the four `bottle-recovery80-start/evaluation-*/metrics.json` files.

Integrity verification passed both datasets before inference. macOS had offloaded29 experimental snapshot files to iCloud; these were restored from the matching local clean snapshot after checking image hashes, label rows and split assignments. The resulting full verification passed without annotation or split changes.

The next continuation preflight passed: unstripped checkpoint completed epoch3, optimizer state retained, fixed80-epoch horizon, no native time limit and same audited replay configuration. No next training run has been launched. Continue without restarting:

```sh
.venv/bin/python scripts/focus_three_class.py continue_train \
  --data data/focus-three-class-improvement/bottle-enriched-clean \
  --model model/focus-three-class-improvement/bottle-recovery80-chunk9 \
  --continue-from model/focus-three-class-improvement/bottle-recovery80-start/train/weights/last_resume.pt \
  --epochs 80 --minutes 9 --target-min-ap50 0.8
```

Evaluate selected and latest checkpoints on both fixed126 and expanded226 after this next chunk. In the four-evaluation loop below, replace model directory `bottle-recovery80-start` with `bottle-recovery80-chunk9`. Do not rerun the completed initial training command. The stopping callback uses expanded validation; both sets must be checked separately before claiming the per-class target.

## Completed initial recovery plan: verified 55.12% bottle checkpoint

User selected the earlier fine-tuned model, not the fresh standard run. The selected starting checkpoint is `meal-replay80-chunk9/train/weights/last.pt`, SHA256 `c6cd2f345c4e1bf5f6712184b5a45703c633dd1f914d51dfe1b173d6e4cf7f90`. Its separate fixed126 AP50 is plate74.39%, wrapper67.60%, bottle55.12%; expanded226 is87.93%/92.46%/54.97%. No class target attainment is implied across both sets.

The fresh `bottle-standard80-chunk9` completed epoch6 but is not the chosen continuation. Its separate expanded226 AP50 is76.60%/86.33%/34.46%. Its fixed126 evaluation failed with OS `Operation canceled` and produced no metrics; do not report an inferred result.

The next experiment, **not yet trained**, uses `bottle-recovery80-recipe.json`: same earlier weights, gentle AdamW fine-tuning and source replay policy, plus12 reviewed external photos/20 plastic beverage bottles in the clean dataset. This is a new optimizer phase because the dataset changed, not an exact optimizer resume or a controlled attribution of gains to data alone. Every training image remains included. There are1068 unique training photos and1756 augmented exposures per epoch; the12 external photos receive four views each, which are not independent new examples. Validation images, labels and membership remain unchanged.

Fresh preflight passed all1294 images, box bounds, verified negatives, byte/pixel/original/capture leakage checks, exact source checkpoint hash and training-config membership. Train objects650plate/399wrapper/350bottle; val237/89/45. This validates preparation, not performance.

Run the following from the repository root. Training stops after a completed epoch once nine minutes elapse; setup, integrity checks and evaluation add time. Output must be a fresh directory.

```sh
.venv/bin/python scripts/focus_three_class.py train \
  --data data/focus-three-class-improvement/bottle-enriched-clean \
  --model model/focus-three-class-improvement/bottle-recovery80-start \
  --recipe model/focus-three-class-improvement/bottle-recovery80-recipe.json
```

Separately evaluate both the selected weakest-class checkpoint and the latest checkpoint on both validation sets:

```sh
for checkpoint in best_min_ap50 last; do
  .venv/bin/python scripts/focus_three_class.py evaluate \
    --data data/focus-three-class-improvement/all-reviewed-stack \
    --model model/focus-three-class-improvement/bottle-recovery80-start \
    --eval-name "evaluation-${checkpoint}-fixed126" \
    --eval-weights "model/focus-three-class-improvement/bottle-recovery80-start/train/weights/${checkpoint}.pt" || break
  .venv/bin/python scripts/focus_three_class.py evaluate \
    --data data/focus-three-class-improvement/bottle-enriched-clean \
    --model model/focus-three-class-improvement/bottle-recovery80-start \
    --eval-name "evaluation-${checkpoint}-expanded226" \
    --eval-weights "model/focus-three-class-improvement/bottle-recovery80-start/train/weights/${checkpoint}.pt" || break
done
```

Report per-class precision, recall, AP50 and AP50–95, overall metrics and checkpoint hash from each metrics.json. Require80% AP50 per class on both sets before accepting the target. These reused development sets do not establish independent deployment performance. Preserve the earlier model if results regress. Further chunks must use the new run's `last_resume.pt`, clean dataset and the same80-epoch horizon.

Everything below describes earlier completed experiments or superseded commands.

## Historical status after bottle sourcing

The previous plate/wrapper scores were real saved-checkpoint validation, after actual fine-tuning on282 added meal-training photos;100 additional meal photos were held out. They were not training-set scores. However, expanded226 AP50 of87.93% plate/92.46% wrapper did not generalize to unchanged126 AP50 of74.39%/67.60%. Neither class is established above80% across both benchmarks, and neither development set is an untouched deployment test.

The current clean dataset is **`data/focus-three-class-improvement/bottle-enriched-clean`**, with1294 images:1068train/226val. All1282 baseline image bytes, label bytes, and split assignments are preserved. Twelve reviewed external photos add20 plastic beverage bottle boxes; all are training-only. Train objects:650plate/399wrapper/350bottle; validation237/89/45. No new negative examples are inferred. New examples receive one view per epoch in the standard recipe.

The initial gentle fine-tuning probe `bottle-enriched80-start` is **invalid and rejected**: Ultralytics silently rewrote one training JPEG because it contained a67-byte Samsung SEFT trailer after JPEG EOI. Expanded evaluation correctly stopped on the changed hash. Its partial fixed126 evaluation is not accepted as evidence of improvement. The original download was retained, a separate clean derivative removed only the trailer, and decoded RGB hashes before/after match exactly. The new builder and verifier reject JPEGs that the loader would rewrite; training and continuation also verify dataset integrity again after completion.20 tests pass, including this regression, unsafe external admission, and byte-for-byte preservation of baseline images/labels including empty negatives.

`bottle-standard80-start` begins from original COCO-pretrained `model/weights/yolo11n.pt`, using standard installed Ultralytics augmentation, optimizer, effective-batch and warmup defaults at batch4/imgsz640/MPS. It uses a fixed80-epoch horizon and epoch-complete stopping, so later chunks retain optimizer state without the native wall-time schedule compression encountered previously. This is a fresh run, not a continuation of the earlier gentle fine-tuning optimizer and not a claimed improvement.

Build (fresh directory required):

```sh
.venv/bin/python scripts/focus_three_class.py build_external \
  --frozen-data data/focus-three-class-improvement/meal-expanded \
  --data data/focus-three-class-improvement/bottle-enriched-clean \
  --review-files data/focus-three-class-improvement/bottle-candidates/approved_training_reviews_clean.json
```

Next user-run chunk, approximately9 minutes plus remaining epoch/setup/evaluation:

```sh
.venv/bin/python scripts/focus_three_class.py continue_train \
  --data data/focus-three-class-improvement/bottle-enriched-clean \
  --model model/focus-three-class-improvement/bottle-standard80-chunk9 \
  --continue-from model/focus-three-class-improvement/bottle-standard80-start/train/weights/last_resume.pt \
  --epochs 80 --minutes 9 --target-min-ap50 0.8

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/bottle-enriched-clean \
  --model model/focus-three-class-improvement/bottle-standard80-chunk9 \
  --eval-weights model/focus-three-class-improvement/bottle-standard80-chunk9/train/weights/best_min_ap50.pt

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/all-reviewed-stack \
  --model model/focus-three-class-improvement/bottle-standard80-chunk9 \
  --eval-name evaluation-fixed126 \
  --eval-weights model/focus-three-class-improvement/bottle-standard80-chunk9/train/weights/best_min_ap50.pt
```

Historical results and rejected probes below refer to their named checkpoints and dataset versions; their metrics must not be combined with the new run.

## Dataset and underlying problems

All 3,096 V2 image/label pairs were inventoried with current hash-validated approved overrides. The earlier 947-candidate count counted existing positive labels, not all visually present objects. Reviewing 445 school images whose target labels were empty recovered 382 signable photos; 63 remain ambiguous. Empty labels alone never establish a negative example. Other unreviewed and ambiguous images remain excluded.

The isolated `data/focus-three-class-improvement/meal-expanded` dataset contains **1,282 distinct images**. All earlier 900 images retain identical bytes, labels and split assignments. Corrected V2 labels, plastic-beverage-only bottles, and one box per entire plate stack are used.

| Split | Images | Verified negatives | Plate objects | Wrapper objects | Bottle objects |
|---|---:|---:|---:|---:|---:|
| Train | 1,056 | 84 | 650 | 399 | 330 |
| Validation | 226 | 3 | 237 | 89 | 45 |

Integrity validation checks normalized box coordinates and bounds, signed-off negative status, original-photo groups, image-byte hashes, decoded-pixel hashes, meal pairs, and video capture groups. No checked group crosses splits. Existing V2 grouping takes precedence over Roboflow filename prefixes. Original 11-class labels and checkpoints are untouched.

The unchanged 126-image development benchmark contains 137 plate, 17 wrapper and 45 bottle objects. It remains essential: adding 100 validation meal photos changes the aggregate distribution. Separate evaluation on both sets prevents favorable new examples from concealing regressions. These repeatedly used development sets are not an independent deployment test; fresh held-out cafeteria captures will be needed to establish deployment performance.

## Findings

The last cold-start nine-minute run was undertrained; native Ultralytics time budgeting compressed its intended 80-epoch horizon to seven epochs. Extending that altered horizon raised the learning rate and regressed. New chunks preserve the same 80-epoch schedule and optimizer, stop after a complete epoch, and save `last_resume.pt` before optimizer stripping.

Adding meal images with uniform sampling improved expanded-set scores but regressed on the unchanged benchmark: AP50 67.39% plate / 38.05% wrapper / 53.53% bottle, versus the earlier checkpoint's 71.47% / 73.99% / 52.15%. The uniformly sampled checkpoint is rejected as a replacement.

The new controlled trial restarts from the stronger earlier checkpoint and retains all 1,056 training images. It repeats selected plate/trash field contexts (maximum six views per image), giving 1,708 augmented training views per epoch. These are repeated exposures, not additional independent images. This explicitly addresses context imbalance without deleting difficult validation examples. Its first epoch took about 162 seconds including in-training validation. The recipe records source exposure counts and warmup adjustment.


## Separately evaluated short-trial results

Both evaluations use checkpoint SHA256 `8526865d6171dc30caf395e79abf07a056e626ad04a337ffafdce35e54ae967e`. Neither passes all classes. Balanced replay recovers some regression relative to uniform meal sampling, but wrapper AP50 still trails the earlier stronger checkpoint (73.99%). This is an experimental continuation candidate, not a replacement or proven improvement.

### Unchanged 126-image benchmark

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 90.45% | 55.33% | 73.02% | 51.76% |
| wrapper | 46.32% | 52.94% | 54.58% | 44.26% |
| bottle | 69.82% | 48.89% | 55.84% | 37.31% |
| Overall | 68.87% | 52.39% | 61.15% | 44.44% |

### Expanded 226-image validation

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 92.37% | 76.66% | 85.70% | 71.64% |
| wrapper | 82.32% | 80.90% | 84.81% | 57.82% |
| bottle | 66.95% | 49.53% | 55.72% | 37.26% |
| Overall | 80.55% | 69.03% | 75.41% | 55.57% |

## Historical build and training commands (completed)

Rebuild into a fresh destination; existing output directories are intentionally protected:

```sh
.venv/bin/python scripts/focus_three_class.py build_all \
  --data data/focus-three-class-improvement/meal-expanded \
  --frozen-data data/focus-three-class-improvement/all-reviewed-stack \
  --one-box-per-stack \
  --review-files data/focus-three-class-improvement/expanded_review_with_school.json
```

The completed short trial used:

```sh
.venv/bin/python scripts/focus_three_class.py train \
  --data data/focus-three-class-improvement/meal-expanded \
  --model model/focus-three-class-improvement/meal-replay80-start \
  --recipe model/focus-three-class-improvement/meal-replay80-recipe.json
```

The replay chunk below has now completed; do not rerun it unchanged. Longer runs are reserved for the user. Epoch-complete nine-minute limits can exceed nine minutes by the remaining epoch plus setup/evaluation. Every chunk requires a fresh output directory. Continue the same horizon and replay configuration:

```sh
.venv/bin/python scripts/focus_three_class.py continue_train \
  --data data/focus-three-class-improvement/meal-expanded \
  --model model/focus-three-class-improvement/meal-replay80-chunk9 \
  --continue-from model/focus-three-class-improvement/meal-replay80-start/train/weights/last_resume.pt \
  --epochs 80 --minutes 9 --target-min-ap50 0.8
```

Evaluate exactly the saved selected checkpoint on both datasets:

```sh
.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/meal-expanded \
  --model model/focus-three-class-improvement/meal-replay80-chunk9 \
  --eval-weights model/focus-three-class-improvement/meal-replay80-chunk9/train/weights/best_min_ap50.pt

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/all-reviewed-stack \
  --model model/focus-three-class-improvement/meal-replay80-chunk9 \
  --eval-name evaluation-fixed126 \
  --eval-weights model/focus-three-class-improvement/meal-replay80-chunk9/train/weights/best_min_ap50.pt
```

Selection during training uses the weakest class on expanded validation. It does not imply the unchanged benchmark target passed. Inspect both `metrics.json` files and their `all_classes_meet_target` flags. No performance improvement or target attainment is inferred from training loss or extra epochs.

Verification: all 18 focused unit tests passed, script compilation passed, and both evaluations revalidated the dataset before inference. The optimizer checkpoint retains epoch horizon80 and the audited replay YAML. A regression test verifies continuation preserves that YAML rather than silently reverting to uniform sampling.

## Completed user replay chunk: epochs2–5

The nine-minute epoch-complete continuation took about616 seconds of training. Weakest-class selection retained the inherited epoch1 checkpoint because subsequent bottle AP50 never exceeded it. The latest epoch5 is evaluated separately below, not presented as the selected winner.

### Unchanged126, latest epoch5, 640

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 76.31% | 70.52% | 74.39% | 53.14% |
| wrapper | 72.58% | 64.71% | 67.60% | 56.16% |
| bottle | 64.42% | 51.11% | 55.12% | 40.32% |
| Overall | 71.10% | 62.11% | 65.70% | 49.87% |

### Expanded226, latest epoch5, 640

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 87.82% | 82.70% | 87.93% | 74.41% |
| wrapper | 91.72% | 87.08% | 92.46% | 65.37% |
| bottle | 65.55% | 50.76% | 54.97% | 40.19% |
| Overall | 81.70% | 73.51% | 78.45% | 59.99% |

### Unchanged126, latest epoch5, 960

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 65.36% | 56.46% | 59.77% | 36.97% |
| wrapper | 54.97% | 58.82% | 45.10% | 33.15% |
| bottle | 60.40% | 53.33% | 50.78% | 36.53% |
| Overall | 60.24% | 56.21% | 51.88% | 35.55% |

Higher-resolution inference is rejected: all three classes worsened. Repeating unchanged training is not recommended from this evidence. Bottle coverage analysis is saved in `meal-replay80-chunk9/bottle_coverage.json`: 107/330 training bottles (32.4%) versus36/45 validation bottles (80.0%) occupy below4% image area; only14/301 training bottle images (4.7%) versus7/36 validation bottle images (19.4%) contain multiple bottles. The382 recovered meal photos added no confirmed beverage bottles. These counts identify a coverage gap; they do not prove it is the only cause.

A short controlled `small-object80-start` trial keeps every unique image and the same replay policy, enables mosaic0.5 and scale0.7 at640, and starts from epoch5. This explicitly starts a new optimizer phase; it is not an exact optimizer continuation. The stored recipe documents the changes. Augmentation cannot create new independent bottle examples. Neither validation set is an untouched test set.

## Small-object augmentation test: rejected

One additional epoch completed in about169 seconds. Saved-checkpoint evaluation gave the following results; no further run of this recipe is recommended from this short test. This does not establish that all possible augmentation schedules would fail.

### Unchanged126

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 88.35% | 55.37% | 75.88% | 53.93% |
| wrapper | 67.31% | 64.71% | 67.01% | 51.74% |
| bottle | 84.20% | 42.22% | 53.44% | 40.49% |
| Overall | 79.95% | 54.10% | 65.44% | 48.72% |

### Expanded226

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 92.33% | 76.21% | 89.11% | 75.46% |
| wrapper | 84.91% | 89.89% | 91.77% | 64.39% |
| bottle | 83.99% | 46.62% | 52.75% | 40.04% |
| Overall | 87.08% | 70.91% | 77.87% | 59.96% |

The80% target remains unmet. Keep the earlier checkpoints; neither higher-resolution inference nor the short mosaic/scale trial is a demonstrated replacement. Existing unresolved images must not be admitted simply to improve a score. Useful new independent training examples would include plastic drink bottles in natural cafeteria/litter scenes, at small apparent sizes, behind objects, crushed or transparent, and with several bottles in one frame. Review all three target classes in each photo. Split entire original captures and video sessions together; reserve fresh sessions for an untouched test. An additional local source folder was requested; no new data or long training job has been assumed available.

## Clean standard run: saved-checkpoint evaluation after epoch1

This is the first epoch of a fresh COCO-pretrained80-epoch schedule, not the earlier fine-tuned model. It is not a replacement or a performance gain yet. Post-training integrity verification passed on all1294 images; optimizer state is retained in last_resume.pt, total horizon80, native timeNone.

### Unchanged126 development benchmark

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 61.82% | 51.82% | 54.72% | 32.59% |
| wrapper | 32.64% | 47.06% | 43.56% | 33.95% |
| bottle | 48.41% | 44.44% | 39.38% | 27.08% |
| Overall | 47.62% | 47.78% | 45.89% | 31.20% |

Checkpoint SHA256: `f6d963a6ab1243d45bec31f852660097486ddaffd0f99c141dca826062873999`. All classes meet80%: **False**.

### Expanded226 development validation

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 72.64% | 70.58% | 76.25% | 58.79% |
| wrapper | 76.63% | 77.53% | 80.40% | 48.81% |
| bottle | 46.33% | 44.44% | 38.34% | 26.43% |
| Overall | 65.20% | 64.18% | 64.99% | 44.68% |

Checkpoint SHA256: `f6d963a6ab1243d45bec31f852660097486ddaffd0f99c141dca826062873999`. All classes meet80%: **False**.


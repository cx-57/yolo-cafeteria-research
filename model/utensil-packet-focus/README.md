# Utensil / condiment packet specialist

## Recovery milestone: 50% AP50 per class (2026-10-21)

The immediate target is now 50% AP50 **for each class**, measured on the unchanged
development validation set. The previous selected checkpoint already exceeds
50% for utensils (52.17%) but not packets (15.88%). This is not an independent
cafeteria-session test.

A normalization counterfactual restored the pre-context running statistics while
keeping trained parameters. It worsened development AP50 to 43.27% utensils and
1.50% packets, ruling out this particular simple normalization repair. Evidence:
`context-v4/normalization-counterfactual.json`.

The isolated `coco-restart-v5` experiment reuses 493/499 compatible tensors from
the original COCO YOLO11n, with a fresh two-class output classifier. It uses the
unchanged, integrity-checked context-v4 dataset: 355 training images (227 real,
128 composites), 190 utensil boxes and 234 packet boxes; 764 validation images,
296 utensil boxes and 42 packet boxes. There are only 40 real packet-positive
training photos. No images move across splits and no validation labels change.

```sh
export PYTHONPATH="$HOME/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts"
# Initialization is already prepared; refuse accidental replacement.
.venv/bin/python scripts/utensil_packet.py train --coco-restart --minutes 2
# Continue a prepared checkpoint for a short user-run experiment:
.venv/bin/python scripts/utensil_packet.py train --coco-restart --resume --minutes 8
```

Check `coco-restart-v5/latest.json` and `selected.json` for actual saved-model
scores. A fresh output classifier starts with poor AP; initialization does not
preserve the separate COCO fork/knife/spoon union diagnostic's 72.33% score.

The completed two-epoch pilot did not reach the target. Saved epoch 2 scored
55.91% utensil / 12.11% packet AP50; selected epoch 1 scored 32.53% / 20.28%.
Do not combine the best score of each class into a fictitious joint model.

The `--head-only` experiment starts from the selected COCO-restart epoch-1
checkpoint and trains only detection module 23. Native Ultralytics freezes
modules 0–22, including their BatchNorm running statistics. Everything else,
including data, augmentation and learning-rate schedule, stays the same.
Outputs are isolated in `head-only-v6`:

```sh
.venv/bin/python scripts/utensil_packet.py train --head-only --resume --minutes 8
```

Neither experiment modifies the eight-class run or the prior selected weights.

Head-only-v6 completed three epochs. Its latest saved model gives **67.26%
utensil / 15.86% packet AP50**. A checkpoint audit found all 378 frozen backbone
tensors bit-identical to initialization (`head-only-v6/freeze-audit.json`). The
weaker-class selector retained initialization (32.53% / 20.28%); these are distinct
models. This demonstrates a stronger utensil result, not a solved packet class.

Real-only-v7 tests a specific training-distribution problem: 192/234 packet
training boxes in context-v4 were composites derived from seven mayonnaise
photos. `--real-only` removes all 128 composites, retaining all 227 real reviewed
training photos, 190 utensil boxes, 42 packet boxes and 65 verified joint negatives.
It preserves every validation record exactly (764 images, 296/42 boxes). It
starts from head-only-v6's latest saved model, retains the frozen backbone and
trains the detection head. This is a controlled experiment, not a claimed gain.

The completed real-only-v7 three-epoch pilot's **selected epoch 2** gives:

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| utensil | 85.52% | 55.85% | 71.06% | 43.83% |
| condiment_packet | 43.73% | 30.95% | 24.74% | 16.85% |

Overall AP50 47.90%, AP50–95 30.34%. Both scores come from the same saved
checkpoint, SHA256 `1e748428bf76e4202561b34b092bf668fa6929d24fe772d39acc34743a826184`.
Latest epoch 3 gives 68.91% / 23.82%. These are repeated-development-validation
gains over expanded-v3's selected 52.17% / 15.88%, not independent test gains.
Initialization, training duration and data distribution all changed across the
sequence, so the gains cannot be attributed solely to composite removal.
The 50% per-class target remains unmet for packets. The remaining data limitation
is 40 real packet-positive training photos, with few independent products/scenes.
The next proposed run is an eight-minute continuation, not a full 80-epoch run.

```sh
.venv/bin/python scripts/utensil_packet.py verify --real-only
.venv/bin/python scripts/utensil_packet.py train --real-only --resume --minutes 8
```

### Packet sampling experiment (balanced-real-v8)

A further four-epoch real-only continuation (epochs 4–7) did not beat the selected
24.74% packet checkpoint; epoch 7 scored 70.99% utensil / 18.84% packet AP50.
This is evidence against recommending more of that same run without a change.

`--balanced-real` tests fourfold sampling of the 40 real packet-positive training
photos. It creates 347 training entries from the same 227 unique real photos,
with 199 sampled utensil boxes / 168 sampled packet boxes. These repetitions are
not new independent data. Original SHA/pixel/capture identities and parent records
remain bound to training; the validation record list is exactly unchanged.
No synthetic packet composites are reintroduced. The frozen-backbone model starts
from real-only-v7's selected epoch 2. A candidate with utensil AP50 below 50% is
ineligible for selection, even if its packet score rises. Ten focused tests pass,
including training-only sampling ancestry and the utensil selection safeguard.

```sh
.venv/bin/python scripts/utensil_packet.py verify --balanced-real
.venv/bin/python scripts/utensil_packet.py train --balanced-real --resume --minutes 8
```

Actual saved-model results are in `balanced-real-v8/latest.json` and `selected.json`.
The three-epoch sampling pilot failed: latest saved model 70.60% utensil /
12.93% packet AP50. Selection retained its 71.06% / 24.74% initialization.
Do not treat the continuation command above as a recommendation for a full run.
The additional real-only continuation and real-positive repetition have not
improved the selected packet model.

Inference probes use the unchanged selected real-only-v7 weights and validation:
`evaluate --real-only --tta` and `evaluate --real-only --eval-imgsz 960`.
Their reports explicitly record image size and test-time augmentation. The
training-fit diagnostic is separate and must never be presented as held-out AP.

The 85% per-class AP50 target is **not achieved**. These are saved-model development-validation results. All phases retain exactly the same 764 validation images and target boxes (296 utensils, 42 packets). No new independent cafeteria-session test exists, and inherited validation annotation completeness remains provisional. Repeated validation use selects/tunes these models; do not present this as an untouched test.

The selected expanded-v3 model gives:

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| utensil | 54.24% | 54.86% | 52.17% | 24.60% |
| condiment_packet | 9.24% | 45.24% | 15.88% | 8.96% |

Overall AP50: 34.02%; AP50–95: 16.78%.
This table uses one checkpoint, not each class's best result from different epochs. Selection maximizes the weaker class's AP50. The last expanded-v3 checkpoint has 59.37% utensil / 7.28% packet AP50 and is retained separately.

## What changed and what remains wrong

The old A checkpoint starts at 20.51% utensil / 2.49% packet AP50 when projected to two outputs. `scripts/utensil_packet.py` preserves its learned localization and classifier channels 3/7 exactly; decoded-box and class-score equivalence tests pass. Original V2 labels, the 11-class pipeline and plate/wrapper specialist are preserved.

Only training photos reviewed for both targets are used, including 32 verified joint negatives. This permits ordinary Ultralytics detection loss and standard augmentation, including mosaic, without treating unreviewed targets as absent. Train/validation checks bind image bytes, labels, review files, YAML, exact directory membership, original/capture identities and independently decoded pixels. Nested unreviewed files are rejected.

The first two-class trial's epoch-4 packet AP50 was 92.52% on training photos versus 6.65% on validation. Utensils were 78.27% versus 49.96%. The packet gap is a substantial generalization problem. Forty-four older training photos repeatedly show the same red/orange Burger King packets on the same floor.

Balanced-v2 continues that four-epoch model, retaining four representative red/orange front/back views instead of all 44. It keeps every other reviewed training photo and every validation record. This is a combined training/sampling continuation, not a matched experiment isolating the cap alone. Its selected saved model scored 47.47% utensil / 11.43% packet AP50.

Expanded-v3 continues balanced-v2's selected checkpoint and adds eight native-resolution reviewed mayonnaise-packet photos (eight packet boxes and one visible eating fork). Source URLs, photographers, licenses and hashes are in `data/utensil-packet-focus/extra-reviews-v3.json`; box QA is `model/utensil-focus/preflight/mayo-box-qa.jpg`. Seven additions share an existing training photographer and one is from another photographer; they are eight new appearances, not eight independent capture groups. Validation photographers are excluded. Salt/pepper, drug sachets, food wrappers, bulk pouches and artwork candidates were not imported as positives.

Expanded-v3 has **192 training images, 188 utensil boxes, 42 packet boxes and 32 verified joint negatives**. Validation stays **764 images / 296 utensil / 42 packet boxes**; empty inherited validation annotations are not certified negatives. Packet training variety and the small, provisional validation support remain limitations. Further training may help; 85% is not established or guaranteed.

## Commands

Run from `/Users/Charles1/Desktop/yolo-cafeteria-research`:

```sh
export PYTHONPATH="$HOME/Library/Caches/yolo-cafeteria-runtime/py312"
.venv/bin/python scripts/utensil_packet.py verify --expanded-packets
.venv/bin/python scripts/utensil_packet.py train --expanded-packets --resume
.venv/bin/python scripts/utensil_packet.py evaluate --expanded-packets
```

The prepared expanded-v3 pilot completed three epochs of its 80-epoch schedule. `--resume` restores its full optimizer and EMA checkpoint and continues that phase to epoch 80, using YOLO11n, batch 4, 640, MPS, fresh-phase AdamW lr0=0.0003, nbs=64, warmup_bias_lr=0, warmup=3, standard augmentation, no early stopping. It evaluates both latest and selected saved weights automatically after training. `--minutes` is intentionally omitted from the full run. The short-test budget is checked after completed training epochs; saved-model evaluation adds time.

The runtime outside iCloud contains the exact installed package versions from official PyPI wheels, verified against published wheel SHA256 hashes. Desktop/iCloud had offloaded native libraries and produced SIGBUS crashes. No dependency versions were changed.

The build/initialize commands used for this prepared phase were:

```sh
.venv/bin/python scripts/utensil_packet.py build --expanded-packets
.venv/bin/python scripts/utensil_packet.py initialize --expanded-packets
```

They refuse to overwrite existing outputs. They require the existing reviewed V7 source and selected balanced-v2 checkpoint. They have already completed successfully here.

For diagnosis on training images (not a held-out score):

```sh
.venv/bin/python scripts/utensil_packet.py evaluate --expanded-packets --split train
```

Results: `expanded-v3/selected.json`, `expanded-v3/latest.json`, `expanded-v3/selection.json`, `expanded-v3/train/results.csv`, `expanded-v3/best.pt`, `expanded-v3/last_resume.pt`. Full phase reports are in `experiment-comparison.json`. Five focused integrity and head-transfer tests pass; independent code/data review found no blocking issue.

## Further diagnosis and context-v4 (2026-10-20)

An error analysis of expanded-v3's selected model found 23 packet matches and 208 unmatched packet detections at confidence 0.25 / IoU 0.5, including 108 unmatched detections on carton images. These are annotation-relative development diagnostics, not a certification of every unmatched prediction as a semantic error. See `diagnosis-v3/matches.json`.

Context-v4 adds 35 visually reviewed negative training photos and 128 training composites made from seven masked, real training packet photos. It excludes obvious conveyor-belt and PXL container capture setups shared with validation. Each synthetic example binds all parents to training-only image hashes; rotation/alpha geometry determines its labels. These 128 composites are synthetic augmentation, not new real-world captures. Validation remains entirely real and unchanged.

The pretraining QA audit discovered two missed metal spoons in previously accepted joint-negative training photos. Hash-bound corrections in `data/utensil-packet-focus/training-corrections-v4.json` add those boxes only in context-v4 and exclude those images from composite backgrounds. The earlier datasets remain unchanged. The corrected context-v4 dataset contains 227 real photos plus 128 composites, 190 utensil boxes, 234 packet boxes, and 65 joint negatives.

The two-epoch context-v4 pilot **did not improve the weaker class**: latest saved model 55.60% utensil / 7.66% packet AP50. Selection retained its initialization (52.17% / 15.88%). No accuracy gain is claimed for context-v4. Seven focused tests and an independent code/data review passed after the spoon corrections. Current corrected QA is `diagnosis-v3/context-qa.jpg`; the separately named before-audit collage is historical evidence.

A separate untrained COCO YOLO11n diagnostic merged its fork, knife and spoon scores before NMS and reached **72.33% utensil AP50** (82.97% precision, 66.22% recall, 45.64% AP50–95) on the same development validation images. It makes no packet predictions and is **not a verified two-class replacement**. This indicates that the current fine-tuning path loses useful pretrained utensil detection ability. Its exact mapping, weight hash and dataset fingerprint are in `pretrained-utensil-control.json`.

Context-v4 uses the same script with `--packet-context`:

```sh
.venv/bin/python scripts/utensil_packet.py verify --packet-context
.venv/bin/python scripts/utensil_packet.py train --packet-context --resume
.venv/bin/python scripts/utensil_packet.py evaluate --packet-context
```

This is an experimental continuation, not a demonstrated improvement or a recommendation that 80 epochs will reach the target. Source annotations and generation code are versioned; generated dataset copies, raw local caches and checkpoints are not included in this experiment commit. Existing reviewed V7 source files and locally saved initialization/checkpoints are required to run these commands.

# Bottle reviewed backgrounds and Open Images implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Inline execution is authorized by the user’s repeated instruction to continue without stopping; no approval handoff.

**Goal:** Continue toward one genuinely saved/fresh bottle checkpoint AP50>=0.90 and AP50–95>=0.64, preserving the frozen246/67 benchmark.

**Architecture:** V31’s first3 neck-training epochs regressed; archive optimizer and all saved checks. Return to balancedV19. Build V33 from V31 positives, a newly AI-reviewed negative pool, and23 Open Images photos/34 original boxes. Hold unreviewed/ambiguous negative photos. Redistribute exactly749 prior negative exposures across the reviewed pool, preserving aggregate negative training volume. Train only the detection head with no mosaic and gentler geometry.

**Tech stack:** Existing cached Python3.12, Ultralytics8.4.175/Torch2.14.1/MPS; PIL/OpenCV source audits; shared V30 exact joint-metric policy/evaluator.

**Spec:** User target90%/64–70%, do not force results, ignore README workflow. Agent choice: data admission and conservative geometry after actual saved-model regression; this is a combined experiment, not a causal ablation.

## Global constraints

Frozen validation photos/labels/split unchanged; development model selection only, no independent test claim. No source annotations edited. Keep generic Bottle source annotations intact; admit whole photo only when every original box is eligible and none visibly omitted. Original labels/absence flags are evidence, not trusted certification. Negative admission requires new full-scene AI review; AI is not human verification. Retain all positives and exact positive exposure counts. Negative exposures exactly749. New reviewed Open Images positives4x. All fixed inference ordinary640,batch4,NMS0.5,noTTA and immutable hash-bound reports retained. One MPS training process at a time.

## Review focus

Unreviewed false negatives must not sneak into admitted backgrounds. Changed image bytes, pixel hashes, box/metadata/filter provenance or source coordinates must fail admission. Source groups/collages and held overlaps rejected. Duplicated accepted source photos/empty negative labels checked. Native accuracy is not fresh saved-model verification. Exact joint target requires both metrics on one checkpoint.

## Tasks

- [x] Inspect V31 saved-model results: selected82.3376%/59.7363%,latest78.0905%/54.9265%; archive after3epochs, keep balanced baseline84.0711%/62.0606%.
- [x] Stage Open Images using official training human-positive labels and Bottle boxes, original photo metadata, rotation0; view60scenes and54target crops. Independent second AI review holds8 (missing fourth bottle); overlap screen holds25. Final23photos/34boxes.
- [x] Finish second review of84 legacy absence candidates and96 uniform seed43 sampled unknown negatives. Record immutable review evidence and conservative holds. Legacy84 retains80 after holds50,65,67,68.
- [x] Observe failing admission tests for Open Images scope/provenance, box coordinates, all indices, overlap and changed bytes. Implement minimal whole-photo admission.
- [x] Observe failing background-exposure tests: preserve all positive exposures; accept only hash-bound reviewed negatives; exactly749 negative exposures, deterministic balanced repeats; reject unknown/positive records masquerading as reviewed absences.
- [x] Build and verify V33 data; compare all frozen validation hashes and protected baseline/legacy checkpoint hashes.
- [x] Reuse shared joint evaluator/policy/controller; balanced88877...1518,freeze23,AdamW1e-5,batch4/nbs16,640,seed43,mosaic0,scale0.25,translate0.05,BNfixed. Save separate AP50/localization/joint/latest roles,3epoch phases up to40 with honest3phase patience.
- [ ] Run full project and experiment checks, independent code review, push changes. Remote GitHub CI currently cannot start due account billing/spending-limit error; local checks must be reported accurately.
- [ ] Launch V33 and continue actual saved-checkpoint assessment. A failed individual trial never becomes a success claim or an automatic end of the overall user task.

## Ledger

Ruling: no new worktree; active experiments bind absolute paths in this workspace. New artifacts isolated by experiment directory. User authorized continued experiments and pushes earlier.
Ruling: absenceFalse does not prove a bad label; absenceTrue with generic notes does not prove correct absence. Use direct AI review to gate this experimental pool, and disclose remaining annotation uncertainty.

User steering on2026-10-24: pause before full training. No launch is authorized now. Final preflight237project/11focused tests passed, frozen246file hashes and protected checkpoints unchanged. V33 training remains unstarted; launch task intentionally pending this explicit user pause.

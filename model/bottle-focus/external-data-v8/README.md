# Combined external bottle supplement

Prepared and verified; **not trained yet**. This combines the two LVIS candidate
rounds documented in `../external-data-v7/`: 35 selected photos with 65 plastic
beverage bottle boxes. Decisions are AI-assisted experimental training reviews,
not human verification. Source photograph URLs, source license IDs, original
annotations, hashes and excluded candidate records are retained.

All 1,056 original training photos remain, giving 1,091 unique photos.
Existing context positives and each admitted new photo have four exposures;
all other originals have one, giving 1,568 exposures per epoch.

Validation remains the original 246 photos and 67 bottle boxes with protocol
`29423591896198d41f5bd16392b51f06c7c3d7254f8a903b3dda74569cfe7de7`.
The source bytes, labels and splits are checked during build and verification.
Exact and heuristic feature screening against every validation photo is
recorded in `duplicate-screen.json`; capture independence is not guaranteed.
The related birthday scene was excluded by visual review.

No accuracy improvement or independent NHS test performance is claimed.

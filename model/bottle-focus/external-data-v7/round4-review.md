# Additional LVIS candidate review — 2026-10-23

The fourth staging batch (seed 42, source-pool offset 180, count 60) yielded
40 decodable, dimension-consistent photos. Codex visually reviewed all 40
full scenes and enlarged 21 target crops from 14 uncertain scenes. This is
AI review, not human verification.

17 photos / 20 source boxes are eligible for a future experimental training
copy. They are **not included in V20 or the active V24 dataset**. The other
23 photos are held for unclear material or identity, reusable containers,
incomplete target annotations, or very small uncertain targets. Photo
lvis_000000057403 also triggered the validation dHash threshold and is held.

All candidates were screened against the fixed 246-photo validation set
using image bytes, decoded pixels, rotation-aware dHash and mutual SIFT
matches with RANSAC geometry. This is a heuristic overlap screen, not proof
of full source independence. No validation image, label, or split changed.
All original source annotation boxes pass image-bound checks. Accepted
annotations and decisions are bound to source SHA-256 and screen SHA-256
in round4-visual-reviews.json. Original COCO/Flickr URLs, license IDs and
LVIS annotation provenance remain in lvis-round4-downloads.json. A source
repository's code license is not treated as a photograph license.

A second AI review held photos 17, 24 and 34: unresolved backpack-bottle
identity, an ambiguous clipped annotation, and an incomplete cola-bottle box.
The remaining counts above reflect those holds. No original annotations were
changed and no candidate has entered training.

# Open Images staging reconnaissance

These files stage public source candidates only. No image has been admitted
to V31 or any training/validation dataset. V31 continues independently.

Official source: https://storage.googleapis.com/openimages/web/download_v7.html
Box annotations: official training split, Bottle MID /m/04dr76w.
Human image labels: official training split, Plastic bottle /m/02rlncx and
Water bottle /m/0118n_9r. Class IDs checked in official V7 descriptions.

Generic first sample: 57 zero-rotation images with1–6 boxes, checked full
scene previews. Most are glass, reusable, non-drink containers or depictions;
not suitable as extra positive data. Potential disposable-plastic scenes
require enlarged per-box review and overlap screening before use.

Targeted pass: 500 human-positive plastic/water image IDs from a bounded
source prefix; 134 have official Bottle boxes, including39 with both human
positive concepts. Missing Bottle boxes are not inferred or fabricated.
Image-level labels only filter candidates; they do not establish each
object’s material, eligibility, exhaustive annotation or box correctness.

CSV processing streams bounded prefixes without storing full archives;
prefix hashes are explicitly not full-source-file hashes. Metadata retains
original photo URL, author, landing URL, license and rotation. Canonical
CVDF JPEGs are inspected only for rotation0 and recognized CC metadata.
Source code licensing is not treated as photograph licensing.

Before future admission: per-object scope and whole-scene absence review,
source geometry, annotation completeness, source-photo/byte/pixel/near-
duplicate overlap checks against the frozen benchmark and prior sources,
and independent second AI QA. AI image review is not human verification.
No performance claim is made from this staging work.

# Future bottle source staging

This directory stages possible future data. It is not a training experiment,
has no training data YAML, and does not alter the active V34 campaign. The
separate full-training run remains paused. No result here establishes 90% AP50.

Official Open Images candidates came from the previously retained targeted
image-level plastic/water filter and complete generic Bottle annotation groups.
Thirty-one additional photos were inspected. Two independent AI reviews retained
17 photos / 26 source boxes for overlap screening. Index27 was additionally held
because its source box cuts through the visible cap/neck; the box was not edited.
Ten further generic photos passed two AI absence reviews; two uncertain photos
remain held. These counts precede overlap screening and are not admitted counts.

Sixty unused official TACO clear-plastic-bottle candidates were staged with
original annotations. Five failed orientation/geometry checks. Whole-scene and
target-crop AI review, followed by independent AI review, retained44 photos /
50 source boxes pending further checks. A brown beach bottle remained materially
uncertain; one repeated capture was held. The official category alone was never
treated as material or target-scope proof. Flickr per-photo JSON-LD provided
license/author/page metadata for48 of60 source IDs; the remainder remain metadata
holds. Of the44 visually retained photos,38 /44boxes have the checked Creative
Commons metadata and enter overlap screening. No blanket image license is inferred
from TACO's software license.

Overlap screening checks the original1,302-photo training/benchmark manifest,
the permanent36-photo V34 monitor and the current567 training photos. Byte
deduplication yields1,448 protected photos, with every role/alias retained in the
receipt. Expected source hashes are verified; the exact review, source-audit,
exposure-list, program and image identities are checked again before publication.
Every exact-byte, decoded-pixel, rotation-aware dHash or SIFT/RANSAC flag is held.
Candidates also join the comparison pool, including held candidates. Screening
is conservative and heuristic, not proof of independent origins. All monitor
photos must remain excluded from any future training dataset.

Five real image/input controls pass locally: identical featureless image,
lossless container alias, re-encoded crop, replaced protected image and changed
review file. Independent review caught two receipt-identity gaps; both received
RED-to-GREEN controls and a clean follow-up review. This is screening evidence,
not final dataset admission. Any future builder must separately verify source
annotations, geometry, whole-photo review and permanent monitor exclusions.

Source records and attribution URLs are retained in JSON. Original photos and
preview sheets remain local staging artifacts.

Screen completed65 candidates:36 passed (16 Open Images positives /25boxes,
9 Open Images backgrounds,11 TACO positives /15boxes);29 flags remain held,
with no overrides. The complete receipt binds1,448 distinct protected photos
and1,905 role aliases. Separate V36 admission subsequently verified these36
photos /40boxes against the original reviewed bytes, official source lists,
metadata and geometry. V35 remains staging only; V34 was never modified.

Sources: [Open Images](https://storage.googleapis.com/openimages/web/download_v7.html),
[official TACO repository](https://github.com/pedropro/TACO). Per-photo Flickr
license evidence is recorded in taco-photo-metadata.json, with page hashes and
retrieval times; unresolved metadata remains a hold.

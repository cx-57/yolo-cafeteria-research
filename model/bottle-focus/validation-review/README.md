# Frozen-validation annotation diagnostic

Read-only AI review of the protected small model's predictions at score at
least 0.05 on images with empty bottle labels. Sixteen photos were reviewed
as full scenes; six enlarged crops cover five uncertain scenes. These are
AI reviews, not human verification. No image, label, manifest or split changed.

One clear missing target was confirmed twice: validation photo
plate__train__lvis_000000384981.jpg contains a disposable clear water bottle
with a green screw cap. Its label and manifest rows are empty. File hashes
and proposal provenance are recorded in empty-image-reviews.json. Three
other photos have unresolved material/type and are not claimed label errors.
Most remaining predictions are glass bottles, condiments, wrappers or cups.

Keep all historical 246-image/67-box scores on their frozen protocol. A
corrected benchmark would require a separately versioned annotation review
and fresh evaluations of all comparison models. This diagnostic does not
constitute an exhaustive validation audit and does not create a new score.

All 55 positive validation photos (67 labels) were also visually reviewed,
with enlarged regions for nine potentially ambiguous scenes. Photo 22,
trash_detection__train__000058_JPG.rf.1fc27d0bbaaed624a92504e831f05147.jpg,
box 0 is a strong visual scope mismatch: a rectangular spirits bottle appears
to be glass. Two AI reviewers agreed; this is not physical material testing.
Photo 41 remains uncertain. The two overlapping labels in photo 49 cover
two distinct crushed bottles; no duplicate-label error is claimed. Other
full-scene reviews found no additional clear issue at their review resolution.
See positive-image-reviews.json for per-photo limits and findings.

frozen-files-verification.json confirms all 246 image hashes and all label
rows still match the frozen manifest, and protected/runtime model hashes
still match the 84.0784% incumbent. These findings are a limitation on the
benchmark annotations, not grounds to revise its historical score or claim
that the model has reached 85%.

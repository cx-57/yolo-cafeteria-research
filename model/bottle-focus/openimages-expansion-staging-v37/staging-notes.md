# Broader source staging

Source-only work; no training dataset, model measurement or admission. V36 has
completed below target. The existing monitor and benchmark are unchanged, and
full training remains paused.

Streamed the official human-image-label prefix to find1,000 further plastic/
water-bottle-positive source IDs, excluding the prior500 candidates and every
permanent monitor ID. Processed2,036,139,879bytes /43,889,583rows; the retained
hash covers the processed prefix, not the complete upstream CSV. Complete generic
Bottle groups exist for463 IDs /1,310boxes. All463 IDs have official photo
metadata; source metadata was streamed without keeping the full upstream file.

After zero-rotation, Creative Commons metadata,1–6 boxes and no source group/
depiction flags,380 sources were eligible for visual staging. Fixedseed1043
selected120 without inspecting model outputs. All120 source JPEGs loaded with
matching geometry/normal orientation. Image-level labels are only a filter and
never establish every object's material or complete target annotation.

Root AI whole-scene/selected-crop review proposed63photos /108boxes. Independent
AI review held44(Smartwater lower body cut) and102(foreground bottle base cut),
leaving61photos /104boxes pending conservative overlap screening and admission.
No box was edited. Reusable containers, uncertain materials, mixed source scope,
possible missing bottles and other geometry issues remain whole-photo holds.
The original reviewed image hashes and review-file binding are retained.

These Open Images photos mainly broaden clean/drinking scenes. They do not by
themselves resolve the litter-domain coverage gap identified after V34/V36.
No monitor is selected using benchmark predictions. Fresh litter-source staging
is separate; all eventual monitor groups/byte/pixel/near aliases must stay out
of future training. Public-source screens remain heuristic, not independent NHS
testing or a guarantee against overfitting.

Completed combined overlap screen:61twice-reviewed Open Images positives produced58passing parents; indices15,91,97held. No overrides or model-score filtering. Complete source annotation groups, official plastic/water human image labels, and per-photo metadata remain separate admission requirements. V42 prepares admission of these58photos (100boxes), not a claim of improved accuracy.

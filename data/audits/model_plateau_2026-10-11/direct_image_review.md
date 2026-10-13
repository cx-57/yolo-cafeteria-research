# Additional visual QA: dry/wet waste 'napkin' labels

Read-only inspection on 2026-10-11 of committed JPEGs and their existing manifest.

The source manifest assigns current class **5 = napkin** to 34 images
from `data/processed/dry_wet_waste/`, and the entire source is train-only.
The five `paper*` image files below were inspected directly. They show
torn sheets of paper, often with page-like edges and perforations, rather
than confidently identifiable disposable cafeteria napkins. Several appear
clearly incompatible with the project's NHS-style napkin target.

| Filename (in `data/processed/dry_wet_waste/images/`) | Review finding |
| --- | --- |
| `test__paper60_jpg.rf.73d291a8166688bf2c5608b92092418f.jpg` | Torn white sheet with page/perforated-looking edge; not a verified napkin |
| `train__paper191_jpg.rf.a2f073f30f0e628c0f1cb3405c90077f.jpg` | Torn white sheet with a similar edge; not a verified napkin |
| `train__paper528_jpg.rf.671a8f48554d5bdc824dfb5e9a7562e5.jpg` | Torn white paper sheet, no demonstrated napkin context |
| `train__paper538_jpg.rf.c3d5b8043687ba0f72e92a82e44b39aa.jpg` | Crumpled, torn sheet of paper, not a reliably identifiable napkin |
| `train__paper63_jpg.rf.d5f2acae59c3188321f106cb261a9d53.jpg` | Torn paper sheet, not a verified napkin |

This is **semantic class-label quality**, not a numerical bounding-box bug.
Valid `x_center y_center width height` coordinates can still label the wrong
object category. The remaining 29 images assigned napkin from this source
have not all been reviewed individually; do not extrapolate an error rate
from these five.

**Action:** Review all 34 class-5 images from this source at full resolution.
For confirmed non-napkins, quarantine the complete image/label pairs from
active training and update the source manifest. Do not reassign the images
to `napkin` merely because the upstream source says `dry waste`.
There is no configured generic `paper` class, so don't silently relabel
them. Preserve original downloaded source images and audit decisions.

Other directly inspected pictures confirmed the existing report's findings:
LVIS plate photo `train__lvis_000000346262.jpg` visibly contains an
unannotated sandwich and white spoon. Condiment source example
`train__464122_0_jpg.rf.962d58608576425f97087f8885069bc1.jpg`
contains a packet but is isolated on a plain background; usefulness for
natural NHS scenes remains unmeasured. The Tetra Pak example
`-long-life-milk-in-200ml-sizes-and-a-tasteful-refreshing-and-healthy-drink-1_jpg.rf.031f9a8a4b777733bc7f2c7511680568.jpg`
shows three retail cartons on a white background, another domain-shift risk.

No image or training label was changed by this review.

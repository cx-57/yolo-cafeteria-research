# Bottle experiment literature check — 2026-10-23

The higher reported numbers were not systematically checked during the prior
head-blending trials. This review checks original papers, separates datasets
and metrics, and changes the next experiment selection.

| Original paper | Reported result | Relevant method and comparison limits |
| --- | --- | --- |
| [Yasiri et al., 2025](https://jurnal.polgan.ac.id/index.php/sinkron/article/download/14242/2971) | Bottle AP50 89.2%; AP50–95 66.3% | YOLOv5, 100 epochs, batch 16, 460 pixels and train-only augmentation. The abstract says 500 images; methods describe 208 original images. Separate authors' dataset, not our benchmark. |
| [Samak & Tulum, 2026](https://dergipark.org.tr/en/download/article-file/5020373) | Plastic bottle AP50 99%, AP50–95 61% | YOLOv8n, reported 100 epochs, batch 16, learning rate 0.001, 640 pixels, augmentation and ten categories. Table 4 lists 93 bottle-positive evaluation images. Its overall dataset/split counts are inconsistent, and it does not establish a matching capture-group protocol. |
| [Kunwar et al., WaDaBa](https://arxiv.org/pdf/2412.20232) | YOLO11n AP50 99.2% over polymer categories | 4,000 original photos, oversampled to 11,000, 80:20 split and 20 YOLO epochs. PET/PE-HD/PP/PS/Others classification differs from finding disposable beverage bottles amid unrelated objects. Oversampling is described before splitting; capture-group separation cannot be established from this text. Do not infer proven leakage. |
| [Akyon et al., SAHI, ICIP 2022](https://arxiv.org/pdf/2202.06934) | Detector-dependent AP50 gains on VisDrone/xView; no bottle result | Original-resolution overlapping patches, aspect-preserving enlargement, optional full-image prediction, coordinate restoration and NMS. The paper tests 25% overlap and NMS IoU 0.5. This addresses our small-object problem directly, but its gain is not a predicted bottle gain. |

Our current saved ordinary-inference result is 84.0784% AP50 / 60.9019%
AP50–95 on 246 development photos with 67 bottle boxes. These external scores
are numerically higher, but none is a head-to-head evaluation on this split.
They should motivate controlled experiments, not a claim of universal ranking.

The V24 larger-model head trial completed six epochs and freshly scored
68.8293% ordinary / 72.7199% augmented AP50. Increasing model size alone did
not beat the incumbent. Its short, frozen-feature schedule is not a replication
of either 100-epoch recipe. Sustained feature adaptation remains a separate
prepared option; do not present six-epoch screening as full convergence.

Next, test official SAHI on the protected small-model weights. Our earlier
size audit found 51/67 bottle boxes below 4% of photo area; that threshold is
our diagnostic definition, not COCO's small-object category. Slice original
photos at 384 pixels with 25% overlap and retain full-image predictions. Check
coordinates against the actual native loader before using the unchanged
validator. If it fails, retain the ordinary model and proceed to sustained
feature adaptation; do not change validation to reach the target.

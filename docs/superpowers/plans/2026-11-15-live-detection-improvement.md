# Live wrong-label improvement

Request: improve incorrect live object classes, while retaining all seven classes.
Preserve datasets, original checkpoint bytes, frozen research config and published results. No training is authorized by this implementation.

Approach: compare the configured unified YOLO11n at 640/800, its existing parent/pilot alternatives, and the four frozen specialists on the exact same 186-image development manifest. Use class-aware exact matches at confidence .60, class-error matching only after reserving correct matches, AP and recall. The development set is exposed and provisionally annotated; do not claim live or independent accuracy from it.

If the specialists improve wrong-class/precision without losing recall, expose them as an optional persistent live detector. Reuse the existing seven-class name remapping and native resolution settings. Keep the unified checkpoint selectable; do not silently swap it. Measure live latency against the existing freshness watchdog before using the alternate detector. Camera tests and observed class errors remain necessary.

- [x] Run actual single-model comparisons without writing research assets.
- [x] Evaluate frozen specialists on the same cohort and record evidence.
- [x] Add failing tests for seven-class ownership/remapping, provenance and result validity; implement optional specialist inference if warranted.
- [x] Verify actual app inference, model hashes, full tests and camera startup; document speed/accuracy tradeoff.

Ruling: specialists gain recall/F1 and reduce the measured residual class-error count, but aggregate precision drops1.45 percentage points. Keep unified as the default and expose specialists explicitly for live comparison; do not claim a universal accuracy gain. Actual bread/utensil camera verification remains pending.

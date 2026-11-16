# Optional food fallback

The user wants progress on unseen live objects and has repeatedly requested autonomous execution. Add an opt-in detection-only inference policy using the existing parent and frozen food specialist. Preserve the default unified mode, threshold, original weights, annotations and splits.

Keep all parent output when it contains food at the configured display confidence. Otherwise retain its other-six predictions and take food from the specialist. Reuse the existing validated food-owner profile and native class maps; never remap a utensil or bottle to food. Return the normal Results object so existing image display, ROI reprojecting and diagnostics work.

Approaches considered: unconditional food replacement loses an orange scene; another training run requires new authorization and previously regressed other classes. Conditional fallback recovers rice and bowl bread while preserving confident parent food and other-six outputs. It adds inference latency whenever consulted and cannot fix all localization errors. It is an opt-in experimental detection mode, with no hardware operation or automatic default promotion.

Require the exact existing two-owner profile layout, include the routing rule in the model signature, propagate expert failures, and reject manual/autonomous mode before loading models. Check routing at the configured threshold, empty output, model failures, class maps, provenance, CLI exclusion, real external images, fixed186-image metrics and latency. Exposed diagnostics do not establish independent field accuracy.

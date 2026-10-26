# Best verified bottle ensemble

Two frozen models with equal-weight box fusion: **85.3558% mAP50 / 62.1329% mAP50–95** on the unchanged246-photo/67-box development benchmark. Both metrics improved over the84.0711/62.0606single-model baseline. This is an ensemble result, not onecheckpoint's score. No independent cafeteria test has been established.

Exact modelpaths/hashes,640/batch4/MPSsettings,fusionpolicy,benchmarkreceipt andruntimecode are in recipe.json. V51runs both ordinary modelviews on the same input; Ultralytics augment=True dispatches thecustomdualforward and doesnotapply imageflips/scales. Both physical checkpoints remain local and Git-ignored. Original single-model checkpoints are preserved.

Reproduce verified evaluation from repository root (cached scores require exact receipt match):

```bash
YOLO_OFFLINE=true PYTHONPATH="/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts" .venv/bin/python model/bottle-focus/dual-model-fusion-v51/campaign.py
```

This command verifies/replays the evaluation campaign; deployment inference must use the same two-model fusion path. Loading the firstcheckpoint alone doesnotproduce the ensemblescore.

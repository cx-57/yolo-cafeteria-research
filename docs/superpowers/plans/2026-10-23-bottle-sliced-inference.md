# Bottle Sliced Inference Implementation Plan

> Implement inline with superpowers:executing-plans; preserve the ongoing authorized bottle research workflow.

**Goal:** Measure original-resolution sliced inference on the unchanged bottle benchmark.
**Architecture:** Official SAHI handles tiling, prediction and NMS; a thin adapter maps its original-image boxes into the existing validator's letterbox coordinate frame.
**Tech Stack:** Python 3.12, Ultralytics 8.4.175, SAHI 0.12.8, PyTorch/MPS.
**Spec:** model/bottle-focus/sliced-inference-v26/README.md

- [x] Write and run failing independent known-coordinate and real-loader tests for adapter.py.
- [x] Implement the minimum adapter and obtain passing tests with source-byte checks.
- [x] Integrate official SAHI in run.py, retaining exact fixed validation and original-resolution inputs; review before inference.
- [x] Run fresh saved-weight evaluation, verify 246 images/67 boxes and unchanged data/weights, record honest inference configuration and elapsed time.
- [x] Retain the higher verified candidate, update progress/research notes, commit and push.

# Food Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make the verified conditional food fallback available as an explicit detection-only mode.

**Architecture:** Subclass the existing SpecialistDetector to reuse model hash and native-class verification. Keep the original first model result when food is confident; otherwise combine its other-six outputs with specialist food. Reuse the existing live inference wrapper and diagnostics.

**Tech Stack:** Python, Ultralytics, Torch, OpenCV, unittest.

**Spec:** docs/superpowers/specs/2026-11-15-food-fallback-design.md

## Task: Optional inference mode

- [x] Add failing tests for skipping the expert when food is confident; threshold-dependent consultation; preserving other classes; rejecting incompatible profiles; propagating failures; empty output; hardware-mode exclusion and routing provenance.
- [x] Confirm expected red results.
- [x] Implement FoodFallbackDetector and explicit CLI selection in scripts/live_sorter.py. Defaults remain unchanged.
- [x] Run targeted tests, then the full suite. Fix failures before reporting completion.
- [x] Verify actual routing on all22 external photos, use the fixed186-image protocol for food and other-six nonregression, and report warm latency/call frequency.
- [x] Document exact opt-in command, remaining misses, latency cost and no hardware eligibility. Verify original hashes and no background training/camera process.

## Review focus

Confident parent food must never be replaced; threshold comes from config. Reject an incompatible owner layout. Empty detections retain valid tensor shape. No expert class IDs escape the food-only mapping. A new routing rule gets a different signature. Hardware exclusion happens before loading or serial access.

No commits, pushes, retraining or deployment are part of this plan.

Completed: nine fallback tests RED→GREEN; native-map review finding fixed with a failing regression test; full suite222tests,221pass/one existing skip. Actual adapter CPU/MPS parity22photos and full186-image validation passed. Original assets/configs protected; no camera, servo, training or default promotion.

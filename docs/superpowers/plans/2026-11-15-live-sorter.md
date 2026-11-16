# Live sorter implementation plan

Goal: connect an existing verified seven-class checkpoint to camera detection and an Uno sorter without changing research assets.
Architecture: one standalone scripts/live_sorter.py, configs/sorter.json, firmware/waste_sorter/waste_sorter.ino. Standard-library state logic and nonblocking serial polling keep camera updates active during mechanical cycles. Detection is the default; hardware requires explicit mode, reviewed disposal/physical mapping and calibrated ROI. No automatic recovery after faults.
Spec: user-supplied Complete Autonomous YOLO11n Cafeteria Waste Sorter request, 2026-11-15.

User explicitly requested autonomous implementation without a design approval pause. Work in the existing checkout to preserve access to existing untracked research modules; do not alter, stage, commit or push those changes.

- [x] Write and run failing tests for normalized ROI eligibility/occupancy, same-object stability, multi-object ambiguity, manual freshness, repeat latch and fault latch.
- [x] Implement state logic, strict config validation and exact checkpoint class validation; test with unittest.
- [x] Write failing protocol/logging tests; implement persistent serial readiness, calibration check, BUSY/DONE matching, timeout/reset/disconnect fault handling, simulator and CSV outcomes.
- [x] Implement camera/image CLI, bounded camera probing, live annotations and ROI selection, fail-closed cleanup.
- [x] Implement fixed-buffer nonblocking Uno firmware; test actual sketch using a host Servo/Arduino harness, and AVR compile if toolchain is available. No upload or movement from this session.
- [x] Run all repository tests, actual checkpoint image inference, simulated workflow and camera availability checks; document exact commands and physical acceptance checklist.

Review focus: detections below trigger threshold must prevent premature clear/rearm; pending manual approval must be invalidated by changed observations; reset READY must never be mistaken for DONE; uncertain cycles must never retry; camera or logging failures must prevent subsequent commands.

Verification: 187 tests, 1 preexisting skip; Uno compile passed. Camera indices 0/1/2 unavailable and no Uno USB port found. See docs/sorter-verification.md for evidence and physical acceptance tasks.

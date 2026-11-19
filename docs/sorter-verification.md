# Sorter integration verification — 2026-11-15

Software integration is implemented. **Camera-to-physical-bin sorting has
not been demonstrated.** No physical success rate or optimized detection
threshold is claimed.

## Repository findings

The existing system uses four frozen specialists (`scripts/specialists.py`),
local disposal suggestions (`scripts/disposal.py`), and offline evaluation,
replay, and validation scripts. No existing camera/Arduino controller or `.ino`
was found. The checkout already contained unrelated modified/untracked V5
research files; they were left in place without edits, staging or commits.

Dataset definitions in the [archived V2 `data.yaml`](https://github.com/cx-57/yolo-cafeteria-research/blob/fae3d929ef9c1cfa917fa8ec2660f2bfcc6bf37f/experiments/yolo-v2-704/data.yaml) and the external
`data/combined-seven-reviewed-20261031/data.yaml` match the requested seven
classes. Original datasets, labels, splits and trained weights were not edited.

The configured unified checkpoint was found outside the Git repository:

```text
../yolo-cafeteria-research-assets/artifacts/cumulative20-100views-20261031/train/weights/best.pt
SHA256: 19e1294484efe9bbabbc71866ccd57f59942899cc385937dc7aeed55fc373795
```

It loads as scale `n`, `yolo11n.yaml`, with exact IDs:
`food, plate, utensil, hinged_container, drink_carton, wrapper, bottle`.
The hash matched again after verification. This is an existing trained
checkpoint selected for compatibility; it is not claimed to be the most
accurate candidate. The separate combined-pilot checkpoint also has these
seven names. Older `runs/yolo11n-corrected-baseline-100` and
`runs/v2-control-n-reviewed-baseline` checkpoints contain **11 classes**;
the new application's loader correctly rejects them rather than remapping
IDs silently. No generic pretrained model was substituted.

Installed Python 3.12.3 had Ultralytics **8.4.80**, OpenCV **4.13.0**, PyTorch
**2.12.1**, PySerial **3.5**. Research requirements pin Ultralytics **8.4.175**;
they were preserved. Explicit OpenCV and PySerial requirements were added.
Tests ran with the installed packages; a clean install of the pinned
requirements was not performed. Arduino CLI was initially absent; CLI 1.6.0,
AVR core 1.8.8 and Servo 1.3.0 were downloaded under `/tmp/cafeteria-arduino-cli`
for compilation only. No firmware was uploaded.

## Passed software checks

- Full repository command: `python3 -m unittest discover -s tests -v`:
  **187 tests run, 186 passed, 1 skipped**, no failures. The preexisting
  `test_real_bundle_verifies_without_inference_when_present` skips because its
  conversation bundle is unavailable. Baseline before changes: 161 run, 1 skipped.
- New coverage: 25 Python tests plus one C++ firmware harness test. Includes
  empty/low confidence/outside ROI, ROI edges, multiple same/different-class
  boxes, spatial continuity, stale manual approval, repeat latch, clear rearm,
  fault latch, exact names, missing model, invalid geometry, camera failure,
  bounded capture watchdog, latest-frame draining, cleanup despite UI failure,
  slow CSV abort, serial disconnection, malformed/fragmented responses,
  calibration mismatch, missing/late acknowledgements, late completion and
  completion timeout.
- Actual PySerial over a POSIX pseudo-terminal: one persistent connection,
  newline PING/SORT, fragmented READY/BUSY/DONE. This is a software serial test,
  not proof of USB/Uno communication.
- Actual configured model through the new prediction decoder on three existing
  validation images: 1, 1 and 4 detections, with valid finite confidence and
  normalized bounding boxes. Local receipt:
  `runs/sorter/verified-image-predictions.json`. These are inference smoke tests,
  without newly scored accuracy metrics.
- Image CLI produced `runs/sorter/model-smoke.jpg`. No source image was changed.
  CLI also rejected an actual 11-class custom checkpoint.
- `--self-test` ran synthetic observations through state selection, serial
  simulator, all three slots, completion logging, repeat latch and clear rearm.
  Local CSV: `runs/sorter/verification-simulated.csv` (explicitly simulated).
- Actual `.ino` compiled for **arduino:avr:uno**: **4,768 bytes flash / 301 bytes
  global RAM**. Build output: `/tmp/cafeteria-arduino-build.log`.
- The actual sketch compiled and executed in a host C++ clock/Serial/Servo
  harness: safe boot targets, all three sequences, chute-before-opening timing,
  direct opening, incremental closing/return, invalid/overlapping commands,
  overflow, line timeout, CENTER and PING passed. Servo writes are simulated.
- `python3 -m py_compile scripts/live_sorter.py` and `git diff --check` passed.

The host harness required an explicit macOS SDK C++ header path because the
local compiler's default header search could not find `cassert`; the test now
supplies that path on macOS. This did not indicate a firmware compilation bug.

## Unavailable or unverified

Camera indices **0, 1 and 2 failed to open** through AVFoundation in this
session. The exact cause (permissions/session/device availability) is not
established. Built-in live video, rendered overlay, camera identity, ROI
alignment, actual webcam disconnect handling and camera FPS still require an
interactive camera-authorized run. The Logitech was not available.

Serial enumeration found only debug/Bluetooth devices, **no Uno USB device**.
No real PING, invalid-command response, servo cycle, category/angle calibration,
mechanical delivery or physical outcome was verified. A successful compile,
host harness or simulated DONE does not establish physical functionality.

No disposal or physical-bin arrangement was inferred. All seven destinations
and three slots remain unset, and all three verification flags remain false.
The 0.60 confidence / 5 stable frames / 10 clear frames defaults are provisional.
The camera can miss objects, and the Uno has no servo-position/jam feedback.

## Physical acceptance checklist

1. Follow [setup instructions](live-sorter.md); allow camera access in the
   launching application and visually identify each readable camera index.
2. Mount the camera, select/inspect the trapdoor ROI, save normalized coordinates.
3. Verify school disposal rules for all seven classes and physically identify
   slots 1/2/3. Enter mappings and review the three verification flags.
4. Upload the sketch with Python/Serial Monitor closed. Unloaded, verify safe
   boot targets and PING, invalid commands, CENTER, SORT 1/2/3 with completion.
5. Confirm chute reaches each destination before opening, then trapdoor closes
   and chute returns without jamming. Adjust constants and re-upload as needed.
6. In manual mode, place a single object, observe stability and press Space.
   Verify the actual bin and record the outcome under the attempt ID. Test
   empty platform, two objects, low confidence, objects outside/at ROI edge,
   and a leftover object that must not trigger again.
7. Test camera loss and USB loss with an unloaded mechanism. Confirm no new
   commands or automatic recovery. An accepted firmware cycle can continue;
   quitting Python is not an emergency servo-power stop.
8. Only after manual delivery tests pass, explicitly enable autonomous mode and
   repeat using the Logitech index. Collect independent physical outcomes
   before calculating correct-bin success rates.

No GitHub changes were pushed. Implementation and existing research changes
remain uncommitted in the working directory.

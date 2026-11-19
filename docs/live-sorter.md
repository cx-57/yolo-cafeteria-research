# Live camera → Arduino waste sorter

The implementation is [scripts/live_sorter.py](../scripts/live_sorter.py),
[configs/sorter.json](../configs/sorter.json), and
[firmware/waste_sorter/waste_sorter.ino](../firmware/waste_sorter/waste_sorter.ino).
[Verification record](sorter-verification.md) separates software checks from physical acceptance.
Existing offline research scripts and their disposal policy remain independent.

## Python setup

From the repository root, use the existing environment if available:

```sh
source ../yolo-cafeteria-research-assets/venv/bin/activate
python -m pip install -r requirements.txt
```

Or create a separate environment (Python 3.12 recommended):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The inspected system Python had Ultralytics 8.4.80, OpenCV 4.13.0,
PyTorch 2.12.1 and PySerial 3.5; the research requirements pin Ultralytics
8.4.175. Verification used the installed environment without upgrading it.
A fresh install of the pinned environment was not tested here.

## Model and image inference

The default weights path is relative to the **repository root**, including
when using an alternate config file. Inspect or change `weights` in the JSON,
or supply `--weights /absolute/path/to/best.pt` on any inference command.
No checkpoint is downloaded. Missing files and any differing class ID order
abort startup. The loader prints architecture scale, class names and SHA256.

The located custom seven-class YOLO11n checkpoint is:

```text
../yolo-cafeteria-research-assets/artifacts/cumulative20-100views-20261031/train/weights/best.pt
```

It matches IDs 0–6: food, plate, utensil, hinged_container, drink_carton,
wrapper, bottle. Two older `artifacts/runs/*/weights/best.pt` models have
11 classes and are rejected. The four specialist files in `weights/` are
also not substitutes for a unified seven-class checkpoint.

Run image inference without opening a camera or serial port:

```sh
python scripts/live_sorter.py --image /absolute/path/to/photo.jpg \
  --output runs/sorter/photo-annotated.jpg
```

Output files must be new paths; original images cannot be overwritten.
Predictions print as normalized boxes, class IDs, confidence and configured
category. `null` means no disposal rule has been entered. This command
verifies inference, not accuracy. There is no retraining step.

## Camera and ROI calibration

```sh
python scripts/live_sorter.py --probe-cameras 0,1,2
python scripts/live_sorter.py --camera 0
```

The default mode is `detection`; it never opens serial or sends movement
commands. Probe results only establish readability, not camera identity.
After the Logitech arrives, rerun probing and view each readable index:

```sh
python scripts/live_sorter.py --camera 2 --mode detection
```

Here `2` is an example: use the index you actually verified. Indices may
change after USB changes; never assume Logitech is index 1. On macOS allow
the launching app (Terminal, IDE or Codex as applicable) in System Settings →
Privacy & Security → Camera, restart that app, and close competing camera
apps. An absent permission entry may require launching from Terminal first.
Camera indices 0/1/2 were unavailable in this agent session.

Press **r** in detection mode, drag a rectangle around the loading area,
then press Enter/Space to accept; Escape/cancel leaves it unchanged. Copy the
printed `[left, top, right, bottom]` normalized coordinates into `roi` in
`configs/sorter.json`. This does not modify the config automatically.
Check alignment on the actual 8 × 8 cm trapdoor, then set `roi_verified: true`.
Calibrate again after changing camera position, camera index or resolution.
`--roi 0.2 0.2 0.8 0.8` is available for detection experiments; these are
example coordinates, and an override clears ROI verification for that run.

Boxes must be **fully inside** the ROI to become eligible. Any box touching
the ROI interior at or above `occupancy_confidence` blocks empty detection.
Two boxes, including two with the same class or a second low-confidence box,
block sorting. Same-class overlapping boxes are retained when YOLO NMS
retains them; different-class boxes are not suppressed together.

## Existing specialist alternative

The unified custom YOLO11n remains the default. To use the repository's four
frozen custom specialists together, explicitly select:

```sh
python scripts/live_sorter.py --detector specialists --mode detection --camera 0
```

This preserves all seven output class IDs. Native specialist IDs are remapped
using the existing research helper; every checkpoint hash must match
`configs/specialists.json`. All four models stay loaded. The utensil specialist
retains its native 800-pixel setting; the others use the configured image size.
This takes more computation than the unified model and must satisfy the same
frame-freshness watchdog. `--weights` applies only to `--detector unified`.
The CSV model signature in specialist mode hashes the full model specifications,
including all verified checkpoint hashes and per-model settings.

Visible labels now use the configured `confidence` threshold (default 0.60).
Predictions down to `occupancy_confidence` (0.15) still block ambiguous sorting
or premature rearming, even when hidden from the display. Filtering labels does
not retrain a model or prove improved recognition. See
[live detector comparison](https://github.com/cx-57/yolo-cafeteria-research/blob/36e302f71e4d66faa5f8897bbee7993a53f776ae/docs/live-detection-comparison.md) for measured development
results and limitations.

## Optional fixed-ROI zoom and diagnostic snapshots

Full-frame inference remains the default. To give smaller platform objects
more of the model input, select the **entire fixed loading area** with **r**,
then press **z** to toggle ROI inference. Do this calibration once before the
objects are placed; do not draw a new tight crop around each test item. The
same model and confidence scores are used. Boxes return to full-camera
coordinates, and objects cut by the crop edge cannot trigger sorting.

For a saved, verified ROI, `--crop-roi` explicitly enables this inference path.
It does not recover detail absent from the sensor and has not established
real-crumb accuracy. [Robustness session evidence](live-robustness-session.md)
records the rejected tiling experiment and the synthetic scale check.

In detection mode, **s** saves a raw PNG and JSON under
`runs/sorter/camera-diagnostics/`. `--capture-dir` changes that location;
`--capture-group` identifies one physical capture group. Keep related frames
and views in one group if later preparing data splits. Ground truth remains
unset; model predictions are not annotations. Snapshots include model/config
hashes, native model sizes, raw/visible predictions, frame age and block reason.

Camera cold startup uses a separate bounded allowance, and inference warmup
finishes before any serial connection. On macOS the camera runs in a separate
process and exposure settles before frames are published. Detection-only slow
frames are marked STALE and reset stability; hardware modes retain their strict
freshness cutoff. Optional operator-only `--signal-snapshot` permits one
SIGUSR1 snapshot request; it is rejected outside detection-only camera mode.
It is not needed for normal keyboard operation.

## Disposal rules and physical bins

All seven class mappings and all three category-to-slot assignments start
as `null`. Enter rules you have checked against your school policy:

- `class_to_bin`: each exact class name → `Recycle`, `Compost` or `Trash`.
- `bin_to_position`: each category → a distinct slot number `1`, `2` or `3`.
- Slot angles are `chute_angles: [60, 90, 120]`, indexed by slot number.

The model class ID, disposal category and physical slot are different values.
Identify the actual bins physically; no category has been assigned to a
side. The existing `configs/disposal.json` contains earlier experimental
suggestions, including manual review for several classes; it is not evidence
that all seven classes have a verified motor destination. Do not copy those
suggestions blindly. Leave unsupported classes unset and stay in detection
mode until rules are resolved.

Before real manual or autonomous operation, all seven destinations and three
unique slots must be entered. Set `disposal_verified`,
`physical_bins_verified` and `roi_verified` to `true` only after those checks.
Flags are operator attestations; software cannot verify material or policy.
Any config change requires reviewing the relevant attestation again.
An alternate config can be passed with `--config /path/to/sorter.json`.

## Arduino setup and first mechanical tests

Wire trapdoor signal to **D9**, chute signal to **D10**. Servo power comes
from the external regulated 5 V supply; join servo, supply and Arduino
**grounds**. The firmware commands the closed and center targets on boot,
which can physically move servos, but never runs a sorting cycle on reset.

Install Arduino IDE or Arduino CLI. With CLI on PATH:

```sh
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli lib install Servo
arduino-cli board list
arduino-cli compile --fqbn arduino:avr:uno firmware/waste_sorter
```

Quit Python and close Serial Monitor before uploading so the port is free:

```sh
arduino-cli upload --fqbn arduino:avr:uno \
  --port /dev/cu.usbmodemYOUR_DEVICE firmware/waste_sorter
```

Replace the example port with the current Uno device. Arduino IDE alternative:
open the `.ino`, select Arduino Uno and its USB port, then Upload.
Test unloaded first, using Serial Monitor with **9600 baud / newline**:

```text
PING
CENTER
SORT 1
SORT 2
SORT 3
```

Wait for `DONE` after each command. Check each slot physically and keep hands
clear of the mechanism. `PING` returns `READY 60 90 120`. Sort replies are
`BUSY SORT n` then `DONE SORT n`. Center replies `BUSY CENTER` then
`DONE CENTER`. Unknown, overlapping, overlong and timed-out commands return
`ERROR ...`. A line timeout discards input through the next newline; send
an extra newline before retrying a malformed partial line. Python does not
retry uncertain commands. Close Serial Monitor before opening Python.

Sequence: move chute one degree per 25 ms → settle 1000 ms → command
trapdoor directly to 160° → allow 300 ms opening travel plus 1500 ms hold →
close to 90° one degree per 25 ms → settle 1000 ms → return chute to 90° →
`DONE`. The nonblocking `millis()` scheduler keeps reading serial while busy.
`servo.write` and `DONE` describe commanded targets; there are no position
sensors confirming movement. Mechanical jams are not detected.

Edit pin/angle/timing constants at the top of the sketch. If changing chute
angles, also change `chute_angles` in JSON; the Python handshake rejects a
mismatch. Recompile/upload after firmware edits. If increasing cycle time,
increase `serial_cycle_seconds` beyond the measured worst case. Never use
timing changes to mask a communication fault.

## Manual, autonomous and simulated operation

List ports without model/camera startup:

```sh
python scripts/live_sorter.py --list-ports
```

After the three configuration attestations and firmware/slot tests:

```sh
python scripts/live_sorter.py --mode manual --camera 0 \
  --port /dev/cu.usbmodemYOUR_DEVICE
```

Place one object on the platform. Wait for `READY_TO_SORT`; press **Space**
to approve the current stable prediction. Changing/ambiguous observations
revoke readiness. Check the entire physical delivery and log its outcome.
Only after manual end-to-end tests succeed, enable autonomous mode explicitly:

```sh
python scripts/live_sorter.py --mode autonomous --camera 0 \
  --port /dev/cu.usbmodemYOUR_DEVICE
```

Change `--camera` to the verified Logitech index for both commands. There is
one persistent 9600-baud connection. Python waits for Uno reset, performs a
fresh PING and checks angles. There is no automatic reconnect or fault resume.

Software-only integration test (no model, camera or USB needed):

```sh
python scripts/live_sorter.py --self-test --log runs/sorter/simulated.csv
```

This uses synthetic observations and explicit fixture mappings for all three
positions. It does not verify disposal rules or physical movement. For live
camera with simulated serial, use a mapped config and:

```sh
python scripts/live_sorter.py --mode autonomous --simulate --camera 0
```

Simulation never opens the configured hardware port. Only real hardware modes
require the three attestations; simulated operation still needs ROI and
class/slot mappings to select a destination. Simulated rows are marked.

Press **q** or close the window to stop. Ctrl+C is also handled. New commands
stop immediately; Python gives an already accepted cycle a bounded chance
to finish and records its response before closing USB. A camera or serial
failure exits and requires a deliberate restart. A blocked camera driver is
isolated from serial supervision. **q is not a physical emergency stop**:
the Uno completes an accepted cycle independently, and closing/reopening USB
can reset it. Cut servo power if an urgent physical stop is necessary.

## Adjust detection and measure outcomes

Defaults are starting values, not optimized thresholds:

| Setting | Default | Purpose |
| --- | --- | --- |
| `confidence` / `--confidence` | 0.60 | Minimum confidence to sort |
| `occupancy_confidence` | 0.15 | Lower threshold blocking clear/rearm and ambiguity |
| `stable_frames` / `--stable-frames` | 5 | Consecutive matching observations |
| `clear_frames` / `--clear-frames` | 10 | Consecutive clear observations after completion |
| `tracking_iou` | 0.4 | Spatial overlap required between consecutive boxes |
| `imgsz` / `--imgsz` | 640 | Inference resolution; positive multiple of 32 |
| `max_frame_seconds` | 2.0 | Acquisition/decision watchdog |
| `camera` / `--camera` | 0 | Verified camera index |
| `serial_ack_seconds` | 2.0 | BUSY acknowledgement deadline |
| `serial_cycle_seconds` | 20.0 | Overall cycle deadline |

Live display shows boxes/classes/confidences, ROI, selected bin, state,
status and processing FPS. `--device mps` can be tested on a compatible Mac;
CPU is the default. Camera capture continuously drains to a single latest
frame, with no application queue. Timestamps measure read completion,
not sensor exposure. Objects the detector misses, occlusions, confidence
below 0.15 and mechanical faults cannot be ruled out by this logic. Inspect
real clear/rearm behavior; the camera does not measure physical bin delivery.

`IDLE → DETECTING → READY_TO_SORT → SORTING → WAITING_FOR_CLEAR → IDLE`.
`FAULT` exits without more commands. Only one object can be eligible; after
sorting, detections cannot trigger again until the ROI clears. Sorts are
never triggered by one frame with the default stability setting.

CSV defaults to `runs/sorter/attempts.csv`. Each attempt has a `STARTED` row
flushed before the command and a `COMPLETED` or `ERROR` row with the same ID.
Rows contain UTC timestamp, class, confidence, intended bin, slot/angle,
stable-frame count, host cycle duration, Arduino response, manual flag,
mode/simulation flag and model/config hashes. Duration includes serial
acknowledgement and host polling; it is not sensor-measured actuator time.
A crash may leave a lone STARTED row; its physical result is unknown.

After completion, keys **1** Recycle, **2** Compost, **3** Trash, **f** Failed
and **u** Unknown append an OUTCOME for the displayed last attempt. Or later:

```sh
python scripts/live_sorter.py --record-outcome ATTEMPT_ID \
  --outcome Compost --notes 'Observed physical delivery'
```

Use `--log /path/to/attempts.csv` when needed. Outcome entries are append-only;
the latest OUTCOME for an ID is its current observation. Join by attempt ID
when analyzing results; do not count STARTED/COMPLETED as two attempts.
Exclude simulation rows and unverified/unknown outcomes from a claimed
physical success rate. No physical success rate has been measured here.

## Optional food fallback

`python3 scripts/live_sorter.py --detector food-fallback --mode detection --device mps`

Keeps confident original food and all other-six predictions; consults the existing food specialist when food is missing. Experimental, detection-only, and slower when consulted. The default remains unified. See [measured results and limitations](https://github.com/cx-57/yolo-cafeteria-research/blob/36e302f71e4d66faa5f8897bbee7993a53f776ae/docs/food-fallback.md); individual crumb localization and some cutlery/wrapper misses remain unresolved.

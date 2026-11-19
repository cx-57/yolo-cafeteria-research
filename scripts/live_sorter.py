"""Live seven-class sorter. Hardware imports are deferred so logic tests run offline."""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['food', 'plate', 'utensil', 'hinged_container', 'drink_carton', 'wrapper', 'bottle']
BINS = {'Recycle', 'Compost', 'Trash'}
LOG = logging.getLogger('sorter')


def verify_names(names):
    actual = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if actual != NAMES:
        raise ValueError(f'Custom checkpoint must have exact IDs {dict(enumerate(NAMES))}; got {names}')


def valid_roi(roi):
    return (isinstance(roi, (list, tuple)) and len(roi) == 4
            and all(type(v) in (float, int) and math.isfinite(v) for v in roi)
            and 0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)


def validate_config(c, hardware=False):
    for key, default in (('camera_startup_seconds', 8.), ('detector_warmup_seconds', 30.)):
        value = c.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('confidence', 'occupancy_confidence', 'tracking_iou'):
        if type(c[key]) not in (float, int) or not math.isfinite(c[key]) or not 0 < c[key] <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    if c['occupancy_confidence'] > c['confidence']:
        raise ValueError('Occupancy threshold must not exceed trigger confidence')
    for key in ('stable_frames', 'clear_frames', 'imgsz'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['imgsz'] % 32:
        raise ValueError('imgsz must be a multiple of 32')
    for key in ('max_frame_seconds', 'serial_ready_seconds', 'serial_ack_seconds', 'serial_cycle_seconds'):
        if type(c[key]) not in (float, int) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if c['roi'] is not None and not valid_roi(c['roi']):
        raise ValueError('ROI must be normalized [left, top, right, bottom] within [0, 1]')
    if set(c['class_to_bin']) != set(NAMES) or any(v not in BINS | {None} for v in c['class_to_bin'].values()):
        raise ValueError('class_to_bin must explicitly contain all seven classes; values are Recycle/Compost/Trash/null')
    positions = c['bin_to_position']
    if set(positions) != BINS or any(v is not None and (type(v) is not int or v not in (1, 2, 3)) for v in positions.values()):
        raise ValueError('bin_to_position must map Recycle/Compost/Trash to 1/2/3/null')
    angles = c['chute_angles']
    if (not isinstance(angles, list) or len(angles) != 3
            or any(type(a) is not int or not 0 <= a <= 180 for a in angles)
            or len(set(angles)) != 3):
        raise ValueError('chute_angles must contain three distinct integer servo angles')
    if type(c.get('crop_roi', False)) is not bool or (c.get('crop_roi') and c['roi'] is None):
        raise ValueError('ROI crop requires crop_roi boolean and an explicit valid ROI')
    if hardware:
        if not all(c.get(k) is True for k in ('roi_verified', 'disposal_verified', 'physical_bins_verified')):
            raise ValueError('Review ROI, disposal rules and physical bins; set their three verified flags true')
        if c['roi'] is None or None in c['class_to_bin'].values() or set(positions.values()) != {1, 2, 3}:
            raise ValueError('Hardware requires calibrated ROI, seven explicit destinations and three distinct physical bins')


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple
    edge_clipped: bool = False

    def validate(self):
        if (type(self.edge_clipped) is not bool or type(self.class_id) is not int or not 0 <= self.class_id < 7
                or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
                or not valid_roi(self.box)):
            raise ValueError(f'Invalid prediction: {self}')


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def contained(a, b):
    return b[0] <= a[0] and b[1] <= a[1] and a[2] <= b[2] and a[3] <= b[3]


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)


class SortingMachine:
    def __init__(self, config):
        self.c = config
        self.state = 'IDLE'
        self.streak = self.clear = 0
        self.candidate = None
        self.message = 'Waiting for an object'
        self.blocking_reason = 'empty'
        self.occupant_count = 0

    def observe(self, detections, stale=False):
        for d in detections:
            d.validate()
        if self.state in ('FAULT', 'SORTING'):
            return
        roi = self.c['roi']
        if stale:
            self.streak = self.clear = 0
            self.candidate = None
            if self.state != 'WAITING_FOR_CLEAR':
                self.state = 'DETECTING'
            self.occupant_count = sum(d.confidence >= self.c['occupancy_confidence']
                                      and intersects(d.box, roi) for d in detections) if roi else 0
            self.blocking_reason = 'stale_frame'
            self.message = 'STALE: diagnostic only; stability reset'
            return
        if roi is None:
            self.blocking_reason = 'roi_unconfigured'
            self.occupant_count = 0
            self.message = 'ROI unconfigured: press r in detection mode'
            return
        occupants = [d for d in detections if d.confidence >= self.c['occupancy_confidence'] and intersects(d.box, roi)]
        self.occupant_count = len(occupants)
        if self.state == 'WAITING_FOR_CLEAR':
            self.blocking_reason = 'waiting_for_clear'
            self.clear = 0 if occupants else self.clear + 1
            if self.clear >= self.c['clear_frames']:
                self.state = 'IDLE'
                self.blocking_reason = 'empty'
                self.message = 'Platform clear; rearmed'
            return
        eligible = (len(occupants) == 1 and occupants[0].confidence >= self.c['confidence']
                    and contained(occupants[0].box, roi) and not occupants[0].edge_clipped)
        if not eligible:
            self.state = 'DETECTING' if occupants else 'IDLE'
            self.streak = 0
            self.candidate = None
            if len(occupants) > 1:
                self.blocking_reason = 'multiple_occupants'
                self.message = f'Multiple ROI occupants: {len(occupants)} (including low confidence boxes)'
            elif occupants and occupants[0].confidence < self.c['confidence']:
                self.blocking_reason = 'low_confidence'
                self.message = 'ROI occupant below trigger confidence'
            elif occupants:
                self.blocking_reason = 'roi_edge'
                self.message = 'Object crosses ROI edge or is cut by the inference crop'
            else:
                self.blocking_reason = 'empty'
                self.message = 'Platform empty'
            return
        d = occupants[0]
        same = self.candidate and self.candidate.class_id == d.class_id and iou(self.candidate.box, d.box) >= self.c['tracking_iou']
        self.streak = self.streak + 1 if same else 1
        self.candidate = d
        self.state = 'READY_TO_SORT' if self.streak >= self.c['stable_frames'] else 'DETECTING'
        self.blocking_reason = None if self.state == 'READY_TO_SORT' else 'stabilizing'
        self.message = f'Stable frames: {self.streak}/{self.c["stable_frames"]}'

    def begin_sort(self):
        if self.state != 'READY_TO_SORT':
            return None
        destination = self.c['class_to_bin'][NAMES[self.candidate.class_id]]
        position = self.c['bin_to_position'].get(destination)
        if destination is None or position is None:
            self.blocking_reason = 'unmapped_destination'
            self.message = 'Unmapped class/bin; review configuration'
            return None
        self.state = 'SORTING'
        self.blocking_reason = 'sorting'
        return self.candidate, destination, position, self.streak

    def completed(self):
        self.state = 'WAITING_FOR_CLEAR'
        self.blocking_reason = 'waiting_for_clear'
        self.clear = self.streak = 0
        self.candidate = None
        self.message = 'Cycle complete; remove any remaining object'

    def fail(self, message):
        self.state = 'FAULT'
        self.blocking_reason = 'fault'
        self.message = str(message) + ' — restart required'
        LOG.error(self.message)


class SimulatedArduino:
    """Protocol simulator only; never represents physical servo verification."""
    def __init__(self, clock=time.monotonic, duration=7.5, angles=(60, 90, 120)):
        self.clock, self.duration, self.angles = clock, duration, angles
        self.buffer = bytearray()
        self.finish_at = None
        self.position = None
        self.is_open = True

    @property
    def in_waiting(self):
        if not self.is_open:
            raise OSError('Simulated serial disconnected')
        if self.finish_at is not None and self.clock() >= self.finish_at:
            self.buffer.extend(f'DONE SORT {self.position}\n'.encode())
            self.finish_at = None
        return len(self.buffer)

    def read(self, count):
        data = bytes(self.buffer[:count])
        del self.buffer[:count]
        return data

    def write(self, data):
        command = data.decode().strip()
        if self.finish_at is not None:
            reply = 'ERROR BUSY'
        elif command == 'PING':
            reply = 'READY ' + ' '.join(map(str, self.angles))
        elif command in ('SORT 1', 'SORT 2', 'SORT 3'):
            self.position = int(command[-1])
            self.finish_at = self.clock() + self.duration
            reply = f'BUSY SORT {self.position}'
        else:
            reply = 'ERROR UNKNOWN'
        self.buffer.extend((reply + '\n').encode())
        return len(data)

    def close(self):
        self.is_open = False


class ArduinoLink:
    """One connection; one outstanding command; any uncertain response latches fault."""
    def __init__(self, transport, config, clock=time.monotonic):
        self.transport, self.c, self.clock = transport, config, clock
        self.buffer = bytearray()
        self.pending = None
        self.ack = False
        self.ready = self.fault = False
        self.started = 0.

    def _lines(self):
        waiting = self.transport.in_waiting
        if waiting > 1024:
            raise RuntimeError('Serial input flood')
        self.buffer.extend(self.transport.read(waiting))
        lines = []
        while b'\n' in self.buffer:
            raw, _, rest = self.buffer.partition(b'\n')
            self.buffer[:] = rest
            if len(raw) > 80:
                raise RuntimeError('Oversized serial response')
            line = raw.decode('ascii').strip()
            if line:
                LOG.info('Arduino: %s', line)
                lines.append(line)
        if len(self.buffer) > 80:
            raise RuntimeError('Unterminated serial response')
        return lines

    def _write(self, command):
        payload = (command + '\n').encode('ascii')
        LOG.info('Sending: %s', command)
        if self.transport.write(payload) != len(payload):
            raise OSError('Incomplete serial write')

    def handshake(self):
        try:
            self._write('PING')
            deadline = self.clock() + self.c['serial_ready_seconds']
            expected = 'READY ' + ' '.join(map(str, self.c['chute_angles']))
            while self.clock() < deadline:
                lines = self._lines()
                if lines:
                    if lines != [expected]:
                        raise RuntimeError(f'Expected {expected!r}; received {lines!r}. Check firmware/calibration.')
                    self.ready = True
                    return
                time.sleep(.01)
            raise TimeoutError('Arduino readiness timeout')
        except Exception:
            self.fault = True
            raise

    def start(self, position):
        if self.fault or not self.ready or self.pending is not None:
            raise RuntimeError('Arduino unavailable, faulted or busy')
        if type(position) is not int or position not in (1, 2, 3):
            raise ValueError('Position must be 1, 2 or 3')
        try:
            if self._lines() or self.buffer:
                raise RuntimeError('Unsolicited serial data before command; possible reset')
            self.pending = position
            self.ack = False
            self.started = self.clock()
            self._write(f'SORT {position}')
        except Exception:
            self.fault = True
            raise

    def poll(self):
        try:
            if self.fault:
                raise RuntimeError('Arduino fault is latched; restart required')
            # A late buffered response cannot erase an expired deadline.
            if self.pending is not None:
                elapsed = self.clock() - self.started
                if not self.ack and elapsed > self.c['serial_ack_seconds']:
                    raise TimeoutError('Arduino acknowledgement timeout; cycle outcome unknown')
                if elapsed > self.c['serial_cycle_seconds']:
                    raise TimeoutError('Arduino cycle timeout; cycle outcome unknown')
            response = None
            for line in self._lines():
                if self.pending is not None and not self.ack and line == f'BUSY SORT {self.pending}':
                    self.ack = True
                elif self.pending is not None and self.ack and line == f'DONE SORT {self.pending}':
                    response = line
                    self.pending = None
                else:
                    raise RuntimeError(f'Unexpected Arduino response: {line}; no automatic retry')
            return response
        except Exception:
            self.fault = True
            raise

    def close(self):
        self.transport.close()


class AttemptLog:
    """Append-only CSV: STARTED precedes motion; COMPLETED/ERROR and OUTCOME share an ID."""
    fields = ['attempt_id', 'timestamp', 'event', 'predicted_class', 'confidence', 'intended_bin',
              'position', 'chute_angle', 'stable_frames', 'cycle_seconds', 'response',
              'manual_confirmation', 'mode', 'simulated', 'weights_sha256', 'config_sha256',
              'actual_outcome', 'notes']

    def __init__(self, path, metadata):
        self.path = Path(path)
        self.metadata = metadata
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline='') as f:
                if next(csv.reader(f), []) != self.fields:
                    raise ValueError('Existing CSV has an incompatible header')
        else:
            self._append(None)

    def _append(self, row):
        import os
        with self.path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if row is None:
                writer.writeheader()
            else:
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def begin(self, selection, config, manual):
        d, destination, position, streak = selection
        row = dict(self.metadata, attempt_id=uuid.uuid4().hex[:12],
                   timestamp=datetime.now(timezone.utc).isoformat(), event='STARTED',
                   predicted_class=NAMES[d.class_id], confidence=f'{d.confidence:.6f}',
                   intended_bin=destination, position=position, chute_angle=config['chute_angles'][position-1],
                   stable_frames=streak, manual_confirmation=manual)
        self._append(row)
        return row

    def finish(self, attempt, duration, response, error=False):
        self._append(dict(attempt, timestamp=datetime.now(timezone.utc).isoformat(),
                          event='ERROR' if error else 'COMPLETED', cycle_seconds=f'{duration:.3f}', response=response))

    def outcome(self, attempt_id, outcome, notes=''):
        if outcome not in BINS | {'Failed', 'Unknown'}:
            raise ValueError('Outcome must be Recycle, Compost, Trash, Failed or Unknown')
        with self.path.open(newline='') as f:
            if not any(r['attempt_id'] == attempt_id and r['event'] == 'STARTED' for r in csv.DictReader(f)):
                raise ValueError('Unknown attempt ID')
        self._append(dict(attempt_id=attempt_id, timestamp=datetime.now(timezone.utc).isoformat(),
                          event='OUTCOME', actual_outcome=outcome, notes=notes))


def resolve_path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_model(weights):
    path = resolve_path(weights)
    if not path.is_file():
        raise FileNotFoundError(f'Custom weights not found: {path}. Supply --weights /absolute/path/best.pt; no model will be downloaded.')
    from ultralytics import YOLO
    model = YOLO(str(path))
    verify_names(model.names)
    model.checkpoint_sha256 = file_sha(path)
    LOG.info('Loaded %s; scale=%s; SHA256=%s; names=%s', path, model.model.yaml.get('scale'), file_sha(path), model.names)
    return model


LEGACY_UTENSIL_SHA256 = 'ce4956547f0c421db1dc4b3d2fa7c28d9d9ae2fc44ddf5acf333ff211c3d4c7f'


def adapt_legacy_pooling(model, checkpoint_sha256):
    if checkpoint_sha256 != LEGACY_UTENSIL_SHA256:
        return 0
    import importlib.util
    spec = importlib.util.spec_from_file_location('live_legacy_pooling', ROOT/'experiments/yolo-v4-4class/v4_hybrid_inference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.repair_legacy_head_pooling(model)


class SpecialistDetector:
    """Keep existing custom detectors loaded with one explicit owner per class."""
    def __init__(self, config_path=ROOT/'configs/specialists.json'):
        from ultralytics import YOLO
        from specialists import map_boxes, inference_settings
        specs = json.loads(Path(config_path).read_text())['models']
        if sorted(name for spec in specs for name in spec['classes']) != sorted(NAMES):
            raise ValueError('Exactly one owner is required for each of the seven classes')
        # Check all files before loading any model. Never download replacements.
        for spec in specs:
            native = spec.get('source_classes', spec['classes'])
            if (not isinstance(native, list) or len(set(native)) != len(native)
                    or not set(native) <= set(NAMES) or not set(spec['classes']) <= set(native)):
                raise ValueError('Invalid native checkpoint class scope or ownership')
            inference_settings({}, spec)  # validate allowed keys and native resolution
            path = resolve_path(spec['weights'])
            if not path.is_file() or file_sha(path) != spec['sha256']:
                raise ValueError(f'Specialist checkpoint missing or hash changed: {path}')
        self.models = []
        for spec in specs:
            model = YOLO(str(resolve_path(spec['weights'])))
            adapt_legacy_pooling(model, spec['sha256'])
            native = spec.get('source_classes', spec['classes'])
            ordered = [model.names[i] for i in range(len(model.names))]
            if ordered != native:
                raise ValueError(f'Native checkpoint class order differs: {ordered} versus {native}')
            map_boxes([], model.names, native)
            self.models.append((spec, model))
        self.names = dict(enumerate(NAMES))
        self.checkpoint_sha256 = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
        LOG.info('Loaded %d existing custom detectors; combined signature=%s', len(self.models), self.checkpoint_sha256)

    def predict(self, frame, config, device):
        import torch
        from ultralytics.engine.results import Results
        from specialists import map_boxes
        rows = []
        for spec, model in self.models:
            result = model.predict(frame, imgsz=spec.get('settings', {}).get('imgsz', config['imgsz']),
                                   conf=config['occupancy_confidence'], iou=.7, max_det=300,
                                   device=device, verbose=False)[0]
            native_rows = result.boxes.data.cpu().tolist()
            height, width = frame.shape[:2]
            decode_detections(native_rows, width, height)  # fail before map_boxes can coerce/drop IDs
            if any(row[5] >= len(model.names) for row in native_rows):
                raise ValueError('Invalid native specialist class ID')
            mapped = map_boxes(native_rows, model.names, spec.get('source_classes', spec['classes']))
            owned = {NAMES.index(name) for name in spec['classes']}
            rows.extend(row for row in mapped if int(row[5]) in owned)
        return Results(frame, path='live-specialists', names=self.names,
                       boxes=torch.tensor(rows, dtype=torch.float32).reshape(-1, 6))


class FoodFallbackDetector(SpecialistDetector):
    """Detection-only candidate; consult frozen food expert when primary food is missing."""
    routing_policy = 'primary_food_else_expert_v1'

    def __init__(self, config_path=ROOT/'configs/food-expert-candidate.json'):
        specs = json.loads(Path(config_path).read_text()).get('models', [])
        if (len(specs) != 2
                or specs[0].get('source_classes') != NAMES
                or specs[0].get('classes') != NAMES[1:]
                or specs[1].get('classes') != ['food']):
            raise ValueError('Incompatible food fallback profile: require unified other-six then food expert')
        super().__init__(config_path)
        self.checkpoint_sha256 = hashlib.sha256(
            (self.checkpoint_sha256 + ':' + self.routing_policy).encode()).hexdigest()

    def predict(self, frame, config, device):
        import torch
        from ultralytics.engine.results import Results
        from specialists import map_boxes
        primary_spec, primary = self.models[0]
        food_spec, food = self.models[1]
        settings = dict(conf=config['occupancy_confidence'], iou=.7, max_det=300,
                        device=device, verbose=False)
        result = primary.predict(frame, imgsz=primary_spec.get('settings', {}).get('imgsz', config['imgsz']),
                                 **settings)[0]
        rows = result.boxes.data.cpu().tolist()
        height, width = frame.shape[:2]
        decode_detections(rows, width, height)
        if any(row[5] == 0 and row[4] >= config['confidence'] for row in rows):
            return result
        expert = food.predict(frame, imgsz=food_spec.get('settings', {}).get('imgsz', config['imgsz']),
                              **settings)[0]
        food_rows = expert.boxes.data.cpu().tolist()
        decode_detections(food_rows, width, height)
        if any(row[5] >= len(food.names) for row in food_rows):
            raise ValueError('Invalid native food expert class ID')
        mapped_food = map_boxes(food_rows, food.names, food_spec['source_classes'])
        combined = [row for row in rows if row[5] != 0] + [row for row in mapped_food if row[5] == 0]
        return Results(frame, path='food-fallback', names=self.names,
                       boxes=torch.tensor(combined, dtype=torch.float32).reshape(-1, 6))


def visible_predictions(result, confidence):
    # Preserve the original lower-confidence boxes for occupancy/ambiguity logic.
    return result[result.boxes.conf >= confidence]


def decode_detections(rows, width, height):
    detections = []
    for row in rows:
        if len(row) != 6 or not all(math.isfinite(v) for v in row) or row[5] != int(row[5]):
            raise ValueError('Malformed YOLO prediction')
        x1, y1, x2, y2, conf, cls = row
        d = Detection(int(cls), float(conf), (x1/width, y1/height, x2/width, y2/height))
        d.validate()
        detections.append(d)
    return detections


def predict(model, frame, config, device):
    height, width = frame.shape[:2]
    source = frame
    crop = config.get('crop_roi', False)
    left = top = 0
    if crop:
        if not valid_roi(config['roi']):
            raise ValueError('Configure a fixed loading-area ROI before enabling ROI inference')
        x1, y1, x2, y2 = config['roi']
        left, top, right, bottom = round(x1*width), round(y1*height), round(x2*width), round(y2*height)
        if right-left < 4 or bottom-top < 4:
            raise ValueError('Inference ROI must be at least four pixels in each dimension')
        source = frame[top:bottom, left:right]
    if isinstance(model, SpecialistDetector):
        result = model.predict(source, config, device)
    else:
        result = model.predict(source, imgsz=config['imgsz'], conf=config['occupancy_confidence'],
                               iou=.7, agnostic_nms=False, device=device, verbose=False)[0]
    rows = result.boxes.data.cpu().tolist()
    source_h, source_w = source.shape[:2]
    decode_detections(rows, source_w, source_h)  # validate before translating any geometry
    clipped = [crop and (r[0] <= 2 or r[1] <= 2 or r[2] >= source_w-2 or r[3] >= source_h-2) for r in rows]
    global_rows = [[r[0]+left, r[1]+top, r[2]+left, r[3]+top, r[4], r[5]] for r in rows]
    detections = [Detection(d.class_id, d.confidence, d.box, bool(edge))
                  for d, edge in zip(decode_detections(global_rows, width, height), clipped)]
    if crop:
        import torch
        from ultralytics.engine.results import Results
        result = Results(frame, path='roi-inference', names=dict(enumerate(NAMES)),
                         boxes=torch.tensor(global_rows, dtype=torch.float32).reshape(-1, 6))
    return result, detections


def open_camera(index, cv):
    if sys.platform == 'darwin' and hasattr(cv, 'CAP_AVFOUNDATION'):
        cap = cv.VideoCapture(index, cv.CAP_AVFOUNDATION)
    else:
        cap = cv.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Camera {index} unavailable. Check index, USB connection, and macOS System Settings > Privacy & Security > Camera for the launching app; restart it after permission changes.')
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f'Camera {index} opened but frame read failed; check permissions or another app using it')
        return cap, frame
    except BaseException:
        cap.release()
        raise


class ThreadCameraStream:
    """Continuously drain the camera into one latest frame; never queue old scenes.

    The worker owns capture/release. A stuck backend cannot block the control
    loop. Timestamp is acquisition completion, not a hardware exposure timestamp.
    """
    def __init__(self, index, cv):
        import threading
        self.index, self.cv = index, cv
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        cap = None
        try:
            cap, frame = open_camera(self.index, self.cv)
            sequence = 0
            while not self.stop.is_set():
                sequence += 1
                with self.condition:
                    self.latest = (sequence, time.monotonic(), frame)
                    self.condition.notify_all()
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError('Camera disconnected or frame read failed')
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception as exc:
                    LOG.error('Camera release error: %s', exc)

    def get(self, after_sequence, timeout):
        with self.condition:
            self.condition.wait_for(lambda: self.error or (self.latest and self.latest[0] > after_sequence), timeout)
            if self.error:
                raise RuntimeError(str(self.error))
            return self.latest if self.latest and self.latest[0] > after_sequence else None

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.5)
        if self.thread.is_alive():
            LOG.warning('Camera backend is blocked; capture worker will end when the process exits')


def _settled_camera_source(index):
    import cv2 as cv
    cap, frame = open_camera(index, cv)
    try:
        # Initial AVFoundation frames can be nearly black while exposure settles.
        for _ in range(45):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError('Camera failed while exposure was settling')
        return cap, frame
    except BaseException:
        cap.release()
        raise


def _camera_process_worker(index, connection, source_factory):
    """Camera-only process: do not import Torch or load a detector here."""
    cap = None
    try:
        cap, frame = source_factory(index)
        sequence = 0
        while True:
            sequence += 1
            captured_at = time.monotonic()
            if connection.poll():
                command = connection.recv()
                if command == 'STOP':
                    break
                if command != 'FRAME':
                    raise RuntimeError('Invalid internal camera request')
                connection.send(('FRAME', sequence, captured_at, frame))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError('Camera disconnected or frame read failed')
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send(('ERROR', str(exc)))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if cap is not None:
            cap.release()
        connection.close()


class ProcessCameraStream:
    """Bounded request/reply capture, isolated from Torch and native model runtimes.

    The child continuously drains the camera. Frames are sent only on request,
    so inference never consumes a queued backlog. NumPy arrays cross losslessly.
    """
    def __init__(self, index, source_factory=_settled_camera_source):
        import multiprocessing
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.pending = False
        self.closed = False
        self.process = context.Process(target=_camera_process_worker,
            args=(index, child, source_factory), daemon=True)
        self.process.start()
        child.close()

    def get(self, after_sequence, timeout):
        deadline = time.monotonic() + timeout
        while not self.closed:
            if not self.pending:
                self.connection.send('FRAME')
                self.pending = True
            if not self.connection.poll(max(0., deadline-time.monotonic())):
                if not self.process.is_alive():
                    raise RuntimeError(f'Camera process exited ({self.process.exitcode})')
                return None
            try:
                packet = self.connection.recv()
            except (EOFError, OSError) as exc:
                raise RuntimeError('Camera process connection lost') from exc
            self.pending = False
            if packet[0] == 'ERROR':
                raise RuntimeError(packet[1])
            if packet[0] != 'FRAME':
                raise RuntimeError('Malformed internal camera response')
            if packet[1] > after_sequence:
                return packet[1], packet[2], packet[3]
            if time.monotonic() >= deadline:
                return None
        raise RuntimeError('Camera stream is closed')

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                self.connection.send('STOP')
                self.process.join(timeout=2.)
            if self.process.is_alive():
                LOG.warning('Stopping unresponsive camera process')
                self.process.terminate()
                self.process.join(timeout=1.)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1.)
        finally:
            self.connection.close()


def CameraStream(index, cv):
    # Keep native camera/CoreML loading out of the Torch process on macOS.
    if sys.platform == 'darwin' and getattr(cv, '__name__', None) == 'cv2':
        return ProcessCameraStream(index)
    return ThreadCameraStream(index, cv)


def warmup_detector(model, frame, config, device, timeout=None):
    """Bound cold initialization separately; never open hardware on warmup failure."""
    import threading
    errors = []
    def work():
        try:
            predict(model, frame, config, device)
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(config.get('detector_warmup_seconds', 30.) if timeout is None else timeout)
    if worker.is_alive():
        raise TimeoutError('Detector warmup timed out; no hardware connection opened')
    if errors:
        raise errors[0]


def prepare_live_camera(stream, model, config, device):
    startup = config.get('camera_startup_seconds', 8.)
    first = stream.get(-1, startup)
    if first is None:
        raise TimeoutError('Camera startup timed out; check camera permission and index')
    warmup_detector(model, first[2], config, device)
    # Discard frames acquired during warmup, then await a new acquisition.
    latest = stream.get(first[0], 0)
    fresh = stream.get(latest[0] if latest else first[0], startup)
    if fresh is None:
        raise TimeoutError('Camera did not produce a fresh frame after warmup')
    return fresh


def detector_metadata(model, config, args):
    from importlib.metadata import version, PackageNotFoundError
    versions = {'python': sys.version.split()[0]}
    for package in ('torch', 'ultralytics', 'opencv-python', 'numpy'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    sizes = ({spec['id']: spec.get('settings', {}).get('imgsz', config['imgsz'])
              for spec, _ in model.models} if isinstance(model, SpecialistDetector)
             else {'unified': config['imgsz']})
    signature = getattr(model, 'checkpoint_sha256', None) or file_sha(resolve_path(args.weights or config['weights']))
    checkpoints = ([{'id': spec['id'], 'weights': spec['weights'], 'sha256': spec['sha256']}
                    for spec, _ in model.models] if isinstance(model, SpecialistDetector)
                   else [{'id': 'unified', 'weights': str(args.weights or config['weights']), 'sha256': signature}])
    return {'kind': config.get('detector', 'unified'), 'device': args.device,
            'signature': signature, 'checkpoints': checkpoints,
            'effective_model_sizes': sizes, 'runtime': versions,
            'routing_policy': getattr(model, 'routing_policy', None),
            'legacy_pooling_adapter': (isinstance(model, SpecialistDetector) and
                any(spec['sha256'] == LEGACY_UTENSIL_SHA256 for spec, _ in model.models))}


def save_camera_diagnostic(args, config, metadata, frame, detections, machine,
                           sequence, captured_at, infer_seconds, clock=time.monotonic):
    """Explicit detection-only observation; predictions never become ground truth."""
    if args.mode != 'detection':
        return None
    import cv2 as cv
    directory = resolve_path(args.capture_dir or 'runs/sorter/camera-diagnostics')
    directory.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex
    path = directory/(stem + '.json')
    image_path = path.with_suffix('.png')
    rows = [{'class_id': d.class_id, 'class': NAMES[d.class_id], 'confidence': d.confidence,
             'box_normalized_xyxy': list(d.box), 'edge_clipped': d.edge_clipped} for d in detections]
    frame_age = clock()-captured_at
    value = {'schema_version': 1, 'capture_group': args.capture_group,
             'session_id': getattr(args, 'capture_session', args.capture_group), 'ground_truth': None,
             'timestamp_utc': datetime.now(timezone.utc).isoformat(),
             'frame': {'file': image_path.name, 'sequence': sequence, 'width': frame.shape[1],
                       'height': frame.shape[0], 'acquired_monotonic': captured_at,
                       'age_seconds': frame_age, 'inference_seconds': infer_seconds,
                       'stale': frame_age > config['max_frame_seconds'], 'diagnostic_only': True},
             'occupancy_predictions': rows,
             'visible_predictions': [r for r in rows if r['confidence'] >= config['confidence']],
             'eligibility': {'roi': config['roi'], 'state': machine.state, 'streak': machine.streak,
                             'blocking_reason': machine.blocking_reason, 'occupant_count': machine.occupant_count,
                             'visible_count': sum(r['confidence'] >= config['confidence'] for r in rows),
                             'raw_count': len(rows), 'message': machine.message},
             'config': config, 'detector': metadata,
             'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True,
                 separators=(',', ':'), allow_nan=False).encode()).hexdigest()}
    encoded_ok, encoded = cv.imencode('.png', frame)
    if not encoded_ok:
        raise OSError('Could not encode raw diagnostic frame')
    with image_path.open('xb') as f:
        f.write(encoded.tobytes())
    try:
        value['frame']['sha256'] = file_sha(image_path)
        with path.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
    except BaseException:
        image_path.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        raise
    LOG.info('Saved observation %s (group %s; ground truth unset)', path, args.capture_group)
    return path


def dispatch_sort(machine, journal, link, config, manual, captured_at, clock=time.monotonic):
    selection = machine.begin_sort()
    if selection is None:
        return None, 0.
    if clock() - captured_at > config['max_frame_seconds']:
        raise TimeoutError('Frame expired before logging; no command sent')
    attempt = journal.begin(selection, config, manual=manual)
    started = clock()
    try:
        if clock() - captured_at > config['max_frame_seconds']:
            raise TimeoutError('Frame expired during logging; no command sent')
        link.start(selection[2])
    except BaseException as exc:
        journal.finish(attempt, clock()-started, f'ABORTED/UNKNOWN: {exc}', error=True)
        raise
    return attempt, started


def probe_cameras(indices):
    """Each probe is isolated so a blocked driver cannot hang the whole scan."""
    import subprocess
    for index in indices:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-one', str(index)],
                                    capture_output=True, text=True, timeout=8)
            print(result.stdout.strip() or f'Camera {index}: unavailable ({result.stderr.strip()[-300:]})')
        except subprocess.TimeoutExpired:
            print(f'Camera {index}: timed out; check Camera permissions, then restart the launching app')


def open_link(args, config):
    if args.simulate:
        transport = SimulatedArduino(angles=config['chute_angles'])
    else:
        import serial
        transport = serial.Serial(args.port, baudrate=9600, timeout=0, write_timeout=1)
        # Uno normally resets on DTR when opening USB. Discard boot messages,
        # then require one fresh PING response; never queue retries.
        time.sleep(2)
        transport.reset_input_buffer()
    link = ArduinoLink(transport, config)
    try:
        link.handshake()
        return link
    except BaseException:
        link.close()
        raise


def self_test(config, log_path):
    """Synthetic predictions -> state machine -> serial protocol -> CSV, no camera/model."""
    import copy
    c = copy.deepcopy(config)
    c.update(roi=[0, 0, 1, 1], stable_frames=5, clear_frames=3)
    # Fixture destinations exercise position numbers; these are not disposal recommendations.
    c['class_to_bin'] = dict.fromkeys(NAMES, 'Trash')
    c['bin_to_position'] = {'Recycle': 1, 'Compost': 2, 'Trash': 3}
    now = [0.]
    machine = SortingMachine(c)
    link = ArduinoLink(SimulatedArduino(clock=lambda: now[0]), c, clock=lambda: now[0])
    link.handshake()
    journal = AttemptLog(log_path, {'mode': 'self-test', 'simulated': True})
    for position, category in enumerate(('Recycle', 'Compost', 'Trash'), 1):
        c['class_to_bin']['food'] = category
        d = Detection(0, .85, (.3, .3, .6, .6))
        for _ in range(5):
            machine.observe([d])
        selection = machine.begin_sort()
        assert selection is not None
        attempt = journal.begin(selection, c, manual=False)
        link.start(position)
        link.poll()
        now[0] += 8
        response = link.poll()
        assert response == f'DONE SORT {position}'
        journal.finish(attempt, 8, response)
        machine.completed()
        for _ in range(10):
            machine.observe([d])
        assert machine.state == 'WAITING_FOR_CLEAR'
        for _ in range(3):
            machine.observe([])
        assert machine.state == 'IDLE'
    link.close()
    print(f'SIMULATED positions 1/2/3 completed; repeat prevention and rearm passed. CSV: {log_path}')


def run_live(args, config, model):
    import cv2 as cv
    cap = link = journal = attempt = None
    machine = SortingMachine(config)
    started = 0.
    last_attempt = None
    try:
        cap = CameraStream(args.camera, cv)
        first = prepare_live_camera(cap, model, config, args.device)
        sequence = first[0]
        last_capture = first[1]
        metadata = detector_metadata(model, config, args) if args.mode == 'detection' else None
        window = 'Cafeteria sorter — q quit'
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        if args.mode != 'detection':
            journal = AttemptLog(resolve_path(args.log or config['log']), {
                'mode': args.mode, 'simulated': args.simulate,
                'weights_sha256': getattr(model, 'checkpoint_sha256', None) or file_sha(resolve_path(args.weights or config['weights'])),
                'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
            link = open_link(args, config)
        while True:
            if link:
                response = link.poll()
                if response:
                    if attempt is None:
                        raise RuntimeError('Completion without a logged attempt')
                    journal.finish(attempt, time.monotonic()-started, response)
                    last_attempt = attempt['attempt_id']
                    attempt = None
                    machine.completed()
            packet = cap.get(sequence, .05)
            if packet is None:
                if time.monotonic() - last_capture > config['max_frame_seconds']:
                    raise TimeoutError('Camera frame watchdog expired; no new commands')
                if cv.waitKey(1) & 0xff == ord('q'):
                    break
                continue
            sequence, captured_at, frame = packet
            last_capture = captured_at
            frame_start = time.monotonic()
            infer_start = frame_start
            result, detections = predict(model, frame, config, args.device)
            infer_seconds = time.monotonic() - infer_start
            frame_age = time.monotonic() - captured_at
            stale = frame_age > config['max_frame_seconds']
            if stale and args.mode != 'detection':
                raise RuntimeError('Camera/inference too slow for fresh decisions; inspect device or max_frame_seconds')
            machine.observe(detections, stale=stale)
            display = visible_predictions(result, config['confidence']).plot()
            h, w = frame.shape[:2]
            if config['roi'] is not None:
                x1, y1, x2, y2 = config['roi']
                cv.rectangle(display, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255, 180, 0), 2)
            candidate = (f'{NAMES[machine.candidate.class_id]} {machine.candidate.confidence:.2f} '
                         f'({machine.streak}/{config["stable_frames"]})' if machine.candidate else 'none')
            selected = config['class_to_bin'][NAMES[machine.candidate.class_id]] if machine.candidate else None
            freshness = 'STALE / diagnostic only' if stale else 'FRESH'
            lines = [f'{args.mode.upper()} {"SIMULATED" if args.simulate else ""} | {machine.state} | Bin: {selected or "unverified/none"}',
                     machine.message + f' | ROI zoom: {config.get("crop_roi", False)} | Raw display >= {config["confidence"]:.0%} | Candidate: {candidate}', f'Camera+inference FPS: {1/max(time.monotonic()-frame_start, .001):.1f} | Inference FPS: {1/max(infer_seconds, .001):.1f}',
                     f'{freshness} | Frame age: {frame_age:.2f}s | Inference: {infer_seconds:.2f}s',
                     'q quit | SPACE approve (manual) | r ROI, z ROI zoom, s save raw observation (detection)',
                     f'Last attempt: {last_attempt or "none"} | outcome: 1 Recycle, 2 Compost, 3 Trash, f Failed, u Unknown']
            for n, text in enumerate(lines):
                y = 22 + n*23
                cv.rectangle(display, (0, y-18), (min(w, 1150), y+5), (0, 0, 0), -1)
                cv.putText(display, text, (6, y), cv.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv.LINE_AA)
            cv.imshow(window, display)
            key = cv.waitKey(1) & 0xff
            if key == ord('q') or cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                break
            if args.mode == 'detection' and (key == ord('s') or getattr(args, 'snapshot_requested', False)):
                args.snapshot_requested = False
                save_camera_diagnostic(args, config, metadata, frame, detections, machine,
                                       sequence, captured_at, infer_seconds)
            if key == ord('z') and args.mode == 'detection':
                if config['roi'] is None:
                    LOG.warning('Select the fixed loading-area ROI with r before enabling zoom')
                else:
                    config['crop_roi'] = not config.get('crop_roi', False)
                    machine = SortingMachine(config)
                    LOG.info('Fixed ROI inference crop: %s', config['crop_roi'])
                continue
            if key == ord('r') and args.mode == 'detection':
                x, y, rw, rh = cv.selectROI(window, frame, showCrosshair=True, fromCenter=False)
                if rw and rh:
                    config['roi'] = [x/w, y/h, (x+rw)/w, (y+rh)/h]
                    config['roi_verified'] = False
                    machine = SortingMachine(config)
                    print('Copy this ROI into configs/sorter.json after checking platform alignment:', json.dumps(config['roi']), flush=True)
                continue
            outcomes = {ord('1'): 'Recycle', ord('2'): 'Compost', ord('3'): 'Trash', ord('f'): 'Failed', ord('u'): 'Unknown'}
            if key in outcomes and last_attempt and journal:
                journal.outcome(last_attempt, outcomes[key])
                LOG.info('Actual outcome %s recorded for %s', outcomes[key], last_attempt)
            approve = args.mode == 'autonomous' or (args.mode == 'manual' and key == ord(' '))
            if approve and link and machine.state == 'READY_TO_SORT':
                attempt, started = dispatch_sort(machine, journal, link, config,
                                                 args.mode == 'manual', captured_at)
    except BaseException as exc:
        machine.fail(exc)
        raise
    finally:
        # Window/camera errors must not bypass serial cleanup and attempt logging.
        try:
            if cap is not None:
                cap.close()
        except Exception as exc:
            LOG.error('Camera cleanup error: %s', exc)
        try:
            cv.destroyAllWindows()
        except Exception as exc:
            LOG.error('Window cleanup error: %s', exc)
        if link is not None:
            try:
                # Quit/failure prevents further commands. Give the already accepted
                # firmware cycle a bounded opportunity to finish before closing USB.
                if attempt is not None:
                    LOG.warning('No more commands. Waiting for current firmware cycle; this is not an emergency stop.')
                    response = None
                    while not link.fault and link.pending is not None:
                        response = link.poll()
                        if response:
                            break
                        time.sleep(.02)
                    journal.finish(attempt, time.monotonic()-started,
                                   response or 'UNKNOWN: interrupted or communication fault', error=response is None)
            except Exception as exc:
                LOG.error('Cleanup/current-cycle error: %s', exc)
                if attempt is not None:
                    journal.finish(attempt, time.monotonic()-started, f'UNKNOWN: {exc}', error=True)
            finally:
                link.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/sorter.json'))
    parser.add_argument('--detector', choices=['unified', 'specialists', 'food-fallback'], default='unified', help='Existing custom detectors; food-fallback is experimental and detection-only')
    parser.add_argument('--specialist-config', help='Optional explicit model ownership profile; requires --detector specialists')
    parser.add_argument('--weights', help='Existing custom seven-class .pt; never downloaded')
    parser.add_argument('--mode', choices=['detection', 'manual', 'autonomous'], default='detection')
    parser.add_argument('--camera', type=int)
    parser.add_argument('--signal-snapshot', action='store_true', help='Allow one manual SIGUSR1 snapshot request in detection-only camera mode')
    parser.add_argument('--capture-dir', help='Detection-only raw PNG/JSON observations saved with s')
    parser.add_argument('--capture-group', help='Session/group identifier; keep each group in one dataset split')
    parser.add_argument('--crop-roi', action='store_true', help='Infer on the fixed loading-area ROI; restores boxes to full-frame coordinates')
    parser.add_argument('--roi', type=float, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    parser.add_argument('--confidence', type=float)
    parser.add_argument('--stable-frames', type=int)
    parser.add_argument('--clear-frames', type=int)
    parser.add_argument('--imgsz', type=int)
    parser.add_argument('--device', default='cpu', help='cpu or mps, after testing')
    parser.add_argument('--port', help='Explicit serial device; use --list-ports')
    parser.add_argument('--list-ports', action='store_true')
    parser.add_argument('--probe-cameras', default=None, help='Comma-separated indices, e.g. 0,1,2')
    parser.add_argument('--probe-one', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--image', help='Image inference only; no serial connection')
    parser.add_argument('--output', help='Optional annotated image (must not exist)')
    parser.add_argument('--simulate', action='store_true', help='Simulated serial, never opens a hardware port')
    parser.add_argument('--self-test', action='store_true', help='Synthetic end-to-end logic/serial/CSV test, no hardware')
    parser.add_argument('--log', help='CSV path override')
    parser.add_argument('--record-outcome', metavar='ATTEMPT_ID')
    parser.add_argument('--outcome', choices=sorted(BINS | {'Failed', 'Unknown'}))
    parser.add_argument('--notes', default='')
    args = parser.parse_args(argv)
    if args.detector == 'food-fallback' and args.mode != 'detection':
        parser.error('food-fallback requires detection mode; hardware operation is not validated')
    args.snapshot_requested = False
    if args.signal_snapshot:
        import signal
        if (args.mode != 'detection' or args.image or args.self_test or args.record_outcome
                or args.list_ports or args.probe_cameras is not None or args.probe_one is not None
                or not hasattr(signal, 'SIGUSR1')):
            parser.error('--signal-snapshot requires detection-only camera mode on a POSIX system')
        # Set a flag only; encode/save the exact next processed frame in the main loop.
        signal.signal(signal.SIGUSR1, lambda *_: setattr(args, 'snapshot_requested', True))
        import faulthandler
        faulthandler.register(signal.SIGUSR2, all_threads=True)
    args.capture_session = uuid.uuid4().hex
    args.capture_group = args.capture_group or args.capture_session
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.list_ports:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f'{p.device}\t{p.description}\t{p.hwid}')
        return
    if args.probe_cameras is not None:
        probe_cameras([int(v) for v in args.probe_cameras.split(',')])
        return
    if args.probe_one is not None:
        import cv2 as cv
        cap, frame = open_camera(args.probe_one, cv)
        try:
            print(f'Camera {args.probe_one}: readable {frame.shape[1]}x{frame.shape[0]}; verify device identity visually')
        finally:
            cap.release()
        return
    config = json.loads(Path(args.config).expanduser().read_text())
    for name in ('camera', 'roi', 'confidence', 'stable_frames', 'clear_frames', 'imgsz'):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    # An overridden ROI has not been physically verified merely by supplying coordinates.
    if args.roi is not None:
        config['roi_verified'] = False
    config['crop_roi'] = args.crop_roi or config.get('crop_roi', False)
    args.camera = config['camera']
    config.setdefault('camera_startup_seconds', 8.)
    config.setdefault('detector_warmup_seconds', 30.)
    validate_config(config, hardware=False)
    if args.record_outcome:
        if args.outcome is None:
            parser.error('--record-outcome requires --outcome')
        path = resolve_path(args.log or config['log'])
        if not path.is_file():
            raise FileNotFoundError(path)
        AttemptLog(path, {}).outcome(args.record_outcome, args.outcome, args.notes)
        return
    if args.self_test:
        self_test(config, resolve_path(args.log or 'runs/sorter/simulated.csv'))
        return
    if args.image:
        if args.mode != 'detection' or args.port:
            parser.error('--image runs only in detection mode without --port')
    elif args.mode != 'detection':
        validate_config(config, hardware=not args.simulate)
        if not args.simulate and not args.port:
            parser.error('Hardware modes require --port; use --list-ports')
    if args.specialist_config and args.detector != 'specialists':
        parser.error('--specialist-config requires --detector specialists')
    if args.detector != 'unified' and args.weights:
        parser.error('--weights applies to --detector unified; other detectors use their verified profiles')
    import torch
    torch.set_num_threads(2)
    config['detector'] = args.detector
    if args.detector == 'food-fallback':
        model = FoodFallbackDetector()
    elif args.detector == 'specialists':
        model = SpecialistDetector(resolve_path(args.specialist_config)) if args.specialist_config else SpecialistDetector()
    else:
        model = load_model(args.weights or config['weights'])
    if args.image:
        import cv2 as cv
        frame = cv.imread(str(Path(args.image).expanduser()))
        if frame is None:
            raise ValueError(f'Cannot read image: {args.image}')
        result, ds = predict(model, frame, config, args.device)
        print(json.dumps([{'class_id': d.class_id, 'class': NAMES[d.class_id], 'confidence': d.confidence,
                           'box_normalized_xyxy': d.box, 'bin': config['class_to_bin'][NAMES[d.class_id]]} for d in ds], indent=2))
        if args.output:
            output = Path(args.output).expanduser()
            if output.exists() or output.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                raise ValueError('Output must be a new .jpg/.jpeg/.png path')
            output.parent.mkdir(parents=True, exist_ok=True)
            encoded_ok, encoded = cv.imencode(output.suffix, visible_predictions(result, config['confidence']).plot())
            if not encoded_ok:
                raise OSError('Could not encode annotated image')
            with output.open('xb') as f:
                f.write(encoded.tobytes())
    else:
        run_live(args, config, model)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped. No additional movement commands will be sent.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        LOG.error('%s', exc)
        sys.exit(1)

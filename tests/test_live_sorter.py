"""Safety behavior of the live sorter, without model/camera/serial hardware."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import live_sorter as s
except ModuleNotFoundError:
    s = None


class SorterLogicTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(s, 'live_sorter application must exist')
        self.config = json.loads((Path(__file__).resolve().parents[1] / 'configs/sorter.json').read_text())
        self.config.update(roi=[.2, .2, .8, .8], stable_frames=3, clear_frames=3)
        self.config['class_to_bin']['food'] = 'Compost'
        self.config['bin_to_position'] = {'Compost': 1, 'Recycle': 2, 'Trash': 3}
        self.box = s.Detection(0, .8, (.3, .3, .5, .5))
        self.machine = s.SortingMachine(self.config)

    def ready(self):
        for _ in range(3):
            self.machine.observe([self.box])
        self.assertEqual(self.machine.state, 'READY_TO_SORT')

    def test_empty_low_confidence_and_outside_do_not_sort(self):
        for detections in [[], [s.Detection(0, .5, self.box.box)], [s.Detection(0, .9, (0, 0, .1, .1))]]:
            for _ in range(5):
                self.machine.observe(detections)
            self.assertNotEqual(self.machine.state, 'READY_TO_SORT')

    def test_multi_object_same_or_different_class_is_ambiguous(self):
        for cls in [0, 1]:
            for _ in range(5):
                self.machine.observe([self.box, s.Detection(cls, .4, (.55, .55, .7, .7))])
            self.assertNotEqual(self.machine.state, 'READY_TO_SORT')

    def test_partial_roi_blocks_without_becoming_eligible(self):
        for _ in range(5):
            self.machine.observe([s.Detection(0, .9, (.1, .3, .4, .5))])
        self.assertNotEqual(self.machine.state, 'READY_TO_SORT')

    def test_stable_then_changed_frame_revokes_manual_approval(self):
        self.ready()
        self.machine.observe([])
        self.assertIsNone(self.machine.begin_sort())

    def test_spatially_different_same_class_resets_stability(self):
        self.machine.observe([self.box])
        self.machine.observe([s.Detection(0, .8, (.6, .6, .7, .7))])
        self.machine.observe([self.box])
        self.assertEqual(self.machine.streak, 1)

    def test_repeat_latch_and_low_confidence_occupancy(self):
        self.ready()
        self.assertIsNotNone(self.machine.begin_sort())
        self.machine.completed()
        for _ in range(5):
            self.machine.observe([s.Detection(0, .3, self.box.box)])
        self.assertEqual(self.machine.state, 'WAITING_FOR_CLEAR')
        for _ in range(2):
            self.machine.observe([])
        self.assertEqual(self.machine.state, 'WAITING_FOR_CLEAR')
        self.machine.observe([])
        self.assertEqual(self.machine.state, 'IDLE')
        self.ready()

    def test_fault_never_rearms(self):
        self.machine.fail('camera lost')
        for _ in range(5):
            self.machine.observe([])
            self.machine.observe([self.box])
        self.assertEqual(self.machine.state, 'FAULT')
        self.assertIsNone(self.machine.begin_sort())

    def test_mapping_and_exact_model_class_order_required(self):
        s.verify_names(dict(enumerate(s.NAMES)))
        with self.assertRaises(ValueError):
            s.verify_names(dict(enumerate(list(reversed(s.NAMES)))))
        with self.assertRaises(ValueError):
            s.validate_config(self.config, hardware=True)
        self.config.update(disposal_verified=True, physical_bins_verified=True, roi_verified=True)
        with self.assertRaises(ValueError):
            s.validate_config(self.config, hardware=True)  # remaining class mappings unset
        self.config['class_to_bin'] = dict.fromkeys(s.NAMES, 'Trash')
        s.validate_config(self.config, hardware=True)

    def test_invalid_prediction_faults(self):
        with self.assertRaises(ValueError):
            self.machine.observe([s.Detection(0, float('nan'), self.box.box)])

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(s, 'ArduinoLink'), 'persistent serial protocol implementation missing')
        self.c = json.loads((s.ROOT / 'configs/sorter.json').read_text())
        self.now = 0.
        self.sim = s.SimulatedArduino(clock=lambda: self.now, duration=5)
        self.link = s.ArduinoLink(self.sim, self.c, clock=lambda: self.now)
        self.link.handshake()

    def test_ack_completion_and_no_overlap(self):
        self.link.start(1)
        with self.assertRaises(RuntimeError):
            self.link.start(2)
        self.assertIsNone(self.link.poll())
        self.now = 6
        self.assertEqual(self.link.poll(), 'DONE SORT 1')
        self.link.start(3)

    def test_missing_ack_faults_without_retry(self):
        self.link.start(2)
        self.sim.buffer.clear()
        self.now = 3
        with self.assertRaises(TimeoutError):
            self.link.poll()
        with self.assertRaises(RuntimeError):
            self.link.start(2)

    def test_done_without_ack_and_wrong_bin_are_faults(self):
        for reply in [b'DONE SORT 1\n', b'BUSY SORT 2\n', b'READY 60 90 120\n', b'ERROR UNKNOWN\n']:
            sim = s.SimulatedArduino(clock=lambda: self.now)
            link = s.ArduinoLink(sim, self.c, clock=lambda: self.now)
            link.handshake()
            link.start(1)
            sim.buffer[:] = reply
            with self.assertRaises(RuntimeError):
                link.poll()
            with self.assertRaises(RuntimeError):
                link.start(1)

    def test_completion_timeout_and_disconnect(self):
        self.link.start(1)
        self.link.poll()
        self.sim.finish_at = 100
        self.now = 21
        with self.assertRaises(TimeoutError):
            self.link.poll()
        self.assertTrue(self.link.fault)

    def test_calibration_mismatch_rejected(self):
        self.c['chute_angles'] = [70, 90, 110]
        link = s.ArduinoLink(s.SimulatedArduino(), self.c)
        with self.assertRaises(RuntimeError):
            link.handshake()

    def test_simulator_full_state_cycle_and_csv_outcome(self):
        c = copy.deepcopy(self.c)
        c.update(roi=[0, 0, 1, 1], stable_frames=2, clear_frames=2)
        c['class_to_bin']['food'] = 'Compost'
        c['bin_to_position']['Compost'] = 3
        machine = s.SortingMachine(c)
        d = s.Detection(0, .9, (.2, .2, .4, .4))
        with tempfile.TemporaryDirectory() as td:
            journal = s.AttemptLog(Path(td)/'attempts.csv', {'mode': 'autonomous', 'simulated': True})
            machine.observe([d]); machine.observe([d])
            attempt = journal.begin(machine.begin_sort(), c, manual=False)
            self.link.start(3)
            self.link.poll()
            self.now = 6
            response = self.link.poll()
            journal.finish(attempt, 6, response)
            machine.completed()
            for _ in range(10):
                machine.observe([d])
            self.assertEqual(machine.state, 'WAITING_FOR_CLEAR')
            machine.observe([]); machine.observe([])
            self.assertEqual(machine.state, 'IDLE')
            journal.outcome(attempt['attempt_id'], 'Compost', 'physical operator entry')
            with journal.path.open() as f:
                rows = list(s.csv.DictReader(f))
            self.assertEqual([r['event'] for r in rows], ['STARTED', 'COMPLETED', 'OUTCOME'])
            self.assertEqual(rows[1]['chute_angle'], '120')
            self.assertEqual(rows[1]['response'], 'DONE SORT 3')
            self.assertEqual(rows[2]['actual_outcome'], 'Compost')

class InputTests(unittest.TestCase):
    def test_missing_weights_never_download(self):
        self.assertTrue(hasattr(s, 'load_model'), 'model loader missing')
        with self.assertRaises(FileNotFoundError):
            s.load_model('/missing/yolo11n.pt')

    def test_shared_class_ids_and_single_checkpoint_hash(self):
        """The live camera reuses the frozen class list and file-hash implementation."""
        from types import SimpleNamespace
        from unittest.mock import patch
        import specialists
        self.assertIs(s.NAMES, specialists.NAMES)
        self.assertIs(s.file_sha, specialists.sha)

        class FakeYOLO:
            def __init__(self, path):
                self.names = dict(enumerate(s.NAMES))
                self.model = SimpleNamespace(yaml={'scale': 'n'})

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'weights.pt'
            path.write_bytes(b'local test checkpoint')
            with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=FakeYOLO)}):
                with patch.object(s, 'file_sha', wraps=s.file_sha) as hash_file:
                    model = s.load_model(path)
                    self.assertEqual(hash_file.call_count, 1)
            self.assertEqual(model.checkpoint_sha256, specialists.sha(path))

    def test_prediction_geometry_and_names(self):
        self.assertTrue(hasattr(s, 'decode_detections'), 'prediction decoder missing')
        rows = [[10, 20, 50, 60, .9, 2]]
        ds = s.decode_detections(rows, 100, 100)
        self.assertEqual(ds[0].box, (.1, .2, .5, .6))
        for bad in [[[10, 20, 50, 60, .9, 2.5]], [[10, 20, 5, 60, .9, 2]], [[0, 0, 101, 20, .9, 2]]]:
            with self.assertRaises(ValueError):
                s.decode_detections(bad, 100, 100)

    def test_camera_read_failure_releases_camera(self):
        self.assertTrue(hasattr(s, 'open_camera'), 'camera opener missing')
        class BrokenCamera:
            released = False
            def isOpened(self): return True
            def read(self): return False, None
            def release(self): self.released = True
        class CV:
            def VideoCapture(self, *a): return cap
        cap = BrokenCamera()
        with self.assertRaises(RuntimeError):
            s.open_camera(0, CV())
        self.assertTrue(cap.released)

class DeadlineAndCameraTests(unittest.TestCase):
    def config(self):
        return json.loads((s.ROOT/'configs/sorter.json').read_text())

    def test_late_ack_and_late_done_are_faults_even_when_buffered(self):
        for acknowledge_first, late_time in [(False, 3), (True, 21)]:
            now = [0.]
            c = self.config()
            sim = s.SimulatedArduino(clock=lambda: now[0], duration=1)
            link = s.ArduinoLink(sim, c, clock=lambda: now[0])
            link.handshake(); link.start(1)
            if acknowledge_first:
                link.poll()
            now[0] = late_time
            with self.assertRaises(TimeoutError):
                link.poll()
            self.assertTrue(link.fault)

    def test_disk_delay_aborts_before_movement(self):
        self.assertTrue(hasattr(s, 'dispatch_sort'), 'freshness-checked dispatch missing')
        now = [0.]
        c = self.config()
        c.update(roi=[0, 0, 1, 1], stable_frames=1)
        c['class_to_bin']['food'] = 'Trash'
        c['bin_to_position']['Trash'] = 1
        machine = s.SortingMachine(c)
        machine.observe([s.Detection(0, .9, (.2, .2, .4, .4))])
        class SlowLog:
            def begin(self, *a, **k):
                now[0] = 3.
                return {'attempt_id': 'slow'}
            def finish(self, *a, **k): self.finished = True
        class Link:
            sent = False
            def start(self, *a): self.sent = True
        journal, link = SlowLog(), Link()
        with self.assertRaises(TimeoutError):
            s.dispatch_sort(machine, journal, link, c, False, 0., clock=lambda: now[0])
        self.assertFalse(link.sent)
        self.assertTrue(journal.finished)

    def test_capture_worker_keeps_latest_frame_and_closes(self):
        self.assertTrue(hasattr(s, 'CameraStream'), 'bounded latest-frame capture missing')
        import time
        class Camera:
            released = False
            n = 0
            def isOpened(self): return True
            def read(self):
                time.sleep(.002)
                self.n += 1
                return True, self.n
            def release(self): self.released = True
        class CV:
            def VideoCapture(self, *a): return cap
        cap = Camera()
        stream = s.CameraStream(0, CV())
        try:
            first = stream.get(-1, .2)
            time.sleep(.04)
            second = stream.get(first[0], .2)
            self.assertGreater(second[0], first[0]+1)
            self.assertEqual(second[2], second[0])
        finally:
            stream.close()
        self.assertTrue(cap.released)

    def test_blocked_camera_wait_is_bounded(self):
        self.assertTrue(hasattr(s, 'CameraStream'), 'bounded capture missing')
        import threading
        import time
        gate = threading.Event()
        class Camera:
            def isOpened(self): return True
            def read(self): gate.wait(.5); return False, None
            def release(self): pass
        class CV:
            def VideoCapture(self, *a): return Camera()
        stream = s.CameraStream(0, CV())
        try:
            start = time.monotonic()
            self.assertIsNone(stream.get(-1, .03))
            self.assertLess(time.monotonic()-start, .2)
        finally:
            gate.set()
            stream.close()

class FaultCleanupTests(unittest.TestCase):
    def test_disconnect_and_malformed_response_latch_fault(self):
        c = json.loads((s.ROOT/'configs/sorter.json').read_text())
        for disconnect in [True, False]:
            sim = s.SimulatedArduino()
            link = s.ArduinoLink(sim, c)
            link.handshake()
            link.start(1)
            if disconnect:
                sim.close()
            else:
                sim.buffer[:] = b'X' * 90
            with self.assertRaises((OSError, RuntimeError)):
                link.poll()
            self.assertTrue(link.fault)

    def test_window_cleanup_error_cannot_skip_serial_close(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class Stream:
            n = 0
            closed = False
            def __init__(self, *a): pass
            def get(self, *a):
                raise RuntimeError('camera gone')
            def close(self): self.closed = True
        class Link:
            closed = False
            def poll(self): return None
            def close(self): self.closed = True
        cv = SimpleNamespace(WINDOW_NORMAL=0, namedWindow=lambda *a: None,
                             destroyAllWindows=lambda: (_ for _ in ()).throw(RuntimeError('window cleanup error')))
        link = Link()
        c = json.loads((s.ROOT/'configs/sorter.json').read_text())
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(camera=0, mode='manual', simulate=True, log=str(Path(td)/'log.csv'), weights=None, device='cpu')
            with patch.dict(sys.modules, {'cv2': cv}), patch.object(s, 'CameraStream', Stream), patch.object(s, 'prepare_live_camera', return_value=(1, 0., None)), patch.object(s, 'open_link', return_value=link), patch.object(s, 'file_sha', return_value='fixture'):
                with self.assertRaisesRegex(RuntimeError, 'camera gone'):
                    s.run_live(args, c, None)
        self.assertTrue(link.closed)

class PySerialTransportTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform in ('darwin', 'linux'), 'POSIX pseudo-terminal required')
    def test_real_pyserial_persistent_connection_with_fragmented_replies(self):
        import os
        import select
        import threading
        import time
        import serial
        c = json.loads((s.ROOT/'configs/sorter.json').read_text())
        master, slave = os.openpty()
        port = serial.Serial(os.ttyname(slave), 9600, timeout=0, write_timeout=1)
        stop = threading.Event()
        received = []
        def emulate():
            buffer = bytearray()
            while not stop.is_set():
                if not select.select([master], [], [], .05)[0]:
                    continue
                buffer.extend(os.read(master, 1024))
                while b'\n' in buffer:
                    raw, _, rest = buffer.partition(b'\n')
                    buffer[:] = rest
                    received.append(raw.decode())
                    reply = b'READY 60 90 120\n' if raw == b'PING' else b'BUSY SORT 2\nDONE SORT 2\n'
                    for chunk in (reply[:3], reply[3:]):
                        os.write(master, chunk)
                        time.sleep(.002)
        worker = threading.Thread(target=emulate, daemon=True)
        worker.start()
        link = s.ArduinoLink(port, c)
        try:
            link.handshake()
            link.start(2)
            result = None
            deadline = time.monotonic()+1
            while result is None and time.monotonic() < deadline:
                result = link.poll()
                time.sleep(.005)
            self.assertEqual(result, 'DONE SORT 2')
            self.assertEqual(received, ['PING', 'SORT 2'])
        finally:
            stop.set(); worker.join(timeout=1)
            link.close()
            os.close(master); os.close(slave)

class SpecialistInferenceTests(unittest.TestCase):
    def test_specialist_local_ids_map_to_canonical_seven_classes(self):
        self.assertTrue(hasattr(s, 'SpecialistDetector'), 'optional existing specialist inference missing')
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        import torch
        specs = [(['food', 'hinged_container', 'drink_carton'], 1), (['plate', 'wrapper'], 0),
                 (['utensil'], 0), (['bottle'], 0)]
        class Model:
            def __init__(self, names, cls):
                self.names = dict(enumerate(names)); self.cls = cls
            def predict(self, frame, **kwargs):
                return [SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor([[10, 10, 30, 30, .9, self.cls]])))]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models = []
            for i, (names, cls) in enumerate(specs):
                p = root/f'{i}.pt'; p.write_bytes(str(i).encode())
                models.append({'id': str(i), 'weights': str(p), 'sha256': s.file_sha(p), 'classes': names})
            config = root/'specialists.json'; config.write_text(json.dumps({'models': models}))
            fakes = [Model(*spec) for spec in specs]
            with patch('ultralytics.YOLO', side_effect=fakes):
                detector = s.SpecialistDetector(config)
            c = json.loads((s.ROOT/'configs/sorter.json').read_text())
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            result, detections = s.predict(detector, frame, c, 'cpu')
            self.assertEqual([d.class_id for d in detections], [3, 1, 2, 6])
            self.assertEqual(result.names, dict(enumerate(s.NAMES)))
            self.assertEqual(len(detector.checkpoint_sha256), 64)
            models[0]['sha256'] = '0'*64
            config.write_text(json.dumps({'models': models}))
            with self.assertRaisesRegex(ValueError, 'hash'):
                s.SpecialistDetector(config)

    def test_missing_or_duplicate_class_ownership_is_rejected(self):
        self.assertTrue(hasattr(s, 'SpecialistDetector'), 'optional existing specialist inference missing')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'models.json'
            path.write_text(json.dumps({'models': [{'classes': s.NAMES[:-1]}]}))
            with self.assertRaisesRegex(ValueError, 'owner'):
                s.SpecialistDetector(path)

class DisplayConfidenceTests(unittest.TestCase):
    def test_low_confidence_boxes_are_hidden_without_removing_safety_occupancy(self):
        self.assertTrue(hasattr(s, 'visible_predictions'), 'display confidence filtering missing')
        import numpy as np
        import torch
        from ultralytics.engine.results import Results
        result = Results(np.zeros((100, 100, 3), dtype=np.uint8), path='fixture', names=dict(enumerate(s.NAMES)),
                         boxes=torch.tensor([[10, 10, 30, 30, .8, 0], [40, 40, 60, 60, .4, 5]]))
        visible = s.visible_predictions(result, .6)
        self.assertEqual(len(visible.boxes), 1)
        self.assertEqual(len(result.boxes), 2)
        c = json.loads((s.ROOT/'configs/sorter.json').read_text())
        c['roi'] = [0, 0, 1, 1]
        machine = s.SortingMachine(c)
        ds = s.decode_detections(result.boxes.data.tolist(), 100, 100)
        for _ in range(10):
            machine.observe(ds)
        self.assertNotEqual(machine.state, 'READY_TO_SORT')

class SpecialistValidationTests(unittest.TestCase):
    def test_malformed_native_ids_fault_instead_of_truncating_or_disappearing(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        class Model:
            names = dict(enumerate(s.NAMES))
            def predict(self, *a, **kw):
                return [SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor([[10, 10, 30, 30, .9, native_id]])))]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'weights.pt';p.write_bytes(b'fixture')
            config = Path(td)/'models.json'
            spec = {'id': 'all', 'weights': str(p), 'sha256': s.file_sha(p), 'classes': s.NAMES}
            config.write_text(json.dumps({'models': [spec]}))
            with patch('ultralytics.YOLO', return_value=Model()):
                detector = s.SpecialistDetector(config)
            c = json.loads((s.ROOT/'configs/sorter.json').read_text())
            for native_id in [1.9, 9, -1, float('nan')]:
                with self.assertRaises(ValueError):
                    s.predict(detector, np.zeros((100, 100, 3), dtype=np.uint8), c, 'cpu')

    def test_invalid_native_resolution_is_rejected_before_model_load(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'weights.pt';p.write_bytes(b'fixture')
            config = Path(td)/'models.json'
            for size in [799, 0, True]:
                spec = {'id': 'all', 'weights': str(p), 'sha256': s.file_sha(p), 'classes': s.NAMES,
                        'settings': {'imgsz': size}}
                config.write_text(json.dumps({'models': [spec]}))
                with patch('ultralytics.YOLO') as loader:
                    with self.assertRaises(ValueError):
                        s.SpecialistDetector(config)
                    loader.assert_not_called()


class CameraDiagnosticsTests(unittest.TestCase):
    def config(self):
        c = json.loads((s.ROOT/'configs/sorter.json').read_text())
        c['roi'] = [0, 0, 1, 1]
        return c

    def test_distinct_blocking_reasons_preserve_safety(self):
        c = self.config(); machine = s.SortingMachine(c)
        d = s.Detection(0, .8, (.2, .2, .4, .4))
        machine.observe([d, s.Detection(5, .3, (.5,.5,.6,.6))])
        self.assertEqual(getattr(machine, 'blocking_reason', None), 'multiple_occupants')
        self.assertEqual(machine.occupant_count, 2)
        machine.observe([s.Detection(0, .3, d.box)])
        self.assertEqual(machine.blocking_reason, 'low_confidence')
        c['roi'] = [.3,.3,.7,.7]; machine.observe([d])
        self.assertEqual(machine.blocking_reason, 'roi_edge')
        c['roi'] = None; machine.observe([d])
        self.assertEqual(machine.blocking_reason, 'roi_unconfigured')
        self.assertIsNone(machine.begin_sort())

    def test_clear_rearm_has_no_stale_blocking_reason(self):
        c = self.config(); machine = s.SortingMachine(c)
        machine.completed()
        for _ in range(c['clear_frames']):
            machine.observe([])
        self.assertEqual(machine.state, 'IDLE')
        self.assertEqual(machine.blocking_reason, 'empty')

    def test_metadata_records_actual_checkpoint_and_effective_size(self):
        from types import SimpleNamespace
        model = SimpleNamespace(checkpoint_sha256='abc')
        args = SimpleNamespace(device='cpu',weights='/custom/model.pt')
        metadata = s.detector_metadata(model,self.config(),args)
        self.assertIn('checkpoints', metadata, 'individual checkpoint provenance missing')
        self.assertEqual(metadata['checkpoints'], [{'id':'unified','weights':'/custom/model.pt','sha256':'abc'}])
        self.assertEqual(metadata['effective_model_sizes'], {'unified':640})
        self.assertIn('ultralytics', metadata['runtime'])

    def test_warmup_failure_never_opens_serial(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class Stream:
            def __init__(self,*a): pass
            def get(self,*a): return (1,1.,None)
            def close(self): pass
        args = SimpleNamespace(camera=0,device='cpu',mode='manual')
        cv = SimpleNamespace(destroyAllWindows=lambda:None)
        with patch.dict(sys.modules,{'cv2':cv}), patch.object(s,'CameraStream',Stream), patch.object(s,'predict',side_effect=RuntimeError('warmup failed')), patch.object(s,'open_link') as link:
            with self.assertRaisesRegex(RuntimeError,'warmup failed'):
                s.run_live(args,self.config(),object())
        link.assert_not_called()

    def test_lossless_diagnostics_keep_all_predictions_and_session_group(self):
        self.assertTrue(hasattr(s, 'save_camera_diagnostic'), 'diagnostic writer missing')
        import numpy as np
        import cv2
        from types import SimpleNamespace
        frame = np.arange(300, dtype=np.uint8).reshape(10,10,3)
        c = self.config(); machine = s.SortingMachine(c)
        ds = [s.Detection(0,.8,(.1,.1,.3,.3)), s.Detection(5,.3,(.5,.5,.8,.8))]
        machine.observe(ds)
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(mode='detection', capture_dir=td, capture_group='session-one', device='cpu')
            metadata = {'signature':'fixture', 'effective_model_sizes':{'unified':640}, 'runtime':{'torch':'fixture'}}
            paths = [s.save_camera_diagnostic(args,c,metadata,frame,ds,machine,7,10.,.12,clock=lambda:10.5) for _ in range(2)]
            self.assertNotEqual(paths[0], paths[1])
            for path in paths:
                value = json.loads(path.read_text())
                self.assertEqual(value['capture_group'], 'session-one')
                self.assertIsNone(value['ground_truth'])
                self.assertEqual(len(value['occupancy_predictions']), 2)
                self.assertEqual(len(value['visible_predictions']), 1)
                self.assertEqual(value['frame']['age_seconds'], .5)
                self.assertEqual(value['eligibility']['blocking_reason'], 'multiple_occupants')
                self.assertEqual(value['config'], c)
                self.assertEqual(value['detector'], metadata)
                self.assertTrue(np.array_equal(cv2.imread(str(path.with_suffix('.png'))), frame))
            args.mode = 'manual'
            self.assertIsNone(s.save_camera_diagnostic(args,c,metadata,frame,ds,machine,7,10.,.12))
            self.assertEqual(len(list(Path(td).glob('*.json'))), 2)

    def test_startup_uses_separate_bound_and_discards_warmup_frames(self):
        self.assertTrue(hasattr(s, 'prepare_live_camera'), 'cold startup helper missing')
        from unittest.mock import patch
        c = self.config(); c['max_frame_seconds'] = .01; c['camera_startup_seconds'] = 8.
        calls = []
        class Stream:
            def get(self, after, timeout):
                calls.append((after,timeout))
                return [(1,1.,'warmup'), (20,20.,'discard'), (21,21.,'fresh')][len(calls)-1]
        with patch.object(s,'warmup_detector') as warmup:
            packet = s.prepare_live_camera(Stream(),object(),c,'cpu')
        self.assertEqual(packet, (21,21.,'fresh'))
        self.assertEqual(calls, [(-1,8.),(1,0),(20,8.)])
        self.assertEqual(warmup.call_args.args[1], 'warmup')

    def test_warmup_has_bounded_failure(self):
        self.assertTrue(hasattr(s, 'warmup_detector'), 'bounded warmup missing')
        import threading
        from unittest.mock import patch
        gate = threading.Event()
        def blocked(*a): gate.wait(1)
        try:
            with patch.object(s,'predict',side_effect=blocked):
                with self.assertRaisesRegex(TimeoutError,'warmup'):
                    s.warmup_detector(object(),None,self.config(),'cpu',timeout=.01)
        finally:
            gate.set()

    def test_legacy_adapter_preserves_pool_math_and_supports_channels_last(self):
        self.assertTrue(hasattr(s, 'adapt_legacy_pooling'), 'legacy adapter missing')
        import torch
        from types import SimpleNamespace
        model = torch.nn.Module(); model.cv3 = torch.nn.Sequential(torch.nn.MaxPool3d((3,1,1)))
        x = torch.arange(96.).reshape(1,6,4,4)
        expected = model.cv3(x)
        s.adapt_legacy_pooling(SimpleNamespace(model=model), 'ce4956547f0c421db1dc4b3d2fa7c28d9d9ae2fc44ddf5acf333ff211c3d4c7f')
        self.assertTrue(torch.equal(model.cv3(x),expected))
        self.assertTrue(torch.equal(model.cv3(x.contiguous(memory_format=torch.channels_last)),expected))


class StaleDiagnosticTests(unittest.TestCase):
    def config(self):
        c = json.loads((s.ROOT/'configs/sorter.json').read_text())
        c.update(roi=[0,0,1,1],stable_frames=2,clear_frames=2)
        return c

    def test_stale_observation_resets_stability_and_never_rearms_or_clears_latches(self):
        import inspect
        self.assertIn('stale', inspect.signature(s.SortingMachine.observe).parameters)
        c = self.config(); d = s.Detection(0,.9,(.2,.2,.4,.4)); machine = s.SortingMachine(c)
        machine.observe([d]); machine.observe([d]); self.assertEqual(machine.state,'READY_TO_SORT')
        machine.observe([d],stale=True)
        self.assertEqual(machine.streak,0); self.assertIsNone(machine.begin_sort())
        machine.observe([d]); self.assertEqual(machine.streak,1)
        self.assertEqual(machine.state,'DETECTING')
        machine.completed(); machine.observe([])
        machine.observe([],stale=True); machine.observe([])
        self.assertEqual(machine.state,'WAITING_FOR_CLEAR')
        for state in ('FAULT','SORTING'):
            machine.state = state
            machine.observe([],stale=True)
            self.assertEqual(machine.state,state)

    def run_live_fixture(self, mode, slow, keys, writer=None, request_snapshot=False):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        frame = np.full((12,12,3),7,dtype=np.uint8)
        warmup = np.zeros_like(frame)
        packets = iter([(1,8.,warmup),(8,9.,warmup),(9,10.,warmup),(10,11.,frame),(11,12.,frame)])
        now = [11.]
        class Stream:
            def __init__(self,*a): pass
            def get(self,*a): return next(packets)
            def close(self): pass
        texts = []
        cv = SimpleNamespace(WINDOW_NORMAL=0,WND_PROP_VISIBLE=0,FONT_HERSHEY_SIMPLEX=0,LINE_AA=0,
            namedWindow=lambda *a:None,destroyAllWindows=lambda:None,
            rectangle=lambda *a:None,putText=lambda image,text,*a:texts.append(text),
            imshow=Mock(),waitKey=Mock(side_effect=[ord(k) for k in keys]),getWindowProperty=lambda *a:1)
        result = SimpleNamespace(plot=lambda:frame.copy())
        def prediction(*a):
            now[0] += 3. if slow else .1
            return result,[s.Detection(0,.9,(.2,.2,.4,.4))]
        link = SimpleNamespace(poll=lambda:None,start=Mock(),close=Mock())
        c = self.config()
        args = SimpleNamespace(mode=mode,camera=0,device='cpu',simulate=True,weights=None,log=None,capture_group='g',capture_dir=None,snapshot_requested=request_snapshot)
        observed = {}
        def capture(*a):
            observed['frame'] = a[3]; observed['sequence'] = a[6]; observed['captured_at'] = a[7]
            observed['state'] = a[5].state; observed['reason'] = a[5].blocking_reason
        with tempfile.TemporaryDirectory() as td:
            args.log = str(Path(td)/'attempts.csv')
            with patch.dict(sys.modules,{'cv2':cv}), patch.object(s,'CameraStream',Stream), patch.object(s,'warmup_detector'), patch.object(s,'detector_metadata',return_value={}), patch.object(s,'predict',side_effect=prediction), patch.object(s,'visible_predictions',side_effect=lambda result,*a:result), patch.object(s.time,'monotonic',side_effect=lambda:now[0]), patch.object(s,'open_link',return_value=link) as opened, patch.object(s,'file_sha',return_value='fixture'), patch.object(s,'save_camera_diagnostic',side_effect=capture) as saved:
                if mode != 'detection' and slow:
                    with self.assertRaisesRegex(RuntimeError,'too slow'):
                        s.run_live(args,c,object())
                else:
                    s.run_live(args,c,object())
                if mode == 'detection': opened.assert_not_called()
                link.start.assert_not_called()
        observed['snapshot_requested'] = args.snapshot_requested
        return texts,cv,saved,observed,frame

    def test_detection_slow_frame_displays_stale_can_capture_and_quit_without_serial(self):
        texts,cv,saved,observed,frame = self.run_live_fixture('detection',True,'sq')
        self.assertTrue(any('STALE' in text and 'diagnostic' in text for text in texts))
        self.assertTrue(any('age' in text.lower() and 'inference' in text.lower() for text in texts))
        self.assertGreaterEqual(cv.imshow.call_count,2)
        self.assertEqual(saved.call_count,1)
        self.assertEqual(observed['reason'],'stale_frame')
        self.assertNotEqual(observed['state'],'READY_TO_SORT')

    def test_operator_requested_snapshot_is_detection_only_and_consumed_once(self):
        texts,cv,saved,observed,frame = self.run_live_fixture('detection',False,'xq',request_snapshot=True)
        self.assertEqual(saved.call_count,1)
        self.assertFalse(observed['snapshot_requested'])
        self.assertIs(observed['frame'],frame)

    def test_both_hardware_modes_fault_on_slow_frame_without_a_command(self):
        for mode in ('manual','autonomous'):
            with self.subTest(mode=mode): self.run_live_fixture(mode,True,'q')

    def test_save_hotkey_uses_same_postwarmup_packet_frame_sequence_and_timestamp(self):
        texts,cv,saved,observed,frame = self.run_live_fixture('detection',False,'sq')
        self.assertIs(observed['frame'],frame)
        self.assertEqual(observed['sequence'],10)
        self.assertEqual(observed['captured_at'],11.)

    def test_saved_config_digest_tracks_roi_changes_and_is_canonical(self):
        import numpy as np
        import hashlib
        from types import SimpleNamespace
        c = self.config(); machine = s.SortingMachine(c)
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(mode='detection',capture_dir=td,capture_group='g')
            a = s.save_camera_diagnostic(args,c,{},np.zeros((2,2,3),dtype=np.uint8),[],machine,1,0.,0.,clock=lambda:0.)
            first = json.loads(a.read_text())
            self.assertIn('config_sha256',first,'snapshot config digest missing')
            expected = hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            self.assertEqual(first['config_sha256'],expected)
            c['roi'] = [.1,.1,.9,.9]
            b = s.save_camera_diagnostic(args,c,{},np.zeros((2,2,3),dtype=np.uint8),[],machine,2,0.,0.,clock=lambda:0.)
            self.assertNotEqual(first['config_sha256'],json.loads(b.read_text())['config_sha256'])

class RoiInferenceTests(unittest.TestCase):
    def test_roi_crop_restores_global_coordinates_and_marks_internal_edge_fragments(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from ultralytics.engine.results import Results
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        c = json.loads((s.ROOT/'configs/sorter.json').read_text())
        c.update(roi=[.25,.2,.75,.8],crop_roi=True,stable_frames=1)
        shapes=[]
        class Model:
            names=dict(enumerate(s.NAMES))
            def predict(self, image, **kwargs):
                shapes.append(image.shape)
                return [Results(image,path='fixture',names=self.names,
                    boxes=torch.tensor([[10,10,40,40,.9,0],[0,20,15,40,.9,0]]))]
        result,ds=s.predict(Model(),frame,c,'cpu')
        self.assertEqual(shapes,[(60,100,3)])
        self.assertEqual(result.orig_shape,(100,200))
        self.assertEqual(ds[0].box,(.3,.3,.45,.6))
        self.assertFalse(ds[0].edge_clipped)
        self.assertTrue(ds[1].edge_clipped)
        machine=s.SortingMachine(c)
        machine.observe([ds[1]])
        self.assertNotEqual(machine.state,'READY_TO_SORT')
        self.assertEqual(machine.blocking_reason,'roi_edge')

    def test_roi_crop_requires_an_explicit_valid_roi(self):
        from unittest.mock import Mock
        import numpy as np
        c=json.loads((s.ROOT/'configs/sorter.json').read_text())
        c['crop_roi']=True
        model=Mock()
        with self.assertRaises(ValueError):s.predict(model,np.zeros((20,20,3),dtype=np.uint8),c,'cpu')
        model.predict.assert_not_called()


class SnapshotSignalTests(unittest.TestCase):
    def test_signal_snapshot_rejected_in_hardware_modes_before_loading(self):
        from unittest.mock import patch
        from contextlib import redirect_stderr
        from io import StringIO
        for mode in ['manual','autonomous']:
            output=StringIO()
            with patch.object(s,'load_model') as loader, redirect_stderr(output):
                with self.assertRaises(SystemExit) as error:
                    s.main(['--signal-snapshot','--mode',mode])
                self.assertEqual(error.exception.code,2)
                self.assertIn('requires detection-only camera mode',output.getvalue())
                loader.assert_not_called()

class _SyntheticProcessCapture:
    def __init__(self): self.n=0
    def read(self):
        import numpy as np,time
        time.sleep(.005);self.n+=1
        return True,np.full((8,10,3),self.n%256,dtype=np.uint8)
    def release(self): pass


def _synthetic_process_source(index):
    capture=_SyntheticProcessCapture()
    _,frame=capture.read()
    return capture,frame


def _blocked_process_source(index):
    import time
    time.sleep(20)
    return _synthetic_process_source(index)


def _crashing_process_source(index):
    import os
    os._exit(17)


class ProcessCameraTests(unittest.TestCase):
    def test_process_capture_delivers_fresh_lossless_frames_and_closes(self):
        self.assertTrue(hasattr(s,'ProcessCameraStream'),'isolated camera backend missing')
        import time,numpy as np
        stream=s.ProcessCameraStream(0,source_factory=_synthetic_process_source)
        try:
            first=stream.get(-1,5)
            self.assertIsNotNone(first)
            time.sleep(.04)
            second=stream.get(first[0],2)
            self.assertGreater(second[0],first[0])
            self.assertGreater(second[1],first[1])
            self.assertEqual(second[2].shape,(8,10,3))
            self.assertEqual(np.unique(second[2]).size,1)
        finally:stream.close()
        self.assertFalse(stream.process.is_alive())

    def test_native_camera_process_failure_is_contained(self):
        stream=s.ProcessCameraStream(0,source_factory=_crashing_process_source)
        try:
            with self.assertRaises((RuntimeError,OSError)):
                stream.get(-1,5)
        finally:stream.close()
        self.assertFalse(stream.process.is_alive())
        self.assertEqual(stream.process.exitcode,17)

    def test_blocked_camera_startup_does_not_block_parent_or_cleanup(self):
        self.assertTrue(hasattr(s,'ProcessCameraStream'),'isolated camera backend missing')
        import time
        stream=s.ProcessCameraStream(0,source_factory=_blocked_process_source)
        start=time.monotonic()
        self.assertIsNone(stream.get(-1,.03))
        self.assertLess(time.monotonic()-start,.3)
        stream.close()
        self.assertFalse(stream.process.is_alive())

class FoodOwnerProfileTests(unittest.TestCase):
    def test_explicit_native_classes_route_only_owned_outputs(self):
        import numpy as np,torch
        from types import SimpleNamespace
        from unittest.mock import patch
        primary_owned=[name for name in s.NAMES if name!='food']
        native_food=['food','hinged_container','drink_carton']
        class Model:
            def __init__(self,names,rows):self.names=dict(enumerate(names));self.rows=rows
            def predict(self,*a,**kw):return [SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor(self.rows)))]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);models=[]
            for i,(owned,native) in enumerate([(primary_owned,s.NAMES),(['food'],native_food)]):
                p=root/f'{i}.pt';p.write_bytes(str(i).encode())
                models.append(dict(id=str(i),weights=str(p),sha256=s.file_sha(p),classes=owned,source_classes=native))
            profile=root/'profile.json';profile.write_text(json.dumps({'models':models}))
            fakes=[Model(s.NAMES,[[10,10,30,30,.99,0],[40,40,60,60,.9,2]]),
                   Model(native_food,[[10,10,30,30,.8,0],[65,65,85,85,.95,2]])]
            with patch('ultralytics.YOLO',side_effect=fakes):detector=s.SpecialistDetector(profile)
            c=json.loads((s.ROOT/'configs/sorter.json').read_text())
            result,ds=s.predict(detector,np.zeros((100,100,3),dtype=np.uint8),c,'cpu')
            self.assertEqual([d.class_id for d in ds],[2,0])
            self.assertAlmostEqual(ds[1].confidence,.8,places=5)
            models[0]['source_classes']=list(reversed(s.NAMES));profile.write_text(json.dumps({'models':models}))
            with patch('ultralytics.YOLO',side_effect=fakes):
                with self.assertRaisesRegex(ValueError,'Native checkpoint'):
                    s.SpecialistDetector(profile)


class FoodFallbackTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(s, 'FoodFallbackDetector'), 'conditional food fallback is missing')
        self.config = json.loads((s.ROOT/'configs/sorter.json').read_text())

    def detector(self, primary_rows, expert_rows, expert_names=None):
        import numpy as np, torch
        from ultralytics.engine.results import Results
        from unittest.mock import patch
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        class Model:
            def __init__(inner, names, rows):
                inner.names = dict(enumerate(names)); inner.calls = 0
                inner.result = Results(self.frame, path='fixture', names=inner.names,
                                       boxes=torch.tensor(rows, dtype=torch.float32).reshape(-1, 6))
            def predict(inner, *a, **kw):
                inner.calls += 1
                return [inner.result]
        specs = json.loads((s.ROOT/'configs/food-expert-candidate.json').read_text())['models']
        if expert_names is not None:
            specs[1]['source_classes'] = expert_names
        primary = Model(s.NAMES, primary_rows)
        expert = Model(specs[1]['source_classes'], expert_rows)
        def init(obj, *a, **kw):
            obj.models = list(zip(specs, [primary, expert]))
            obj.names = dict(enumerate(s.NAMES)); obj.checkpoint_sha256 = 'original-profile'
        with patch.object(s.SpecialistDetector, '__init__', init):
            detector = s.FoodFallbackDetector()
        return detector, primary, expert

    def test_confident_primary_food_keeps_result_and_skips_expert(self):
        detector, primary, expert = self.detector([[10,10,30,30,.8,0],[40,40,60,60,.9,2]], [])
        result, ds = s.predict(detector, self.frame, self.config, 'cpu')
        self.assertIs(result, primary.result)
        self.assertEqual(expert.calls, 0)
        self.assertEqual([d.class_id for d in ds], [0,2])

    def test_missing_food_uses_expert_without_remapping_other_classes(self):
        detector, primary, expert = self.detector(
            [[10,10,30,30,.55,0],[40,40,60,60,.9,2],[60,60,90,90,.7,6]],
            [[10,10,30,30,.85,0],[50,50,80,80,.99,2]])
        _, ds = s.predict(detector, self.frame, self.config, 'cpu')
        self.assertEqual([d.class_id for d in ds], [2,6,0])
        self.assertAlmostEqual(ds[0].confidence, .9, places=6)
        self.assertAlmostEqual(ds[-1].confidence, .85, places=6)
        self.assertEqual(expert.calls, 1)

    def test_consultation_uses_configured_display_threshold(self):
        detector, _, expert = self.detector([[10,10,30,30,.75,0]], [[10,10,30,30,.9,0]])
        self.config['confidence'] = .8
        _, ds = s.predict(detector, self.frame, self.config, 'cpu')
        self.assertEqual(expert.calls, 1)
        self.assertAlmostEqual(ds[0].confidence, .9, places=6)

    def test_empty_output_retains_valid_shape(self):
        detector, _, _ = self.detector([], [])
        result, ds = s.predict(detector, self.frame, self.config, 'cpu')
        self.assertEqual(ds, [])
        self.assertEqual(tuple(result.boxes.data.shape), (0,6))

    def test_expert_food_is_mapped_by_native_name_not_id_zero(self):
        detector, _, _ = self.detector([], [[10,10,30,30,.99,0],[40,40,60,60,.85,1]],
                                       ['drink_carton','food','hinged_container'])
        _, ds = s.predict(detector, self.frame, self.config, 'cpu')
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0].class_id, 0)
        self.assertAlmostEqual(ds[0].confidence, .85, places=6)
        self.assertAlmostEqual(ds[0].box[0], .4, places=6)

    def test_expert_failure_propagates(self):
        from unittest.mock import patch
        detector, _, expert = self.detector([], [])
        with patch.object(expert, 'predict', side_effect=RuntimeError('expert failed')):
            with self.assertRaisesRegex(RuntimeError, 'expert failed'):
                s.predict(detector, self.frame, self.config, 'cpu')

    def test_incompatible_profile_rejected_before_loading(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'profile.json'; path.write_text(json.dumps({'models':[]}))
            with patch.object(s.SpecialistDetector, '__init__') as loaded:
                with self.assertRaisesRegex(ValueError, 'food fallback profile'):
                    s.FoodFallbackDetector(path)
                loaded.assert_not_called()

    def test_routing_has_distinct_signature_and_metadata(self):
        from types import SimpleNamespace
        detector, _, _ = self.detector([], [])
        self.assertNotEqual(detector.checkpoint_sha256, 'original-profile')
        args = SimpleNamespace(device='cpu', weights=None)
        metadata = s.detector_metadata(detector, dict(self.config, detector='food-fallback'), args)
        self.assertEqual(metadata['routing_policy'], detector.routing_policy)
        self.assertEqual(len(metadata['checkpoints']), 2)

    def test_hardware_modes_rejected_before_detector_load(self):
        import contextlib, io
        from unittest.mock import patch
        for mode in ['manual', 'autonomous']:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), patch.object(s, 'FoodFallbackDetector') as loaded:
                with self.assertRaises(SystemExit):
                    s.main(['--detector','food-fallback','--mode',mode,'--simulate'])
                loaded.assert_not_called()
            self.assertIn('food-fallback requires detection mode', err.getvalue())


if __name__ == '__main__':
    unittest.main()

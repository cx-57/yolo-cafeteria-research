import sys
import unittest
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import packet_big_run as b


class PacketBigRunTests(unittest.TestCase):
    def test_packet_boxes_are_validated_before_training(self):
        for box in [[0,0,0,1], [-.1,0,.2,.5], [.5,.2,.4,.8], [0,0,1.1,1]]:
            row = dict(name='a', split='train', group='g', sha256='s', pixels='p', boxes=[box])
            with self.assertRaises(ValueError):b.ps.validate_records([row])

    def test_packet_coordinates_keep_the_same_object_extent(self):
        self.assertEqual(b.ps.yolo_row([.2,.1,.8,.9]),
                         '0 0.500000000 0.500000000 0.600000000 0.800000000')

    def test_packet_source_aliases_cannot_cross_existing_source_splits(self):
        a = dict(name='a', split='train', group='one', sha256='s', pixels='p', boxes=[])
        c = dict(name='b', split='val', group='two', sha256='t', pixels='q', boxes=[])
        b.ps.validate_records([a,c])
        with self.assertRaises(ValueError):b.ps.validate_records([a,dict(c,pixels='p')])

    def records(self):
        return [dict(name=str(i), group=str(i), sha256=str(i), pixels=str(i),
                     boxes=[[.1,.1,.2,.2]] if i % 2 else [], provenance={}) for i in range(30)]

    def test_monitor_is_deterministic_disjoint_and_positive_supported(self):
        rows = self.records()
        train, monitor = b.grouped_monitor_split(rows)
        self.assertEqual((train, monitor), b.grouped_monitor_split(rows))
        self.assertEqual(len(train) + len(monitor), len(rows))
        self.assertFalse({r['group'] for r in train} & {r['group'] for r in monitor})
        self.assertTrue(any(r['boxes'] for r in train))
        self.assertTrue(any(r['boxes'] for r in monitor))

    def test_transitive_aliases_cannot_cross_monitor_split(self):
        rows = self.records()
        rows[1]['pixels'] = rows[0]['pixels']
        rows[2]['provenance'] = dict(pageid=123)
        rows[1]['provenance'] = dict(pageid=123)
        train, monitor = b.grouped_monitor_split(rows)
        for split in (train, monitor):
            names = {r['name'] for r in split}
            self.assertTrue(not names.intersection({'0','1','2'}) or {'0','1','2'}.issubset(names))

    def test_insufficient_group_support_rejected(self):
        rows = self.records()
        for r in rows:r['group'] = 'same-session'
        with self.assertRaises(ValueError):b.grouped_monitor_split(rows)

    def test_prepared_data_rejects_changed_labels_yaml_and_image_links(self):
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, source = root/'run', root/'source'
            train, monitor = b.grouped_monitor_split(self.records())
            for split, rows in [('train', train), ('val', monitor)]:
                for kind in ['images', 'labels']:(run/'data'/kind/split).mkdir(parents=True)
                for r in rows:
                    p = source/'images/train'/r['name'];p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(r['name'].encode());r['sha256'] = b.s.sha(p)
                    (run/'data/images'/split/r['name']).symlink_to(p)
                    (run/'data/labels'/split/(r['name']+'.txt')).write_text(''.join(b.ps.yolo_row(box)+'\n' for box in r['boxes']))
            yml = run/'data/data.yaml'
            original = yaml.safe_dump(dict(path=str(run/'data'),train='images/train',val='images/val',names=['condiment_packet']))
            yml.write_text(original)
            b.verify_prepared_files(run, source, train, monitor)
            label = run/'data/labels/train'/(train[0]['name']+'.txt')
            old = label.read_text();label.write_text('0 .5 .5 .9 .9\n')
            with self.assertRaises(ValueError):b.verify_prepared_files(run, source, train, monitor)
            label.write_text(old);yml.write_text(original.replace('images/val','images/train'))
            with self.assertRaises(ValueError):b.verify_prepared_files(run, source, train, monitor)
            yml.write_text(original)
            image = run/'data/images/train'/train[0]['name'];image.unlink();image.symlink_to(source/'images/train'/train[1]['name'])
            with self.assertRaises(ValueError):b.verify_prepared_files(run, source, train, monitor)

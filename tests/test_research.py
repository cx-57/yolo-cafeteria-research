"""Research-audit and bounded candidate experiment tests."""

# --- test_research_audit.py ---
import sys
import tempfile
import unittest
import random
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_audit as a


class ResearchAuditTests(unittest.TestCase):
    def test_freeze_rejects_changed_external_annotation_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'labels.json'
            manifest.write_text('{"rows": []}')
            freeze = dict(protected_sha256={}, manifests={
                'test': dict(path=str(manifest), sha256=a.s.sha(manifest))})
            a.verify_freeze(freeze, root)
            manifest.write_text('{"rows": ["changed"]}')
            with self.assertRaisesRegex(ValueError, 'test'):
                a.verify_freeze(freeze, root)

    def test_bin_format_search_records_hashes_but_does_not_assert_human_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'bins.csv').write_text('image,disposal_bin\na.jpg,recycling\n')
            (root/'classes.yaml').write_text('names: [compost, recycling, landfill]\n')
            (root/'objects.xml').write_text('<annotation><object><name>Plastic Bottle</name></object></annotation>')
            result = a.search_label_formats(root)
            self.assertEqual(result['counts'], dict(csv=1, xml=1, yaml=1))
            self.assertEqual(len(result['file_hashes']), 3)
            self.assertEqual(result['bin_class_candidates'], ['classes.yaml'])
            self.assertEqual(result['csv_header_candidates'][0]['path'], 'bins.csv')
            self.assertNotIn('verified_human_labels', result)

    def test_school_review_figures_are_not_original_candidate_photographs(self):
        self.assertTrue(a.is_candidate_path('data/school-submission-20261027/images/IMG_3403.jpg'))
        self.assertTrue(a.is_candidate_path('data/school-submission-20261031/originals/IMG_4066.HEIC'))
        self.assertFalse(a.is_candidate_path('data/school-submission-20261027/review/sheet.jpg'))
        self.assertFalse(a.is_candidate_path('artifacts/run/plot.png'))

    def test_perceptual_index_agrees_with_brute_force_radius_search(self):
        rng = random.Random(42)
        codes = [rng.getrandbits(64) for _ in range(100)]
        index = a.PerceptualIndex()
        for i, code in enumerate(codes):
            index.add(code, str(i))
        for query in [codes[0], codes[1]^31, codes[2]^255, rng.getrandbits(64)]:
            expected = min((query^code).bit_count() for code in codes)
            found = index.find([query], 5)
            self.assertEqual(found is None, expected > 5)
            if found:
                self.assertEqual(found['distance'], expected)

    def test_reencoding_is_pixel_duplicate_and_rotation_is_near_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new('RGB', (31, 23))
            image.putdata([((x*17+y*3)%256, (x*7+y*19)%256, (x*y)%256)
                           for y in range(23) for x in range(31)])
            image.save(root/'a.png', compress_level=0)
            image.save(root/'b.png', compress_level=9)
            image.transpose(Image.Transpose.ROTATE_90).save(root/'rotated.png')
            first, second, rotated = [a.fingerprint(root/p) for p in ['a.png','b.png','rotated.png']]
            self.assertNotEqual(first['sha256'], second['sha256'])
            self.assertEqual(first['pixel_sha256'], second['pixel_sha256'])
            self.assertEqual(a.hash_distance(first['dhashes'], rotated['dhashes']), 0)

    def test_capture_day_overlap_blocks_even_distinct_images(self):
        candidate = dict(sha256='unseen', pixel_sha256='different', groups=['school-day-2026-10-29'], dhashes=[0])
        reference = dict(sha256='trained', pixel_sha256='other', groups=['school-day-2026-10-29'], dhashes=[2**64-1])
        reasons = a.overlap_reasons(candidate, reference)
        self.assertIn('capture_or_group_overlap', reasons)
        self.assertNotIn('exact_bytes', reasons)

    def test_no_match_does_not_certify_independence(self):
        self.assertEqual(a.independence_status([], False), 'unresolved_history')
        self.assertEqual(a.independence_status(['exact_bytes'], True), 'exposure_or_duplicate')
        self.assertEqual(a.independence_status([], True), 'eligible_pending_human_provenance_review')

    def test_decode_failure_is_not_evidence_of_training_exposure(self):
        self.assertEqual(a.independence_status(['decode_unavailable_no_visual_screen']),
                         'unresolved_visual_screen')

    def test_fn_diagnostics_separate_threshold_localization_and_no_detection(self):
        truth = [[.1,.1,.3,.3,0], [.4,.4,.6,.6,0], [.7,.7,.9,.9,0]]
        predictions = [[.1,.1,.3,.3,.49,0], [.5,.4,.7,.6,.8,0]]
        result = a.error_diagnostics(truth, predictions, 0)
        self.assertEqual(result['false_negatives']['below_automatic_threshold'], 1)
        self.assertEqual(result['false_negatives']['localization_candidate'], 1)
        self.assertEqual(result['false_negatives']['no_same_class_overlap'], 1)
        self.assertEqual(result['false_positives']['localization_candidate'], 1)

    def test_duplicate_detections_are_not_new_true_positives(self):
        result = a.error_diagnostics([[.1,.1,.3,.3,6]],
                                    [[.1,.1,.3,.3,.9,6],[.1,.1,.3,.3,.8,6]],6)
        self.assertEqual(result['tp'], 1)
        self.assertEqual(result['false_positives']['duplicate_or_competing_box'], 1)


# --- test_explore_candidates.py ---
"""Exploratory runner tests. No real YOLO checkpoints or independent test labels."""
import hashlib
import io
import json
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import explore_candidates as ex


def make_zip(path, changed_hash=False, missing=False, duplicate=False):
    image=Image.new('RGB',(50,40),'orange')
    bio=io.BytesIO();image.save(bio,format='JPEG');data=bio.getvalue()
    sha=hashlib.sha256(data).hexdigest()
    row=dict(path='images/test.jpg',sha256='0'*64 if changed_hash else sha,
             source_hint='bottle',source='synthetic')
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('manifest.json',json.dumps({'independent_certified_images':0,
                   'images':[row]+([row] if duplicate else [])}))
        if not missing:z.writestr('images/test.jpg',data)


class ExploratoryWorkflowTests(unittest.TestCase):
    def test_real_bundle_verifies_without_inference_when_present(self):
        actual=Path('/mnt/data/cafeteria-44-image-review-kit-final.zip')
        if not actual.exists():self.skipTest('Conversation bundle not in CI')
        with tempfile.TemporaryDirectory() as td:
            manifest=ex.unpack_checked(actual,Path(td))
            self.assertEqual(44,len(manifest['images']))
            self.assertEqual(0,manifest['independent_certified_images'])

    def test_rejects_tampered_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.zip';make_zip(p,changed_hash=True)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                ex.unpack_checked(p,Path(td)/'expanded')

    def test_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.zip';make_zip(p,missing=True)
            with self.assertRaisesRegex(ValueError,'identity'):
                ex.unpack_checked(p,Path(td)/'expanded')

    def test_rejects_duplicate_manifest_paths(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.zip';make_zip(p,duplicate=True)
            with self.assertRaisesRegex(ValueError,'Repeated image path'):
                ex.unpack_checked(p,Path(td)/'expanded')

    def test_synthetic_summary_explicitly_has_no_accuracy(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source/images';source.mkdir(parents=True)
            Image.new('RGB',(60,70),'white').save(source/'test.jpg')
            absname=str((source/'test.jpg').resolve())
            manifest=dict(images=[dict(path='images/test.jpg',source='synthetic',source_hint='bottle')])
            decisions={absname:[dict(object_class='bottle',detection_confidence=.9,
                 category='recycling',suggested_category='recycling',reasons=[],
                 box_xyxy_normalized=[.2,.3,.8,.9])]}
            report=ex.summarize(decisions,manifest,root,'test.zip')
            self.assertEqual(1,report['detected_objects_at_display_floor'])
            self.assertEqual({'bottle':1},report['detections_by_object_class'])
            self.assertIsNone(report['map50'])
            self.assertIsNone(report['actual_correct_bin_accuracy'])
            self.assertFalse(report['independent_performance_estimated'])
            self.assertTrue((root/'index.html').is_file())

    def test_frozen_hash_tamper_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'configs').mkdir();(root/'weights').mkdir()
            protected={}
            paths=['configs/specialists.json','configs/disposal.json',
                   'weights/food-container-carton.pt','weights/plate-wrapper.pt',
                   'weights/utensil.pt','weights/bottle.pt']
            for path in paths:
                p=root/path;p.write_bytes(b'known-fixed-data')
                protected[path]=hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'configs/research-freeze.json').write_text(json.dumps({'protected_sha256':protected}))
            with patch.object(ex.s,'sha',side_effect=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()):
                self.assertEqual(protected,ex.check_frozen(root))
                (root/'weights/bottle.pt').write_bytes(b'altered')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    ex.check_frozen(root)


if __name__=='__main__':unittest.main()


# --- test_guard_candidate.py ---
import copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import guard_candidate as g


class CandidateGuardTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "images": 30, "manifest_sha256": "frozen-labels",
            "AP50": .75, "AP50_95": .55,
            "per_class": {
                "food": {"AP50": .70, "AP50_95": .50, "objects": 50},
                "bottle": {"AP50": .80, "AP50_95": .60, "objects": 25},
            },
        }
        self.candidate = copy.deepcopy(self.base)
        self.candidate["AP50"] = .76
        self.candidate["AP50_95"] = .558
        self.candidate["per_class"]["bottle"]["AP50"] = .81
        self.candidate["per_class"]["bottle"]["AP50_95"] = .615

    def test_target_improves_without_other_regression(self):
        result = g.check_candidate(self.base, self.candidate, "bottle")
        self.assertTrue(result["passed"])
        self.assertEqual(result["blockers"], [])

    def test_other_class_regression_is_blocked(self):
        self.candidate["per_class"]["food"]["AP50_95"] = .49
        result = g.check_candidate(self.base, self.candidate, "bottle")
        self.assertFalse(result["passed"])
        self.assertTrue(any("food" in x for x in result["blockers"]))

    def test_candidate_must_retain_frozen_label_manifest(self):
        self.candidate["manifest_sha256"] = "changed-labels"
        self.assertFalse(g.check_candidate(self.base, self.candidate, "bottle")["passed"])
        self.candidate["manifest_sha256"] = "frozen-labels"
        self.candidate["per_class"]["bottle"]["objects"] = 24
        self.assertFalse(g.check_candidate(self.base, self.candidate, "bottle")["passed"])

    def test_target_gain_and_aggregate_both_required(self):
        self.candidate["per_class"]["bottle"]["AP50_95"] = .601
        self.assertFalse(g.check_candidate(self.base, self.candidate, "bottle")["passed"])
        self.candidate["per_class"]["bottle"]["AP50_95"] = .615
        self.candidate["AP50_95"] = .54
        self.assertFalse(g.check_candidate(self.base, self.candidate, "bottle")["passed"])

    def test_historical_retention_can_pass_without_extra_target_gain(self):
        old = copy.deepcopy(self.base)
        new = copy.deepcopy(self.base)
        self.assertTrue(g.check_candidate(old, new, "bottle",
                                          minimum_gain=0)["passed"])
        new["per_class"]["food"]["AP50_95"] -= .005
        self.assertFalse(g.check_candidate(old, new, "bottle",
                                           minimum_gain=0)["passed"])

    def test_invalid_metrics_rejected(self):
        self.candidate["per_class"]["bottle"]["AP50"] = float("nan")
        with self.assertRaises(ValueError):
            g.check_candidate(self.base, self.candidate, "bottle")
        with self.assertRaises(ValueError):
            g.check_candidate(self.base, self.base, "missing")


# --- test_hinged_candidate.py ---
"""Tests for separate 3-class candidate construction. Synthetic labels only."""

import hinged_candidate as h


def fake(root, name, classes):
    folder=root/'images'/'train'
    folder.mkdir(parents=True,exist_ok=True)
    image=folder/(name+'.jpg')
    image.write_bytes(('synthetic-'+name).encode())
    label=root/'labels'/'train'/(name+'.txt')
    label.parent.mkdir(parents=True,exist_ok=True)
    label.write_text(''.join(f'{c} 0.5 0.5 0.5 0.5\n' for c in classes))
    return dict(image=str(image),label=str(label),
                sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                label_sha256=hashlib.sha256(label.read_bytes()).hexdigest())


class HingedTests(unittest.TestCase):
    def make_info(self,root):
        train=[
            *[fake(root,f'hinge{i}',[3,0]) for i in range(6)],
            *[fake(root,f'carton{i}',[4]) for i in range(4)],
            *[fake(root,f'food{i}',[0]) for i in range(4)],
            *[fake(root,f'plate{i}',[1]) for i in range(4)],
            fake(root,'both',[3,4,1]) ]
        # Make a distinct separate val
        validation=[fake(root,'val_hinge',[3]),fake(root,'val_food',[0]),
                    fake(root,'val_carton',[4]),fake(root,'val_plate',[1])]
        return {'records':{'train':train,'val':validation}}

    def test_select_all_hinges_and_plate_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            info=self.make_info(Path(tmp)/'src')
            selected,hinges,negs=h.choose_train(info['records']['train'],
                                                  seed=7,support_cap=2,background_cap=3)
            self.assertEqual(7,hinges)
            self.assertEqual(3,negs)
            self.assertEqual(14,len(selected))
            self.assertTrue(all(r['image'] not in {v['image'] for v in info['records']['val']} for r in selected))

    def test_three_class_remap_and_validation_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=self.make_info(root/'src')
            output=root/'snapshot'
            report=h.export_dataset(info,output,repeat=3,support_cap=2,background_cap=2)
            self.assertEqual(7,report['train_hinged_views'])
            self.assertEqual(4,report['val_images'])
            config=__import__('yaml').safe_load((output/'data.yaml').read_text())
            self.assertEqual(['food','hinged_container','drink_carton'],config['names'])
            val_files=list((output/'labels'/'val').glob('*.txt'))
            self.assertEqual(4,len(val_files))
            self.assertEqual(1,sum('1 ' in x.read_text() for x in val_files))
            train_files=list((output/'labels'/'train').glob('*.txt'))
            self.assertEqual(7*3 + 2+2+2,len(train_files))
            self.assertTrue(all(int(row.split()[0])<3
                for p in train_files for row in p.read_text().splitlines()))
            self.assertEqual('',next(p for p in val_files if p.read_text()=='').read_text())

    def test_corrected_frozen_validation_overrides_stale_disk_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=self.make_info(root/'src')
            corrected={r['sha256']:[] for r in info['records']['val']}
            corrected[info['records']['val'][-1]['sha256']]=['3 0.5 0.5 0.5 0.5']
            out=root/'snapshot'
            h.export_dataset(info,out,repeat=1,support_cap=1,background_cap=1,
                             val_rows_by_sha=corrected)
            val_files=list((out/'labels'/'val').glob('*.txt'))
            self.assertEqual(1,sum(x.read_text().startswith('1 ') for x in val_files))
            self.assertEqual(3,sum(not x.read_text().strip() for x in val_files))

    def test_cannot_reuse_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=self.make_info(root/'src')
            out=root/'snapshot';out.mkdir()
            with self.assertRaises(FileExistsError):
                h.export_dataset(info,out)

    def test_invalid_repeat_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=self.make_info(root/'src')
            with self.assertRaises(ValueError):
                h.export_dataset(info,root/'snapshot',repeat=20)

    def test_no_hinge_cannot_be_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            info=self.make_info(Path(tmp)/'src')
            info['records']['train']=[r for r in info['records']['train']
                                      if not any(c==3 for c,_ in h.parse_rows(r['label']))]
            with self.assertRaisesRegex(ValueError,'hinged'):
                h.choose_train(info['records']['train'])

    def test_extra_annotated_checks_and_train_only(self):
        import hashlib, zipfile
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            src=root/'extra.zip'
            img=b'fake-image-bytes'
            label=b'1 0.5 0.5 0.6 0.6\n'
            image_hash=hashlib.sha256(img).hexdigest()
            label_hash=hashlib.sha256(label).hexdigest()
            entries=[]
            with zipfile.ZipFile(src,'w') as z:
                for i in range(29):
                    photo=img+str(i).encode()
                    digest=hashlib.sha256(photo).hexdigest()
                    z.writestr(f'images/train/ex{i}.jpg',photo)
                    z.writestr(f'labels/train/ex{i}.txt',label)
                    entries.append(dict(image=f'images/train/ex{i}.jpg',
                        label=f'labels/train/ex{i}.txt',image_sha256=digest,
                        label_sha256=label_hash,source_gallery_index=i,
                        boxes=[dict(class_id=1)]))
                z.writestr('manifest.json',json.dumps(dict(
                    status='AI_VISUAL_DRAFT_NOT_HUMAN_VERIFIED',
                    classes=['food','hinged_container','drink_carton'],images=entries)))
            dest=root/'out';(dest/'images/train').mkdir(parents=True)
            (dest/'labels/train').mkdir(parents=True)
            r=h.append_new_annotated_train(src,dest,set(),set(),repeat=2)
            self.assertEqual(r['images_added'],29)
            self.assertEqual(r['training_views_added'],58)
            self.assertEqual(len(list((dest/'labels/train').glob('*.txt'))),58)
            with self.assertRaisesRegex(ValueError,'overlaps'):
                h.append_new_annotated_train(src,root/'out2',{entries[0]['image_sha256']},set(),repeat=2)

    def test_v4_parent_checkpoint_provenance_and_comparison(self):
        import specialists
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            parent=root/'v3'; (parent/'fit/weights').mkdir(parents=True)
            w=parent/'fit/weights/best.pt';w.write_bytes(b'synthetic-v3-model')
            baseline=specialists.read(specialists.ROOT/'results/baseline.json')
            config=copy.deepcopy(specialists.read(specialists.CONFIG))
            primary=next(x for x in config['models'] if x['id']=='food-container-carton')
            primary['weights']=str(w.resolve())
            primary['sha256']=specialists.sha(w)
            (parent/'experiment.json').write_text(json.dumps(dict(
                status='TRAINED_NOT_PROMOTED',candidate_weight=str(w),
                candidate_sha256=specialists.sha(w),
                original_frozen_checkpoint_sha256=baseline['checkpoint_sha256']['food-container-carton'],
                corrected_validation_manifest_sha256=baseline['manifest_sha256'])))
            (parent/'candidate-specialists.json').write_text(json.dumps(config))
            (parent/'candidate-validation.json').write_text(json.dumps(baseline))
            (parent/'stress-comparison').mkdir()
            prior_stress=dict(images=44,
                original_checkpoint_sha256=baseline['checkpoint_sha256']['food-container-carton'],
                source_hint_image_counts={'hinged_container_candidate':12,'drink_carton':6},
                class_hits_by_source_hint={
                   'frozen': {'hinged_container_candidate':4,'drink_carton':5},
                   'candidate': {'hinged_container_candidate':8,'drink_carton':6}})
            (parent/'stress-comparison/comparison.json').write_text(json.dumps(prior_stress))
            result=h.verify_parent_candidate(parent,baseline)
            self.assertEqual(8,result['parent_stress_hinged_hits'])
            self.assertEqual(specialists.sha(w),result['initial_weights_sha256'])
            validation=h.parent_validation_report(parent,baseline,baseline)
            self.assertAlmostEqual(0,validation['delta_AP50'])
            fresh=copy.deepcopy(prior_stress)
            fresh['class_hits_by_source_hint']['candidate']['hinged_container_candidate']=9
            self.assertEqual(1,h.parent_stress_report(parent,fresh)['change'])
            w.write_bytes(b'altered-unpinned')
            with self.assertRaisesRegex(ValueError,'SHA'):
                h.verify_parent_candidate(parent,baseline)

    def test_v4_parent_validation_compares_identical_support(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)
            original=specialists.read(specialists.ROOT/'results/baseline.json')
            (parent/'candidate-validation.json').write_text(json.dumps(original))
            candidate=copy.deepcopy(original)
            candidate['AP50']=original['AP50']+0.002
            candidate['per_class']['food']['AP50']+=0.014
            report=h.parent_validation_report(parent,candidate,original)
            self.assertAlmostEqual(0.002,report['delta_AP50'])
            self.assertAlmostEqual(0.014,report['per_class_delta']['food']['AP50'])

    def test_v4_rejects_wrong_parent_class_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'fit/weights').mkdir(parents=True)
            p=root/'fit/weights/best.pt';p.write_bytes(b'one')
            baseline=specialists.read(specialists.ROOT/'results/baseline.json')
            cfg=copy.deepcopy(specialists.read(specialists.CONFIG))
            model=next(x for x in cfg['models'] if x['id']=='food-container-carton')
            model['classes']=['food','plate','drink_carton']
            model['weights']=str(p);model['sha256']=specialists.sha(p)
            (root/'experiment.json').write_text(json.dumps(dict(
                status='TRAINED_NOT_PROMOTED',candidate_weight=str(p),
                candidate_sha256=specialists.sha(p),
                original_frozen_checkpoint_sha256=baseline['checkpoint_sha256']['food-container-carton'],
                corrected_validation_manifest_sha256=baseline['manifest_sha256'])))
            (root/'candidate-specialists.json').write_text(json.dumps(cfg))
            (root/'candidate-validation.json').write_text(json.dumps(baseline))
            (root/'stress-comparison').mkdir()
            (root/'stress-comparison/comparison.json').write_text(json.dumps(dict(
                images=44,original_checkpoint_sha256=baseline['checkpoint_sha256']['food-container-carton'],
                source_hint_image_counts={'hinged_container_candidate':12},
                class_hits_by_source_hint={'candidate':{'hinged_container_candidate':8}})))
            with self.assertRaisesRegex(ValueError,'three-class'):
                h.verify_parent_candidate(root,baseline)

if __name__=='__main__':unittest.main()

"""Real source-ledger mutation tests; no files or images are modified."""
from pathlib import Path
import importlib.util
import copy
from unittest.mock import patch
import unittest
p=Path(__file__).with_name('sources.py');spec=importlib.util.spec_from_file_location('sources42test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SourceTests(unittest.TestCase):
 def test_stale_second_review_rejected(self):
  original=m.helper.snapshot
  def changed(path):
   obj,digest=original(path)
   if str(path).endswith('openimages-expansion-staging-v37/second-review.json'):
    obj=copy.deepcopy(obj);obj['first_review_sha256']='stale'
   return obj,digest
  # Prior sources are independently tested in V40; avoid decoding them again here.
  prior=dict(records=[],source_input_shas={},permanent_monitor_records=[])
  def snapshot(path):
   if str(path).endswith('source-coverage-diagnostic-v40/dataset.json'):
    return dict(source_input_shas={},selection=dict(monitor=[],remaining=[]),records=[]),'fixture'
   return changed(path)
  with patch.object(m.v40,'compile_all',return_value=prior),patch.object(m.helper,'snapshot',side_effect=snapshot),patch.object(m.helper,'check_inputs'):
   with self.assertRaises(AssertionError):m.compile_all()
 def test_capture_group_comes_from_flickr_identity(self):
  record=dict(source_type='openimages',photo_metadata=dict(OriginalLandingURL='https://www.flickr.com/photos/someone/123456/'))
  self.assertEqual(m.group(record),'flickr/123456')
 def test_missing_capture_metadata_fails_closed(self):
  with self.assertRaises(AssertionError):m.group(dict(source_type='openimages',photo_metadata=dict(OriginalLandingURL='https://example.com/no-photo-id')))
if __name__=='__main__':unittest.main()

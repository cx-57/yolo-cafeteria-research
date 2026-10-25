from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib,json
import unittest
import run
import finish
class LifecycleTests(unittest.TestCase):
 def test_actual_runner_finalizes_terminal_crash_gap_without_training(self):
  for epoch,ap,stop_marker in [(6,.9,False),(2,.8,True),(2,.8,False)]:
   with self.subTest(epoch=epoch,stop_marker=stop_marker),TemporaryDirectory() as folder:
    root=Path(folder);cache=root/'cache';last=cache/'train/weights/last.pt';last.parent.mkdir(parents=True);last.write_bytes(b'saved-checkpoint')
    declaration=root/'declaration.json';declaration.write_text('{}');digest=run.sha(declaration)
    run.save(root/'authorization.json',dict(declaration_sha=digest,optimization_authorized=True,maximum_epochs_authorized=20))
    point=dict(epoch=epoch,AP50=ap,AP50_95=.7,weights_sha=run.sha(last),declaration_sha=digest)
    run.save(root/'progress.json',point)
    if stop_marker:run.save(root/'retention-stop.json',point)
    with patch.object(run,'RUN',root),patch.object(run.data,'CACHE',cache),patch.object(run,'check_declaration'),patch.object(run.data,'verify',return_value={}),patch.object(run.evaluate,'cohorts',return_value=[{}]),patch.object(run.evaluate,'evaluate',return_value=dict(AP50=.9,AP50_95=.7)),patch.object(run,'YOLO',side_effect=AssertionError('Training model must not load')),patch.object(finish,'main') as final:
     run.main();final.assert_called_once();self.assertTrue((root/'training-complete.json').exists())
if __name__=='__main__':unittest.main()

import json
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'scripts/v5_experiment.py'


class CLITests(unittest.TestCase):
    def test_standalone_notebook_contains_executable_six_class_workflow(self):
        notebook = ROOT / 'experiments/yolo-v5-original-plus-v4/YOLO_V5_Six_Class_Unified_First.ipynb'
        doc = json.loads(notebook.read_text())
        payload = None
        for index, cell in enumerate(doc['cells']):
            if cell['cell_type'] != 'code': continue
            source = ''.join(cell['source'])
            compile(source, f'notebook-cell-{index}', 'exec')
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == 'SOURCES' for n in node.targets):
                    payload = ast.literal_eval(node.value)
        self.assertIsNotNone(payload)
        with tempfile.TemporaryDirectory() as d:
            for name, source in payload.items():
                (Path(d) / name).write_text(source)
            result = subprocess.run([sys.executable, str(Path(d) / 'v5_experiment.py'), 'plan'],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['order'], ['baseline', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6'])

    def test_plan_lists_baseline_and_six_variants_without_loading_models(self):
        result = subprocess.run([sys.executable, str(CLI), 'plan'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['order'], ['baseline', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6'])
        self.assertEqual(plan['names'], ['food', 'plate', 'utensil', 'hinged_container', 'drink_carton', 'wrapper'])
        self.assertFalse(plan['specialists_allowed'])

    def test_missing_corrected_source_fails_without_creating_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'prepared'
            missing = Path(d) / 'YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip'
            result = subprocess.run([sys.executable, str(CLI), 'prepare', '--archive', str(missing),
                                     '--exclusions', str(Path(d) / 'missing.csv'), '--output', str(out)],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unavailable', result.stderr)
            self.assertFalse(out.exists())

    def test_audit_does_not_fallback_to_any_other_zip(self):
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run([sys.executable, str(CLI), 'audit', '--archive', str(Path(d) / 'missing.zip')],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unavailable', result.stderr)


if __name__ == '__main__': unittest.main()

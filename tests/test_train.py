"""Fast synthetic tests for split integrity, 1x default, optional 10x, and annotation safety."""
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.processed = root / "processed"
        self.output = root / "training"
        self.patch_processed = patch.object(train, "DATA", root)
        self.patch_output = patch.object(train, "OUTPUT", self.output)
        self.patch_processed.start()
        self.patch_output.start()
        self.addCleanup(self.patch_processed.stop)
        self.addCleanup(self.patch_output.stop)

    def add_image(self, source, name, color, row="0 .5 .5 .4 .4"):
        image_dir = self.processed / source / "images"
        label_dir = self.processed / source / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = image_dir / (name + ".png")
        Image.new("RGB", (36, 24), color).save(filename)
        (label_dir / (name + ".txt")).write_text(row + "\n")
        return filename

    def populate(self):
        for i in range(45):
            # Colors intentionally differ: don't group unrelated files by pixel.
            self.add_image("school_lunch", f"food_{i:02d}", (i * 4, 50, 75))

    def test_ten_views_per_training_original_and_untouched_validation(self):
        self.populate()
        first_label = (self.processed / "school_lunch/labels/food_00.txt").read_text()
        train.build_dataset(augment_factor=10, seed=42)
        result = json.loads((self.output / "split_summary.json").read_text())
        t, v = result["train_originals"], result["val_originals"]
        self.assertEqual(t + v, 45)
        self.assertEqual((t, v), (36, 9))  # Exact 80/20 originals, not 80/20 of eligible only.
        self.assertEqual(result["train_views"], 10 * t)
        self.assertEqual(result["test_originals"], 0)
        self.assertGreater(v, 0)
        self.assertEqual(len(list((self.output / "images/train").glob("*.png"))), t * 10)
        self.assertEqual(len(list((self.output / "images/val").glob("*.png"))), v)
        self.assertEqual(len(list((self.output / "labels/train").glob("*.txt"))), t * 10)
        self.assertFalse(any("__view" in p.stem for p in (self.output / "images/val").iterdir()))
        self.assertEqual((self.processed / "school_lunch/labels/food_00.txt").read_text(), first_label)
        original = next(p for p in (self.output / "images/train").iterdir() if "__view" not in p.stem)
        mirrored = original.with_name(original.stem + "__view01" + original.suffix)
        self.assertEqual(os.stat(original).st_ino, os.stat(mirrored).st_ino)

    def test_ratio_applies_to_all_sources_without_leaking_train_only_photos(self):
        self.populate()
        for i in range(15):
            self.add_image("food_american", f"additional_{i:02d}",
                           (i * 3, 100, 15))
        train.build_dataset(augment_factor=1, seed=42)
        result = json.loads((self.output / "split_summary.json").read_text())
        self.assertEqual(result["original_images"], 60)
        self.assertEqual(result["target_validation_originals"], 12)
        self.assertEqual(result["val_originals"], 12)
        self.assertEqual(result["train_originals"], 48)
        self.assertEqual(result["test_originals"], 0)
        self.assertFalse(any(p.name.startswith("food_american__")
                             for p in (self.output / "images/val").iterdir()))

    def test_deterministic_grouped_split(self):
        self.populate()
        train.build_dataset(augment_factor=1, seed=9)
        before = sorted(p.name for p in (self.output / "images/val").iterdir())
        train.build_dataset(augment_factor=1, seed=9)
        after = sorted(p.name for p in (self.output / "images/val").iterdir())
        self.assertEqual(before, after)
        self.assertEqual(train.group_for_split("school_lunch",
                         "train__photo.rf.abc"), train.group_for_split("school_lunch",
                         "valid__photo.rf.xyz"))

    def test_tracked_near_duplicate_audit_is_used_by_default(self):
        self.populate()
        train.build_dataset()
        from_train = next((self.output / "images/train").glob("school_lunch__*.png"))
        from_val = next((self.output / "images/val").glob("school_lunch__*.png"))
        def original(path):
            return "school_lunch/images/" + path.name.removeprefix("school_lunch__")
        near_file = train.DATA / "audits/day1/near_duplicates.csv"
        near_file.parent.mkdir(parents=True)
        near_file.write_text("image_a,image_b\n" + original(from_train) +
                             "," + original(from_val) + "\n")
        train.build_dataset()
        summary = json.loads((self.output / "split_summary.json").read_text())
        self.assertEqual(summary["group_audit"]["matched_pairs"], 1)
        self.assertEqual(summary["group_audit"]["merged_groups"], 1)
        train_names = {p.name for p in (self.output / "images/train").iterdir()}
        val_names = {p.name for p in (self.output / "images/val").iterdir()}
        self.assertEqual(from_train.name in train_names, from_val.name in train_names)
        self.assertEqual(from_train.name in val_names, from_val.name in val_names)

    def test_exact_duplicate_cross_source_cannot_enter_validation(self):
        self.populate()
        # New food source is forced train-only. Duplicate pixel with old source
        # must drag the old source original into training too.
        self.add_image("food_american", "duplicate", (0, 50, 75))
        train.build_dataset(augment_factor=1, seed=42)
        self.assertTrue((self.output / "images/train/school_lunch__food_00.png").is_file())
        self.assertTrue((self.output / "images/train/food_american__duplicate.png").is_file())

    def test_source_labels_are_converted_only_in_generated_copy(self):
        self.populate()
        image = self.add_image("macworx", "box", (200, 20, 80),
                               "5 .5 .5 .3 .3")  # Historical ID 5 -> current ID 4.
        train.build_dataset(augment_factor=1)
        label = list((self.output / "labels/train").glob("macworx__box.txt"))
        if not label:
            label = list((self.output / "labels/val").glob("macworx__box.txt"))
        self.assertTrue(label)
        self.assertTrue(label[0].read_text().startswith("4 "))
        self.assertTrue(image.with_name(image.name).exists())
        self.assertTrue((self.processed / "macworx/labels/box.txt").read_text().startswith("5 "))

    def test_invalid_geometry_fails_before_training(self):
        self.populate()
        self.add_image("food_american", "bad", (100, 200, 100), "0 .95 .5 .4 .4")
        with self.assertRaises(ValueError):
            train.build_dataset()

    def test_roundoff_correction_does_not_edit_source(self):
        self.populate()
        original = "0 0.500000 0.502695 1.000000 0.994629"
        self.add_image("food_american", "rounding", (109, 97, 32), original)
        train.build_dataset(augment_factor=1)
        generated = self.output / "labels/train/food_american__rounding.txt"
        self.assertTrue(generated.exists())  # New imports are train-only.
        parts = generated.read_text().split()
        x, y, w, h = map(float, parts[1:])
        self.assertLessEqual(y + h / 2, 1.000001)
        self.assertEqual((self.processed / "food_american/labels/rounding.txt").read_text(),
                         original + "\n")
        report = json.loads((self.output / "split_summary.json").read_text())
        self.assertEqual(report["minor_boundary_rounding_clips_in_generated_labels"]["food_american"], 1)

    def test_orphan_label_and_missing_image_fail(self):
        self.populate()
        self.add_image("food_american", "one", (100, 100, 100))
        (self.processed / "food_american/labels/orphan.txt").write_text("0 .5 .5 .2 .2\n")
        with self.assertRaisesRegex(ValueError, "Label without image"):
            train.build_dataset()

    def test_factor_one_and_invalid_factor(self):
        self.populate()
        with self.assertRaises(ValueError):
            train.build_dataset(augment_factor=0)
        train.build_dataset()
        report = json.loads((self.output / "split_summary.json").read_text())
        self.assertEqual(report["augment_factor"], 1)
        self.assertEqual(report["train_originals"], report["train_views"])

    def test_audited_candidate_pairs_stay_in_one_split_in_an_alternate_output(self):
        self.populate()
        train.build_dataset()
        current_summary = (self.output / "split_summary.json").read_bytes()
        current_validation = sorted(p.name for p in (self.output / "images/val").iterdir())
        val_name = current_validation[0].removeprefix("school_lunch__")
        train_name = next((self.output / "images/train").iterdir()).name.removeprefix("school_lunch__")
        audit = Path(self.tmp.name) / "near_duplicates.csv"
        audit.write_text("image_a,image_b,status\n"
                         f"school_lunch/images/{val_name},school_lunch/images/{train_name},review_candidate_do_not_delete\n"
                         "absent/images/one.png,absent/images/two.png,review_candidate_do_not_delete\n")
        alternate = Path(self.tmp.name) / "grouped"
        train.build_dataset(group_audit=audit, output=alternate)
        split_by_name = {p.name: split for split in ("train", "val")
                         for p in (alternate / "images" / split).iterdir()}
        self.assertEqual(split_by_name["school_lunch__" + val_name],
                         split_by_name["school_lunch__" + train_name])
        self.assertEqual((self.output / "split_summary.json").read_bytes(), current_summary)
        self.assertEqual(sorted(p.name for p in (self.output / "images/val").iterdir()),
                         current_validation)
        grouping = json.loads((alternate / "split_summary.json").read_text())["group_audit"]
        self.assertEqual((grouping["candidate_pairs"], grouping["matched_pairs"],
                          grouping["out_of_corpus_pairs"], grouping["merged_groups"]),
                         (2, 1, 1, 1))
        self.assertEqual(grouping["path"], str(audit.resolve()))

    def test_malformed_group_audit_fails_before_replacing_generated_output(self):
        self.populate()
        train.build_dataset()
        original = (self.output / "split_summary.json").read_bytes()
        audit = Path(self.tmp.name) / "bad.csv"
        for malformed in ("other,columns\na,b\n", "image_a,image_b\n,foo/images/x.png\n",
                          "image_a,image_b\n../images/x.png,foo/images/y.png\n",
                          "image_a,image_b\nfoo/images/x.png,foo/images/y.png,extra\n"):
            with self.subTest(malformed=malformed):
                audit.write_text(malformed)
                with self.assertRaisesRegex(ValueError, "group audit"):
                    train.build_dataset(group_audit=audit)
                self.assertEqual((self.output / "split_summary.json").read_bytes(), original)

    def test_missing_class_warnings_are_visible_after_build(self):
        self.populate()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            train.build_dataset()
        self.assertIn("Missing training classes:", stdout.getvalue())
        self.assertIn("Missing validation classes:", stdout.getvalue())
        self.assertIn("plate", stdout.getvalue())

    def test_build_refuses_originals_and_unrelated_existing_output_directories(self):
        self.populate()
        unrelated = Path(self.tmp.name) / "notes"
        unrelated.mkdir()
        sentinel = unrelated / "keep.txt"
        sentinel.write_text("keep")
        for output in (train.PROJECT_ROOT, train.DATA, self.processed,
                       self.processed / "school_lunch", train.DATA / "raw",
                       train.DATA / "raw/one", train.MODEL, unrelated):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "output"):
                    train.build_dataset(output=output)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertTrue((self.processed / "school_lunch/images/food_00.png").exists())

    def test_build_does_not_delete_output_referenced_by_a_source_symlink(self):
        self.populate()
        train.build_dataset()
        generated = next((self.output / "images/train").iterdir())
        source = self.processed / "school_lunch/images" / generated.name.removeprefix("school_lunch__")
        source.unlink()
        source.symlink_to(generated)
        with self.assertRaisesRegex(ValueError, "output.*source"):
            train.build_dataset()
        self.assertTrue(generated.exists())

    def test_build_does_not_delete_output_referenced_by_a_source_label_symlink(self):
        self.populate()
        train.build_dataset()
        generated = next((self.output / "labels/train").iterdir())
        source = self.processed / "school_lunch/labels" / generated.name.removeprefix("school_lunch__")
        source.unlink()
        source.symlink_to(generated)
        summary = (self.output / "split_summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "output.*source"):
            train.build_dataset()
        self.assertTrue(generated.exists())
        self.assertEqual((self.output / "split_summary.json").read_bytes(), summary)


class TrainingControlsTests(unittest.TestCase):
    """Exercise the CLI/report boundary without downloading or fitting a detector."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_dir = self.root / "model"
        self.data_file = self.root / "data.yaml"
        self.data_file.write_text(train.yaml.safe_dump({
            "names": train.class_names(), "val": "images/val"}))
        self.models = []
        self.checkpoint_names = dict(enumerate(train.class_names()))
        self.metrics_mutation = None
        self.legacy_validator_support = False
        case = self

        class RecordingYOLO:
            def __init__(self, weights):
                self.weights = weights
                self.loaded_from = Path.cwd()
                self.names = case.checkpoint_names
                self.overrides = {"imgsz": 896, "batch": 16}
                self.callbacks = {}
                self.trainer = SimpleNamespace(save_dir=case.model_dir / "fit")
                case.models.append(self)

            def train(self, **options):
                self.training_options = options

            def add_callback(self, event, callback):
                self.callbacks[event] = callback

            def val(self, **options):
                self.validation_options = options
                save_dir = Path(options["project"]) / options["name"]
                save_dir.mkdir(parents=True, exist_ok=True)
                effective = {**self.overrides, **options}
                effective["imgsz"] = ((effective["imgsz"] + 31) // 32) * 32
                if "on_val_end" in self.callbacks:
                    self.callbacks["on_val_end"](SimpleNamespace(
                        args=SimpleNamespace(**effective)))
                metrics = SimpleNamespace(
                    save_dir=save_dir,
                    box=SimpleNamespace(
                        mp=0.5, mr=0.35, map50=0.55, map=0.375,
                        ap_class_index=[0, 4], ap50=[0.8, 0.3], ap=[0.6, 0.15],
                        p=[0.8, 0.2], r=[0.6, 0.1], f1=[24 / 35, 2 / 15]),
                    nt_per_class=[3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    nt_per_image=[2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                if case.metrics_mutation:
                    case.metrics_mutation(metrics)
                if case.legacy_validator_support:
                    self.callbacks["on_val_end"](SimpleNamespace(
                        args=SimpleNamespace(**effective),
                        nt_per_class=metrics.nt_per_class,
                        nt_per_image=metrics.nt_per_image))
                    del metrics.nt_per_class
                    del metrics.nt_per_image
                return metrics

        model_patch = patch.object(train, "MODEL", self.model_dir)
        module_patch = patch.dict(sys.modules, {
            "ultralytics": SimpleNamespace(YOLO=RecordingYOLO)})
        model_patch.start()
        module_patch.start()
        self.addCleanup(model_patch.stop)
        self.addCleanup(module_patch.stop)

    def run_cli(self, *args):
        with patch.object(sys, "argv", ["train.py", str(self.data_file), *args]):
            train.main()

    def report(self, name):
        return json.loads((self.model_dir / "runs" / name / "metrics.json").read_text())

    def test_evaluation_flags_reach_validator_and_report_effective_settings(self):
        self.run_cli("--evaluate", "best.pt", "--imgsz", "961", "--batch", "4",
                     "--workers", "2", "--eval-name", "comparison")
        options = self.models[0].validation_options
        self.assertEqual((options["imgsz"], options["batch"], options["workers"]),
                         (961, 4, 2))
        result = self.report("comparison")
        self.assertEqual((result["imgsz"], result["batch"]), (992, 4))
        self.assertEqual(result["weights"], str(Path("best.pt").resolve()))
        self.assertEqual(result["data"], str(self.data_file.resolve()))

    def test_legacy_validator_counts_are_used_when_metrics_lack_support(self):
        self.legacy_validator_support = True
        train.evaluate("best.pt", self.data_file)
        result = self.report("evaluation_val")
        self.assertEqual(result["per_class"]["food"]["support"], 3)
        self.assertEqual(result["per_class"]["hinged_container"]["images"], 1)

    def test_fine_tune_can_skip_initial_warmup(self):
        self.run_cli("--warmup-epochs", "0", "--epochs", "3")
        self.assertEqual(self.models[0].training_options["warmup_epochs"], 0)

    def test_omitted_evaluation_imgsz_uses_checkpoint_size_and_name_is_explicit(self):
        self.run_cli("--evaluate", "best.pt")
        self.assertNotIn("imgsz", self.models[0].validation_options)
        self.assertEqual(self.report("evaluation_val")["imgsz"], 896)
        self.run_cli("--evaluate", "best.pt", "--name", "chosen")
        self.assertEqual(self.models[-1].validation_options["name"], "chosen")

    def test_report_aligns_metrics_and_does_not_invent_ap_for_unsupported_classes(self):
        train.evaluate("best.pt", self.data_file)
        result = self.report("evaluation_val")
        self.assertEqual(result["map50"], 0.55)
        food = result["per_class"]["food"]
        self.assertEqual((food["ap50"], food["map50_95"]), (0.8, 0.6))
        self.assertEqual((food["precision"], food["recall"]), (0.8, 0.6))
        self.assertAlmostEqual(food["f1"], 24 / 35)
        self.assertEqual((food["support"], food["images"], food["status"]),
                         (3, 2, "supported"))
        container = result["per_class"]["hinged_container"]
        self.assertEqual((container["ap50"], container["precision"]), (0.3, 0.2))
        plate = result["per_class"]["plate"]
        self.assertEqual((plate["support"], plate["images"], plate["status"]),
                         (0, 0, "unsupported"))
        for metric in ("ap50", "map50_95", "precision", "recall", "f1"):
            self.assertIsNone(plate[metric])
        self.assertEqual(result["coverage"]["supported_class_count"], 2)
        self.assertEqual(result["coverage"]["supported_classes"], ["food", "hinged_container"])
        self.assertEqual(result["coverage"]["unsupported_class_count"], 9)
        self.assertFalse(result["coverage"]["all_classes_evaluable"])
        self.assertIn("Provisional validation", result["interpretation"])

    def test_validator_refuses_corrupt_or_misaligned_metrics(self):
        cases = (
            ("missing_ap_row", lambda m: setattr(m.box, "ap_class_index", [0]), "AP rows"),
            ("no_targets", lambda m: setattr(m, "nt_per_class", [0] * 11), "AP rows"),
            ("wrong_support_length", lambda m: setattr(m, "nt_per_image", [2]), "class support"),
            ("nan_ap", lambda m: m.box.ap.__setitem__(0, float("nan")), "nonfinite metrics"),
            ("inflated_macro", lambda m: setattr(m.box, "map50", 0.99), "macro ap50"),
            ("unsupported_ap", lambda m: m.box.ap_class_index.__setitem__(1, 3), "AP rows"),
        )
        for name, mutation, message in cases:
            with self.subTest(name=name):
                self.metrics_mutation = mutation
                with self.assertRaisesRegex(ValueError, message):
                    train.evaluate("best.pt", self.data_file)
        self.assertFalse((self.model_dir / "runs/evaluation_val/metrics.json").exists())

    def test_checkpoint_class_names_and_ids_are_checked_before_validation(self):
        for names in ({0: "other"}, {1: "food"},
                      {**dict(enumerate(train.class_names())), 11: "extra"}):
            with self.subTest(names=names):
                self.checkpoint_names = names
                with self.assertRaisesRegex(ValueError, "checkpoint.*class names or IDs"):
                    train.evaluate("old.pt", self.data_file)
                self.assertFalse(hasattr(self.models[-1], "validation_options"))

    def test_training_controls_and_matching_final_evaluation_are_forwarded(self):
        self.run_cli("--epochs", "30", "--imgsz", "960", "--batch", "4",
                     "--optimizer", "AdamW", "--lr0", "0.001", "--lrf", "0.05",
                     "--cos-lr", "--patience", "12", "--mosaic", "0.5",
                     "--close-mosaic", "7", "--workers", "2", "--name", "fit",
                     "--eval-name", "fit_validation")
        options = self.models[0].training_options
        self.assertEqual({key: options[key] for key in (
            "optimizer", "lr0", "lrf", "cos_lr", "patience", "mosaic", "close_mosaic", "workers")},
            {"optimizer": "AdamW", "lr0": 0.001, "lrf": 0.05, "cos_lr": True,
             "patience": 12, "mosaic": 0.5, "close_mosaic": 7, "workers": 2})
        validation = self.models[-1].validation_options
        self.assertEqual((validation["imgsz"], validation["batch"], validation["name"]),
                         (960, 4, "fit_validation"))

    def test_reweighting_and_freeze_options_are_opt_in(self):
        self.run_cli("--cls-pw", "0.25", "--freeze", "10", "--epochs", "3")
        training = self.models[0].training_options
        self.assertEqual(training["cls_pw"], 0.25)
        self.assertEqual(training["freeze"], 10)

    def test_invalid_class_weight_and_freeze_rejected(self):
        for args in (("--cls-pw", "-0.01"), ("--cls-pw", "1.1"),
                     ("--cls-pw", "nan"), ("--freeze", "-1")):
            self.models.clear()
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.run_cli(*args)
                self.assertEqual(self.models, [])

    def test_coverage_aware_training_requires_no_mosaic_and_passes_custom_trainer(self):
        sentinel = object()
        with patch.dict(sys.modules, {"partial_labels": SimpleNamespace(
                make_trainer=lambda path: sentinel)}):
            self.run_cli("--epochs", "2", "--coverage-policy", "policy.json",
                         "--mosaic", "0", "--batch", "4", "--workers", "0")
        training = self.models[0].training_options
        self.assertIs(training["trainer"], sentinel)
        self.assertEqual((training["mosaic"], training["mixup"],
                          training["copy_paste"], training["cutmix"]), (0, 0.0, 0.0, 0.0))

    def test_coverage_mask_refuses_mixed_source_training_or_evaluation(self):
        for argv in (("--coverage-policy", "policy.json"),
                     ("--coverage-policy", "policy.json", "--evaluate", "best.pt")):
            self.models.clear()
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    self.run_cli(*argv)
                self.assertIn("--coverage-policy", stderr.getvalue())
                self.assertEqual(self.models, [])

    def test_default_training_behavior_is_preserved(self):
        self.run_cli()
        options = self.models[0].training_options
        self.assertEqual((options["imgsz"], options["batch"], options["name"]),
                         (640, 8, "yolo11n-1x"))
        self.assertEqual((options["optimizer"], options["patience"], options["mosaic"],
                          options["close_mosaic"], options["workers"]),
                         ("auto", 25, 0.3, 10, 8))
        self.assertNotIn("lr0", options)
        self.assertNotIn("lrf", options)
        self.assertNotIn("cls_pw", options)
        self.assertNotIn("freeze", options)
        self.assertEqual(self.models[-1].validation_options["imgsz"], 640)

    def test_invalid_and_nonfinite_options_fail_before_loading_a_model(self):
        cases = [("--imgsz", "0"), ("--batch", "0"), ("--workers", "-1"),
                 ("--patience", "-1"), ("--mosaic", "1.1"), ("--mosaic", "nan"),
                 ("--close-mosaic", "-1"), ("--lrf", "inf"), ("--lrf", "-0.1"),
                 ("--lrf", "1.1"), ("--optimizer", "AdamW", "--lr0", "nan"),
                 ("--optimizer", "AdamW", "--lr0", "0")]
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                self.models.clear()
                with self.assertRaises(SystemExit):
                    self.run_cli("--evaluate", "best.pt", *arguments)
                self.assertEqual(self.models, [])

    def test_manual_lr_requires_a_manual_optimizer(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            self.run_cli("--lr0", "0.001")
        self.assertIn("--lr0 requires an explicit --optimizer", stderr.getvalue())
        self.assertEqual(self.models, [])

    def test_build_options_with_prepared_data_fail_before_loading_model(self):
        for arguments in (("--output", str(self.root / "alternate")),
                          ("--group-audit", str(self.root / "near.csv"))):
            stderr = io.StringIO()
            with self.subTest(arguments=arguments), redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    self.run_cli(*arguments)
                self.assertIn("only applies when building", stderr.getvalue())
                self.assertEqual(self.models, [])

    def test_all_bare_yolo11_names_load_and_download_inside_weights(self):
        weights = self.model_dir / "weights"
        weights.mkdir(parents=True)
        for filename in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11x.pt"):
            with self.subTest(filename=filename):
                local = weights / filename
                local.touch()
                loaded = train.load_model(filename)
                self.assertEqual(loaded.weights, str(local))
                local.unlink()
                original = Path.cwd()
                downloaded = train.load_model(filename)
                self.assertEqual(downloaded.loaded_from, weights.resolve())
                self.assertEqual(Path.cwd(), original)


class VerifiedAnnotationTests(unittest.TestCase):
    """Real source-label regression checks; catches accidental class remapping."""

    def test_verified_missing_target_repairs_keep_legal_boxes(self):
        examples = (
            ("plate", "train__lvis_000000346262.txt", [1, 0, 3]),
            ("plate", "train__lvis_000000004386.txt", [1, 0, 0]),
            ("trash_detection",
             "train__000090_jpg.rf.73d3f6dab3183f7c267d3f62ab8a22c1.txt",
             [9, 8, 8]),
        )
        for source, file, expected in examples:
            with self.subTest(file=file):
                path = ROOT / "data/processed" / source / "labels" / file
                rows, clipped = train.read_labels(path, source, train.class_names())
                # The original 346262 plate box exceeds y=1 by ~5e-10,
                # so read_labels legitimately clips that existing row once.
                self.assertEqual(clipped, int(file == "train__lvis_000000346262.txt"))
                self.assertEqual([int(row.split()[0]) for row in rows], expected)
                for row in rows:
                    _, x, y, w, h = map(float, row.split())
                    self.assertGreater(w, 0)
                    self.assertGreater(h, 0)
                    self.assertGreaterEqual(x - w / 2, -1e-6)
                    self.assertGreaterEqual(y - h / 2, -1e-6)
                    self.assertLessEqual(x + w / 2, 1 + 1e-6)
                    self.assertLessEqual(y + h / 2, 1 + 1e-6)


if __name__ == "__main__":
    unittest.main()

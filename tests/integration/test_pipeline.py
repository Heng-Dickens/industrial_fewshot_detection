import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import industrial_fewshot.runner as pipeline
from tests.support import FakeClient, fixture, response, make_image, make_reference_pair


class PipelineTest(unittest.TestCase):
    def test_score_threshold_filters_export_and_visualization_without_losing_cached_candidates(self):
        from industrial_fewshot.evaluation.visualize import render_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw['evaluation'] = {'score_threshold': 0.5}
            pipeline.write_json(path, raw)
            payload = json.loads(response(confidence=0.5))
            payload['objects'].append({'label': 'target', 'bbox_2d': [10, 20, 50, 100],
                                       'confidence': 0.49, 'hanging_object_confirmed': True})
            payload['target_count'] = 2
            client = FakeClient([json.dumps(payload)])
            first = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
            first_dir = Path(first['run_dir'])
            before = (first_dir / 'records.jsonl').read_bytes()
            exported = pipeline.read_json(first_dir / 'predictions/frame.json')['shapes']
            self.assertEqual([shape['confidence'] for shape in exported], [0.5])
            self.assertEqual(pipeline.read_json(first_dir / 'predictions_score_0.7/frame.json')['shapes'], [])
            primary_before = (first_dir / 'predictions/frame.json').read_bytes()
            raw_before = (first_dir / 'records.jsonl').read_bytes()
            (first_dir / 'predictions_score_0.7/frame.json').unlink()
            old_metrics = {key: value for key, value in first['metrics'].items()
                           if key != 'score_threshold_results'}
            pipeline.write_json(first_dir.parent / 'comparison.json',
                                {'groups': [{**first, 'metrics': old_metrics,
                                             'run_dir': str(first_dir.parent / 'moved' / first_dir.name)}],
                                 'rows': []})
            from industrial_fewshot.evaluation.threshold_outputs import supplement_saved_run
            supplemented = supplement_saved_run(first_dir)
            self.assertEqual(supplemented['score_threshold_results']['0.7']['tp'], 0)
            self.assertEqual(pipeline.read_json(first_dir / 'predictions_score_0.7/frame.json')['shapes'], [])
            self.assertEqual((first_dir / 'predictions/frame.json').read_bytes(), primary_before)
            self.assertEqual((first_dir / 'records.jsonl').read_bytes(), raw_before)
            comparison = pipeline.read_json(first_dir.parent / 'comparison.json')
            self.assertIn('0.7', comparison['groups'][0]['metrics']['score_threshold_results'])
            self.assertEqual(comparison['groups'][0]['run_dir'], str(first_dir))
            self.assertIn('score>=0.7', (first_dir.parent / 'comparison.txt').read_text(encoding='utf-8'))
            visual = render_run(first_dir, root / 'visual_first')
            self.assertEqual(visual['score_threshold'], 0.5)
            self.assertEqual(visual['images'][0]['prediction_count'], 1)
            self.assertEqual(visual['images'][0]['filtered_count'], 1)
            with Image.open(visual['images'][0]['output']) as image:
                self.assertEqual(image.getpixel((100, 100)), (255, 0, 0))
                self.assertEqual(image.getpixel((10, 10)), (128, 128, 128))
            high_visual = render_run(first_dir, root / 'visual_07', score_threshold=0.7)
            self.assertEqual(high_visual['images'][0]['prediction_count'], 0)
            with Image.open(high_visual['images'][0]['output']) as image:
                self.assertEqual(image.getpixel((100, 100)), (0, 255, 0))

            raw['evaluation']['score_threshold'] = 0.75
            pipeline.write_json(path, raw)
            def no_request(**_):
                self.fail('Changing export threshold must reuse the original predictions')
            second = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=no_request))
            second_dir = Path(second['run_dir'])
            self.assertEqual(second['executed_inference_count'], 0)
            self.assertEqual(pipeline.read_json(second_dir / 'predictions/frame.json')['shapes'], [])
            record = pipeline.read_json(second_dir / 'summary.json')['annotations'][0]
            self.assertEqual([obj['confidence'] for obj in record['objects']], [0.5, 0.49])
            self.assertEqual(first['metrics']['coco']['stats'], second['metrics']['coco']['stats'])
            self.assertEqual(first['metrics']['tp'], 1)
            self.assertEqual(second['metrics']['tp'], 0)
            visual = render_run(second_dir, root / 'visual_second')
            self.assertEqual(visual['images'][0]['prediction_count'], 0)
            self.assertEqual(visual['images'][0]['filtered_count'], 2)
            with Image.open(visual['images'][0]['output']) as image:
                self.assertEqual(image.getpixel((100, 100)), (0, 255, 0))
            self.assertEqual((first_dir / 'records.jsonl').read_bytes(), before)

    def test_default_sibling_exports_train_and_test_results_with_valid_image_paths(self):
        for mode in ("train", "test"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, raw = fixture(root, mode=mode)
                raw.pop("output_dir")
                pipeline.write_json(path, raw)
                original = (root / "images/frame.json").read_bytes()
                client = FakeClient([response()])
                report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
                run_dir = Path(report["run_dir"])
                self.assertEqual(run_dir.parent, (root / "images_annotations").resolve())
                destination = run_dir / "predictions/frame.json"
                prediction = pipeline.read_json(destination)
                self.assertEqual((destination.parent / prediction["imagePath"]).resolve(), (root / "images/frame.jpg").resolve())
                self.assertEqual((root / "images/frame.json").read_bytes(), original)
                self.assertEqual("metrics" in report, mode == "train")
                self.assertEqual(pipeline.read_json(run_dir / "run.json")["output_dir"], str(run_dir.parent))

    def test_cli_output_override_is_reported_by_read_only_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root, mode="test")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = pipeline.main(["--config", str(path), "--output-dir", str(root / "chosen"), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["output_dir"], str((root / "chosen").resolve()))
            self.assertFalse((root / "chosen").exists())

    def test_both_passes_receive_shared_reference_inputs_without_modifying_them(self):
        for backend in ("generic", "diaowu", "diaogou"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, raw = fixture(root, backend=backend)
                raw.update(max_reference_examples=1, max_reference_crops=1, max_reference_negatives=1)
                raw["inference"]["enable_completeness_review"] = True
                pipeline.write_json(path, raw)
                ref = root / "references"
                make_reference_pair(ref, "positive", label="target")
                make_image(ref / "positive_crop_01_target.png", (40, 80))
                make_image(ref / "negative.jpg")
                before = {p: p.read_bytes() for p in ref.iterdir()}
                config = pipeline.load_config(path)
                client = FakeClient([response(), response()])
                report = asyncio.run(pipeline.run(config, client_factory=lambda **_: client))
                self.assertEqual(report["metrics"]["mAP50"], 1)
                prepared = config["targets"][0]["config"]
                refs = [prepared[key][0]["rendered_data_url"] for key in ("reference_examples", "reference_crops", "reference_negatives")]
                self.assertEqual(len(client.requests), 2)
                for request in client.requests:
                    urls = [b["image_url"]["url"] for b in request["messages"][0]["content"] if b["type"] == "image_url"]
                    self.assertEqual(urls[:-1], refs)
                    self.assertEqual(len(urls), 4)
                self.assertEqual({p: p.read_bytes() for p in ref.iterdir()}, before)

    def test_train_preserves_truth_and_exports_metrics_and_separate_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            original = (root / "images/frame.json").read_bytes()
            client = FakeClient([response()])
            report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
            self.assertEqual((root / "images/frame.json").read_bytes(), original)
            self.assertEqual(report["metrics"]["mAP50_95"], 1)
            self.assertEqual(report["metrics"]["loss"]["total"], 0)
            self.assertTrue(client.closed)
            run_dir = Path(report["run_dir"])
            prediction = pipeline.read_json(run_dir / "predictions/frame.json")
            self.assertEqual(prediction["shapes"][0]["points"], [[100.0, 100.0], [800.0, 450.0]])
            self.assertEqual(prediction["shapes"][0]["confidence"], 0.9)
            self.assertEqual((run_dir / "predictions" / prediction["imagePath"]).resolve(), (root / "images/frame.jpg").resolve())
            self.assertNotIn("secret-for-test", (run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertNotIn("secret-for-test", (run_dir / "summary.json").read_text(encoding="utf-8"))

    def test_test_mode_ignores_even_invalid_existing_json_and_uses_backend(self):
        for backend in ("diaowu", "diaogou", "generic"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, _ = fixture(root, "test", backend)
                (root / "images/frame.json").write_text("original non-LabelMe content")
                client = FakeClient([response()])
                report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
                self.assertNotIn("metrics", report)
                self.assertEqual(report["success_request_count"], 1)
                self.assertEqual((root / "images/frame.json").read_text(), "original non-LabelMe content")

    def test_external_prompt_is_used_for_both_passes_and_requires_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw["inference"]["enable_completeness_review"] = True
            pipeline.write_json(path, raw)
            client = FakeClient([response(confidence=0.6), response(confidence=0.95)])
            report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
            self.assertEqual(len(client.requests), 2)
            for request in client.requests:
                text = "".join(b["text"] for b in request["messages"][0]["content"] if b["type"] == "text")
                self.assertIn("CUSTOM_EXTERNAL_RULE target special appearance", text)
                required = request["response_format"]["json_schema"]["schema"]["properties"]["objects"]["items"]["required"]
                self.assertIn("confidence", required)
                self.assertEqual(len(required), len(set(required)))
            second = client.requests[1]["messages"][0]["content"]
            self.assertIn("CUSTOM_REVIEW", str(second))
            summary = pipeline.read_json(Path(report["run_dir"]) / "summary.json")
            self.assertEqual(summary["annotations"][0]["objects"][0]["confidence"], 0.95)

    def test_missing_gt_still_infers_image_but_excludes_it_from_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            Image.new("RGB", (1000, 500)).save(root / "images/unlabeled.jpg")
            client = FakeClient([response(), response()])
            report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
            self.assertEqual(report["success_request_count"], 2)
            self.assertEqual(report["metrics"]["evaluated_image_count"], 1)
            self.assertEqual(report["metrics"]["unlabeled_image_count"], 1)
            self.assertEqual(report["metrics"]["fp"], 0)

    def test_missing_gt_error_before_model_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw["dataset"]["missing_ground_truth"] = "error"
            pipeline.write_json(path, raw)
            Image.new("RGB", (1000, 500)).save(root / "images/unlabeled.jpg")
            with self.assertRaisesRegex(ValueError, "缺少真值"):
                asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: self.fail("API must not be called")))
            self.assertFalse((root / "results").exists())

    def test_failed_inference_keeps_ground_truth_in_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            client = FakeClient([RuntimeError("offline")])
            report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
            self.assertEqual(report["failed_request_count"], 1)
            self.assertEqual(report["metrics"]["fn"], 1)
            self.assertEqual(report["metrics"]["evaluation_status"], "incomplete")
            self.assertFalse((Path(report["run_dir"]) / "predictions/frame.json").exists())

    def test_missing_or_invalid_confidence_fails_instead_of_faking_scores(self):
        for score in (None, -0.1, 1.1, float("nan"), True):
            with self.subTest(score=score), tempfile.TemporaryDirectory() as directory:
                path, _ = fixture(Path(directory))
                client = FakeClient([response(confidence=score)])
                report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
                self.assertEqual(report["failed_request_count"], 1)

    def test_dry_run_does_not_write_or_call_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            report = asyncio.run(pipeline.run(pipeline.load_config(path), dry_run=True, client_factory=lambda **_: self.fail("API called")))
            self.assertTrue(report["dry_run"])
            self.assertFalse((root / "results").exists())

    def test_review_failure_retains_initial_valid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            raw["inference"]["enable_completeness_review"] = True
            pipeline.write_json(path, raw)
            client = FakeClient([response(), RuntimeError("review offline")])
            report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
            self.assertEqual(report["metrics"]["mAP50"], 1)
            records = pipeline.read_json(Path(report["run_dir"]) / "summary.json")["annotations"]
            self.assertEqual(records[0]["review_error"], "review offline")
            self.assertFalse(records[0]["review_performed"])

    def test_missing_confidence_retries_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = fixture(Path(directory))
            raw["inference"].update(max_attempts=2, retry_delay_seconds=0)
            pipeline.write_json(path, raw)
            client = FakeClient([response(confidence=None), response()])
            report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(report["failed_request_count"], 0)

    def test_existing_empty_ground_truth_is_a_real_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            pipeline.write_json(root / "images/frame.json", {"shapes": []})
            client = FakeClient([response()])
            report = asyncio.run(pipeline.run(pipeline.load_config(path), client_factory=lambda **_: client))
            self.assertEqual(report["metrics"]["evaluated_image_count"], 1)
            self.assertEqual(report["metrics"]["fp"], 1)
            self.assertIsNone(report["metrics"]["mAP50"])

    def test_same_stem_images_are_rejected_before_overwriting_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            Image.new("RGB", (1000, 500)).save(root / "images/frame.png")
            with self.assertRaisesRegex(ValueError, "同一标注文件"):
                pipeline.inspect_dataset(pipeline.load_config(path))

    def test_distinct_ground_truth_root_and_nested_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            nested = root / "images/sub"
            nested.mkdir()
            (root / "images/frame.jpg").rename(nested / "frame.jpg")
            gt = root / "truth/sub"
            gt.mkdir(parents=True)
            (root / "images/frame.json").rename(gt / "frame.json")
            raw["dataset"]["ground_truth_dir"] = "truth"
            pipeline.write_json(path, raw)
            items = pipeline.inspect_dataset(pipeline.load_config(path))
            self.assertEqual(items[0]["relative_path"], "sub/frame.jpg")
            self.assertEqual(len(items[0]["ground_truth"]), 1)

    def test_ground_truth_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            truth = pipeline.read_json(root / "images/frame.json")
            truth["imageWidth"] = 200
            pipeline.write_json(root / "images/frame.json", truth)
            with self.assertRaisesRegex(ValueError, "尺寸不匹配"):
                pipeline.inspect_dataset(pipeline.load_config(path))

    def test_switching_scene_configs_switches_all_scene_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("scene_a", "scene_b"):
                (root / name).mkdir()
            first, _ = fixture(root / "scene_a")
            second, raw = fixture(root / "scene_b", mode="test", backend="diaogou")
            raw.update(scene_name="different_scene", model="second-model", base_url="http://localhost:9000/v1")
            raw["labels"] = [{"name": "hook", "description": "different object"}]
            (root / "scene_b/prompt.txt").write_text("SECOND_SCENE {{label_name}}", encoding="utf-8")
            pipeline.write_json(second, raw)
            configs = [pipeline.load_config(first), pipeline.load_config(second)]
            clients = [FakeClient([response()]), FakeClient([response(label="hook")])]
            reports, factories = [], []
            for config, client in zip(configs, clients):
                def factory(**kwargs):
                    factories.append(kwargs)
                    return client
                reports.append(asyncio.run(pipeline.run(config, client_factory=factory)))
            self.assertEqual(reports[0]["metrics"]["tp"], 1)
            self.assertNotIn("metrics", reports[1])
            self.assertNotEqual(reports[0]["scene_name"], reports[1]["scene_name"])
            self.assertNotEqual(configs[0]["image_dir"], configs[1]["image_dir"])
            self.assertNotEqual(reports[0]["targets"][0]["reference_image_dir"], reports[1]["targets"][0]["reference_image_dir"])
            self.assertNotEqual(reports[0]["targets"][0]["prompt_file"], reports[1]["targets"][0]["prompt_file"])
            self.assertNotEqual(factories[0]["base_url"], factories[1]["base_url"])
            self.assertEqual(clients[0].requests[0]["model"], "fake-model")
            self.assertEqual(clients[1].requests[0]["model"], "second-model")
            self.assertIn("SECOND_SCENE hook", str(clients[1].requests[0]["messages"]))

    def test_main_requires_explicit_scene_config_and_runs_configured_mode(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as directory:
            path, _ = fixture(Path(directory))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = pipeline.main(["--config", str(path), "--dry-run"])
            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["mode"], "train")
            self.assertEqual(report["config_file"], str(path.resolve()))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.main([])

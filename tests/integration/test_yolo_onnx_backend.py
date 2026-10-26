import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.evaluation.compare import compare_runs
from industrial_fewshot.evaluation.evaluator import evaluate_saved_run
from industrial_fewshot.evaluation.visualize import render_run
from industrial_fewshot.runner import run
from tests.support import fixture, FakeClient, response


class FakeRunner:
    setup_seconds = 0.01
    reference_encoding_seconds = 0.0

    def __init__(self, config):
        self.config = config
        self.calls = 0
        if 'TEST_TRUTH_SECRET' in json.dumps(config):
            raise AssertionError('Ground truth leaked into detector config')

    def predict(self, data):
        self.calls += 1
        return dict(annotation_status='annotated', reason_code='target_detected', reason='local', target_count=1,
            objects=[dict(label='target', confidence=0.9, bbox_pixels_xyxy=[100.125, 100.25, 800.5, 450.125])])

    async def close(self):
        pass


def onnx_fixture(root):
    path, raw = fixture(root)
    raw.update(backend='yolo_onnx', reference_image_dir=None,
               yolo_onnx=dict(weights='model.onnx', class_names=['target']))
    (root / 'model.onnx').write_bytes(b'FAKE_ONNX_USED_ONLY_WITH_MOCK')
    write_json(path, raw)
    return path, raw


class YOLOONNXIntegrationTest(unittest.TestCase):
    def execute(self, path, *, forbid=False, dry_run=False, force=False):
        runners = []
        def factory(config):
            if forbid:
                self.fail('Cache/dry-run must not load the model')
            runner = FakeRunner(config)
            runners.append(runner)
            return runner
        with patch('industrial_fewshot.runner.inference.request_annotation', side_effect=AssertionError('No network')):
            result = asyncio.run(run(load_config(path), dry_run=dry_run, force=force, local_runner_factory=factory))
        return result, runners

    def test_flat_exports_feedback_pixel_precision_and_evaluation_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = onnx_fixture(root)
            planned, _ = self.execute(path, forbid=True, dry_run=True)
            expected = root / 'results/model_监督检测'
            self.assertEqual(Path(planned['run_dir']), expected)
            self.assertEqual(planned['targets'][0]['reference_count']['reference_examples'], 0)
            self.assertFalse(expected.exists())
            report, runners = self.execute(path)
            self.assertEqual(runners[0].calls, 1)
            self.assertEqual(report['metrics']['recall'], 1)
            for filename in ('config.resolved.json', 'records.jsonl', 'metrics.json', 'evaluation.txt',
                             'predictions/frame.json', 'feedback/frame.json', 'summary.json', 'run.json'):
                self.assertTrue((expected / filename).is_file(), filename)
            self.assertEqual(read_json(expected / 'predictions/frame.json')['shapes'][0]['points'][0], [100.125, 100.25])
            feedback = read_json(expected / 'feedback/frame.json')
            self.assertEqual(feedback['objects'][0]['reason_status'], 'unsupported')
            reevaluated = evaluate_saved_run(expected)
            self.assertEqual(reevaluated['metrics']['recall'], 1)
            rendered = render_run(expected, root / 'preview')
            self.assertTrue(Path(rendered['images'][0]['output']).is_file())
            raw.update(backend='generic', output_dir='vlm_results', reference_mode='text_only')
            write_json(path, raw)
            vlm = asyncio.run(run(load_config(path), client_factory=lambda **_: FakeClient([response()])))
            comparison = compare_runs([expected, Path(vlm['run_dir'])], root / 'comparison')
            self.assertEqual(comparison['rows'][0]['method'], 'YOLO ONNX（监督检测）')
            self.assertEqual(comparison['image_count'], 1)

    def test_cache_reuses_evaluation_but_invalidates_weights_and_inference_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = onnx_fixture(root)
            first, _ = self.execute(path)
            raw['evaluation'] = dict(score_threshold=0.95)
            write_json(path, raw)
            evaluated, _ = self.execute(path, forbid=True)
            self.assertEqual(evaluated['reused_prediction_count'], 1)
            self.assertEqual(evaluated['metrics']['recall'], 0)
            self.assertEqual(first['targets'][0]['request_signature'], evaluated['targets'][0]['request_signature'])
            raw['yolo_onnx']['confidence'] = 0.3
            write_json(path, raw)
            changed, _ = self.execute(path)
            self.assertEqual(changed['executed_inference_count'], 1)
            self.assertNotEqual(first['targets'][0]['request_signature'], changed['targets'][0]['request_signature'])
            (root / 'model.onnx').write_bytes(b'CHANGED_FAKE_ONNX')
            weights, _ = self.execute(path)
            self.assertNotEqual(changed['targets'][0]['request_signature'], weights['targets'][0]['request_signature'])
            forced, _ = self.execute(path, force=True)
            self.assertEqual(forced['executed_inference_count'], 1)

    def test_supervised_model_rejects_reference_prompts_and_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = onnx_fixture(Path(directory))
            for changes in (dict(max_reference_examples=1), dict(ablation={'enabled': True})):
                write_json(path, {**raw, **changes})
                with self.assertRaises(ValueError):
                    load_config(path)

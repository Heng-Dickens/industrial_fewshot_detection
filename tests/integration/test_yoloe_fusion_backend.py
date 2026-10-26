import asyncio
import copy
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

from industrial_fewshot.config import load_config
from industrial_fewshot.configuration import load_document
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.detectors.yoloe_fusion import YOLOEFusionRunner, fuse_prototypes, yoloe_fusion_settings
from industrial_fewshot.detectors.yoloe_text import YOLOETextRunner
from industrial_fewshot.evaluation.compare import compare_runs
from industrial_fewshot.evaluation.evaluator import evaluate_saved_run
from industrial_fewshot.prompts.references import prepare_reference_examples
from industrial_fewshot.runner import run
from tests.integration.test_yoloe_backend import FakeRunner
from tests.integration.test_yoloe_text_backend import fixture as text_fixture


def fixture(root):
    path, raw = text_fixture(root)
    raw['backend'] = 'yoloe_fusion'
    raw['yoloe_fusion'] = {**raw.pop('yoloe_text'), 'positive_count': 5, 'visual_weight': 0.75, 'text_weight': 0.25}
    write_json(path, raw)
    return path, raw


class YOLOEFusionBackendTest(unittest.TestCase):
    def execute(self, path, factory=FakeRunner, dry=False):
        return asyncio.run(run(load_config(path), local_runner_factory=factory, dry_run=dry))

    def test_pipeline_provenance_cache_and_saved_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            dry = self.execute(path, lambda _: self.fail('Dry run must not load model'), dry=True)
            self.assertFalse((root / 'results').exists())
            self.assertTrue(dry['run_dir'].endswith('weights_5shot_裁剪图0.75+文字0.25'))
            first = self.execute(path)
            directory = Path(first['run_dir'])
            target = first['targets'][0]
            snapshot = read_json(directory / 'prompt.json')
            self.assertEqual(snapshot['kind'], 'text_visual')
            self.assertEqual(snapshot['fusion']['visual_weight'], 0.75)
            self.assertEqual(snapshot['fusion']['text_weight'], 0.25)
            self.assertEqual(len(snapshot['references']), 5)
            self.assertTrue(all(row['crops'] for row in snapshot['references']))
            self.assertEqual(target['reference_count']['reference_originals'], 0)
            self.assertEqual(target['reference_count']['reference_negatives'], 0)
            self.assertGreaterEqual(target['reference_count']['reference_crops'], 5)
            self.assertEqual((directory / 'prompt.txt').read_text(encoding='utf-8'), raw['yoloe_fusion']['text_prompt'])
            self.assertIn('裁剪视觉权重=0.75', (directory / 'evaluation.txt').read_text(encoding='utf-8'))
            self.assertEqual(read_json(directory / 'feedback/frame.json')['feedback_source'], 'unsupported_by_detector')
            cached = self.execute(path, lambda _: self.fail('Should reuse fusion predictions'))
            self.assertEqual(cached['reused_prediction_count'], 1)
            self.assertEqual(evaluate_saved_run(directory)['metrics']['mAP50'], 1)
            compared = compare_runs([directory, Path(cached['run_dir'])], root / 'comparison')
            self.assertTrue(all(row['method'] == 'YOLOE（裁剪+文字融合）' for row in compared['rows']))
            self.assertTrue(all(row['positive_original_count'] == 5 and row['json_reference_count'] == 0
                                for row in compared['rows']))
            raw['evaluation'] = {'score_threshold': 0.95}
            write_json(path, raw)
            evaluated = self.execute(path, lambda _: self.fail('Evaluation settings do not change inference'))
            self.assertEqual(evaluated['reused_prediction_count'], 1)
            raw['yoloe_fusion'].update(visual_weight=0.6, text_weight=0.4)
            write_json(path, raw)
            weighted = self.execute(path)
            self.assertNotEqual(target['request_signature'], weighted['targets'][0]['request_signature'])
            raw['yoloe_fusion']['text_prompt'] = 'another description.'
            write_json(path, raw)
            changed = self.execute(path)
            self.assertNotEqual(weighted['targets'][0]['request_signature'], changed['targets'][0]['request_signature'])
            crop = snapshot['references'][0]['crops'][0]
            Image.new('RGB', (crop['width'], crop['height']), 'red').save(crop['image_path'])
            recropped = self.execute(path)
            self.assertEqual(recropped['executed_inference_count'], 1)
            self.assertNotEqual(changed['targets'][0]['request_signature'], recropped['targets'][0]['request_signature'])
            self.assertIn('裁剪视觉+文字', (root / 'results/prompt_results.md').read_text(encoding='utf-8'))

    def test_configuration_uses_same_caption_and_validated_weights(self):
        root = Path(__file__).resolve().parents[2]
        raw, _ = load_document(root / 'configs/specific/yoloe_fusion_5shot.yaml')
        baseline, _ = load_document(root / 'configs/specific/yoloe_text.yaml')
        settings = yoloe_fusion_settings(raw['yoloe_fusion'], root)
        self.assertEqual(settings['text_prompt'], baseline['yoloe_text']['text_prompt'])
        self.assertEqual((settings['visual_weight'], settings['text_weight'], settings['positive_count']), (0.75, 0.25, 5))
        for bad in ({'visual_weight': 0.8}, {'visual_weight': -1}, {'text_weight': float('nan')},
                    {'text_weight': True}, {'positive_count': 0}, {'text_prompt': ''}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                yoloe_fusion_settings({**raw['yoloe_fusion'], **bad}, root)

    def test_stale_crop_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            config = load_config(path)['targets'][0]['config']
            prepare_reference_examples(config)
            selected = config['reference_examples'][0]['sample_id']
            manifest = read_json(root / 'prepared/manifest.json')
            row = next(row for row in manifest['images'] if row['image'] == selected)
            row['crops'][0]['bbox_xyxy'][0] += 1
            write_json(root / 'prepared/manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, '裁剪清单'):
                self.execute(path, dry=True)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch required for prototype math')
    def test_actual_fusion_path_normalizes_and_balances_shots_before_query(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            config = load_config(path)['targets'][0]['config']
            prepare_reference_examples(config)
            # Force different numbers of crops to verify equal per-shot weight.
            config['reference_examples'][0]['crops'].append(copy.deepcopy(config['reference_examples'][0]['crops'][0]))
            count = sum(len(row['crops']) for row in config['reference_examples'])
            vectors = [torch.tensor([[[float(i + 1), 1.0]]]) for i in range(count)]
            crop_calls, query_calls = [], []

            class Model:
                def __init__(self):
                    self.model = torch.nn.Linear(1, 1)
                    self.model.pe = torch.tensor([[[0.0, 2.0]]])
                    self.model.model = [types.SimpleNamespace(is_fused=False)]
                def predict(self, image, **kwargs):
                    if 'prompts' in kwargs:
                        height, width = image.shape[:2]
                        np.testing.assert_array_equal(kwargs['prompts']['bboxes'], [[0, 0, width, height]])
                        assert kwargs['return_vpe']
                        self.predictor = types.SimpleNamespace(vpe=vectors[len(crop_calls)])
                        crop_calls.append(kwargs)
                    else:
                        query_calls.append(kwargs)
                    return [types.SimpleNamespace(boxes=None)]
                def set_classes(self, names, prototype):
                    self.names, self.prototype = names, prototype

            def text_setup(runner, config):
                runner.config, runner.settings = config, config['yoloe_text']
                runner.np, runner.torch = np, torch
                runner.model = Model()
                runner.text_encoding_seconds = runner.setup_seconds = 0.01

            predictor = types.ModuleType('ultralytics.models.yolo.yoloe.predict_vp')
            predictor.YOLOEVPSegPredictor = type('VisualPredictor', (), {})
            with patch.object(YOLOETextRunner, '__init__', text_setup), patch.dict('sys.modules', {
                    'ultralytics.models.yolo.yoloe.predict_vp': predictor}):
                runner = YOLOEFusionRunner(config)
                unit = lambda v: torch.nn.functional.normalize(v, dim=-1)
                offset, per_image = 0, []
                for example in config['reference_examples']:
                    n = len(example['crops'])
                    per_image.append(unit(torch.stack([unit(v) for v in vectors[offset:offset+n]]).mean(0)))
                    offset += n
                expected = unit(0.75 * unit(torch.stack(per_image).mean(0)) + 0.25 * torch.tensor([[[0.0, 1.0]]]))
                self.assertTrue(torch.allclose(runner.model.prototype, expected))
                self.assertEqual(runner.model.names, ['target'])
                self.assertEqual(len(crop_calls), count)
                self.assertFalse(query_calls)
                self.assertIsNone(runner.model.predictor)
                self.assertIn('联合提示', runner.predict((root / 'images/frame.jpg').read_bytes())['reason'])
                self.assertEqual(len(query_calls), 1)
                self.assertNotIn('prompts', query_calls[0])
                crop = config['reference_examples'][0]['crops'][0]
                Path(crop['image_path']).write_bytes(b'changed-after-preparation')
                with self.assertRaisesRegex(ValueError, '准备后发生变化'):
                    YOLOEFusionRunner(config)
            with self.assertRaises(ValueError):
                fuse_prototypes(torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[-1.0, 0.0]]]), 0.5, 0.5)

import asyncio
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from industrial_fewshot.config import load_config
from industrial_fewshot.configuration import load_document
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.detectors.yoloe_text import YOLOETextRunner, yoloe_text_settings
from industrial_fewshot.evaluation.compare import compare_runs
from industrial_fewshot.evaluation.evaluator import evaluate_saved_run
from industrial_fewshot.evaluation.visualize import render_run
from industrial_fewshot.prompts.references import prepare_reference_examples
from industrial_fewshot.runner import run
from tests.integration.test_yoloe_backend import FakeRunner, local_fixture


def fixture(root):
    path, raw = local_fixture(root)
    raw.update(backend='yoloe_text', max_reference_examples=0, max_reference_crops=0,
               max_reference_negatives=0, yoloe_text=dict(source_dir='checkout', weights='weights.pt',
               text_encoder_weights='encoder.pt', text_prompt='a suspended cylindrical load.'))
    (root / 'encoder.pt').write_bytes(b'FAKE_TEXT_ENCODER')
    (root / 'checkout/ultralytics/nn').mkdir(parents=True)
    (root / 'checkout/ultralytics/nn/text_model.py').write_text('# interface fixture', encoding='utf-8')
    write_json(path, raw)
    return path, raw


class YOLOETextBackendTest(unittest.TestCase):
    def execute(self, path, factory=FakeRunner, dry=False):
        return asyncio.run(run(load_config(path), local_runner_factory=factory, dry_run=dry))

    def test_text_only_outputs_cache_evaluation_comparison_and_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            first = self.execute(path)
            saved = Path(first['run_dir'])
            target = first['targets'][0]
            self.assertEqual(target['reference_mode'], 'yoloe_text')
            self.assertTrue(all(v == 0 for v in target['reference_count'].values()))
            self.assertEqual(target['reference_sample_ids'], [])
            self.assertEqual(saved.name, 'weights_0shot')
            self.assertEqual(target['prompt_snapshot']['kind'], 'text')
            self.assertEqual((saved / 'prompt.txt').read_text(encoding='utf-8'), raw['yoloe_text']['text_prompt'])
            self.assertIn('YOLOE 文字 prompt', (saved / 'evaluation.txt').read_text(encoding='utf-8'))
            self.assertEqual(first['metrics']['mAP50'], 1)
            self.assertEqual(read_json(saved / 'feedback/frame.json')['feedback_source'], 'unsupported_by_detector')
            repeated = self.execute(path, lambda _: self.fail('Cache should avoid model initialization'))
            self.assertEqual(repeated['reused_prediction_count'], 1)
            self.assertEqual(evaluate_saved_run(saved)['metrics']['mAP50'], 1)
            comparison = compare_runs([saved, Path(repeated['run_dir'])], root / 'comparison')
            self.assertTrue(all(row['method'] == 'YOLOE（文字提示）' for row in comparison['rows']))
            self.assertEqual(len(render_run(saved, root / 'rendered')['images']), 1)
            self.assertIn(raw['yoloe_text']['text_prompt'], (root / 'results/prompt_results.md').read_text(encoding='utf-8'))

    def test_only_inference_inputs_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            first = self.execute(path)
            raw['evaluation'] = {'score_threshold': 0.95}
            write_json(path, raw)
            evaluated = self.execute(path, lambda _: self.fail('Evaluation-only changes should reuse boxes'))
            self.assertEqual(evaluated['reused_prediction_count'], 1)
            self.assertEqual(evaluated['metrics']['recall'], 0)
            raw['yoloe_text']['text_prompt'] = 'a different suspended object.'
            write_json(path, raw)
            changed = self.execute(path)
            self.assertEqual(changed['executed_inference_count'], 1)
            self.assertNotEqual(changed['targets'][0]['request_signature'], first['targets'][0]['request_signature'])
            self.assertEqual((Path(first['run_dir']) / 'prompt.txt').read_text(encoding='utf-8'), 'a suspended cylindrical load.')
            (root / 'encoder.pt').write_bytes(b'CHANGED_ENCODER')
            encoder = self.execute(path)
            self.assertNotEqual(changed['targets'][0]['request_signature'], encoder['targets'][0]['request_signature'])

    def test_dry_run_never_loads_encoder_and_rejects_visual_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            self.execute(path, lambda _: self.fail('Dry-run must not load model'), dry=True)
            self.assertFalse((root / 'results').exists())
            raw['max_reference_examples'] = 1
            write_json(path, raw)
            with self.assertRaises(ValueError):
                load_config(path)
            for bad in ({}, {'text_prompt': ''}, {'text_prompt': 'load', 'positive_count': 0},
                        {'text_prompt': 'load', 'text_encoder_weights': ''},
                        {'text_prompt': 'load', 'confidence': float('nan')}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    yoloe_text_settings(bad, root)

    def test_actual_configuration_caption_matches_groundingdino_and_paths_are_anchored(self):
        root = Path(__file__).resolve().parents[2]
        raw, _ = load_document(root / 'configs/specific/yoloe_text.yaml')
        grounding, _ = load_document(root / 'configs/specific/zero_shot.yaml')
        self.assertEqual(raw['yoloe_text']['text_prompt'], grounding['groundingdino']['text_prompt'])
        self.assertEqual(Path(raw['yoloe_text']['text_encoder_weights']), root / 'weights/yoloe/mobileclip_blt.pt')
        self.assertEqual(raw['evaluation']['score_threshold'], 0.5)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch required for embedding math')
    def test_encoder_uses_exact_caption_normalization_alignment_and_no_visual_api(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            config = load_config(path)['targets'][0]['config']
            prepare_reference_examples(config)
            texts, alignment_inputs, image_calls = [], [], []

            class Head(torch.nn.Module):
                def get_tpe(self, features):
                    alignment_inputs.append(features.clone())
                    return features + 0.25

            class Model:
                def __init__(self, weights):
                    self.model = torch.nn.Linear(1, 1)
                    self.model.args = {'text_model': 'mobileclip:blt'}
                    self.model.model = [Head()]
                def to(self, device):
                    self.model.to(device)
                def set_classes(self, names, embeddings):
                    self.names, self.embeddings = names, embeddings
                def predict(self, image, **kwargs):
                    image_calls.append(kwargs)
                    if 'prompts' in kwargs or 'predictor' in kwargs:
                        raise AssertionError('Visual prompt API must not be used')
                    return [types.SimpleNamespace(boxes=None)]

            class Encoder:
                def eval(self):
                    return self
                def encode_text(self, tokens):
                    return torch.tensor([[3.0, 4.0]], device=tokens.device)

            class Tokenizer:
                context_length = 77
                tokenizer = types.SimpleNamespace(encode=lambda text: text.split())
                def __call__(self, captions):
                    texts.extend(captions)
                    return torch.tensor([[1, 2]])

            package = types.ModuleType('ultralytics')
            package.__file__ = str(root / 'checkout/ultralytics/__init__.py')
            package.YOLOE = Model
            mobileclip = types.ModuleType('mobileclip')
            mobileclip.create_model_and_transforms = lambda *a, **kw: (Encoder(), None, None)
            mobileclip.get_tokenizer = lambda _: Tokenizer()
            with patch.dict('sys.modules', {'ultralytics': package, 'mobileclip': mobileclip}):
                runner = YOLOETextRunner(config)
                self.assertEqual(texts, [config['yoloe_text']['text_prompt']])
                self.assertTrue(torch.allclose(alignment_inputs[0], torch.tensor([[[0.6, 0.8]]])))
                self.assertTrue(torch.allclose(runner.model.embeddings, torch.tensor([[[0.85, 1.05]]])))
                self.assertFalse(image_calls)  # No image was used for prompt encoding.
                prediction = runner.predict((root / 'images/frame.jpg').read_bytes())
                self.assertEqual(len(image_calls), 1)
                self.assertEqual(prediction['reason'], 'YOLOE 纯文字提示检测')
                config['yoloe_text']['text_prompt'] = 'word ' * 100
                with self.assertRaisesRegex(ValueError, 'tokens'):
                    YOLOETextRunner(config)

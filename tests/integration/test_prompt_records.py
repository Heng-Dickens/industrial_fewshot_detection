import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from industrial_fewshot.cache import request_signature
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.prompt_records import backfill_prompts
from industrial_fewshot.runner import run
from tests.integration.test_yoloe_backend import local_fixture, FakeRunner
from tests.integration.test_yolo_onnx_backend import onnx_fixture
from tests.support import FakeClient, response


class EmptyRunner(FakeRunner):
    def predict(self, data):
        return dict(annotation_status='no_target', reason_code='no_target', reason='none', target_count=0, objects=[])


def grounding_fixture(root):
    path, raw = local_fixture(root)
    raw.update(backend='groundingdino', groundingdino=dict(checkpoint='absent', text_prompt='a suspended mold.'))
    write_json(path, raw)
    return path, raw


def execute(path, factory=FakeRunner):
    return asyncio.run(run(load_config(path), local_runner_factory=factory))


class PromptRecordsIntegrationTest(unittest.TestCase):
    def test_missing_historical_text_stays_unknown_after_repeated_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / 'results/old'
            write_json(saved / 'run.json', {'targets': [{'id': 'old', 'prompt_sha256': '0' * 64}]})
            backfill_prompts(root / 'results')
            backfill_prompts(root / 'results')
            snapshot = read_json(saved / 'prompt.json')
            self.assertEqual(snapshot['kind'], 'unknown')
            self.assertIsNone(snapshot['text'])
            self.assertIsNone(snapshot['text_sha256'])
            self.assertEqual(read_json(root / 'results/prompt_results.json')['runs'][0]['prompt']['kind'], 'unknown')

    def test_real_text_snapshot_reports_index_and_cached_run_preserve_association(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = grounding_fixture(root)
            first = execute(path)
            first_dir = Path(first['run_dir'])
            text = raw['groundingdino']['text_prompt']
            self.assertEqual((first_dir / 'prompt.txt').read_text(encoding='utf-8'), text)
            snapshot = read_json(first_dir / 'prompt.json')
            self.assertEqual(snapshot['text'], text)
            self.assertEqual(snapshot['text_sha256'], hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(snapshot['request_signature'], first['targets'][0]['request_signature'])
            self.assertIn(text, (first_dir / 'evaluation.txt').read_text(encoding='utf-8'))
            descriptor = dict(first['targets'][0])
            descriptor.pop('prompt_snapshot')
            self.assertEqual(request_signature(descriptor), first['targets'][0]['request_signature'])
            raw['groundingdino']['text_prompt'] = 'a hollow cylinder.'
            write_json(path, raw)
            second = execute(path, EmptyRunner)
            second_dir = Path(second['run_dir'])
            self.assertNotEqual(first['targets'][0]['request_signature'], second['targets'][0]['request_signature'])
            self.assertEqual((first_dir / 'prompt.txt').read_text(encoding='utf-8'), text)
            self.assertEqual((second_dir / 'prompt.txt').read_text(encoding='utf-8'), 'a hollow cylinder.')
            third = execute(path, lambda _: self.fail('Expected cached boxes without model initialization'))
            self.assertEqual(third['reused_prediction_count'], 1)
            rows = read_json(root / 'results/prompt_results.json')['runs']
            first_row = next(row for row in rows if row['run_dir'] == first_dir.name)
            second_row = next(row for row in rows if row['run_dir'] == second_dir.name)
            self.assertEqual((first_row['prompt']['text'], first_row['AP50']), (text, 1))
            self.assertEqual((second_row['prompt']['text'], second_row['AP50']), ('a hollow cylinder.', 0))

    def test_historical_recovery_ignores_current_config_and_preserves_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = grounding_fixture(root)
            reports = [execute(path)]
            raw['groundingdino']['text_prompt'] = 'a hollow cylinder.'
            write_json(path, raw)
            reports.append(execute(path))
            original = {}
            for report in reports:
                run_dir = Path(report['run_dir'])
                metadata = read_json(run_dir / 'run.json')
                metadata['targets'][0].pop('prompt_snapshot')
                write_json(run_dir / 'run.json', metadata)
                (run_dir / 'prompt.txt').write_text('', encoding='utf-8')
                (run_dir / 'prompt.json').unlink()
                (run_dir / 'evaluation.txt').write_text('ORIGINAL METRIC TABLE\n', encoding='utf-8')
                original[run_dir] = {name:(run_dir / name).read_bytes()
                                     for name in ('run.json', 'summary.json', 'metrics.json', 'records.jsonl')}
            raw['groundingdino']['text_prompt'] = 'the current config is not historical evidence.'
            write_json(path, raw)
            with patch('industrial_fewshot.config.load_config', side_effect=AssertionError('No current config')):
                planned = backfill_prompts(root / 'results', dry_run=True)
                self.assertEqual(len(planned), 2)
                self.assertFalse((Path(reports[0]['run_dir']) / 'prompt.json').exists())
                backfill_prompts(root / 'results')
                backfill_prompts(root / 'results')  # recovery is idempotent
            for report in reports:
                run_dir = Path(report['run_dir'])
                text = report['targets'][0]['local_model']['settings']['text_prompt']
                self.assertEqual((run_dir / 'prompt.txt').read_text(encoding='utf-8'), text)
                evaluation = (run_dir / 'evaluation.txt').read_text(encoding='utf-8')
                self.assertTrue(evaluation.endswith('ORIGINAL METRIC TABLE\n'))
                self.assertEqual(evaluation.count('【提示词记录】'), 1)
                for name, content in original[run_dir].items():
                    self.assertEqual((run_dir / name).read_bytes(), content, name)

    def test_visual_and_supervised_models_explicitly_have_no_text_prompt(self):
        for backend in ('yoloe', 'yolo_onnx'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, _ = local_fixture(root) if backend == 'yoloe' else onnx_fixture(root)
                report = execute(path)
                run_dir = Path(report['run_dir'])
                snapshot = read_json(run_dir / 'prompt.json')
                self.assertIsNone(snapshot['text'])
                self.assertIsNone(snapshot['text_sha256'])
                self.assertEqual(snapshot['kind'], 'visual' if backend == 'yoloe' else 'none')
                self.assertIn('无文字 prompt', (run_dir / 'prompt.txt').read_text(encoding='utf-8'))
                if backend == 'yoloe':
                    self.assertEqual(len(snapshot['references']), 5)
                    self.assertIn('bboxes', snapshot['references'][0])

    def test_conflicting_legacy_text_aborts_before_any_backfill_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = grounding_fixture(root)
            ground = execute(path)
            ground_dir = Path(ground['run_dir'])
            (ground_dir / 'prompt.txt').write_text('', encoding='utf-8')
            (ground_dir / 'prompt.json').unlink()
            raw.update(backend='generic', reference_mode='text_only', require_box_reason=False)
            write_json(path, raw)
            vlm = asyncio.run(run(load_config(path), client_factory=lambda **_: FakeClient([response()])))
            vlm_dir = Path(vlm['run_dir'])
            metadata = read_json(vlm_dir / 'run.json')
            metadata['targets'][0].pop('prompt_snapshot')
            write_json(vlm_dir / 'run.json', metadata)
            (vlm_dir / 'prompt.txt').write_text('WRONG HISTORICAL PROMPT', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '指纹不一致'):
                backfill_prompts(root / 'results')
            self.assertFalse((ground_dir / 'prompt.json').exists())
            self.assertEqual((ground_dir / 'prompt.txt').read_bytes(), b'')

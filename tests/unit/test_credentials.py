import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from industrial_fewshot.config import load_config
from industrial_fewshot.configuration import load_document
from industrial_fewshot.credentials import resolve_api_key
from industrial_fewshot.data.labelme import write_json
from industrial_fewshot.runner import run
from tests.support import FakeClient, fixture, response


class CredentialsTest(unittest.TestCase):
    def test_environment_file_literals_and_process_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            model = {'env_file': str(path), 'api_key_env': 'QWEN_MLLM_API_KEY'}
            for value in ('project-key # comment', '"project-key" # comment', "'project-key'"):
                path.write_text('\ufeff# comment\nexport QWEN_MLLM_API_KEY = ' + value + '\n', encoding='utf-8')
                self.assertEqual(resolve_api_key(model), 'project-key')
                self.assertNotIn('QWEN_MLLM_API_KEY', os.environ)
            with patch.dict(os.environ, {'QWEN_MLLM_API_KEY': 'process-key'}):
                path.unlink()
                self.assertEqual(resolve_api_key(model), 'process-key')

    def test_environment_file_does_not_expand_or_execute_values(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            value = '${OPENAI_API_KEY}$(echo secret)`echo secret`'
            path.write_text('QWEN_MLLM_API_KEY=' + value, encoding='utf-8')
            self.assertEqual(resolve_api_key({'env_file': str(path), 'api_key_env': 'QWEN_MLLM_API_KEY'}), value)

    def test_missing_or_invalid_environment_key_fails_without_echoing_content(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            model = {'env_file': str(path), 'api_key_env': 'QWEN_MLLM_API_KEY', 'api_key': 'EMPTY'}
            with self.assertRaises(RuntimeError):
                resolve_api_key(model)
            path.write_bytes(b'QWEN_MLLM_API_KEY=SECRET\xff')
            with self.assertRaises(RuntimeError) as error:
                resolve_api_key(model)
            self.assertNotIn('SECRET', str(error.exception))
            for text in ('', 'OTHER=SECRET', 'QWEN_MLLM_API_KEY=', 'QWEN_MLLM_API_KEY=EMPTY', 'SECRET BAD LINE', 'QWEN_MLLM_API_KEY="SECRET'):
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text), self.assertRaises(RuntimeError) as error:
                    resolve_api_key(model)
                self.assertNotIn('SECRET', str(error.exception))

    def test_environment_file_is_anchored_in_declaring_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model').mkdir()
            (root / 'model/base.yaml').write_text('env_file: ../.env\n', encoding='utf-8')
            (root / 'experiment.yaml').write_text('extends: model/base.yaml\n', encoding='utf-8')
            raw, _ = load_document(root / 'experiment.yaml')
            self.assertEqual(Path(raw['env_file']), root / '.env')

    def test_pipeline_reads_environment_file_without_persisting_secret(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            path, raw = fixture(root)
            raw['model'].update(api_key_env='QWEN_MLLM_API_KEY', env_file='.env')
            write_json(path, raw)
            secret = 'ENV-FILE-TEST-KEY'
            (root / '.env').write_text('QWEN_MLLM_API_KEY=' + secret, encoding='utf-8')
            captured = []
            def factory(**kwargs):
                captured.append(kwargs)
                return FakeClient([response()])
            report = asyncio.run(run(load_config(path), client_factory=factory))
            self.assertEqual(captured[0]['api_key'], secret)
            for artifact in Path(report['run_dir']).rglob('*'):
                if artifact.is_file() and artifact.suffix in {'.json', '.jsonl', '.txt'}:
                    self.assertNotIn(secret, artifact.read_text(encoding='utf-8'))
            (root / '.env').unlink()
            cached = asyncio.run(run(load_config(path), client_factory=lambda **_: self.fail('Cached run called API')))
            self.assertEqual(cached['reused_prediction_count'], 1)

    def test_flat_model_forwards_environment_file_and_dry_run_needs_no_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = fixture(Path(directory))
            model = raw['model']
            raw.update(model=model['name'], base_url=model['base_url'], env_file='missing.env', api_key_env='QWEN_MLLM_API_KEY')
            write_json(path, raw)
            config = load_config(path)
            self.assertEqual(Path(config['targets'][0]['model_config']['env_file']), Path(directory) / 'missing.env')
            self.assertTrue(asyncio.run(run(config, dry_run=True))['dry_run'])

    def test_scoped_environment_wins_without_opening_file(self):
        model = {'api_key_env': 'QWEN_MLLM_API_KEY', 'api_key_file': 'missing.json'}
        with patch.dict(os.environ, {'QWEN_MLLM_API_KEY': 'project-key'}, clear=True):
            self.assertEqual(resolve_api_key(model), 'project-key')

    def test_project_file_is_independent_of_codex_and_legacy_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'key.json'
            write_json(path, {'api_key': 'project-only-key'})
            model = {'api_key_env': 'QWEN_MLLM_API_KEY', 'api_key_file': str(path)}
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'codex-key', 'VLLM_API_KEY': 'other-key'}, clear=True):
                self.assertEqual(resolve_api_key(model), 'project-only-key')

    def test_missing_project_file_does_not_fall_back_to_codex_key(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'codex-secret'}, clear=True):
            with self.assertRaises(RuntimeError) as error:
                resolve_api_key({'api_key_env': 'QWEN_MLLM_API_KEY', 'api_key_file': 'missing.json'})
        self.assertIn('QWEN_MLLM_API_KEY', str(error.exception))
        self.assertNotIn('codex-secret', str(error.exception))

    def test_invalid_project_key_errors_do_not_echo_file_contents(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / 'key.json'
            for text in ('SENSITIVE-CONTENT invalid json', '[]', '{"api_key": ""}', '{"api_key": "EMPTY"}'):
                with self.subTest(text=text):
                    path.write_text(text, encoding='utf-8')
                    with self.assertRaises(RuntimeError) as error:
                        resolve_api_key({'api_key_file': str(path)})
                    self.assertNotIn('SENSITIVE-CONTENT', str(error.exception))

    def test_legacy_key_and_keyless_server_remain_supported(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_api_key({'api_key': 'legacy-key'}), 'legacy-key')
            self.assertEqual(resolve_api_key({}), 'EMPTY')

    def test_key_path_is_anchored_in_its_declaring_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model').mkdir()
            (root / 'model/base.yaml').write_text('api_key_file: private.json\n', encoding='utf-8')
            (root / 'experiment.yaml').write_text('extends: model/base.yaml\n', encoding='utf-8')
            raw, _ = load_document(root / 'experiment.yaml')
            self.assertEqual(raw['api_key_file'], str(root / 'model/private.json'))

    def test_real_pipeline_uses_private_key_without_persisting_it_and_cache_needs_no_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            path, raw = fixture(root)
            raw['model'].pop('api_key')
            raw['model'].update(api_key_env='QWEN_MLLM_API_KEY', api_key_file='private.json')
            write_json(path, raw)
            secret_file = root / 'private.json'
            write_json(secret_file, {'api_key': 'PRIVATE-TEST-KEY'})
            captured = []
            client = FakeClient([response()])
            def factory(**kwargs):
                captured.append(kwargs)
                return client
            report = asyncio.run(run(load_config(path), client_factory=factory))
            self.assertEqual(captured[0]['api_key'], 'PRIVATE-TEST-KEY')
            for artifact in Path(report['run_dir']).rglob('*'):
                if artifact.is_file() and artifact.suffix in {'.json', '.jsonl', '.txt'}:
                    self.assertNotIn('PRIVATE-TEST-KEY', artifact.read_text(encoding='utf-8'))
            secret_file.unlink()
            report = asyncio.run(run(load_config(path), client_factory=lambda **_: self.fail('Cached run called API')))
            self.assertEqual(report['reused_prediction_count'], 1)

    def test_dry_run_does_not_require_or_read_a_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = fixture(Path(directory))
            raw['model'].update(api_key_file='missing.json')
            write_json(path, raw)
            report = asyncio.run(run(load_config(path), dry_run=True))
            self.assertTrue(report['dry_run'])

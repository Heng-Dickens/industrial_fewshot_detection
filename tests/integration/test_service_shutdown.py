import asyncio
import contextlib
import io
import json
import signal
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.runner import run
from industrial_fewshot.service_errors import ModelServiceError
from industrial_fewshot.cli import main as cli_main
from tests.integration.test_reference_suite import suite_fixture
from tests.support import FakeClient, fixture, make_image, response


class ServiceStatusError(Exception):
    def __init__(self, status):
        super().__init__('fake service error')
        self.status_code = status


class ServiceShutdownTest(unittest.TestCase):
    def prepare(self, root, **inference):
        path, raw = fixture(root, mode='test')
        raw['inference'].update(max_attempts=3, retry_delay_seconds=0, **inference)
        make_image(root / 'images/other.jpg', (1000, 500))
        write_json(path, raw)
        return path

    def test_auth_and_missing_model_stop_after_one_request_and_save_failure(self):
        for status in (401, 403, 404):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = self.prepare(root)
                client = FakeClient([ServiceStatusError(status)])
                with self.assertRaises(ModelServiceError) as error:
                    asyncio.run(run(load_config(path), client_factory=lambda **_: client))
                self.assertEqual(error.exception.status_code, status)
                self.assertEqual(len(client.requests), 1)
                self.assertTrue(client.closed)
                summary = read_json(next((root / 'results').glob('*/summary.json')))
                self.assertEqual(summary['status'], 'failed')
                self.assertEqual(summary['failed_request_count'], 1)
                self.assertIn(f'HTTP {status}', summary['annotations'][0]['error'])

    def test_connection_failure_stops_the_run_after_configured_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(Path(directory))
            client = FakeClient([ConnectionError('offline')] * 3)
            with self.assertRaisesRegex(ModelServiceError, '连续 3 次'):
                asyncio.run(run(load_config(path), client_factory=lambda **_: client))
            self.assertEqual(len(client.requests), 3)
            self.assertTrue(client.closed)

    def test_temporary_connection_failure_can_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(Path(directory))
            client = FakeClient([ConnectionError('temporary'), response(), response()])
            result = asyncio.run(run(load_config(path), client_factory=lambda **_: client))
            self.assertEqual(result['success_request_count'], 2)
            self.assertEqual(len(client.requests), 3)

    def test_review_auth_failure_is_not_silently_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.prepare(root, enable_completeness_review=True)
            client = FakeClient([response(), ServiceStatusError(401)])
            with self.assertRaises(ModelServiceError):
                asyncio.run(run(load_config(path), client_factory=lambda **_: client))
            self.assertEqual(len(client.requests), 2)
            summary = read_json(next((root / 'results').glob('*/summary.json')))
            events = summary['annotations'][0]['model_feedback_attempts']
            self.assertEqual([event['stage'] for event in events], ['initial', 'review'])
            self.assertEqual(events[0]['raw_response'], response())

    def test_auth_failure_cancels_other_inflight_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(Path(directory), concurrency=2)
            class Client:
                def __init__(self):
                    self.chat = SimpleNamespace(completions=self)
                    self.started = asyncio.Event()
                    self.calls = 0
                    self.cancelled = self.closed = False
                async def create(self, **kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        self.started.set()
                        try:
                            await asyncio.Event().wait()
                        except asyncio.CancelledError:
                            self.cancelled = True
                            raise
                    await self.started.wait()
                    raise ServiceStatusError(403)
                async def close(self):
                    self.closed = True
            async def exercise():
                client = Client()
                with self.assertRaises(ModelServiceError):
                    await asyncio.wait_for(run(load_config(path), client_factory=lambda **_: client), 3)
                self.assertTrue(client.cancelled)
                self.assertTrue(client.closed)
                self.assertEqual(client.calls, 2)
            asyncio.run(exercise())

    def test_completed_success_survives_service_failure_and_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prepare(Path(directory))
            client = FakeClient([response(), ServiceStatusError(401)])
            with self.assertRaises(ModelServiceError):
                asyncio.run(run(load_config(path), client_factory=lambda **_: client))
            recovered_client = FakeClient([response()])
            recovered = asyncio.run(run(load_config(path), client_factory=lambda **_: recovered_client))
            self.assertEqual(recovered['reused_prediction_count'], 1)
            self.assertEqual(len(recovered_client.requests), 1)

    def test_auth_failure_stops_the_whole_comparison_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = suite_fixture(root, positive_counts=[5])
            client = FakeClient([ServiceStatusError(401)])
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401'):
                asyncio.run(run_ablation(load_config(path), client_factory=lambda **_: client))
            self.assertEqual(len(client.requests), 1)
            comparison = read_json(next((root / 'results').glob('*/comparison.json')))
            self.assertEqual(comparison['status'], 'failed')
            self.assertEqual(comparison['failed_mode'], 'prompt_only')
            self.assertEqual(comparison['groups'], [])
            self.assertEqual(comparison['service_status_code'], 401)

    def test_cancelled_suite_is_saved_as_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = suite_fixture(root, positive_counts=[5])
            async def interrupted(*args, **kwargs):
                raise asyncio.CancelledError
            with patch('industrial_fewshot.experiments.run', interrupted):
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(run_ablation(load_config(path)))
            comparison = read_json(next((root / 'results').glob('*/comparison.json')))
            self.assertEqual(comparison['status'], 'interrupted')
            self.assertEqual(comparison['interrupted_mode'], 'prompt_only')

    def test_cli_ctrl_c_exits_130_closes_client_and_saves_interrupted_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.prepare(root)
            client = FakeClient([])
            async def blocked(**kwargs):
                asyncio.get_running_loop().call_later(0.05, signal.raise_signal, signal.SIGINT)
                await asyncio.Event().wait()
            client.create = blocked
            with patch('industrial_fewshot.dependencies.load_openai_client', return_value=lambda **_: client), contextlib.redirect_stdout(io.StringIO()):
                code = cli_main(['--config', str(path)])
            self.assertEqual(code, 130)
            self.assertTrue(client.closed)
            summary = read_json(next((root / 'results').glob('*/summary.json')))
            self.assertEqual(summary['status'], 'interrupted')

import unittest
import contextlib
import io
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from scripts.run_5shot_batch import service_status
from scripts import run_5shot_batch as batch


class BatchServiceTest(unittest.TestCase):
    def setUp(self):
        self.config = {'targets': [{'model_config': {
            'name': 'qwen3.8-27b', 'base_url': 'http://127.0.0.1:15721/v1',
            'api_key': 'EMPTY', 'api_key_env': 'TEST_BATCH_KEY_NOT_SET'}}]}

    def client(self, status, body):
        client = MagicMock()
        response = MagicMock(status_code=status)
        response.json.return_value = body
        client.get.return_value = client.post.return_value = response
        return client

    def test_listing_requires_the_exact_configured_model(self):
        client = self.client(200, {'data': [{'id': 'different-model'}]})
        with patch('httpx2.Client') as factory:
            factory.return_value.__enter__.return_value = client
            ready, result = service_status(self.config)
        self.assertFalse(ready)
        self.assertEqual(result['available_models'], ['different-model'])
        client.post.assert_not_called()

    def test_proxy_chat_check_uses_original_model_and_no_images(self):
        client = self.client(200, {'choices': [{'message': {'content': 'OK'}}]})
        with patch('httpx2.Client') as factory:
            factory.return_value.__enter__.return_value = client
            ready, result = service_status(self.config, probe_chat=True)
        self.assertTrue(ready)
        self.assertEqual(result['requested_model'], 'qwen3.8-27b')
        args, kwargs = client.post.call_args
        self.assertEqual(args, ('http://127.0.0.1:15721/v1/chat/completions',))
        self.assertEqual(kwargs['json']['model'], 'qwen3.8-27b')
        self.assertEqual(kwargs['json']['messages'], [{'role': 'user', 'content': 'Reply with OK.'}])
        self.assertFalse(kwargs['json']['enable_thinking'])
        client.get.assert_not_called()

    def test_proxy_upstream_403_is_not_ready_and_does_not_expose_body(self):
        client = self.client(403, {'error': 'sensitive upstream diagnostic'})
        with patch('httpx2.Client') as factory:
            factory.return_value.__enter__.return_value = client
            ready, result = service_status(self.config, probe_chat=True)
        self.assertFalse(ready)
        self.assertEqual(result['http_status'], 403)
        self.assertNotIn('sensitive', str(result))

    def test_claude_probe_uses_messages_without_requiring_model_catalog(self):
        self.config['targets'][0]['model_config']['api_protocol'] = 'anthropic_messages'
        client = self.client(200, {'model': 'qwen3.8-27b', 'content': [{'type': 'text', 'text': 'OK'}]})
        with patch('httpx2.Client') as factory:
            factory.return_value.__enter__.return_value = client
            ready, result = service_status(self.config)
        self.assertTrue(ready)
        args, kwargs = client.post.call_args
        self.assertEqual(args, ('http://127.0.0.1:15721/v1/messages',))
        self.assertEqual(kwargs['headers']['anthropic-version'], '2023-06-01')
        self.assertEqual(kwargs['json']['thinking'], {'type': 'disabled'})
        self.assertEqual(result['api_protocol'], 'anthropic_messages')
        client.get.assert_not_called()

    def test_claude_probe_rejects_remapped_model(self):
        self.config['targets'][0]['model_config']['api_protocol'] = 'anthropic_messages'
        client = self.client(200, {'model': 'another-model', 'content': [{'type': 'text', 'text': 'OK'}]})
        with patch('httpx2.Client') as factory:
            factory.return_value.__enter__.return_value = client
            self.assertFalse(service_status(self.config)[0])

    def test_supervisor_does_not_wait_or_launch_after_auth_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {**self.config, 'experiment_suite': {'positive_counts': [5]}}
            argv = ['batch', '--config', str(root / 'test.yaml'), '--job-dir', str(root / 'outputs/job')]
            with patch.object(batch, 'ROOT', root), patch('sys.argv', argv), \
                 patch.object(batch, 'load_config', return_value=config), \
                 patch.object(batch, 'group_specs', return_value=list(range(15))), \
                 patch.object(batch, 'service_status', return_value=(False, {'http_status': 401})), \
                 patch.object(batch.time, 'sleep') as sleep, \
                 patch.object(batch.subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch.main(), 1)
            sleep.assert_not_called()
            launch.assert_not_called()

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import httpx2
from openai import AsyncOpenAI

from industrial_fewshot.cache import request_signature
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import write_json
from industrial_fewshot.detectors.claude_messages import message_body, read_message, complete_message
from industrial_fewshot.runner import run
from industrial_fewshot.service_errors import ModelServiceError
from tests.support import fixture, response


def settings(**extra):
    return dict(model='fake-model', temperature=0.1, top_p=0.9, max_tokens=8192,
                stream=False, enable_thinking=False, **extra)


class ClaudeTransportTest(unittest.TestCase):
    def test_proxy_alias_does_not_change_required_model_and_changes_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path,raw=fixture(Path(directory))
            raw['model'].update(name='qwen3.8-flash',api_protocol='anthropic_messages',
                                proxy_request_model='claude-sonnet-4-6')
            write_json(path,raw)
            config=load_config(path)
            self.assertEqual(config['targets'][0]['config']['model'],'qwen3.8-flash')
            calls=[]
            def handler(request):
                calls.append(json.loads(request.content))
                return httpx2.Response(200,json={'model':'qwen3.8-flash',
                    'content':[{'type':'text','text':response()}],'stop_reason':'end_turn'})
            def factory(**kwargs):
                return AsyncOpenAI(**kwargs,http_client=httpx2.AsyncClient(
                    transport=httpx2.MockTransport(handler),trust_env=False))
            report=asyncio.run(run(config,client_factory=factory))
            self.assertEqual(report['failed_request_count'],0)
            self.assertEqual(calls[0]['model'],'claude-sonnet-4-6')
            descriptor=report['targets'][0]
            self.assertEqual(descriptor['model'],'qwen3.8-flash')
            self.assertEqual(descriptor['proxy_request_model'],'claude-sonnet-4-6')
            direct={k:v for k,v in descriptor.items() if k!='proxy_request_model'}
            self.assertNotEqual(request_signature(direct),request_signature(descriptor))
            with self.assertRaises(ModelServiceError):
                asyncio.run(read_message({'model':'qwen3.8-max','content':[]},False,'qwen3.8-flash'))
            with self.assertRaisesRegex(ModelServiceError,'缺少实际模型'):
                asyncio.run(read_message({'content':[]},False,'qwen3.8-flash',True))

    def test_proxy_alias_rejects_empty_alias_or_non_claude_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path,raw=fixture(Path(directory))
            raw['model']['proxy_request_model']='claude-sonnet-4-6'
            write_json(path,raw)
            with self.assertRaisesRegex(ValueError,'proxy_request_model'):load_config(path)
            raw['model'].update(api_protocol='anthropic_messages',proxy_request_model='')
            write_json(path,raw)
            with self.assertRaisesRegex(ValueError,'proxy_request_model'):load_config(path)

    def test_reference_text_and_images_preserve_order_and_bytes(self):
        content = [{'type': 'text', 'text': 'reference'},
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,cmVm'}},
                   {'type': 'text', 'text': 'query'},
                   {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,cXVlcnk='}}]
        body = message_body(settings(), [{'role': 'user', 'content': content}], {'type': 'json_object'})
        blocks = body['messages'][0]['content']
        self.assertEqual([block['type'] for block in blocks], ['text', 'image', 'text', 'image'])
        self.assertEqual(blocks[1]['source'], {'type': 'base64', 'media_type': 'image/png', 'data': 'cmVm'})
        self.assertEqual(blocks[3]['source']['data'], 'cXVlcnk=')
        self.assertEqual((body['max_tokens'], body['temperature'], body['top_p']), (8192, 0.1, 0.9))
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertIn('JSON', body['system'])
        self.assertNotIn('response_format', body)
        self.assertNotIn('extra_body', body)

    def test_native_schema_and_unknown_image_format(self):
        schema = {'type': 'object', 'properties': {}}
        body = message_body(settings(), [{'role': 'user', 'content': 'hello'}],
                            {'type': 'json_schema', 'json_schema': {'schema': schema}})
        self.assertEqual(body['output_config']['format']['schema'], schema)
        with self.assertRaises(ValueError):
            message_body(settings(), [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'https://example.com/image'}}]}])

    def test_nonstream_answer_usage_and_stop_reason_ignore_thinking(self):
        value = {'model': 'fake-model', 'content': [
            {'type': 'thinking', 'thinking': 'PRIVATE'}, {'type': 'text', 'text': '{"ok":true}'}],
                 'stop_reason': 'max_tokens', 'usage': {'input_tokens': 10, 'output_tokens': 5}}
        text, reason, usage, thinking = asyncio.run(read_message(value, False, 'fake-model'))
        self.assertEqual((text, reason, usage['output_tokens'], thinking), ('{"ok":true}', 'length', 5, None))
        with self.assertRaises(ModelServiceError):
            asyncio.run(read_message(value, False, 'different-model'))

    def test_stream_merges_usage_and_rejects_truncated_stream(self):
        class Events:
            def __init__(self, truncated=False):
                self.closed = False
                self.events = [
                    {'type': 'message_start', 'message': {'model': 'fake-model', 'usage': {'input_tokens': 10}}},
                    {'type': 'content_block_start', 'content_block': {'type': 'text', 'text': '{'}},
                    {'type': 'content_block_delta', 'delta': {'type': 'thinking_delta', 'thinking': 'PRIVATE'}},
                    {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': '}'}},
                    {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 2}},
                    {'type': 'message_stop'}]
                if truncated:
                    self.events.pop()
            def __aiter__(self):
                async def iterate():
                    for event in self.events:
                        yield event
                return iterate()
            async def close(self):
                self.closed = True
        events = Events()
        self.assertEqual(asyncio.run(read_message(events, True, 'fake-model')),
                         ('{}', 'stop', {'input_tokens': 10, 'output_tokens': 2}, None))
        self.assertTrue(events.closed)
        events = Events(truncated=True)
        with self.assertRaises(ValueError):
            asyncio.run(read_message(events, True))
        self.assertTrue(events.closed)

    def test_real_sdk_uses_messages_for_image_review_and_cache_is_protocol_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw = fixture(root)
            openai_descriptor = asyncio.run(run(load_config(path), dry_run=True))['targets'][0]
            raw['model']['api_protocol'] = 'anthropic_messages'
            raw['inference']['enable_completeness_review'] = True
            raw['require_box_reason'] = True
            write_json(path, raw)
            calls = []
            def handler(request):
                calls.append((request.url.path, request.headers, json.loads(request.content)))
                answer = json.loads(response())
                answer['objects'][0]['reason'] = 'Visible object within its bounds'
                return httpx2.Response(200, json={'model': 'fake-model',
                    'content': [{'type': 'text', 'text': json.dumps(answer)}],
                    'stop_reason': 'end_turn', 'usage': {'input_tokens': 12, 'output_tokens': 8}})
            def factory(**kwargs):
                return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(
                    transport=httpx2.MockTransport(handler), trust_env=False))
            report = asyncio.run(run(load_config(path), client_factory=factory))
            self.assertEqual(report['failed_request_count'], 0)
            self.assertEqual(len(calls), 2)
            for endpoint, headers, body in calls:
                self.assertEqual(endpoint, '/v1/messages')
                self.assertEqual(headers['anthropic-version'], '2023-06-01')
                self.assertEqual(body['messages'][0]['content'][-1]['type'], 'image')
                self.assertEqual(body['model'], 'fake-model')
            record = json.loads((Path(report['run_dir']) / 'records.jsonl').read_text(encoding='utf-8'))
            self.assertTrue(record['review_performed'])
            self.assertEqual(record['usage']['input_tokens'], 12)
            self.assertEqual(record['objects'][0]['reason'], 'Visible object within its bounds')
            self.assertNotEqual(request_signature(openai_descriptor), request_signature(report['targets'][0]))
            descriptor = dict(openai_descriptor, api_protocol='anthropic_messages')
            self.assertNotEqual(request_signature(openai_descriptor), request_signature(descriptor))
            old = dict(openai_descriptor)
            old.pop('api_protocol')
            self.assertEqual(request_signature(old), request_signature(openai_descriptor))
            cached = asyncio.run(run(load_config(path), client_factory=lambda **_: self.fail('Cache called API')))
            self.assertEqual(cached['reused_prediction_count'], 1)

    def test_real_sdk_auth_failure_aborts_before_next_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = fixture(Path(directory))
            raw['model']['api_protocol'] = 'anthropic_messages'
            write_json(path, raw)
            def factory(**kwargs):
                return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(
                    transport=httpx2.MockTransport(lambda _: httpx2.Response(401, json={'error': 'denied'})),
                    trust_env=False))
            with self.assertRaises(ModelServiceError) as error:
                asyncio.run(run(load_config(path), client_factory=factory))
            self.assertEqual(error.exception.status_code, 401)

    def test_sdk_reads_anthropic_sse_events_and_closes_response(self):
        events = [
            {'type': 'message_start', 'message': {'model': 'fake-model', 'usage': {'input_tokens': 3}}},
            {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': '{}'}},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 2}},
            {'type': 'message_stop'}]
        payload = ''.join('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n' for event in events)
        captured = []
        def handler(request):
            captured.append(request)
            return httpx2.Response(200, text=payload, headers={'content-type': 'text/event-stream'})
        async def check():
            async with AsyncOpenAI(base_url='http://localhost/v1', api_key='EMPTY',
                    http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))) as client:
                config = settings()
                config['stream'] = True
                return await complete_message(client, config, [{'role': 'user', 'content': 'JSON'}])
        self.assertEqual(asyncio.run(check()), ('{}', 'stop', {'input_tokens': 3, 'output_tokens': 2}, None))
        self.assertEqual(captured[0].url.path, '/v1/messages')

    def test_unknown_protocol_and_unsupported_thinking_fail_before_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = fixture(Path(directory))
            raw['model']['api_protocol'] = 'invalid'
            write_json(path, raw)
            with self.assertRaises(ValueError):
                load_config(path)
            raw['model']['api_protocol'] = 'anthropic_messages'
            raw['inference']['enable_thinking'] = True
            write_json(path, raw)
            with self.assertRaises(ValueError):
                load_config(path)

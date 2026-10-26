import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from industrial_fewshot.cache import PredictionCache, request_signature
from industrial_fewshot.config import load_config
from industrial_fewshot.data.exports import export_labelme, export_model_replies
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.evaluation.feedback import build_feedback
from industrial_fewshot.postprocess.schema import build_json_schema, validate_prediction
from industrial_fewshot.runner import run
from tests.support import FakeClient, fixture, response


def reason_response(reason='可见目标外轮廓，框覆盖完整目标。', box=None):
    data = json.loads(response())
    data['objects'][0]['reason'] = reason
    if box is not None:
        data['objects'][0]['bbox_2d'] = box
    return json.dumps(data, ensure_ascii=False)


def feedback_fixture(root, **overrides):
    path, raw = fixture(root, mode='test')
    raw.pop('require_box_reason')  # Verify the actual default-on MLLM behavior.
    raw.update(overrides)
    write_json(path, raw)
    return path, raw


def execute(path, payloads=None, **kwargs):
    client = FakeClient(payloads or [])
    report = asyncio.run(run(load_config(path), client_factory=lambda **_: client, **kwargs))
    summary = read_json(Path(report['run_dir']) / 'summary.json') if not kwargs.get('dry_run') else None
    return report, client, summary


class ModelFeedbackTests(unittest.TestCase):
    def test_default_requests_distinct_per_box_reasons_and_saves_all_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = feedback_fixture(root)
            truth = root / 'images/frame.json'
            original_truth = truth.read_bytes()
            payload = json.loads(reason_response('左侧目标边界完整，框覆盖其外轮廓。'))
            payload['objects'].append({**payload['objects'][0], 'bbox_2d': [810, 200, 950, 900],
                                       'reason': '右侧独立目标可见，单独框住其完整轮廓。'})
            payload['target_count'] = 2
            report, client, summary = execute(path, [json.dumps(payload)])
            self.assertEqual(report['success_request_count'], 1)
            record = summary['annotations'][0]
            reasons = [obj['reason'] for obj in payload['objects']]
            self.assertEqual([obj['reason'] for obj in record['objects']], reasons)
            self.assertEqual([shape['description'] for shape in read_json(Path(record['labelme_json']))['shapes']], reasons)
            feedback = read_json(Path(record['feedback_json']))
            self.assertEqual([obj['reason'] for obj in feedback['objects']], reasons)
            self.assertIn(reasons[1], Path(record['feedback_text']).read_text(encoding='utf-8'))
            required = client.requests[0]['response_format']['json_schema']['schema']['properties']['objects']['items']['required']
            self.assertEqual(required.count('reason'), 1)
            self.assertEqual(load_config(path)['targets'][0]['prompt'].count('【逐框反馈】'), 1)
            self.assertEqual(truth.read_bytes(), original_truth)
            self.assertNotIn('ground_truth', json.dumps(feedback))
            export_labelme([{**record, 'width': 1000, 'height': 500}], root / 'images', root / 'export', ['target'], 'none')
            self.assertEqual([shape['description'] for shape in read_json(root / 'export/frame.json')['shapes']], reasons)
            export_model_replies([record], root / 'replies')
            reply = read_json(root / 'replies/frame.json')
            self.assertEqual(reply['model_feedback_attempts'], record['model_feedback_attempts'])

    def test_review_updates_box_reason_and_preserves_initial_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = feedback_fixture(Path(directory))
            raw['inference']['enable_completeness_review'] = True
            write_json(path, raw)
            _, client, summary = execute(path, [reason_response('首轮依据'), reason_response('复核后收紧框边界', [150, 200, 750, 900])])
            record = summary['annotations'][0]
            feedback = read_json(Path(record['feedback_json']))
            self.assertEqual(record['objects'][0]['reason'], '复核后收紧框边界')
            self.assertEqual(feedback['initial_annotation']['objects'][0]['reason'], '首轮依据')
            self.assertEqual(feedback['objects'][0]['bbox_pixels_xyxy'], [150, 100, 750, 450])
            self.assertTrue(feedback['review_performed'])
            self.assertEqual([event['stage'] for event in feedback['attempts']], ['initial', 'review'])
            self.assertIn('首轮依据', str(client.requests[1]['messages']))

    def test_invalid_review_retains_first_reason_and_invalid_raw_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = feedback_fixture(Path(directory))
            raw['inference']['enable_completeness_review'] = True
            write_json(path, raw)
            _, _, summary = execute(path, [reason_response('首轮有效依据'), response()])
            record = summary['annotations'][0]
            feedback = read_json(Path(record['feedback_json']))
            self.assertEqual(record['objects'][0]['reason'], '首轮有效依据')
            self.assertFalse(record['review_performed'])
            self.assertIn('逐框 reason', record['review_error'])
            self.assertEqual(feedback['attempts'][-1]['raw_response'], response())
            self.assertEqual(feedback['attempts'][-1]['status'], 'invalid_response')

    def test_missing_reason_retries_and_all_failed_responses_remain_inspectable(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = feedback_fixture(Path(directory))
            raw['inference'].update(max_attempts=2, retry_delay_seconds=0)
            write_json(path, raw)
            report, client, summary = execute(path, [response(), reason_response('重试后提供依据')])
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(report['success_request_count'], 1)
            record = summary['annotations'][0]
            self.assertEqual([event['status'] for event in record['model_feedback_attempts']], ['invalid_response', 'success'])
            report, _, summary = execute(path, ['not json', 'still not json'], force=True)
            self.assertEqual(report['failed_request_count'], 1)
            feedback = read_json(Path(summary['annotations'][0]['feedback_json']))
            self.assertEqual([event['raw_response'] for event in feedback['attempts']], ['not json', 'still not json'])

    def test_cache_reuse_restores_reasons_and_old_protocol_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = feedback_fixture(root)
            first, _, original = execute(path, [reason_response('缓存理由')])
            second, client, summary = execute(path)
            self.assertEqual(second['reused_prediction_count'], 1)
            self.assertFalse(client.requests)
            record = summary['annotations'][0]
            self.assertEqual(record['objects'][0]['reason'], '缓存理由')
            self.assertNotEqual(record['feedback_json'], original['annotations'][0]['feedback_json'])
            self.assertTrue(read_json(Path(record['feedback_json']))['cache_hit'])
            descriptor = first['targets'][0]
            legacy = {key: value for key, value in descriptor.items() if key != 'require_box_reason'}
            self.assertNotEqual(request_signature(legacy), request_signature(descriptor))
            self.assertEqual(request_signature(legacy), request_signature({**legacy, 'require_box_reason': False}))
            cache = PredictionCache(root / 'unused', root / 'images', descriptor)
            broken = {**record, 'objects': [{key: value for key, value in record['objects'][0].items() if key != 'reason'}]}
            self.assertFalse(cache._validated(broken, {'relative_path': 'frame.jpg', 'width': 1000, 'height': 500}))

    def test_no_target_still_saves_image_reason_without_fabricating_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = feedback_fixture(Path(directory))
            payload = {'annotation_status': 'no_target', 'reason_code': 'no_target',
                       'reason': '画面中没有符合定义的目标。', 'target_count': 0, 'objects': []}
            _, _, summary = execute(path, [json.dumps(payload)])
            feedback = read_json(Path(summary['annotations'][0]['feedback_json']))
            self.assertEqual(feedback['objects'], [])
            self.assertEqual(feedback['image_reason'], payload['reason'])

    def test_local_and_legacy_missing_reasons_are_explicitly_unavailable(self):
        record = {**json.loads(response()), 'status': 'success', 'relative_path': 'frame.jpg'}
        record['objects'][0]['bbox_pixels_xyxy'] = [100, 100, 800, 450]
        for backend, expected in [('yoloe', 'unsupported'), ('groundingdino', 'unsupported'), ('generic', 'not_provided')]:
            target = {'backend': backend, 'id': 'target', 'config': {'model': 'test'}}
            feedback = build_feedback(record, target, 1000, 500)
            self.assertEqual(feedback['objects'][0]['reason_status'], expected)
            self.assertIsNone(feedback['objects'][0]['reason'])

    def test_reason_validation_is_optional_for_legacy_and_strict_when_supplied(self):
        self.assertNotIn('reason', build_json_schema({'labels': [{'name': 'target'}]})['properties']['objects']['items']['required'])
        payload = json.loads(response())
        for value in ('', '  ', None, 123):
            payload['objects'][0]['reason'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'reason'):
                validate_prediction(payload, {'target'}, 1000, 500)

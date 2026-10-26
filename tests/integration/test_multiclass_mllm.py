import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from industrial_fewshot.cache import request_signature, image_sha256
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.data.prepare import prepare_dataset
from industrial_fewshot.data.supports import ReferencePool
from industrial_fewshot.evaluation.evaluator import evaluate_saved_run
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.runner import run
from tests.support import FakeClient, fixture, response


class MultiClassMLLMTest(unittest.TestCase):
    def test_one_request_preserves_all_classes_in_gt_cache_and_saved_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, raw = fixture(root)
            raw['labels'].append(dict(name='other', description='second class'))
            write_json(path, raw)
            truth = read_json(root/'images/frame.json')
            truth['shapes'].append({**truth['shapes'][0], 'label': 'other'})
            write_json(root/'images/frame.json', truth)
            payload = json.loads(response())
            payload['objects'].append({**payload['objects'][0], 'label': 'other'})
            payload['target_count'] = 2
            client = FakeClient([json.dumps(payload)])
            report = asyncio.run(run(load_config(path), client_factory=lambda **kw: client))
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(report['targets'][0]['labels'], ['target', 'other'])
            self.assertEqual(report['metrics']['tp'], 2)
            self.assertEqual(set(report['metrics']['per_class']), {'target', 'other'})
            saved = evaluate_saved_run(Path(report['run_dir']))
            self.assertEqual(saved['metrics']['tp'], 2)
            repeated = asyncio.run(run(load_config(path), client_factory=lambda **kw: self.fail('cache miss')))
            self.assertEqual(repeated['reused_prediction_count'], 1)
            descriptor = report['targets'][0]
            self.assertNotEqual(request_signature(descriptor), request_signature({**descriptor, 'labels': ['target', 'changed']}))

    def test_per_class_supports_use_two_variants_and_one_query_request_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, raw = fixture(root)
            raw['labels'].append(dict(name='other', description='second class'))
            raw.update(reference_per_class=True, support_manifest_dir='references')
            raw['experiment_suite'] = dict(enabled=True, include_prompt_only=True, positive_counts=[1],
                positive_variants=['O+B'], negative_variants=[False], selection_seed=0)
            for label, color in [('target', 'red'), ('other', 'blue')]:
                source = root/'originals'/label
                source.mkdir(parents=True)
                Image.new('RGB', (640,360), color).save(source/'ref.jpg')
                write_json(source/'ref.json', dict(imageWidth=640,imageHeight=360,
                    shapes=[dict(label=label,shape_type='rectangle',points=[[10,20],[300,320]])]))
                ref = root/'references'/label
                prepare_dataset(source, ref)
                pool = ReferencePool(dict(reference_image_dir=str(ref),labels=[dict(name=label)]),0)
                row = pool.select(1)[0][0]
                write_json(ref/'1shot/seed_0.json', dict(schema='industrial_fewshot.support.v1',positive_count=1,
                    selection_seed=0,reference_root=str(ref),negative_samples=[],positive_samples=[dict(
                        sample_id=row['sample_id'],image_path=str(pool.path(row['image'])),
                        image_sha256=image_sha256(pool.path(row['image'])),
                        json_path=str(pool.path(row['annotation'])),annotation_sha256=image_sha256(pool.path(row['annotation'])))]))
            write_json(path,raw)
            clients = []
            def factory(**kw):
                client = FakeClient([response()]);clients.append(client);return client
            report = asyncio.run(run_ablation(load_config(path),client_factory=factory))
            self.assertEqual(report['group_count'],2)
            self.assertEqual(report['executed_inference_count'],2)
            plain, visual = (client.requests[0]['messages'][0]['content'] for client in clients)
            self.assertEqual(sum(b['type']=='image_url' for b in plain),1)
            self.assertEqual(sum(b['type']=='image_url' for b in visual),5)
            self.assertIn('类别 target',str(visual))
            self.assertIn('类别 other',str(visual))
            group = report['groups'][1]
            self.assertEqual(len(group['support_manifests']),2)
            self.assertEqual(group['targets'][0]['reference_count']['reference_originals'],2)
            self.assertEqual(group['targets'][0]['reference_count']['reference_boxed_images'],2)
            self.assertEqual(group['experiment']['positive_count'],1)

    def test_duplicate_labels_and_multiclass_local_backend_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,raw = fixture(Path(tmp))
            raw['labels'] *= 2
            write_json(path,raw)
            with self.assertRaisesRegex(ValueError,'重复'):
                load_config(path)
            raw['labels'][1] = dict(name='other',description='different')
            raw['backend'] = 'yoloe'
            write_json(path,raw)
            with self.assertRaisesRegex(ValueError,'generic'):
                load_config(path)

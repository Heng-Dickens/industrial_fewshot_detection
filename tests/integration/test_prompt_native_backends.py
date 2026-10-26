import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from industrial_fewshot.config import load_config
from industrial_fewshot.configuration import load_document
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.detectors.prompt_native import native_settings, native_prediction, PromptNativeRunner
from industrial_fewshot.evaluation.compare import compare_runs
from industrial_fewshot.evaluation.evaluator import evaluate_saved_run
from industrial_fewshot.evaluation.visualize import render_run
from industrial_fewshot.prompts.references import prepare_reference_examples
from industrial_fewshot.runner import run
from tests.integration.test_yoloe_backend import local_fixture, FakeRunner


def fixture(root, backend, visual=False):
    path, raw = local_fixture(root)
    source = root / 'checkout'
    entry = source / ('mmdet/apis/det_inferencer.py' if backend == 'pet_dino' else 'deploy/test_coco_pytorch.py')
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text('# offline interface fixture', encoding='utf-8')
    language = root / 'language'
    language.mkdir()
    write_json(language / 'config.json', {})
    (language / 'tokenizer.json').write_text('{}', encoding='utf-8')
    (language / 'model.safetensors').write_bytes(b'fixture language weights')
    (root / 'model.py').write_text('# fixture model config', encoding='utf-8')
    settings = dict(source_dir='checkout', weights='weights.pt', worker_python=str(sys.executable),
                    language_model_dir='language', text_prompt='suspended load.')
    if backend == 'pet_dino':
        settings.update(model_config='model.py', prompt_mode='visual' if visual else 'text', positive_count=5 if visual else 0)
    raw.update(backend=backend, max_reference_examples=0, max_reference_crops=0,
               max_reference_negatives=0, evaluation={'score_threshold': 0.5, 'score_thresholds': [0.5, 0.7]},
               **{backend:settings})
    write_json(path, raw)
    return path, raw


class NativeBackendTest(unittest.TestCase):
    def execute(self, path, factory=FakeRunner, dry=False):
        with patch('industrial_fewshot.runner.inference.request_annotation', side_effect=AssertionError('No API calls')):
            return asyncio.run(run(load_config(path), local_runner_factory=factory, dry_run=dry))

    def test_all_modes_export_prompts_scores_cache_metrics_and_render(self):
        for backend, visual in [('pet_dino', True), ('pet_dino', False), ('wedetect', False)]:
            with self.subTest(backend=backend, visual=visual), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, raw = fixture(root, backend, visual)
                result = self.execute(path)
                saved = Path(result['run_dir'])
                target = result['targets'][0]
                self.assertEqual(target['prompt_snapshot']['kind'], 'visual' if visual else 'text')
                self.assertEqual(len(target['reference_sample_ids']), 5 if visual else 0)
                if visual:
                    self.assertIn('PET-DINO', (saved / 'prompt.txt').read_text(encoding='utf-8'))
                    self.assertEqual(len(target['prompt_snapshot']['references']), 5)
                else:
                    self.assertEqual((saved / 'prompt.txt').read_text(encoding='utf-8'), raw[backend]['text_prompt'])
                self.assertEqual(saved.name, 'weights_5shot' if visual else 'weights_0shot')
                self.assertEqual(result['metrics']['mAP50'], 1)
                self.assertEqual(read_json(saved / 'feedback/frame.json')['feedback_source'], 'unsupported_by_detector')
                again = self.execute(path, lambda _: self.fail('Should reuse successful prediction'))
                self.assertEqual(again['reused_prediction_count'], 1)
                self.assertEqual(evaluate_saved_run(saved)['metrics']['mAP50'], 1)
                comparison = compare_runs([saved, Path(again['run_dir'])], root / 'comparison')
                self.assertEqual(comparison['rows'][0]['method'], 'PET-DINO' if backend == 'pet_dino' else 'WeDetect')
                self.assertEqual(len(render_run(saved, root / 'rendered')['images']), 1)
                raw['evaluation']['score_threshold'] = 0.95
                write_json(path, raw)
                self.assertEqual(self.execute(path, lambda _: self.fail('Evaluation must reuse'))['metrics']['recall'], 0)
                if not visual:
                    raw[backend]['text_prompt'] = 'another target.'
                else:
                    annotation = Path(load_config(path)['targets'][0]['config']['reference_image_dir'])
                    cfg = load_config(path)['targets'][0]['config']
                    prepare_reference_examples(cfg)
                    annotation = Path(cfg['reference_examples'][0]['json_path'])
                    doc = read_json(annotation)
                    doc['shapes'][0]['points'][0][0] += 0.25
                    write_json(annotation, doc)
                write_json(path, raw)
                changed = self.execute(path)
                self.assertNotEqual(target['request_signature'], changed['targets'][0]['request_signature'])

    def test_dry_run_reports_missing_environment_without_importing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path, raw = fixture(Path(directory), 'pet_dino', True)
            raw['pet_dino']['worker_python'] = 'missing/python.exe'
            write_json(path, raw)
            result = self.execute(path, lambda _: self.fail('Dry run must not load model'), dry=True)
            metadata = result['targets'][0]['local_model']
            self.assertFalse(metadata['runtime_ready'])
            cfg = load_config(path)['targets'][0]['config']
            prepare_reference_examples(cfg)
            with self.assertRaisesRegex(ValueError, '未就绪'):
                PromptNativeRunner(cfg)
            self.assertFalse((Path(directory) / 'results').exists())

    def test_native_pixels_keep_subpixel_coordinates_and_model_scores(self):
        settings = dict(confidence=0.001, nms_iou=0.7, max_detections=100)
        result = native_prediction(dict(boxes=[[1200.125, 500.25, 1900.5, 1000.75],
                                               [1200, 500, 1900, 1000], [-20, 0, -1, 20]],
                                        scores=[0.875, 0.8, 0.9]), 'target', 2048, 1536, settings)
        self.assertEqual(result['target_count'], 1)
        self.assertEqual(result['objects'][0]['bbox_pixels_xyxy'], [1200.125, 500.25, 1900.5, 1000.75])
        self.assertEqual(result['objects'][0]['confidence'], 0.875)
        for rows in [dict(boxes=[[0, 0, 2, 2]], scores=[]),
                     dict(boxes=[[0, 0, 2, float('nan')]], scores=[0.9]),
                     dict(boxes=[[0, 0, 2, 2]], scores=[1.1])]:
            with self.assertRaises(ValueError):
                native_prediction(rows, 'target', 200, 100, settings)

    def test_validation_and_actual_config_text(self):
        root = Path(__file__).resolve().parents[2]
        for backend, bad in [('pet_dino', {'prompt_mode':'other'}), ('pet_dino', {'positive_count':0}),
                             ('pet_dino', {'prompt_mode':'text'}), ('wedetect', {'variant':'uni'}),
                             ('wedetect', {'confidence':float('nan')}), ('wedetect', {'nms_iou':True}),
                             ('wedetect', {'timeout_seconds':0}), ('wedetect', {'mystery':1})]:
            with self.subTest(backend=backend, bad=bad), self.assertRaises(ValueError):
                native_settings(backend, bad, root)
        text, _ = load_document(root / 'configs/specific/pet_dino_text.yaml')
        dino, _ = load_document(root / 'configs/specific/zero_shot.yaml')
        self.assertEqual(text['pet_dino']['text_prompt'], dino['groundingdino']['text_prompt'])
        self.assertEqual(Path(text['pet_dino']['model_config']), root / 'third_party/pet_dino/configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py')

    def test_persistent_worker_protocol_and_shutdown(self):
        import industrial_fewshot.detectors.prompt_native as native
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / 'fake_worker.py'
            worker.write_text('import sys,json\nfor line in sys.stdin:\n m=json.loads(line)\n print(json.dumps(dict(setup_seconds=.01, reference_encoding_seconds=0) if m["op"]=="init" else dict(boxes=[[1200.125,5,1700.5,80]], scores=[.875])), flush=True)\n', encoding='utf-8')
            config = dict(reference_mode='wedetect', wedetect=dict(worker_python=sys.executable, timeout_seconds=3,
                          confidence=0.001, nms_iou=0.7, max_detections=100),
                          reference_examples=[], labels=[{'name':'target'}], local_model={'runtime_ready':True})
            with patch.object(native, 'WORKER', worker):
                runner = PromptNativeRunner(config)
                try:
                    from io import BytesIO
                    data = BytesIO()
                    Image.new('RGB', (2000, 100)).save(data, format='PNG')
                    for _ in range(2):
                        self.assertEqual(runner.predict(data.getvalue())['objects'][0]['bbox_pixels_xyxy'][0], 1200.125)
                    self.assertIsNone(runner.process.poll())
                finally:
                    asyncio.run(runner.close())
                self.assertIsNotNone(runner.process.poll())

    def test_worker_init_failure_and_timeout_leave_no_child_process(self):
        import industrial_fewshot.detectors.prompt_native as native
        import subprocess
        for body, error in [('import sys,json\nfor line in sys.stdin: print(json.dumps(dict(error="load failed")), flush=True)\n', RuntimeError),
                            ('import sys,time\nsys.stdin.readline()\ntime.sleep(20)\n', TimeoutError)]:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                worker = Path(directory) / 'fake_worker.py'
                worker.write_text(body, encoding='utf-8')
                children = []
                popen = subprocess.Popen
                def create(*args, **kwargs):
                    child = popen(*args, **kwargs)
                    children.append(child)
                    return child
                config = dict(reference_mode='wedetect', wedetect=dict(worker_python=sys.executable, timeout_seconds=1),
                              reference_examples=[], local_model={'runtime_ready':True})
                with patch.object(native, 'WORKER', worker), patch.object(native.subprocess, 'Popen', side_effect=create):
                    with self.assertRaises(error):
                        PromptNativeRunner(config)
                self.assertIsNotNone(children[0].poll())


if __name__ == '__main__':
    unittest.main()

"""Isolated upstream PET-DINO and WeDetect inference, using native pixel boxes."""
from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from PIL import Image

from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.supports import ReferencePool
from industrial_fewshot.paths import PROJECT_ROOT
from industrial_fewshot.postprocess.boxes import clip_box, nms

BACKENDS = {'pet_dino', 'wedetect'}
WORKER = Path(__file__).with_name('prompt_native_worker.py')


def native_settings(backend: str, raw, base: Path) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f'{backend} 必须是配置对象')
    values = dict(source_dir=str(PROJECT_ROOT / 'third_party' / backend),
                  weights=str(PROJECT_ROOT / 'weights' / backend / ('pet_dino_swin_t.pth' if backend == 'pet_dino' else 'wedetect_base.pth')),
                  worker_python=str(PROJECT_ROOT / f'.venv-{backend}' / 'Scripts/python.exe'),
                  device='cpu', confidence=0.001, nms_iou=0.7, max_detections=100,
                  torch_threads=4, timeout_seconds=600, exclude_reference_overlap=True)
    if backend == 'pet_dino':
        values.update(prompt_mode='visual', positive_count=5, selection_seed=0,
                      model_config=str(PROJECT_ROOT / 'third_party/pet_dino/configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py'),
                      language_model_dir=str(PROJECT_ROOT / 'weights/pet_dino/bert-base-uncased'), text_prompt='hanging object.')
    else:
        values.update(variant='base', text_prompt='吊物',
                      language_model_dir=str(PROJECT_ROOT / 'weights/wedetect/xlm-roberta-base'))
    unknown = set(raw) - set(values)
    if unknown:
        raise ValueError(f'未知 {backend} 字段: {sorted(unknown)}')
    values.update(raw)
    for key, value in values.items():
        if key in {'confidence', 'nms_iou'}:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{backend}.{key} 必须是 0~1 的有限数字')
        elif key in {'max_detections', 'torch_threads', 'timeout_seconds', 'positive_count', 'selection_seed'}:
            minimum = 0 if key in {'selection_seed', 'positive_count'} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f'{backend}.{key} 必须是 >= {minimum} 的整数')
        elif key == 'exclude_reference_overlap':
            if type(value) is not bool:
                raise ValueError(f'{backend}.{key} 必须是布尔值')
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f'{backend}.{key} 必须是非空字符串')
    if backend == 'pet_dino':
        if values['prompt_mode'] not in {'visual', 'text'}:
            raise ValueError('pet_dino.prompt_mode 只支持 visual 或 text')
        if (values['prompt_mode'] == 'visual') != (values['positive_count'] > 0):
            raise ValueError('PET-DINO visual 模式需要 positive_count > 0；text 模式需要 positive_count: 0')
    elif values['variant'] not in {'tiny', 'base', 'large'}:
        raise ValueError('wedetect.variant 只支持 tiny/base/large')
    for key in ('source_dir', 'weights', 'worker_python', 'model_config', 'language_model_dir'):
        if key in values:
            values[key] = str((base / values[key]).expanduser().resolve())
    return values


def _fingerprint(root: Path, suffixes: set[str]):
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in suffixes) if root.is_dir() else []
    return digest([[p.relative_to(root).as_posix(), image_sha256(p)] for p in files]) if files else None


def _references(config, settings):
    pool = ReferencePool(config, settings.get('selection_seed', 0)) if config.get('reference_image_dir') else None
    examples = []
    if settings.get('prompt_mode') == 'visual':
        if pool is None:
            raise ValueError('PET-DINO 视觉模式缺少 reference_image_dir')
        rows, _ = pool.select(settings['positive_count'])
        labels = {row['name'] for row in config['labels']}
        for row in rows:
            image_path = pool.path(row['image'])
            with Image.open(image_path) as image:
                width, height = image.size
            doc = pool.documents[row['sample_id']]
            if (doc.get('imageWidth'), doc.get('imageHeight')) != (width, height):
                raise ValueError(f'参考标注尺寸不符: {image_path}')
            boxes = []
            for shape in doc['shapes']:
                if shape.get('label') not in labels:
                    continue
                points = shape.get('points')
                if (shape.get('shape_type') not in {'rectangle', 'polygon'} or not isinstance(points, list)
                        or len(points) < 2 or any(not isinstance(p, list) or len(p) != 2 for p in points)
                        or any(type(v) not in (int, float) or not math.isfinite(v) for p in points for v in p)):
                    raise ValueError(f'无效参考框: {row["annotation"]}')
                box = [min(p[0] for p in points), min(p[1] for p in points),
                       max(p[0] for p in points), max(p[1] for p in points)]
                if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    raise ValueError(f'参考框越界或为空: {row["annotation"]}')
                boxes.append(box)
            if not boxes:
                raise ValueError(f'正例缺少目标框: {row["annotation"]}')
            examples.append(dict(sample_id=row['sample_id'], image_path=str(image_path), json_path=str(pool.path(row['annotation'])),
                                 image_sha256=image_sha256(image_path), bboxes=boxes))
    config.update(reference_examples=examples, reference_originals=examples, reference_crops=[],
                  reference_boxed_images=[], reference_negatives=[],
                  reference_pool_image_hashes=sorted(pool.original_hashes) if pool else [])
    return examples


def prepare_references(config):
    backend = config['reference_mode']
    settings = config[backend]
    examples = _references(config, settings)
    visual = settings.get('prompt_mode') == 'visual'
    config['reference_manifest_sha256'] = digest(dict(schema=f'{backend}.v1', examples=examples,
                                                     text=None if visual else settings['text_prompt']))
    source, language = Path(settings['source_dir']), Path(settings['language_model_dir'])
    weights, python = Path(settings['weights']), Path(settings['worker_python'])
    entry = source / ('mmdet/apis/det_inferencer.py' if backend == 'pet_dino' else 'deploy/test_coco_pytorch.py')
    required = [entry, weights, python, language / 'config.json']
    if backend == 'pet_dino':
        required.append(Path(settings['model_config']))
    tokenizer_ready = any((language / name).is_file() for name in ('tokenizer.json', 'vocab.txt', 'sentencepiece.bpe.model'))
    language_weights_ready = backend == 'wedetect' or any(language.glob('*.safetensors')) or any(language.glob('pytorch_model*.bin'))
    missing = [str(p) for p in required if not p.is_file()]
    if not tokenizer_ready:
        missing.append(str(language / '<tokenizer files>'))
    if not language_weights_ready:
        missing.append(str(language / '<BERT weights>'))
    versions = None
    if python.is_file():
        # Check the selected interpreter, not the parent project's environment.
        script = "import importlib.metadata as m,json; names=('torch','torchvision','transformers','mmcv','mmengine','numpy'); d={};\nfor n in names:\n try:d[n]=m.version(n)\n except m.PackageNotFoundError:d[n]=None\nprint(json.dumps(d))"
        try:
            probe = subprocess.run([str(python), '-c', script], capture_output=True, text=True, timeout=15,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            versions = json.loads(probe.stdout) if probe.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
    packages = ('torch', 'torchvision', 'transformers', 'mmcv', 'mmengine') if backend == 'pet_dino' else ('torch', 'torchvision', 'transformers')
    missing += [f'{python}: package {name}' for name in packages if not (versions or {}).get(name)]
    upstream = json.loads((source / 'UPSTREAM.json').read_text(encoding='utf-8')) if (source / 'UPSTREAM.json').is_file() else {}
    config['local_model'] = dict(settings=settings, runtime_ready=not missing, missing_assets=missing,
        library_versions=versions, upstream_revision=upstream.get('revision'),
        weights_sha256=image_sha256(weights) if weights.is_file() else None,
        source_sha256=_fingerprint(source, {'.py', '.json', '.yaml', '.yml'}),
        language_sha256=_fingerprint(language, {'.py', '.json', '.txt', '.model', '.bin', '.safetensors'}),
        model_config_sha256=image_sha256(Path(settings['model_config'])) if backend == 'pet_dino' and Path(settings['model_config']).is_file() else None,
        adapter_sha256=digest([image_sha256(Path(__file__)), image_sha256(WORKER)]))


def native_prediction(rows: dict, label: str, width: int, height: int, settings: dict) -> dict:
    boxes, scores = rows['boxes'], rows['scores']
    if len(boxes) != len(scores):
        raise ValueError('检测框和分数数量不一致')
    objects = []
    for box, score in zip(boxes, scores):
        if (len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('模型输出无效坐标或置信度')
        if score < settings['confidence']:
            continue
        if max(0, box[0]) >= min(width, box[2]) or max(0, box[1]) >= min(height, box[3]):
            continue
        box = clip_box(box, width, height)
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        objects.append(dict(label=label, bbox_pixels_xyxy=box, confidence=score))
    objects = nms(objects, settings['nms_iou'], settings['max_detections'])
    return dict(annotation_status='annotated' if objects else 'no_target',
                reason_code='target_detected' if objects else 'no_target', reason='本地提示检测',
                target_count=len(objects), objects=objects)


class PromptNativeRunner:
    """One persistent worker per model; upstream imports never enter the parent."""
    def __init__(self, config):
        self.config = config
        self.settings = config[config['reference_mode']]
        self.process = None
        if not config['local_model']['runtime_ready']:
            raise ValueError('模型环境或文件未就绪，请按 docs/guides/PET_DINO_WEDETECT.md 准备：\n' +
                             '\n'.join(config['local_model']['missing_assets']))
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        self.process = subprocess.Popen([self.settings['worker_python'], '-u', str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, encoding='utf-8', env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.responses = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        try:
            ready = self._request(dict(op='init', backend=config['reference_mode'], settings=self.settings,
                                       references=config['reference_examples']))
            self.setup_seconds = ready['setup_seconds']
            self.reference_encoding_seconds = ready['reference_encoding_seconds']
        except BaseException:
            self._stop()
            raise

    def _reader(self):
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except Exception as exc:
            self.responses.put(dict(error=f'模型子进程通信失败: {exc}'))
        finally:
            self.responses.put(dict(error='模型子进程已退出，请查看上方错误'))

    def _request(self, payload):
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
            self.process.stdin.flush()
            result = self.responses.get(timeout=self.settings['timeout_seconds'])
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result
        except queue.Empty as exc:
            self._stop()
            raise TimeoutError('本地模型推理超时，已停止子进程') from exc

    def predict(self, data):
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        rows = self._request(dict(op='predict', image_base64=base64.b64encode(data).decode('ascii')))
        return native_prediction(rows, self.config['labels'][0]['name'], width, height, self.settings)

    def _stop(self):
        if self.process is not None:
            # EOF lets an idle worker release its temporary prototype/images.
            # A stuck model still gets a bounded, forceful shutdown.
            if self.process.poll() is None:
                if self.process.stdin:
                    self.process.stdin.close()
                try:
                    self.process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process.wait(timeout=2)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()

    async def close(self):
        self._stop()

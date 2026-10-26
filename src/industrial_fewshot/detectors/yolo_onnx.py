"""Supervised YOLO raw-DFL ONNX detection, without custom training classes."""
from __future__ import annotations

import ast
import importlib.metadata
import math
from pathlib import Path
import time

from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.supports import ReferencePool
from industrial_fewshot.postprocess.boxes import nms


def yolo_onnx_settings(raw: dict, base: Path) -> dict:
    if not isinstance(raw, dict):
        raise ValueError('yolo_onnx 必须是配置对象')
    settings = dict(input_size=[960, 1280], class_names=['hanging object'], target_class_id=0,
                    strides=[8, 16, 32], reg_max=16, confidence=0.2, nms_iou=0.65,
                    max_detections=300, intra_op_threads=4, providers=['CPUExecutionProvider'],
                    exclude_reference_overlap=True)
    allowed = set(settings) | {'weights'}
    if set(raw) - allowed:
        raise ValueError(f'未知 yolo_onnx 配置字段: {sorted(set(raw) - allowed)}')
    settings.update(raw)
    if not isinstance(settings.get('weights'), str) or not settings['weights'].strip():
        raise ValueError('yolo_onnx.weights 必须是本地 ONNX 路径')
    weights = (base / settings['weights']).expanduser().resolve()
    if weights.suffix.lower() != '.onnx':
        raise ValueError('yolo_onnx.weights 必须是 .onnx 文件；PT 包含自定义网络层，不能交给此适配器')
    settings['weights'] = str(weights)
    for key in ('input_size', 'strides'):
        value = settings[key]
        if not isinstance(value, list) or not value or any(type(v) is not int or v < 1 for v in value):
            raise ValueError(f'yolo_onnx.{key} 必须是正整数列表')
    if len(settings['input_size']) != 2 or len(set(settings['strides'])) != len(settings['strides']):
        raise ValueError('input_size 必须是 [高, 宽]，strides 不能重复')
    if any(size % stride for size in settings['input_size'] for stride in settings['strides']):
        raise ValueError('input_size 必须能被每个 stride 整除')
    for key in ('reg_max', 'max_detections', 'intra_op_threads'):
        if type(settings[key]) is not int or settings[key] < (2 if key == 'reg_max' else 1):
            raise ValueError(f'yolo_onnx.{key} 数值无效')
    for key in ('confidence', 'nms_iou'):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'yolo_onnx.{key} 必须是 0~1 的有限数字')
    names = settings['class_names']
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('yolo_onnx.class_names 必须是与模型类别 ID 对应的非空名称列表')
    if type(settings['target_class_id']) is not int or not 0 <= settings['target_class_id'] < len(names):
        raise ValueError('yolo_onnx.target_class_id 超出类别范围')
    providers = settings['providers']
    if (not isinstance(providers, list) or not providers
            or any(not isinstance(v, str) or not v.strip() for v in providers)
            or len(set(providers)) != len(providers)):
        raise ValueError('yolo_onnx.providers 必须是不重复的运行设备名称列表')
    if type(settings['exclude_reference_overlap']) is not bool:
        raise ValueError('yolo_onnx.exclude_reference_overlap 必须是布尔值')
    return settings


def prepare_references(config: dict) -> None:
    """No reference enters this model; optional pool only excludes cohort overlap."""
    settings = config['yolo_onnx']
    hashes = sorted(ReferencePool(config, 0).original_hashes) if config.get('reference_image_dir') else []
    config.update(reference_examples=[], reference_originals=[], reference_crops=[], reference_negatives=[],
                  reference_boxed_images=[], reference_pool_image_hashes=hashes)
    config['reference_manifest_sha256'] = digest({'schema': 'yolo_onnx.supervised.v1', 'references': []})
    versions = {}
    for package in ('onnxruntime', 'opencv-python', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    weights = Path(settings['weights'])
    config['local_model'] = dict(settings=settings, library_versions=versions,
        weights_sha256=image_sha256(weights) if weights.is_file() else None,
        adapter_sha256=image_sha256(Path(__file__)), runtime_ready=weights.is_file(),
        training_regime='supervised', output_format='raw_dfl_nchw',
        preprocessing='RGB float32 /255; bilinear resize; centered letterbox114; NCHW')


def letterbox(image, input_size):
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    target_h, target_w = input_size
    gain = min(target_h / height, target_w / width)
    resized_w, resized_h = round(width * gain), round(height * gain)
    left = round((target_w - resized_w) / 2 - 0.1)
    top = round((target_h - resized_h) / 2 - 0.1)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[top:top + resized_h, left:left + resized_w] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, (gain, left, top, width, height)


def decode_heads(outputs, settings: dict, transform, label: str) -> list[dict]:
    """Decode 4*reg_max DFL logits plus class logits, then NMS in input pixels."""
    import numpy as np
    channels = 4 * settings['reg_max'] + len(settings['class_names'])
    input_h, input_w = settings['input_size']
    shapes = {(input_h // stride, input_w // stride): stride for stride in settings['strides']}
    if len(outputs) != len(shapes):
        raise ValueError('ONNX 检测头数量与 strides 不一致')
    candidates, seen = [], set()
    for output in outputs:
        if (output.ndim != 4 or output.shape[:2] != (1, channels)
                or tuple(output.shape[2:]) not in shapes or tuple(output.shape[2:]) in seen
                or not np.isfinite(output).all()):
            raise ValueError(f'ONNX 原始 DFL 检测头无效: {output.shape}；预期通道={channels}')
        shape = tuple(output.shape[2:])
        seen.add(shape)
        stride = shapes[shape]
        values = output[0].reshape(channels, -1)
        logits = values[4 * settings['reg_max']:]
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        classes = probabilities.argmax(axis=0)
        scores = probabilities.max(axis=0)
        selected = np.flatnonzero((classes == settings['target_class_id']) & (scores >= settings['confidence']))
        if not len(selected):
            continue
        distribution = values[:4 * settings['reg_max'], selected].reshape(4, settings['reg_max'], -1)
        distribution = np.exp(distribution - distribution.max(axis=1, keepdims=True))
        distribution /= distribution.sum(axis=1, keepdims=True)
        distances = (distribution * np.arange(settings['reg_max'])[None, :, None]).sum(axis=1) * stride
        anchors_x = (selected % shape[1] + 0.5) * stride
        anchors_y = (selected // shape[1] + 0.5) * stride
        boxes = np.stack((anchors_x - distances[0], anchors_y - distances[1],
                          anchors_x + distances[2], anchors_y + distances[3]), axis=1)
        for box, score in zip(boxes.tolist(), scores[selected].tolist()):
            if box[0] < box[2] and box[1] < box[3]:
                candidates.append(dict(label=label, confidence=score, bbox_pixels_xyxy=box))
    gain, left, top, width, height = transform
    objects = []
    for candidate in nms(candidates, settings['nms_iou'], settings['max_detections']):
        x1, y1, x2, y2 = candidate['bbox_pixels_xyxy']
        box = [max(0.0, min((value - offset) / gain, bound))
               for value, offset, bound in zip((x1, y1, x2, y2), (left, top, left, top),
                                               (width, height, width, height))]
        if box[0] < box[2] and box[1] < box[3]:
            objects.append({**candidate, 'bbox_pixels_xyxy': box})
    return objects


class YOLOONNXRunner:
    def __init__(self, config: dict):
        self.config, self.settings = config, config['yolo_onnx']
        self.reference_encoding_seconds = 0.0
        started = time.perf_counter()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('此模型需要 ONNX Runtime：python -m pip install -e ".[yolo-onnx]"') from exc
        unavailable = set(self.settings['providers']) - set(ort.get_available_providers())
        if unavailable:
            raise ValueError(f'ONNX 运行设备不可用: {sorted(unavailable)}')
        if not Path(self.settings['weights']).is_file():
            raise ValueError(f"ONNX 权重不存在: {self.settings['weights']}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.settings['intra_op_threads']
        self.session = ort.InferenceSession(self.settings['weights'], sess_options=options,
                                           providers=self.settings['providers'])
        inputs = self.session.get_inputs()
        expected = [1, 3, *self.settings['input_size']]
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or inputs[0].shape != expected:
            raise ValueError(f'ONNX 输入必须是 float32 {expected}，请核对 input_size')
        self.input_name = inputs[0].name
        names = self.session.get_modelmeta().custom_metadata_map.get('names')
        if names:
            parsed = ast.literal_eval(names)
            model_names = [parsed[index] for index in range(len(parsed))] if isinstance(parsed, dict) else parsed
            if model_names != self.settings['class_names']:
                raise ValueError(f"ONNX 类别 {model_names} 与 class_names 配置不一致")
        if self.settings['class_names'][self.settings['target_class_id']] != config['labels'][0]['name']:
            raise ValueError('ONNX 目标类别名称与 labels.name 不一致')
        expected_outputs = {(1, 4 * self.settings['reg_max'] + len(self.settings['class_names']),
                             expected[2] // stride, expected[3] // stride) for stride in self.settings['strides']}
        actual_outputs = [tuple(output.shape) for output in self.session.get_outputs()]
        if len(actual_outputs) != len(expected_outputs) or set(actual_outputs) != expected_outputs:
            raise ValueError(f'ONNX 输出结构与 reg_max/class_names/strides 不一致: {actual_outputs}')
        self.setup_seconds = time.perf_counter() - started

    def predict(self, data: bytes) -> dict:
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('无法解码待测图片')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor, transform = letterbox(image, self.settings['input_size'])
        outputs = self.session.run(None, {self.input_name: tensor})
        objects = decode_heads(outputs, self.settings, transform, self.config['labels'][0]['name'])
        status = 'annotated' if objects else 'no_target'
        return dict(annotation_status=status, reason_code='target_detected' if objects else 'no_target',
                    reason='JSTY 监督训练 YOLO ONNX 检测', target_count=len(objects), objects=objects)

    async def close(self):
        self.session = None

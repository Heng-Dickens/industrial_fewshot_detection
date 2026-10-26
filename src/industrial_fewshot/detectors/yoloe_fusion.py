"""Single-class fusion of YOLOE crop prototypes and aligned text embeddings."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

from PIL import Image

from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.supports import ReferencePool
from .yoloe_text import YOLOETextRunner, prepare_references as prepare_text, yoloe_text_settings
from .yoloe_visual import prepare_references as prepare_visual, yoloe_settings

FUSION_KEYS = {'positive_count', 'selection_seed', 'aggregation', 'visual_weight', 'text_weight'}


def text_settings(settings):
    return {key: value for key, value in settings.items() if key not in FUSION_KEYS}


def yoloe_fusion_settings(raw: dict, base: Path) -> dict:
    if not isinstance(raw, dict):
        raise ValueError('yoloe_fusion 必须是配置对象')
    settings = yoloe_text_settings(text_settings(raw), base)
    visual = yoloe_settings({key: raw[key] for key in ('positive_count', 'selection_seed', 'aggregation') if key in raw}, base)
    settings.update({key: visual[key] for key in ('positive_count', 'selection_seed', 'aggregation')})
    for key, default in (('visual_weight', 0.75), ('text_weight', 0.25)):
        value = raw.get(key, default)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'yoloe_fusion.{key} 必须是 0~1 的有限数字')
        settings[key] = value
    if not math.isclose(settings['visual_weight'] + settings['text_weight'], 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError('visual_weight + text_weight 必须等于 1')
    return settings


def prepare_references(config: dict) -> None:
    settings = config['yoloe_fusion']
    # Reuse the frozen source-image selection and existing annotation checks.
    visual_keys = {'text_prompt', 'text_encoder_weights', 'visual_weight', 'text_weight'}
    visual = {**config, 'yoloe': {key: value for key, value in settings.items() if key not in visual_keys}}
    prepare_visual(visual)
    pool = ReferencePool(config, settings['selection_seed'])
    selected = {row['sample_id']: row for row in pool.select(settings['positive_count'])[0]}
    examples, crops = [], []
    label = config['labels'][0]['name']
    for original in visual['reference_examples']:
        entries = [row for row in selected[original['sample_id']].get('crops', []) if row.get('label') == label]
        if not entries or sorted(row['bbox_xyxy'] for row in entries) != sorted(original['bboxes']):
            raise ValueError(f"参考裁剪清单与当前标注框不一致，请重新准备参考数据: {original['sample_id']}")
        prepared = []
        for entry in entries:
            path = pool.path(entry['image'])
            with Image.open(path) as image:
                width, height = image.size
            bounds = entry['crop_xyxy']
            if width != bounds[2] - bounds[0] or height != bounds[3] - bounds[1] or min(width, height) <= 0:
                raise ValueError(f'参考裁剪尺寸与清单不一致: {path}')
            crop = dict(sample_id=original['sample_id'], image_path=str(path), image_sha256=image_sha256(path),
                        width=width, height=height, bboxes=[[0, 0, width, height]],
                        source_bbox_xyxy=entry['bbox_xyxy'], source_crop_xyxy=bounds)
            prepared.append(crop)
            crops.append(crop)
        examples.append({**original, 'crops': prepared})
    textual = {**config, 'yoloe_text': text_settings(settings)}
    prepare_text(textual)
    config.update(reference_examples=examples, reference_originals=[], reference_crops=crops,
                  reference_negatives=[], reference_boxed_images=[],
                  reference_pool_image_hashes=visual['reference_pool_image_hashes'])
    config['reference_manifest_sha256'] = digest(dict(schema='yoloe_crop_text.v1', examples=examples,
        text_prompt=settings['text_prompt'], visual_weight=settings['visual_weight'], text_weight=settings['text_weight']))
    config['local_model'] = {**textual['local_model'], 'settings': settings,
        'text_adapter_sha256': textual['local_model']['adapter_sha256'],
        'adapter_sha256': image_sha256(Path(__file__)), 'visual_input': 'prepared_target_crops',
        'fusion': 'normalize(visual_weight * normalize(V) + text_weight * normalize(T))'}


def normalized(vector):
    import torch
    if vector.ndim != 3 or tuple(vector.shape[:2]) != (1, 1):
        raise ValueError(f'YOLOE 单类提示特征形状异常: {tuple(vector.shape)}')
    norm = vector.norm(p=2, dim=-1, keepdim=True)
    if not torch.isfinite(vector).all() or (norm < 1e-8).any():
        raise ValueError('YOLOE 提示特征为空或非有限值')
    return vector / norm


def fuse_prototypes(visual, textual, visual_weight, text_weight):
    if visual.shape != textual.shape:
        raise ValueError('YOLOE 视觉与文字提示特征维度不一致')
    return normalized(visual_weight * normalized(visual) + text_weight * normalized(textual))


class YOLOEFusionRunner(YOLOETextRunner):
    def __init__(self, config: dict):
        settings = config['yoloe_fusion']
        # Text setup does not run ordinary model.predict, so its class head has
        # not yet been fused. Keep that head unfused until both prompts exist.
        super().__init__({**config, 'yoloe_text': text_settings(settings)})
        self.config, self.settings = config, settings
        textual = self.model.model.pe.detach().clone()
        from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
        started = time.perf_counter()
        prototypes = []
        with self.torch.inference_mode():
            for example in config['reference_examples']:
                per_image = []
                for crop in example['crops']:
                    data = Path(crop['image_path']).read_bytes()
                    if hashlib.sha256(data).hexdigest() != crop['image_sha256']:
                        raise ValueError(f"参考裁剪图在准备后发生变化: {crop['image_path']}")
                    self.model.predictor = None
                    self.model.predict(self._image(data), prompts={
                        'bboxes': self.np.asarray(crop['bboxes'], dtype=self.np.float32),
                        'cls': self.np.zeros(1, dtype=self.np.int64)}, predictor=YOLOEVPSegPredictor,
                        return_vpe=True, **self._predict_args())
                    per_image.append(normalized(self.model.predictor.vpe.detach()))
                prototypes.append(normalized(self.torch.stack(per_image).mean(dim=0)))
            visual = normalized(self.torch.stack(prototypes).mean(dim=0))
            if getattr(self.model.model.model[-1], 'is_fused', False):
                raise ValueError('YOLOE 检测头过早固化，无法设置联合提示')
            prototype = fuse_prototypes(visual, textual, settings['visual_weight'], settings['text_weight'])
            self.model.set_classes([config['labels'][0]['name']], prototype)
        self.model.predictor = None
        self._sync()
        self.reference_encoding_seconds = time.perf_counter() - started

    def predict(self, data: bytes) -> dict:
        result = super().predict(data)
        result['reason'] = 'YOLOE 裁剪视觉特征与文字特征联合提示检测'
        return result

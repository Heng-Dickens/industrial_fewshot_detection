"""Text-only YOLOE using the pinned upstream MobileCLIP-B(LT) alignment path."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import sys
import time

from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.data.supports import ReferencePool
from industrial_fewshot.paths import PROJECT_ROOT
from .yoloe_visual import YOLOERunner, source_fingerprint, yoloe_settings


def yoloe_text_settings(raw: dict, base: Path) -> dict:
    if not isinstance(raw, dict):
        raise ValueError('yoloe_text 必须是配置对象')
    text_keys = {'text_prompt', 'text_encoder_weights'}
    visual_keys = {'positive_count', 'selection_seed', 'aggregation'}
    if visual_keys & raw.keys():
        raise ValueError('yoloe_text 仅使用文字提示，不接受 positive_count/selection_seed/aggregation')
    text = raw.get('text_prompt')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('yoloe_text.text_prompt 必须是非空字符串')
    encoder = raw.get('text_encoder_weights', '../weights/yoloe/mobileclip_blt.pt')
    if not isinstance(encoder, str) or not encoder.strip():
        raise ValueError('yoloe_text.text_encoder_weights 必须是本地 MobileCLIP-B(LT) 权重路径')
    shared = yoloe_settings({key: value for key, value in raw.items() if key not in text_keys}, base)
    settings = {key: value for key, value in shared.items() if key not in visual_keys}
    settings.update(text_prompt=text, text_encoder_weights=str((base / encoder).expanduser().resolve()))
    return settings


def prepare_references(config: dict) -> None:
    settings = config['yoloe_text']
    # Read image hashes only to retain the existing disjoint evaluation cohort.
    # No support images or their boxes are passed to the encoder or detector.
    hashes = sorted(ReferencePool(config, 0).original_hashes) if config.get('reference_image_dir') else []
    config.update(reference_examples=[], reference_originals=[], reference_crops=[],
                  reference_negatives=[], reference_boxed_images=[], reference_pool_image_hashes=hashes)
    config['reference_manifest_sha256'] = digest({'schema': 'yoloe_text.v1', 'text_prompt': settings['text_prompt']})
    weights, encoder, source = (Path(settings[key]) for key in ('weights', 'text_encoder_weights', 'source_dir'))
    versions = {}
    for package in ('torch', 'mobileclip', 'open-clip-torch', 'timm'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_hash = source_fingerprint(source)
    config['local_model'] = dict(settings=settings,
        weights_sha256=image_sha256(weights) if weights.is_file() else None,
        text_encoder_sha256=image_sha256(encoder) if encoder.is_file() else None,
        source_sha256=source_hash, library_versions=versions,
        adapter_sha256=image_sha256(Path(__file__)),
        shared_adapter_sha256=image_sha256(Path(__file__).with_name('yoloe_visual.py')),
        runtime_ready=weights.is_file() and encoder.is_file() and source_hash is not None
                      and all(versions.values()))


class YOLOETextRunner(YOLOERunner):
    """Reuse pixel output conversion, replacing visual prototype creation entirely."""

    def __init__(self, config: dict):
        self.config, self.settings = config, config['yoloe_text']
        self.reference_encoding_seconds = 0.0
        started = time.perf_counter()
        source = Path(self.settings['source_dir'])
        for key in ('weights', 'text_encoder_weights'):
            if not Path(self.settings[key]).is_file():
                raise ValueError(f'YOLOE 文字模式缺少 {key}: {self.settings[key]}；请参阅 docs/YOLOE_TEXT_GUIDE.md')
        if not (source / 'ultralytics/nn/text_model.py').is_file():
            raise ValueError(f'不是项目使用的 THU-MIG YOLOE 源码: {source}')
        loaded = sys.modules.get('ultralytics')
        if loaded and not Path(loaded.__file__).resolve().is_relative_to(source):
            raise RuntimeError('当前进程已导入其他 ultralytics，请在新进程运行 main.py')
        sys.path.insert(0, str(source))
        os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
        os.environ.setdefault('YOLO_CONFIG_DIR', str(PROJECT_ROOT / 'outputs/runs/runtime/yoloe_settings'))
        try:
            import numpy as np
            import torch
            import mobileclip
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError('YOLOE 文字依赖未就绪，请用当前解释器安装 requirements-yoloe-text.txt。'
                               f' 原因: {exc}') from exc
        self.np, self.torch = np, torch
        torch.set_num_threads(self.settings['torch_threads'])
        self.model = YOLOE(self.settings['weights'])
        self.model.to(self.settings['device'])
        self.model.model.eval()
        if self.model.model.args.get('text_model') != 'mobileclip:blt':
            raise ValueError('当前适配器仅支持使用 mobileclip:blt 训练的 YOLOE 权重')
        self.setup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        device = next(self.model.model.parameters()).device
        encoder = mobileclip.create_model_and_transforms('mobileclip_b',
            pretrained=self.settings['text_encoder_weights'], device=device)[0]
        encoder.eval()
        tokenizer = mobileclip.get_tokenizer('mobileclip_b')
        text = self.settings['text_prompt']
        # OpenCLIP normally truncates silently; never report a full caption as used
        # when its final words did not actually enter the encoder.
        token_count = len(tokenizer.tokenizer.encode(text)) + 2
        if token_count > tokenizer.context_length:
            raise ValueError(f'YOLOE 文字提示过长: {token_count} tokens，最多 {tokenizer.context_length}；请精简描述')
        with torch.inference_mode():
            features = encoder.encode_text(tokenizer([text]).to(device)).float()
            norms = features.norm(p=2, dim=-1, keepdim=True)
            if not torch.isfinite(features).all() or (norms < 1e-8).any():
                raise ValueError('YOLOE 文字编码器输出为空或非有限值')
            # Same normalization and RepRTA as upstream YOLOEModel.get_text_pe;
            # explicit checkpoint path avoids dependence on the process cwd.
            normalized = (features / norms).reshape(1, 1, -1)
            embeddings = self.model.model.model[-1].get_tpe(normalized)
            if embeddings.ndim != 3 or tuple(embeddings.shape[:2]) != (1, 1) or not torch.isfinite(embeddings).all():
                raise ValueError('YOLOE 文字提示特征形状或数值异常')
            self.model.set_classes([config['labels'][0]['name']], embeddings)
        self.model.predictor = None
        self._sync()
        self.text_encoding_seconds = time.perf_counter() - started
        del encoder

    def predict(self, data: bytes) -> dict:
        result = super().predict(data)
        result['reason'] = 'YOLOE 纯文字提示检测'
        return result

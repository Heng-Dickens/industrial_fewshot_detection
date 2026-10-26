"""Private JSON-line worker. Run with the backend's separate Python environment."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


class PETAdapter:
    def __init__(self, settings, references, runtime):
        import numpy as np
        import torch
        from mmengine.config import Config
        from mmdet.apis import DetInferencer
        self.np, self.torch, self.settings = np, torch, settings
        cfg = Config.fromfile(settings['model_config'])
        # Training-only memory bank reads Object365 annotations; inference must
        # never require that dataset or derive prompts from the query's GT.
        cfg.model.memory_bank = {}
        cfg.model.backbone.init_cfg = None
        cfg.model.language_model.name = settings['language_model_dir']
        cfg.model.test_cfg.max_per_img = settings['max_detections']
        self.inferencer = DetInferencer(model=cfg, weights=settings['weights'],
                                       device=settings['device'], palette='random', show_progress=False)
        self.embedding = None
        self.reference_encoding_seconds = 0.0
        if settings['prompt_mode'] == 'visual':
            started = time.perf_counter()
            # Extract each support image through upstream's official Visual-G
            # path; then pool its class embedding across the K support images.
            vectors = []
            for index, row in enumerate(references):
                image_path = Path(row['image_path'])
                if hashlib.sha256(image_path.read_bytes()).hexdigest() != row['image_sha256']:
                    raise ValueError(f'Reference image changed: {image_path}')
                out = Path(runtime) / f'support_{index}'
                self.inferencer(str(image_path), prompt_type='Visual', texts='object.',
                    prompt_bboxes=row['bboxes'], prompt_bboxes_labels=[0] * len(row['bboxes']),
                    extract_visual_embedding=True, out_dir=str(out), no_save_vis=True, no_save_pred=True)
                vectors.append(torch.load(out / '0.pt', map_location='cpu')['visual_embedding'])
            self.embedding = str(Path(runtime) / 'prototype.pt')
            torch.save(dict(visual_embedding=torch.stack(vectors).mean(dim=0), label=0,
                            class_name='hanging object'), self.embedding)
            self.reference_encoding_seconds = time.perf_counter() - started

    def predict(self, image):
        # MMDetection ndarray convention is BGR; PIL is RGB.
        array = self.np.asarray(image)[:, :, ::-1].copy()
        kwargs = dict(texts=self.settings['text_prompt'], custom_entities=True,
                      no_save_vis=True, no_save_pred=True, out_dir='', align_trex_format=True)
        if self.embedding:
            kwargs.update(prompt_type='Visual', texts='object.', prompt_visual_embedding_path=self.embedding)
        else:
            kwargs['prompt_type'] = 'Text'
        with self.torch.inference_mode():
            rows = self.inferencer(array, **kwargs)[0]
        return dict(boxes=rows['boxes'], scores=rows['scores'])


class WeDetectAdapter:
    def __init__(self, settings, references, runtime):
        import torch
        self.torch, self.settings, self.runtime = torch, settings, Path(runtime)
        path = Path(settings['source_dir']) / 'deploy/test_coco_pytorch.py'
        spec = importlib.util.spec_from_file_location('official_wedetect_pytorch', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        encoder = module.XLMRobertaLanguageBackbone(settings['language_model_dir'], settings['weights'])
        encoder.to(settings['device']).eval()
        with torch.inference_mode():
            self.embeddings = torch.nn.functional.normalize(encoder([settings['text_prompt']]), dim=-1)
        del encoder
        self.model = module.SimpleYOLOWorldDetector(settings['variant'], score_thr=settings['confidence'],
            nms_iou=settings['nms_iou'], post_nms_topk=settings['max_detections'])
        # Refuse partially loaded checkpoints: wrong Uni/Ref/variant weights
        # must not silently produce results from random parameters.
        original_load = self.model.load_state_dict
        def checked_load(state, strict=False):
            result = original_load(state, strict=False)
            if result.missing_keys or result.unexpected_keys:
                raise ValueError(f'WeDetect checkpoint incompatible: {result}')
            return result
        self.model.load_state_dict = checked_load
        module.load_vision_checkpoint(self.model, settings['weights'])
        self.model.to(settings['device']).eval()
        self.reference_encoding_seconds = 0.0

    def predict(self, image):
        # The unmodified official forward accepts image paths; lossless PNG
        # preserves the input pixels without passing any query annotation.
        path = self.runtime / 'query.png'
        image.save(path)
        with self.torch.inference_mode():
            rows = self.model([str(path)], self.embeddings, rescale=True)[0]
        return dict(boxes=rows['bboxes'].cpu().tolist(), scores=rows['scores'].cpu().tolist())


def main():
    # Reserve a separate OS descriptor for protocol output. C/native upstream
    # prints as well as Python prints go to stderr, never into the JSON stream.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with contextlib.redirect_stdout(sys.stderr), tempfile.TemporaryDirectory(prefix='prompt_detector_') as runtime:
        adapter = None
        for line in sys.stdin:
            try:
                message = json.loads(line)
                if message['op'] == 'init':
                    settings = message['settings']
                    sys.path.insert(0, settings['source_dir'])
                    import torch
                    torch.set_num_threads(settings['torch_threads'])
                    started = time.perf_counter()
                    cls = PETAdapter if message['backend'] == 'pet_dino' else WeDetectAdapter
                    adapter = cls(settings, message['references'], runtime)
                    response = dict(setup_seconds=time.perf_counter() - started,
                                    reference_encoding_seconds=adapter.reference_encoding_seconds)
                elif message['op'] == 'predict' and adapter is not None:
                    from PIL import Image
                    with Image.open(io.BytesIO(base64.b64decode(message['image_base64'], validate=True))) as source:
                        response = adapter.predict(source.convert('RGB'))
                else:
                    raise ValueError('Invalid worker operation')
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = dict(error=f'{type(exc).__name__}: {exc}')
            protocol.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + '\n')
            protocol.flush()


if __name__ == '__main__':
    main()

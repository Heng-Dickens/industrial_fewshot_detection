"""Offline source/import/operator/asset checks. --load-model adds actual loading, no image inference."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def asset_paths(backend, mode='visual', variant='base', model='yoloe-v8s-seg.pt', root=ROOT):
    root = Path(root)
    if backend == 'yoloe':
        paths = [root/'third_party/yoloe/ultralytics/models/yolo/yoloe/predict_vp.py', root/'weights/yoloe'/model]
        if mode != 'visual':
            paths.append(root/'weights/yoloe/mobileclip_blt.pt')
        return paths
    if backend == 'groundingdino':
        checkpoint = root/'weights'/f'grounding-dino-{variant}'
        return [checkpoint/'config.json', checkpoint/'preprocessor_config.json', checkpoint/'tokenizer.json', checkpoint/'model.safetensors']
    source = root/'third_party'/backend
    language = root/'weights'/backend/('bert-base-uncased' if backend == 'pet_dino' else ('xlm-roberta-large' if variant == 'large' else 'xlm-roberta-base'))
    weights = root/'weights'/backend/('pet_dino_swin_t.pth' if backend == 'pet_dino' else f'wedetect_{variant}.pth')
    paths = [source/('mmdet/apis/det_inferencer.py' if backend == 'pet_dino' else 'deploy/test_coco_pytorch.py'),
             weights, language/'config.json', language/'tokenizer.json']
    if backend == 'pet_dino':
        paths += [source/'configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py', language/'model.safetensors']
    return paths


def import_checks(backend, mode, root=ROOT):
    modules = ['PIL', 'numpy', 'torch', 'torchvision']
    if backend == 'yoloe':
        sys.path.insert(0,str(root/'third_party/yoloe'))
        modules += ['ultralytics']
        if mode != 'visual':
            modules += ['mobileclip', 'open_clip', 'timm']
    elif backend == 'groundingdino':
        modules += ['transformers']
    elif backend == 'pet_dino':
        sys.path.insert(0,str(root/'third_party/pet_dino'))
        modules += ['mmcv', 'mmengine', 'mmdet.apis', 'transformers']
    else:
        modules += ['transformers', 'sentencepiece', 'pycocotools.cocoeval', 'mmengine']
    errors = []
    for name in modules:
        try:
            module = importlib.import_module(name)
            if name == 'ultralytics' and not Path(module.__file__).resolve().is_relative_to(root/'third_party/yoloe'):
                raise RuntimeError('Imported ultralytics is not the pinned source')
            print(f'OK import {name}')
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    try:
        import torch
        from torchvision.ops import nms
        selected = nms(torch.tensor([[0.,0.,10.,10.]]), torch.tensor([.8]), .5)
        if selected.tolist() != [0]:
            raise RuntimeError('Unexpected torchvision NMS result')
        print(f'OK torchvision native NMS, torch={torch.__version__}')
        if backend == 'pet_dino':
            from mmcv.ops import MultiScaleDeformableAttention
            import mmcv._ext
            print('OK compiled MMCV extension and deformable attention import')
        if backend == 'groundingdino':
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        if backend == 'wedetect':
            from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel
    except Exception as exc:
        errors.append(f'native/model API: {type(exc).__name__}: {exc}')
    return errors


def load_model(args):
    if args.backend == 'yoloe':
        from ultralytics import YOLOE
        model = YOLOE(str(ROOT/'weights/yoloe'/args.model))
        model.model.eval()
        if args.mode != 'visual':
            import mobileclip
            encoder = mobileclip.create_model_and_transforms('mobileclip_b',
                pretrained=str(ROOT/'weights/yoloe/mobileclip_blt.pt'),device='cpu')[0]
            encoder.eval()
    elif args.backend == 'groundingdino':
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        directory = ROOT/'weights'/f'grounding-dino-{args.variant}'
        processor = AutoProcessor.from_pretrained(directory,local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(directory,local_files_only=True).eval()
    else:
        path = ROOT/'src/industrial_fewshot/detectors/prompt_native_worker.py'
        spec = importlib.util.spec_from_file_location('_setup_native_worker',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        language = 'bert-base-uncased' if args.backend == 'pet_dino' else ('xlm-roberta-large' if args.variant == 'large' else 'xlm-roberta-base')
        settings = dict(source_dir=str(ROOT/'third_party'/args.backend), variant=args.variant,
            weights=str(ROOT/'weights'/args.backend/('pet_dino_swin_t.pth' if args.backend=='pet_dino' else f'wedetect_{args.variant}.pth')),
            language_model_dir=str(ROOT/'weights'/args.backend/language), device='cpu',
            confidence=.001,nms_iou=.7,max_detections=100, prompt_mode='text',text_prompt='object.',
            model_config=str(ROOT/'third_party/pet_dino/configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py'))
        with tempfile.TemporaryDirectory(prefix='model_setup_check_') as runtime:
            cls = module.PETAdapter if args.backend == 'pet_dino' else module.WeDetectAdapter
            model = cls(settings,[],runtime)
    print('OK actual model load; no target images or annotations were used.')
    if args.backend == 'pet_dino' and args.mode == 'visual':
        print('Visual-G support encoding still requires an actual experiment; this check loads the shared model via Text mode.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['yoloe','groundingdino','pet_dino','wedetect'],required=True)
    parser.add_argument('--mode',choices=['visual','text','fusion'],default='visual')
    parser.add_argument('--variant',choices=['tiny','base','large'],default='base')
    parser.add_argument('--model',default='yoloe-v8s-seg.pt')
    parser.add_argument('--load-model',action='store_true')
    args = parser.parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false',
                      YOLO_CONFIG_DIR=str(ROOT/'outputs/runtime/yoloe_settings'))
    print(f'Checking {args.backend}, Python: {sys.executable}',flush=True)
    errors = []
    for path in asset_paths(args.backend,args.mode,args.variant,args.model):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f'Missing/empty asset: {path}')
        else:
            print(f'OK asset {path.relative_to(ROOT)}')
    errors += import_checks(args.backend,args.mode)
    if args.load_model and not errors:
        try:
            load_model(args)
        except Exception as exc:
            errors.append(f'model load: {type(exc).__name__}: {exc}')
    for message in errors:
        print('FAIL '+message,file=sys.stderr)
    if not errors:
        print('Installation check passed.' + ('' if args.load_model else ' Use -Check -LoadModel for actual model loading.'))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())

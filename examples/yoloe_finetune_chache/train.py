"""Three detection fine-tuning recipes; no training without --execute.

New warm starts: YOLOE fixed head, YOLOE SAVPE, or pretrained YOLOv8-L.
Only the explicit execution branch imports torch/Ultralytics or writes artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

try:
    from .finetune_config import DEFAULT_CONFIG, load_config
except ImportError:
    from finetune_config import DEFAULT_CONFIG, load_config

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--data', type=Path, help='Override prepared dataset YAML.')
    parser.add_argument('--device', help='One visible GPU index, chosen after checking occupancy.')
    parser.add_argument('--batch', type=int, help='Explicit batch after checking available GPU memory.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--method', choices=['yoloe-head', 'yoloe-vp', 'yolov8'], default='yoloe-head')
    parser.add_argument('--name', help='New experiment directory name; set explicitly for 1/10-shot runs.')
    parser.add_argument('--strategy', choices=['head', 'full'], default='head')
    parser.add_argument('--execute', action='store_true', help='Actually initialize models and train.')
    args = parser.parse_args()
    if args.method != 'yoloe-head' and args.strategy != 'head':
        raise ValueError('--strategy applies only to yoloe-head; yolov8 is full tuning and yoloe-vp is SAVPE-only.')
    config = load_config(args.config)
    classes = config['class_names']
    args.name = args.name or f'{args.method}_{config["shot"]}shot_seed{config["training"]["seed"]}_val100_v1'
    paths = {k: (ROOT / config[k]).resolve() for k in ('source_dir', 'weights', 'text_encoder', 'data', 'project')}
    if args.method == 'yolov8':
        paths['weights'] = (ROOT / config['yolov8_weights']).resolve()
    if args.data:
        paths['data'] = args.data.resolve()
    settings = dict(config['training'])
    if args.epochs is not None:
        settings['epochs'] = args.epochs
    if Path(args.name).name != args.name or args.name in {'.', '..', ''} or not re.fullmatch(r'[\w.-]+', args.name):
        raise ValueError('--name must be a simple new experiment name.')
    run_dir = paths['project'] / args.name
    plan = dict(paths={k: str(v) for k, v in paths.items()}, run_dir=str(run_dir),
                method=args.method, strategy=(args.strategy if args.method == 'yoloe-head' else
                                             'savpe_only' if args.method == 'yoloe-vp' else 'full'),
                scene=config['scene'], shot_per_class=config['shot'],
                dev_images=config['dev_images'], dev_manifest=config['dev_manifest'],
                test_manifest=config['test_manifest'],
                class_names=classes, text_prompts=config['text_prompts'],
                device=args.device, batch=args.batch, training=settings, execute=args.execute)
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    # Step 1: default invocation is a pure plan; it does not import model libraries.
    if not args.execute:
        print('Plan only. Add --execute --device GPU_INDEX --batch BATCH only when ready to train.')
        return
    if args.device is None or not re.fullmatch(r'\d+', args.device):
        raise ValueError('This example supports one GPU: explicitly pass --device INDEX. DDP is not implemented.')
    if args.batch is None or args.batch < 1 or settings['epochs'] < 1:
        raise ValueError('Pass a positive --batch and positive epochs.')
    if settings['imgsz'] < 32 or settings['imgsz'] % 32 or settings.get('amp') or settings.get('half'):
        raise ValueError('Use a positive multiple-of-32 input size and FP32 (amp=false, half=false).')
    required = ('weights', 'text_encoder', 'data') if args.method == 'yoloe-head' else ('weights', 'data')
    for name in required:
        if not paths[name].is_file():
            raise FileNotFoundError(f'Missing {name}: {paths[name]}')
    if args.method == 'yoloe-head' and paths['text_encoder'].name != 'mobileclip_blt.pt':
        raise ValueError('The pinned encoder loader expects the filename mobileclip_blt.pt.')
    if not (paths['source_dir'] / 'ultralytics/models/yolo/yoloe/train_pe.py').is_file():
        raise FileNotFoundError('Pinned THU-MIG YOLOE source is missing.')
    if run_dir.exists():
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {run_dir}')

    # Step 2: use this repository's YOLOE, including in any subprocess environment.
    sys.path.insert(0, str(paths['source_dir']))
    os.environ['PYTHONPATH'] = str(paths['source_dir']) + os.pathsep + os.environ.get('PYTHONPATH', '')
    import yaml
    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOE
    from ultralytics.models.yolo.yoloe.train_pe import YOLOEPETrainer

    if not Path(ultralytics.__file__).resolve().is_relative_to(paths['source_dir']):
        raise RuntimeError('Imported a different Ultralytics installation.')
    dataset = yaml.safe_load(paths['data'].read_text(encoding='utf-8-sig'))
    names = dataset['names']
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
    if names != classes:
        raise ValueError('Dataset class order differs from embedding/checkpoint class order.')
    preparation_path = paths['data'].parent / 'preparation.json'
    preparation = json.loads(preparation_path.read_text(encoding='utf-8-sig'))
    if preparation['classes'] != classes or preparation.get('scene', config['scene']) != config['scene']:
        raise ValueError('Preparation manifest classes differ.')
    if preparation['shot_per_class'] != config['shot']:
        raise ValueError('Prepared shot differs from config.shot; update both data and shot in a new configuration.')
    if preparation['splits']['val']['image_count'] != config['dev_expected_images']:
        raise ValueError('Prepared validation image count differs from the selected 100-image configuration.')
    expected_dev = (ROOT / config['dev_manifest']).resolve()
    if not expected_dev.is_file() or preparation.get('dev_manifest_sha256') != file_hash(expected_dev):
        raise ValueError('Prepared dev manifest differs from the configured frozen validation set; prepare new data.')
    if args.batch > preparation['splits']['train']['image_count']:
        raise ValueError('For this small-sample example, set batch <= training image count.')
    if not torch.cuda.is_available() or int(args.device) >= torch.cuda.device_count():
        raise RuntimeError('Selected GPU is unavailable in this Python environment.')
    torch.manual_seed(settings['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Step 3: encode semantic class descriptions once, using the original pretrained head.
    # Keep dataset IDs unchanged; "Chache1" itself is not a meaningful text description.
    # The pinned MobileCLIP loader locates its checkpoint by current working directory.
    embeddings = None
    if args.method == 'yoloe-head':
        teacher = YOLOE(str(paths['weights']))
        teacher.model.float().eval()
        if teacher.model.args.get('text_model') != 'mobileclip:blt':
            raise ValueError('This example expects YOLOE-v8l with MobileCLIP-B(LT).')
        previous_cwd = Path.cwd()
        try:
            os.chdir(paths['text_encoder'].parent)
            embeddings = teacher.get_text_pe(config['text_prompts']).detach().cpu()
        finally:
            os.chdir(previous_cwd)
        if embeddings.ndim != 3 or embeddings.shape[1] != len(classes):
            raise ValueError(f'Unexpected text embedding dimensions: {embeddings.shape}')
        del teacher

    # Step 4: build detection architecture (no masks), and transfer the pretrained weights.
    # yaml_model_load resolves the virtual scale-specific v8l name to bundled yoloe-v8.yaml.
    if args.method == 'yolov8':
        # A .pt checkpoint, never a fresh YAML-only model: this is pretrained fine-tuning.
        if paths['weights'].name != 'yolov8l.pt':
            raise ValueError('The YOLOv8 comparison expects the original detection checkpoint yolov8l.pt.')
        model = YOLO(str(paths['weights']), task='detect')
    else:
        architecture = paths['source_dir'] / 'ultralytics/cfg/models/v8/yoloe-v8l.yaml'
        model = YOLOE(str(architecture), task='detect').load(str(paths['weights']))
    if model.model.yaml.get('scale') != 'l':
        raise ValueError('Expected the same v8-L model scale as the original baseline.')
    head_index = len(model.model.model) - 1
    head = model.model.model[-1]
    if len(head.cv3) != 3 or any(len(branch) != 3 for branch in head.cv3):
        raise ValueError('Unexpected head structure; recheck the pinned train_pe.py recipe.')

    # Step 5: reproduce official linear-probing freezing, or explicitly choose full tuning.
    freeze = []
    if args.method == 'yoloe-head' and args.strategy == 'head':
        freeze = [str(i) for i in range(head_index)]
        freeze += [f'{head_index}.{name}' for name, _ in head.named_children() if name != 'cv3']
        freeze += [f'{head_index}.cv3.{i}.{j}' for i in range(3) for j in (0, 1)]
    elif args.method == 'yoloe-vp':
        if not hasattr(head, 'savpe'):
            raise ValueError('Visual prompting requires an unfused YOLOE head containing SAVPE.')
        freeze = [str(i) for i in range(head_index)]
        freeze += [f'{head_index}.{name}' for name, _ in head.named_children() if name != 'savpe']

    # Step 6: reserve a fresh output directory and save the complete initialization evidence.
    run_dir.mkdir(parents=True, exist_ok=False)
    pe_path = run_dir / 'class_embeddings.pt'
    if embeddings is not None:
        torch.save(dict(names=classes, pe=embeddings), pe_path)
    metadata = {**plan, 'freeze': freeze, 'source': json.loads(
        (paths['source_dir'] / 'UPSTREAM.json').read_text(encoding='utf-8-sig')),
        'input_sha256': {k: file_hash(paths[k]) for k in required},
        'preparation_sha256': file_hash(preparation_path),
        'dev_manifest_sha256': file_hash(expected_dev),
        'test_manifest_sha256': file_hash(ROOT / config['test_manifest']),
        'shot_per_class': preparation['shot_per_class'], 'nbs': args.batch,
        'status': 'initialized', 'note': 'New FP32 warm start; not resume. Independent dev for checkpoint selection.',
        'visual_training_protocol': ('Same-image bbox-conditioned SAVPE adaptation, fixed global class IDs; '
            'cross-image dev validation with training-only support prototypes. Not episodic cross-image training.'
            if args.method == 'yoloe-vp' else None)}
    metadata_path = run_dir / 'finetune_config.json'
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # The trainer rebuilds/fuses the head. Verify actual trainable parameters after that step.
    def check_trainable(trainer):
        trainable = [name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError('No trainable parameters.')
        if args.method == 'yoloe-head' and args.strategy == 'head' and any(not any(name.startswith(f'model.{head_index}.cv3.{i}.2.')
                                                  for i in range(3)) for name in trainable):
            raise RuntimeError(f'Unexpected trainable layers: {trainable}')
        if args.method == 'yoloe-vp' and any(not name.startswith(f'model.{head_index}.savpe.') for name in trainable):
            raise RuntimeError(f'Unexpected trainable layers outside SAVPE: {trainable}')
        (run_dir / 'trainable_parameters.json').write_text(json.dumps(trainable, indent=2) + '\n', encoding='utf-8')
        print('Actual trainable tensors:', len(trainable))

    model.add_callback('on_train_start', check_trainable)
    recipe = dict(load_vp=False)
    if args.method == 'yoloe-head':
        recipe.update(trainer=YOLOEPETrainer, train_pe_path=str(pe_path))
    elif args.method == 'yoloe-vp':
        # This adapter uses ordinary bbox data and a training-support-only validator,
        # avoiding upstream LVIS references and global text-embedding caches.
        from vp_trainer import SupportVPTrainer
        recipe.update(trainer=SupportVPTrainer, load_vp=True)
    # Step 7: this is the only optimization entry point. nbs=batch avoids a hidden 64-image
    # accumulation target; warmup_epochs=0 avoids the pinned trainer's minimum 100-step warmup.
    try:
        model.train(data=str(paths['data']), **recipe,
                    freeze=freeze, device=args.device, batch=args.batch, nbs=args.batch,
                    project=str(paths['project']), name=args.name, exist_ok=True, resume=False, **settings)
    except BaseException:
        metadata['status'] = 'interrupted_or_failed'
        raise
    else:
        metadata['status'] = 'completed'
        print(f'Best checkpoint: {run_dir / "weights/best.pt"}')
    finally:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

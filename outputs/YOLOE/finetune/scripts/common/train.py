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
import time

try:
    from .finetune_config import DEFAULT_CONFIG, load_config
except ImportError:
    from finetune_config import DEFAULT_CONFIG, load_config

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_model_scale(architecture, method):
    scale = architecture.get('scale')
    # Official YOLOv8-L checkpoints predate the consolidated scales YAML format.
    if (scale is None and method == 'yolov8'
            and architecture.get('depth_multiple') == 1.0
            and architecture.get('width_multiple') == 1.0):
        scale = 'l'
    if scale != 'l':
        raise ValueError('Expected the same v8-L model scale as the original baseline.')


def freeze_savpe_bn_statistics(model):
    """Keep pretrained SAVPE running statistics; affine parameters stay trainable."""
    import torch
    head = model.model[-1]
    if not hasattr(head, 'savpe'):
        raise ValueError('Freezing SAVPE BN statistics requires an unfused visual head.')
    count = 0
    for module in head.savpe.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def validate_adaptation(adaptation, method):
    allowed = {'freeze_savpe', 'freeze_savpe_bn_stats', 'scope', 'loss', 'protocol'}
    if not isinstance(adaptation, dict) or set(adaptation)-allowed:
        raise ValueError('Invalid adaptation configuration.')
    adaptation = dict(adaptation)
    for flag in ['freeze_savpe', 'freeze_savpe_bn_stats']:
        if type(adaptation.get(flag, False)) is not bool:
            raise ValueError(f'{flag} must be boolean.')
    if adaptation.get('scope', 'savpe') not in {'savpe','head','neck','full'}:
        raise ValueError('Invalid VP training scope.')
    if adaptation.get('loss','classification') not in {'classification','detection'}:
        raise ValueError('Invalid VP loss.')
    if adaptation.get('protocol','same_image') not in {'same_image','cross_image'}:
        raise ValueError('Invalid VP support/query protocol.')
    if adaptation.get('freeze_savpe'):
        if method != 'yoloe-vp':
            raise ValueError('freeze_savpe requires an unfused visual-prompt model.')
        if adaptation.get('scope','savpe') == 'savpe':
            raise ValueError('Freezing the only trainable SAVPE leaves no trainable parameters.')
        adaptation['freeze_savpe_bn_stats'] = True
    if adaptation.get('freeze_savpe_bn_stats') and method != 'yoloe-vp':
        raise ValueError('freeze_savpe_bn_stats applies only to yoloe-vp.')
    return adaptation


def module_fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def keep_frozen_savpe(trainer):
    """Freeze module state, while preserving autograd through its inputs."""
    trainer.model.model[-1].savpe.eval()
    if any(p.requires_grad for p in trainer.model.model[-1].savpe.parameters()):
        raise RuntimeError('Frozen SAVPE unexpectedly has trainable parameters.')


def sync_frozen_savpe_ema(trainer):
    # EMA arithmetic can round even constant tensors; keep the frozen module exact.
    if trainer.ema is not None:
        trainer.ema.ema.model[-1].savpe.load_state_dict(trainer.model.model[-1].savpe.state_dict())


def main(default_config=DEFAULT_CONFIG):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_config, required=default_config is None)
    parser.add_argument('--data', type=Path, help='Override prepared dataset YAML.')
    parser.add_argument('--device', help='One visible GPU index, chosen after checking occupancy.')
    parser.add_argument('--batch', type=int, help='Explicit batch after checking available GPU memory.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--method', choices=['yoloe-head', 'yoloe-vp', 'yolov8'], required=True,
                        help='Required: the run directory name does not select the training method.')
    parser.add_argument('--name', help='New experiment directory name; set explicitly for 1/10-shot runs.')
    parser.add_argument('--strategy', choices=['head', 'neck', 'full'], default='head')
    parser.add_argument('--execute', action='store_true', help='Actually initialize models and train.')
    parser.add_argument('--initialize-only', action='store_true',
                        help='YOLOv8 only: adapt the target head and save it without optimizer steps.')
    parser.add_argument('--resume', action='store_true', help='Resume this interrupted VP run from its own last.pt.')
    args = parser.parse_args()
    if args.initialize_only and (args.method != 'yolov8' or args.resume):
        raise ValueError('--initialize-only requires a new YOLOv8 run.')
    if args.method != 'yoloe-head' and args.strategy != 'head':
        raise ValueError('--strategy applies only to yoloe-head; VP scope is configured in adaptation.scope.')
    config = load_config(args.config)
    adaptation = validate_adaptation(config.get('adaptation', {}), args.method)
    freeze_vp_bn = adaptation.get('freeze_savpe_bn_stats', False)
    if freeze_vp_bn and args.method != 'yoloe-vp':
        raise ValueError('freeze_savpe_bn_stats applies only to yoloe-vp.')
    classes = config['class_names']
    args.name = args.name or f'{args.method}_{config["shot"]}shot_seed{config["training"]["seed"]}_val100_v1'
    paths = {k: (ROOT / config[k]).resolve() for k in ('source_dir', 'weights', 'text_encoder', 'data', 'project')}
    if args.method == 'yolov8':
        paths['weights'] = (ROOT / config['yolov8_weights']).resolve()
    if args.data:
        paths['data'] = args.data.resolve()
    settings = dict(config['training'])
    # Pinned upstream reuses fitness on skipped validation epochs and can overwrite
    # best.pt with an unvalidated EMA. Validate each epoch unless explicitly configured.
    settings.setdefault('val_interval', 1)
    if args.epochs is not None:
        settings['epochs'] = args.epochs
    if Path(args.name).name != args.name or args.name in {'.', '..', ''} or not re.fullmatch(r'[\w.-]+', args.name):
        raise ValueError('--name must be a simple new experiment name.')
    run_dir = paths['project'] / args.name
    plan = dict(paths={k: str(v) for k, v in paths.items()}, run_dir=str(run_dir),
                method=args.method, strategy=(args.strategy if args.method == 'yoloe-head' else
                                             adaptation.get('scope', 'savpe') if args.method == 'yoloe-vp' else 'full'),
                scene=config['scene'], shot_per_class=config['shot'],
                dev_images=config['dev_images'], dev_manifest=config['dev_manifest'],
                test_manifest=config['test_manifest'],
                class_names=classes, text_prompts=config['text_prompts'],
                class_protocol=config.get('class_protocol'),
                device=args.device, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                batch=args.batch, training=settings, adaptation=adaptation,
                execute=args.execute, initialize_only=args.initialize_only)
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
    previous_metadata = None
    if args.resume:
        if args.method != 'yoloe-vp':
            raise ValueError('Checkpoint resume is currently supported only for the VP trainer.')
        previous_metadata = json.loads((run_dir / 'finetune_config.json').read_text(encoding='utf-8'))
        if previous_metadata.get('status') == 'completed':
            raise ValueError('This run is already completed; nothing to resume.')
        for key in ('method', 'strategy', 'scene', 'shot_per_class', 'class_names', 'training', 'adaptation', 'batch'):
            if previous_metadata.get(key) != plan.get(key):
                raise ValueError(f'Resume configuration differs: {key}')
        for key, expected in previous_metadata['input_sha256'].items():
            if file_hash(paths[key]) != expected:
                raise ValueError(f'Resume input changed: {key}')
        if not (run_dir / 'weights/last.pt').is_file():
            raise FileNotFoundError(run_dir / 'weights/last.pt')
    elif run_dir.exists():
        raise FileExistsError(f'Refusing to overwrite an existing experiment: {run_dir}')

    # Step 2: use this repository's YOLOE, including in any subprocess environment.
    sys.path.insert(0, str(paths['source_dir']))
    os.environ['PYTHONPATH'] = str(paths['source_dir']) + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs/_shared/runtime/yoloe_settings' / args.name))
    # Independent workers need independent settings, but can reuse the local font.
    font_source = Path.home() / '.config/Ultralytics/Arial.ttf'
    font_destination = Path(os.environ['YOLO_CONFIG_DIR']) / 'Arial.ttf'
    if font_source.is_file() and not font_destination.exists():
        import shutil
        font_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(font_source, font_destination)
    import yaml
    import torch
    torch.set_num_threads(2)
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
    if args.resume:
        import vp_trainer  # Custom cross-image checkpoint class must be importable.
        model = YOLOE(str(run_dir / 'weights/last.pt'), task='detect')
        if model.ckpt.get('epoch', -1) < 0 or model.ckpt.get('optimizer') is None:
            raise ValueError('last.pt has no resumable optimizer/epoch state.')
    elif args.method == 'yolov8':
        # A .pt checkpoint, never a fresh YAML-only model: this is pretrained fine-tuning.
        if paths['weights'].name != 'yolov8l.pt':
            raise ValueError('The YOLOv8 comparison expects the original detection checkpoint yolov8l.pt.')
        model = YOLO(str(paths['weights']), task='detect')
    else:
        architecture = paths['source_dir'] / 'ultralytics/cfg/models/v8/yoloe-v8l.yaml'
        model = YOLOE(str(architecture), task='detect').load(str(paths['weights']))
    validate_model_scale(model.model.yaml, args.method)
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
    elif args.method == 'yoloe-head' and args.strategy == 'neck':
        freeze = [str(i) for i in range(10)]
    elif args.method == 'yoloe-vp':
        if not hasattr(head, 'savpe'):
            raise ValueError('Visual prompting requires an unfused YOLOE head containing SAVPE.')
        scope = adaptation.get('scope', 'savpe')
        freeze = [str(i) for i in range(head_index if scope in {'savpe', 'head'} else 10 if scope == 'neck' else 0)]
        if scope == 'savpe':
            freeze += [f'{head_index}.{name}' for name, _ in head.named_children() if name != 'savpe']
        if adaptation.get('freeze_savpe'):
            freeze.append(f'{head_index}.savpe')

    # Step 6: reserve a fresh output directory and save the complete initialization evidence.
    run_dir.mkdir(parents=True, exist_ok=args.resume)
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
        'visual_training_protocol': (f"{adaptation.get('protocol', 'same_image')} bbox-conditioned training; "
            'cross-image dev validation with training-only support prototypes; '
            f"scope={adaptation.get('scope', 'savpe')}, loss={adaptation.get('loss', 'classification')}"
            if args.method == 'yoloe-vp' else None)}
    metadata_path = run_dir / 'finetune_config.json'
    if args.resume:
        for key in ('preparation_sha256', 'dev_manifest_sha256', 'test_manifest_sha256', 'freeze'):
            if previous_metadata.get(key) != metadata.get(key):
                raise ValueError(f'Resume dataset or layer policy changed: {key}')
        metadata['resume_history'] = previous_metadata.get('resume_history', []) + [dict(
            time=time.time(), checkpoint=str(run_dir / 'weights/last.pt'),
            sha256=file_hash(run_dir / 'weights/last.pt'), completed_epochs=model.ckpt['epoch'] + 1,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))]
        metadata['note'] = ('Interrupted run resumed with checkpoint optimizer/EMA/epoch. '
                            'Upstream does not restore RNG or early-stopper history; not bitwise continuous.')
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    if args.initialize_only:
        # Exactly the DetectionTrainer warm-start recipe, with target nc and zero
        # optimization. Only incompatible classification output tensors stay new.
        from copy import deepcopy
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils.torch_utils import init_seeds
        init_seeds(settings['seed'], deterministic=settings.get('deterministic', True))
        original = model.model.float()
        initialized = DetectionModel(deepcopy(original.yaml), nc=len(classes), verbose=False)
        initialized.load(original)
        initial_state, source_state = initialized.state_dict(), original.state_dict()
        transferred = [key for key in initial_state if key in source_state
                       and initial_state[key].shape == source_state[key].shape]
        new_outputs = [key for key in initial_state if key not in transferred]
        expected_outputs = {f'model.{head_index}.cv3.{i}.2.{suffix}'
                            for i in range(3) for suffix in ['weight', 'bias']}
        if set(new_outputs) != expected_outputs:
            raise RuntimeError(f'Unexpected randomly initialized tensors: {new_outputs}')
        if not all(torch.equal(initial_state[key], source_state[key]) for key in transferred):
            raise RuntimeError('A transferred pretrained tensor changed during initialization.')
        initialized.names = dict(enumerate(classes))
        initialized.nc = len(classes)
        initialized.args = dict(model.overrides, task='detect')
        initialized.float().eval()
        checkpoint = run_dir / 'weights/initialized.pt'
        checkpoint.parent.mkdir()
        torch.save(dict(model=initialized, ema=None, optimizer=None, epoch=-1,
                        train_args=initialized.args), checkpoint)
        metadata.update(status='initialized_no_training', shot_per_class=0,
                        optimization_steps=0, checkpoint=str(checkpoint),
                        note='Target classification outputs initialized randomly; all compatible '
                             'pretrained tensors retained exactly. No training or COCO class remapping.',
                        initialization_verification=dict(transferred_tensor_count=len(transferred),
                            all_transferred_tensors_equal=True, new_classification_tensors=new_outputs,
                            seed=settings['seed'], checkpoint_sha256=file_hash(checkpoint)))
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Untrained target-head checkpoint (0 optimizer steps): {checkpoint}', flush=True)
        return

    # The trainer rebuilds/fuses the head. Verify actual trainable parameters after that step.
    pretrained_backbone = model.model.model[0].conv.weight.detach().float().cpu().clone()
    frozen_evidence = {}
    def check_trainable(trainer):
        actual_backbone = trainer.model.model[0].conv.weight.detach().float().cpu()
        if not torch.equal(actual_backbone, pretrained_backbone):
            raise RuntimeError('First backbone tensor does not match the warm-start pretrained weights.')
        (run_dir / 'initialization_verification.json').write_text(json.dumps(dict(
            source_weights_sha256=file_hash(paths['weights']),
            checked_tensor='model.0.conv.weight', pretrained_tensor_equal=True,
            tensor_sha256=hashlib.sha256(actual_backbone.contiguous().numpy().tobytes()).hexdigest(),
            device=str(trainer.device)), indent=2)+'\n')
        trainable = [name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError('No trainable parameters.')
        if args.method == 'yoloe-head' and args.strategy == 'head' and any(not any(name.startswith(f'model.{head_index}.cv3.{i}.2.')
                                                  for i in range(3)) for name in trainable):
            raise RuntimeError(f'Unexpected trainable layers: {trainable}')
        if args.method == 'yoloe-vp' and adaptation.get('scope', 'savpe') == 'savpe' and any(not name.startswith(f'model.{head_index}.savpe.') for name in trainable):
            raise RuntimeError(f'Unexpected trainable layers outside SAVPE: {trainable}')
        if adaptation.get('freeze_savpe'):
            keep_frozen_savpe(trainer)
            frozen_evidence['savpe_initial_sha256'] = module_fingerprint(trainer.model.model[-1].savpe)
            name, parameter = next((n,p) for n,p in trainer.model.named_parameters() if p.requires_grad)
            frozen_evidence.update(monitored_parameter=name,nonzero_gradient_batches=0,
                                   monitored_initial=parameter.detach().cpu().clone())
            def capture_gradient(gradient):
                if gradient.detach().abs().max().item() > 0:
                    frozen_evidence['nonzero_gradient_batches'] += 1
            parameter.register_hook(capture_gradient)
        (run_dir / 'trainable_parameters.json').write_text(json.dumps(trainable, indent=2) + '\n', encoding='utf-8')
        metadata['trainable_parameter_count'] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print('Actual trainable tensors:', len(trainable))

    model.add_callback('on_train_start', check_trainable)
    if adaptation.get('freeze_savpe'):
        model.add_callback('on_train_batch_start', keep_frozen_savpe)
        model.add_callback('on_train_batch_end', sync_frozen_savpe_ema)
        def verify_frozen_module(trainer):
            import torch
            initial = frozen_evidence['savpe_initial_sha256']
            final = module_fingerprint(trainer.model.model[-1].savpe)
            ema_final = module_fingerprint(trainer.ema.ema.model[-1].savpe)
            parameter = dict(trainer.model.named_parameters())[frozen_evidence['monitored_parameter']]
            changed = not torch.equal(parameter.detach().cpu(), frozen_evidence.pop('monitored_initial'))
            result = dict(**frozen_evidence,savpe_final_sha256=final,savpe_ema_sha256=ema_final,
                          savpe_unchanged=initial==final==ema_final,other_parameter_updated=changed)
            (run_dir/'frozen_savpe_verification.json').write_text(json.dumps(result,indent=2)+'\n')
            if not result['savpe_unchanged'] or not changed or not result['nonzero_gradient_batches']:
                raise RuntimeError(f'Frozen-SAVPE training verification failed: {result}')
            metadata['frozen_savpe_verification'] = result
        model.add_callback('on_train_end', verify_frozen_module)
    if freeze_vp_bn:
        # Upstream calls _model_train AFTER on_train_epoch_start. Apply immediately
        # before every batch so an epoch reset cannot re-enable BN running updates.
        def keep_savpe_statistics(trainer):
            count = freeze_savpe_bn_statistics(trainer.model)
            if not count:
                raise RuntimeError('No SAVPE BatchNorm layers found; recheck the checkpoint.')
        model.add_callback('on_train_batch_start', keep_savpe_statistics)
    recipe = dict(load_vp=False)
    if args.method == 'yoloe-head':
        recipe.update(trainer=YOLOEPETrainer, train_pe_path=str(pe_path))
    elif args.method == 'yoloe-vp':
        # This adapter uses ordinary bbox data and a training-support-only validator,
        # avoiding upstream LVIS references and global text-embedding caches.
        from vp_trainer import SupportVPTrainer
        from functools import partial
        recipe.update(trainer=partial(SupportVPTrainer, adaptation=adaptation), load_vp=True)
    # Step 7: this is the only optimization entry point. nbs=batch avoids a hidden 64-image
    # accumulation target; warmup_epochs=0 avoids the pinned trainer's minimum 100-step warmup.
    try:
        model.train(data=str(paths['data']), **recipe,
                    freeze=freeze, device=args.device, batch=args.batch, nbs=args.batch,
                    project=str(paths['project']), name=args.name, exist_ok=True, resume=args.resume, **settings)
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

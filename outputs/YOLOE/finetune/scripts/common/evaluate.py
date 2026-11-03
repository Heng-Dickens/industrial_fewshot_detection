"""Final 300-image evaluation: fixed heads once, visual checkpoints at 5/10 prompt shots.

The training support budget stays fixed. --prompt-shots changes only inference
references, never training data. Model loading/inference requires --execute.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

from finetune_config import DEFAULT_CONFIG, ROOT, load_config
from prepare_data import read, sha256

sys.path.insert(0, str(ROOT / 'src'))


def evaluation_modes(method, prompt_shots):
    if method not in {'yoloe-head', 'yoloe-vp', 'yolov8'}:
        raise ValueError(f'Unknown training method: {method}')
    if method != 'yoloe-vp':
        if prompt_shots:
            raise ValueError('固定检测头不支持 --prompt-shots；5shot/10shot 视觉提示需要用 '
                             '--method yoloe-vp 训练且保留 SAVPE 的权重。')
        return [None]
    return prompt_shots or [5, 10]


def safe_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes data directory: {relative}')
    return path


def load_test(config, metadata, preparation, split='test'):
    from PIL import Image
    from industrial_fewshot.data.labelme import load_evaluation_truth

    manifest_path = (ROOT / config['test_manifest' if split == 'test' else 'dev_manifest']).resolve()
    manifest_hash = sha256(manifest_path)
    if (manifest_hash != metadata['test_manifest_sha256' if split == 'test' else 'dev_manifest_sha256'] or
            manifest_hash != preparation['excluded_test_manifest_sha256' if split == 'test' else 'dev_manifest_sha256']):
        raise ValueError('Test manifest changed since training.')
    manifest = read(manifest_path)
    image_root = manifest_path.parent / 'images'
    items, seen = [], set()
    excluded = {s['image_sha256'] for s in preparation['samples'] if split == 'test' or s['split'] == 'train'}
    for row in sorted(manifest['samples'], key=lambda r: r['relative_path']):
        image = safe_child(image_root, row['relative_path'])
        annotation = image.with_suffix('.json')
        if (sha256(image) != row['image_sha256'] or
                sha256(annotation) != row['annotation_sha256']):
            raise ValueError(f'Test image/annotation bytes changed: {image}')
        if row['image_sha256'] in seen or row['image_sha256'] in excluded:
            raise ValueError(f'Duplicate/test overlaps training or dev: {image}')
        seen.add(row['image_sha256'])
        with Image.open(image) as opened:
            size = opened.size
        if size != (row['width'], row['height']):
            raise ValueError(f'Test dimensions changed: {image}')
        truth, origin = load_evaluation_truth(annotation, *size, set(config['class_names']), 'error')
        if truth != row['ground_truth']:
            raise ValueError(f'Test GT differs from frozen manifest: {annotation}')
        items.append(dict(path=str(image), relative_path=row['relative_path'], width=size[0],
                          height=size[1], image_sha256=row['image_sha256'], ground_truth=truth,
                          ground_truth_source=origin))
    expected = 300 if split == 'test' else config['dev_expected_images']
    if len(items) != expected:
        raise ValueError(f'Expected {expected} frozen images, got {len(items)}.')
    return manifest_path, manifest, items


def load_supports(config, shot, seed, items, preparation):
    from industrial_fewshot.data.labelme import load_ground_truth
    from PIL import Image

    test_hashes = {r['image_sha256'] for r in items}
    dev_hashes = {s['image_sha256'] for s in preparation['samples'] if s['split'] == 'val'}
    supports, sources, seen = {}, [], set()
    train_hashes = {s['image_sha256'] for s in preparation['samples'] if s['split'] == 'train'}
    for label in config['class_names']:
        reference = ROOT / 'data/references' / config['scene']
        if config['scene'] == 'chache':
            reference = reference / label
        frozen_path = reference / f'{shot}shot/seed_{seed}.json'
        frozen = read(frozen_path)
        if frozen['positive_count'] != shot or frozen['selection_seed'] != seed:
            raise ValueError(f'Support selection settings differ: {frozen_path}')
        rows = frozen['positive_samples']
        if len(rows) != shot:
            raise ValueError(f'Incorrect support count: {frozen_path}')
        index = {r['image']: r for r in read(reference / 'manifest.json')['images']}
        sources.append(dict(path=str(frozen_path), sha256=sha256(frozen_path)))
        supports[label] = []
        for chosen in rows:
            source = index[chosen['sample_id']]
            image = safe_child(reference, source['image'])
            annotation = safe_child(reference, source['annotation'])
            checksum = sha256(image)
            if checksum != chosen['image_sha256'] or sha256(annotation) != chosen['annotation_sha256']:
                raise ValueError(f'Frozen support bytes changed: {image}')
            if checksum in seen or checksum in test_hashes or checksum in dev_hashes:
                raise ValueError(f'Duplicate/support overlaps test or dev: {image}')
            # This protocol reuses the declared training support pool only.
            if checksum not in train_hashes:
                raise ValueError(f'Inference support was not in this run\'s training pool: {image}')
            seen.add(checksum)
            with Image.open(image) as opened:
                truth = load_ground_truth(annotation, *opened.size, {label})
            if not truth:
                raise ValueError(f'Support lacks target class: {annotation}')
            supports[label].append(dict(sample_id=chosen['sample_id'], image_path=str(image),
                json_path=str(annotation), image_sha256=checksum,
                annotation_sha256=chosen['annotation_sha256'],
                bboxes=[r['bbox_pixels_xyxy'] for r in truth]))
    return supports, sources


def check_capability(model, method, classes):
    head = model.model.model[-1]
    visual = hasattr(head, 'savpe') and not getattr(head, 'is_fused', False)
    if method == 'yoloe-vp' and not visual:
        raise ValueError('Checkpoint has no usable SAVPE. This is not a visual-prompt checkpoint.')
    if method == 'yoloe-head' and visual:
        raise ValueError('Checkpoint structure conflicts with fixed-head training metadata.')
    names = model.names
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if names != classes:
        raise ValueError(f'Checkpoint class order differs: {names} != {classes}')
    return dict(task=model.task, head=type(head).__name__, has_savpe=hasattr(head, 'savpe'),
                is_fused=getattr(head, 'is_fused', False), names=names)


def infer_fixed(model, items, classes, predict_args, stream):
    import numpy as np
    from PIL import Image
    from industrial_fewshot.data.labelme import validate_pixel_prediction

    records = []
    for number, item in enumerate(items, 1):
        with Image.open(item['path']) as opened:
            image = np.asarray(opened.convert('RGB'))[:, :, ::-1].copy()
        result = model.predict(image, **predict_args)[0]
        objects = []
        discarded = 0
        if result.boxes is not None:
            for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                if cls != int(cls) or not 0 <= int(cls) < len(classes):
                    raise ValueError(f'Invalid class ID: {cls}')
                if not all(math.isfinite(v) for v in [*box, score]):
                    raise ValueError('Nonfinite model prediction.')
                box = [max(0., min(limit, value)) for value, limit in
                       zip(box, [item['width'], item['height']] * 2)]
                if box[0] >= box[2] or box[1] >= box[3]:
                    discarded += 1
                    continue
                objects.append(dict(label=classes[int(cls)], bbox_pixels_xyxy=box, confidence=score))
        payload = dict(annotation_status='annotated' if objects else 'no_target',
                       reason_code='target_detected' if objects else 'no_target',
                       reason='Fine-tuned fixed detector', target_count=len(objects), objects=objects)
        payload = validate_pixel_prediction(payload, set(classes), item['width'], item['height'])
        record = dict(relative_path=item['relative_path'], image_sha256=item['image_sha256'],
                      target_id='all_classes', status='success', discarded_degenerate_count=discarded, **payload)
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        records.append(record)
        if number % 50 == 0:
            print(f'fixed: {number}/{len(items)}', flush=True)
    return records


def infer_visual(settings, supports, items, stream):
    import gc
    from industrial_fewshot.detectors.yoloe_visual import YOLOERunner

    records = []
    for label, examples in supports.items():
        runner = YOLOERunner(dict(yoloe=settings, labels=[dict(name=label)], reference_examples=examples))
        for number, item in enumerate(items, 1):
            payload = runner.predict(Path(item['path']).read_bytes())
            record = dict(relative_path=item['relative_path'], image_sha256=item['image_sha256'],
                          target_id=label, status='success', **payload)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            records.append(record)
            if number % 50 == 0:
                print(f'{label}: {number}/{len(items)}', flush=True)
        runner.model = None
        del runner
        gc.collect()
    return records


def write_summary(output, entries, method, training_shot, split='test', image_count=300):
    fields = ['result', 'training_shot_per_class', 'prompt_shot_per_class', 'mAP50', 'mAP50_95',
              'precision', 'recall', 'f2', 'tp', 'fp', 'fn']
    with (output / 'test_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)
    title = '未训练目标检测头基线评估' if method == 'yolov8' and training_shot == 0 else '微调后评估'
    lines = [f'# {title}：{split}', '', f'实际模型方法：`{method}`；训练预算：每类 {training_shot} 张。',
             f'固定{image_count}张；指标为百分数，P/R/F2 为 micro，IoU=0.5、score≥0.5。',
             'AP使用conf≥0.001候选，NMS IoU=0.7；未根据测试集调参或挑选checkpoint。', '']
    if method != 'yoloe-vp':
        lines += ['该模型是固定检测头，推理不接收参考图。5shot/10shot视觉提示：不适用。',
                  '如要比较5shot/10shot训练预算，需分别训练两个模型。',
                  '如要在同一微调权重上更换视觉提示数量，需从原始权重执行 `--method yoloe-vp`。', '']
    else:
        lines += ['推理shot只改变参考数量，训练预算保持不变。逐类独立查询后合并，不做跨类NMS。', '']
    lines += ['| 结果 | 训练shot/类 | 提示shot/类 | AP50 | AP50:95 | P | R | F2 | TP/FP/FN |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in entries:
        values = [f'{100 * row[k]:.2f}' for k in ['mAP50', 'mAP50_95', 'precision', 'recall', 'f2']]
        lines.append(f"| [{row['result']}]({row['result']}/evaluation.txt) | {training_shot} | "
                     f"{row['prompt_shot_per_class'] or '不适用'} | " + ' | '.join(values) +
                     f" | {row['tp']}/{row['fp']}/{row['fn']} |")
    lines += ['', '固定头为一次多类查询，全图max_det=500（吊物100）；评分每类最多100框。',
              '视觉提示为每类一次查询，每类max_det=100。两者查询与NMS竞争方式不同。',
              '各结果目录保存逐图预测、完整指标、逐类AP及score≥0.7指标、权重和数据哈希。', '']
    (output / 'test_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main(default_config=DEFAULT_CONFIG):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_config, required=default_config is None)
    parser.add_argument('--run-dir', type=Path, required=True, help='Exact completed training directory.')
    parser.add_argument('--checkpoint', choices=['best', 'last', 'initialized'], default='best')
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--pretrained-baseline', action='store_true')
    parser.add_argument('--prompt-shots', nargs='+', type=int, choices=[5, 10])
    parser.add_argument('--device', help='Explicit single CUDA device or cpu; required with --execute.')
    parser.add_argument('--output', type=Path, help='New results directory; existing paths are refused.')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / 'finetune_config.json'
    metadata = read(metadata_path)
    initialized_baseline = (metadata['status'] == 'initialized_no_training'
        and metadata['method'] == 'yolov8' and metadata.get('optimization_steps') == 0
        and args.checkpoint == 'initialized')
    if metadata['status'] != 'completed' and not initialized_baseline:
        raise ValueError('Select a completed training run.')
    if metadata['scene'] != config['scene'] or metadata['class_names'] != config['class_names']:
        raise ValueError('Training run and task configuration differ.')
    modes = evaluation_modes(metadata['method'], args.prompt_shots)
    weights = run_dir / 'weights' / f'{args.checkpoint}.pt'
    if args.pretrained_baseline:
        if metadata['method'] != 'yoloe-vp':
            raise ValueError('Pretrained visual baseline requires visual training metadata as the data contract.')
        weights = ROOT / config['weights']
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output = (args.output or run_dir / ('test_val300' if args.split == 'test' else 'dev_val100')).resolve()
    print(json.dumps(dict(method=metadata['method'], training_shot_per_class=metadata['shot_per_class'],
        checkpoint=str(weights), prompt_shots=modes, output=str(output), execute=args.execute),
        ensure_ascii=False, indent=2), flush=True)
    if metadata['method'] != 'yoloe-vp':
        print('此权重是固定检测头；5shot/10shot推理提示不适用。', flush=True)
    if not args.execute:
        return
    if not args.device:
        parser.error('--execute requires --device')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite results: {output}')
    data_path = Path(metadata['paths']['data'])
    if not data_path.is_file():
        data_path = ROOT / config['data']
    preparation_path = data_path.with_name('preparation.json')
    if sha256(preparation_path) != metadata['preparation_sha256']:
        raise ValueError('Prepared training data record changed since training.')
    if sha256(data_path) != metadata['input_sha256']['data']:
        raise ValueError('Training dataset YAML changed since training.')
    preparation = read(preparation_path)
    manifest_path, manifest, items = load_test(config, metadata, preparation, args.split)
    support_sets = {shot: load_supports(config, shot, preparation['support_seed'], items, preparation)
                    for shot in modes if shot is not None}
    sys.path.insert(0, str(ROOT / config['source_dir']))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs/_shared/runtime/yoloe_settings'))
    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOE
    from industrial_fewshot.cache import digest
    from industrial_fewshot.detectors.yoloe_visual import source_fingerprint
    from industrial_fewshot.evaluation.detection_metrics import check_evaluation_dependencies
    from industrial_fewshot.evaluation.evaluator import evaluate_records
    from industrial_fewshot.evaluation.reporting import format_metrics
    from industrial_fewshot.data.labelme import write_json

    if not Path(ultralytics.__file__).resolve().is_relative_to((ROOT / config['source_dir']).resolve()):
        raise ValueError('Wrong Ultralytics source imported.')
    check_evaluation_dependencies()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = (YOLO if metadata['method'] == 'yolov8' else YOLOE)(str(weights))
    if args.pretrained_baseline:
        head = model.model.model[-1]
        if not hasattr(head, 'savpe') or getattr(head, 'is_fused', False):
            raise ValueError('Baseline lacks visual prompting capability.')
        capability = dict(task=model.task, head=type(head).__name__, has_savpe=True, is_fused=False)
    else:
        capability = check_capability(model, metadata['method'], config['class_names'])
    model.model.float().eval()
    settings = dict(source_dir=str(ROOT / config['source_dir']), weights=str(weights),
                    device=args.device, imgsz=metadata['training']['imgsz'], confidence=.001,
                    nms_iou=.7, max_detections=100, torch_threads=4)
    evaluation = dict(iou_threshold=.5, score_threshold=.5, score_thresholds=[.5, .7],
                      max_detections_per_image=100)
    output.mkdir(parents=True)
    entries = []
    weights_hash = sha256(weights)
    source_hash = source_fingerprint(ROOT / config['source_dir'])
    for shot in modes:
        name = 'fixed_head' if shot is None else f'visual_{shot}shot'
        destination = output / name
        destination.mkdir()
        targets = [dict(id='all_classes', backend='yoloe_fixed', labels=[dict(name=c) for c in config['class_names']])]
        supports, support_sources = support_sets[shot] if shot else ({}, [])
        if shot:
            targets = [dict(id=c, backend='yoloe', label=c) for c in config['class_names']]
        base = dict(schema='industrial_fewshot.finetune_test.v1', scene_name=manifest_path.parent.name,
            method=metadata['method'], run_dir=str(destination), training_run=str(run_dir),
            training_shot_per_class=0 if args.pretrained_baseline else metadata['shot_per_class'], prompt_shot_per_class=shot,
            split=args.split, pretrained_baseline=args.pretrained_baseline,
            checkpoint=str(weights), weights_sha256=weights_hash, checkpoint_structure=capability,
            finetune_config_sha256=sha256(metadata_path), preparation_sha256=sha256(preparation_path),
            source_sha256=source_hash, image_count=len(items), image_dir=str(manifest_path.parent/'images'),
            image_paths=[r['relative_path'] for r in items], ground_truth_dir=str(manifest_path.parent/'images'),
            dataset_manifest=str(manifest_path), dataset_manifest_sha256=sha256(manifest_path),
            dataset_sha256=manifest.get('dataset_sha256'), evaluation=evaluation, targets=targets,
            inference=settings, support_manifests=support_sources, references=supports,
            protocol='independent_single_class_queries' if shot else 'joint_multiclass_query',
            max_det_per_query=100 if shot else 100 * len(config['class_names']),
            status='running', requested_prompt_shots=args.prompt_shots,
            prompt_shot_status='supported' if shot else 'not_applicable_fixed_head')
        if initialized_baseline:
            base.update(untrained_target_head_baseline=True, optimization_steps=0,
                        initialization_verification=metadata['initialization_verification'])
        base['test_identity_sha256'] = digest(dict(weights=weights_hash, dataset=sha256(manifest_path),
            supports=supports, settings=settings, evaluation=evaluation, source=source_hash, method=base['method']))
        write_json(destination / 'run.json', base)
        started = time.perf_counter()
        try:
            with (destination / 'records.jsonl').open('w', encoding='utf-8') as stream:
                if shot:
                    model = None
                    records = infer_visual(settings, supports, items, stream)
                else:
                    predict_device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)
                    predict_args = dict(device=predict_device, imgsz=settings['imgsz'], conf=.001, iou=.7,
                        max_det=base['max_det_per_query'], half=False, rect=False, verbose=False,
                        save=False, save_txt=False, save_crop=False, show=False)
                    records = infer_fixed(model, items, config['class_names'], predict_args, stream)
            if len(records) != len(items) * len(targets):
                raise ValueError('Incomplete prediction records.')
            metrics = evaluate_records(items, records, config['class_names'], evaluation)
            base.update(status='completed', success_request_count=len(records), failed_request_count=0,
                        elapsed_seconds=time.perf_counter()-started,
                        discarded_degenerate_count=sum(r.get('discarded_degenerate_count',0) for r in records),
                        box_policy='Clip to original image; discard zero-area clipped boxes; reject nonfinite predictions.')
            write_json(destination / 'metrics.json', metrics)
            (destination / 'evaluation.txt').write_text(format_metrics(metrics)+'\n', encoding='utf-8')
            entries.append(dict(result=name, training_shot_per_class=0 if args.pretrained_baseline else metadata['shot_per_class'],
                prompt_shot_per_class=shot, **{k: metrics[k] for k in
                ['mAP50', 'mAP50_95', 'precision', 'recall', 'f2', 'tp', 'fp', 'fn']}))
            print(format_metrics(metrics), flush=True)
        except BaseException as exc:
            base.update(status='failed', error=repr(exc))
            raise
        finally:
            write_json(destination / 'run.json', base)
        write_summary(output, entries, metadata['method'], 0 if args.pretrained_baseline else metadata['shot_per_class'], args.split, len(items))


if __name__ == '__main__':
    main()

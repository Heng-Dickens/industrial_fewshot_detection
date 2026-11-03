"""Matched base-four / all-five / pretrained visual-prompt transfer experiment.

Preparation and training use the existing public fine-tuning entrypoints. Novel
supports are explicitly permitted only at evaluation, never inserted into the
base-four preparation record. Test predictions are made for all five labels;
novel/base/all metrics are projections of the same saved records.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / 'outputs/YOLOE/finetune/scripts/common'
sys.path[:0] = [str(COMMON), str(ROOT / 'src')]
from finetune_config import load_config, SCENES
from prepare_data import read, sha256

STAMP = 'novel_chache4_20261102'
RUNTIME = ROOT / 'outputs/_shared/runtime' / STAMP
DATA = ROOT / 'data/finetune' / STAMP
RUNS = ROOT / 'outputs/YOLOE/finetune/chache_val300/runs' / STAMP
REPORT = ROOT / 'outputs/YOLOE/finetune/chache_val300/comparisons' / STAMP
BASE = ['Chache1', 'Chache2', 'Chache3', 'Qianyinche']
NOVEL = ['Chache4']
ALL = SCENES['chache']
JOBS = [dict(id=f'{partition}_{shot}shot', partition=partition, shot=shot, gpu=gpu)
        for gpu, (partition, shot) in enumerate([
            ('base', 5), ('all', 5), ('base', 10), ('all', 10),
            ('pretrained', 5), ('pretrained', 10)])]
NAMES = {'base': '四类微调→新增Chache4提示', 'all': '五类一起微调', 'pretrained': '原始预训练＋视觉提示'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def config_path(job):
    return RUNTIME / 'configs' / f'{job["partition"]}_{job["shot"]}shot.json'


def protocol():
    return dict(schema='industrial_fewshot.heldout_visual_class.v1', novel_classes=NOVEL,
        base_classes=BASE, evaluation_classes=ALL, shots=[5, 10], support_seed=0, training_seed=0,
        train_protocol='same_image', train_scope='full', savpe_weights_trainable=True,
        savpe_bn_running_statistics_frozen=True, validation='identical base-only 88-image subset',
        checkpoint_selection='upstream fitness on base-only dev; Chache4 GT never used',
        inference='independent single-class queries, including Chache4, merged without cross-class NMS',
        evaluation='all frozen 300 test images; novel/base/all GT and predictions projected by label',
        pretrained_seen_categories_unknown=True,
        limitations=['Novel means absent from THIS fine-tuning; no claim of absence from original pretraining.',
          'Image hashes disjoint; video/camera independence is not established.',
          'One training/support seed; one held-out category; exploratory comparison.',
          'Reference annotations contain only declared class; unannotated visual co-occurrence is not certified.',
          'Five-class condition has extra Chache4 gradient supervision; same per-class K, not equal total images.',
          'Validation excludes Chache4-positive images for BOTH trained conditions, so five-class model selection also uses base GT only.'])


def prepare():
    if RUNTIME.exists() or DATA.exists() or REPORT.exists():
        raise FileExistsError('Protocol paths exist. Use --launch/--summarize; do not overwrite preparation.')
    full = read(ROOT / 'data/raw/chache_val100/manifest.json')
    kept = [r for r in full['samples'] if not any(g['label'] in NOVEL for g in r['ground_truth'])]
    if len(kept) != 88 or set(g['label'] for r in kept for g in r['ground_truth']) != set(BASE):
        raise ValueError('Unexpected base-only dev composition; inspect the frozen source.')
    hashes = [r['image_sha256'] for r in kept]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate dev images.')
    dev_root = DATA / 'dev_base_only'
    dev_images = dev_root / 'images'
    dev_images.mkdir(parents=True)
    for row in kept:
        source = ROOT / 'data/raw/chache_val100/images' / row['relative_path']
        for p in [source, source.with_suffix('.json')]:
            target = dev_images / p.name
            # Actual copies preserve originals if a later tool edits generated data.
            shutil.copy2(p, target)
    subset = {**full, 'image_root': str(dev_images), 'ground_truth_root': str(dev_images),
        'samples': kept, 'source_manifest': 'data/raw/chache_val100/manifest.json',
        'source_manifest_sha256': sha256(ROOT/'data/raw/chache_val100/manifest.json'),
        'excluded_class': NOVEL,
        'excluded_images': [r['relative_path'] for r in full['samples'] if r not in kept]}
    subset['dataset_sha256'] = hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()
    write(dev_root/'manifest.json', subset)
    cfg0 = read(ROOT/'outputs/YOLOE/finetune/scripts/chache/config.json')
    for job in JOBS:
        cfg = copy.deepcopy(cfg0)
        partition = 'base' if job['partition'] == 'base' else 'all'
        classes = BASE if partition == 'base' else ALL
        cfg.update(shot=job['shot'], class_names=classes,
            text_prompts=[cfg0['text_prompts'][ALL.index(c)] for c in classes],
            data=str((DATA/job['id']/'dataset.yaml').relative_to(ROOT)),
            dev_images=str(dev_images.relative_to(ROOT)),
            dev_manifest=str((dev_root/'manifest.json').relative_to(ROOT)), dev_expected_images=88,
            project=str(RUNS.relative_to(ROOT)), class_protocol=dict(
                type='heldout_visual_class_v1', base_classes=BASE, novel_classes=NOVEL,
                training_partition=partition, validation_classes=BASE))
        cfg['training'].update(workers=0, plots=False, save_period=-1, val_interval=1)
        cfg['adaptation'] = dict(scope='full', protocol='same_image', loss='detection',
                                  freeze_savpe_bn_stats=True)
        write(config_path(job), cfg)
        load_config(config_path(job))
        if job['partition'] != 'pretrained':
            subprocess.run([sys.executable, str(COMMON/'prepare_data.py'), '--config',
                str(config_path(job)), '--write'], cwd=ROOT, check=True)
    description = protocol()
    description.update(dev_image_count=len(kept), excluded_dev_image_count=100-len(kept),
        dev_box_counts=dict(Counter(g['label'] for r in kept for g in r['ground_truth'])),
        source_test_manifest_sha256=sha256(ROOT/cfg0['test_manifest']),
        dev_manifest_sha256=sha256(dev_root/'manifest.json'), jobs=JOBS)
    write(REPORT/'protocol.json', description)
    (REPORT/'README.md').write_text(documentation(), encoding='utf-8')
    print('Prepared isolated protocol:', REPORT, flush=True)


def documentation():
    return '''# Chache4 新类别视觉提示泛化

## 目的和对照

Chache4 在四类微调期间完全没有标注监督，训练完成后只加入它的参考图片与目标框，不再更新权重。
每类5张、10张各做一档，分别比较四类微调、五类微调、原始预训练权重，共六组。
四类为 Chache1、Chache2、Chache3、Qianyinche；五类条件额外包含 Chache4。
四类训练分别20/40张；五类训练分别25/50张。三种条件使用相同的每类K张冻结参考图（support seed=0）。
五类条件既使用Chache4图做梯度更新，也把同一批图用于推理提示；四类条件只在推理时使用它们。
预训练条件不做任何梯度更新。这里的“新类”仅指微调未见，不能保证原始预训练未见。

## 训练和选权重

沿用现有全量视觉提示微调：YOLOE-v8l-seg预训练权重转检测架构；backbone、neck、检测头及SAVPE参数都更新。
同图框条件训练（same_image）、完整检测损失；SAVPE的BN运行统计冻结，SAVPE可学习参数仍更新。
FP32、640输入、batch=4、AdamW、lr=0.0001、最多100epoch、patience=10、seed=0；无mosaic/mixup。
与既有5shot实验保持训练配方，仅按本实验协议改变训练类别和选权重的验证集。
原始100张验证集排除所有含Chache4标注的12张图，得到共同的88张验证图；包括30张背景图。
三组之间不使用Chache4验证性能调参、早停或挑权重，两种微调均按同一基础四类验证集的上游fitness选best.pt。
五类训练支持中有Chache4，但验证查询图没有Chache4；验证的四类AP用于选权重，不把无GT的Chache4 AP当0平均。
每个微调独占一张GPU：训练池仅20–50张，分布式通信和每卡极小batch开销不划算；不同实验并行。

## 测试口径

冻结的300张测试图全部保留，五类逐类提示查询后合并；不做跨类别NMS，和原有YOLOE评测一致。
每个模型一次评测五类，保存1500条请求；再将同一份GT和预测按标签投影为Chache4、基础四类、全五类三种范围。
Chache4主指标使用全部300张图：30个Chache4真值框，其他类别图/背景图仍计Chache4误检。
AP使用conf≥0.001、NMS IoU=0.7、每类最多100框；P/R/F1/F2使用score≥0.5、IoU=0.5，另保存score≥0.7。
三种条件的参考图、测试图、提示聚合（逐图归一化→均值→归一化）、输入大小和阈值完全相同。
推理时给Chache4生成新的视觉原型，无须把四类固定分类输出改成五类或重新训练。
所有支持图、训练图、验证图与测试图做字节哈希隔离；四类训练/验证禁止出现Chache4标注或Chache4支持图哈希。

## 如何执行

在仓库根目录、industrial-fewshot环境中：

```bash
python scripts/run_yoloe_novel_chache4.py --prepare
python scripts/run_yoloe_novel_chache4.py --launch
python scripts/run_yoloe_novel_chache4.py --status
python scripts/run_yoloe_novel_chache4.py --summarize
```

prepare仅创建新协议和数据；launch会运行四个微调与两组预训练对照，自动完成测试；summarize仅读取已保存预测/指标。
已完成任务不会重跑；已有未完成输出不覆盖。配置和完整日志位于`outputs/_shared/runtime/novel_chache4_20261102/`。
训练/权重保存在`outputs/YOLOE/finetune/chache_val300/runs/novel_chache4_20261102/`。
本页更新为最终汇总后，仍保留完整配置、协议、逐图记录和三种标签范围的指标。

## 如何解释

四类微调高于原始权重：说明基础类别微调有助于该新类的提示迁移；若低于原始权重，说明存在迁移损失或遗忘。
四类接近五类监督结果：支持“少做新类训练、换提示即可扩展”的价值；五类通常是有新类梯度监督的对照，不是无监督上界保证。
一组差异不能证明SAVPE优于所有固定头：此处测的是不增加新类优化步骤的适配能力。
仅一个held-out类别和一个seed，属于探索实验。没有按视频/相机重新划分，不能称为独立视频或新相机泛化。
参考标注只含所声明的类别；未标注的同图物体是否存在Chache4，仍需人工复核。

[协议与哈希](protocol.json)

'''


def load_contract(job):
    """Audit explicit new-class exceptions without changing legacy evaluation rules."""
    from evaluate import load_test, safe_child
    from industrial_fewshot.data.labelme import load_ground_truth
    from PIL import Image
    cfg = load_config(config_path(job))
    pretrained = job['partition'] == 'pretrained'
    train_job = dict(job, partition='all', id=f'all_{job["shot"]}shot') if pretrained else job
    prep_path = DATA/train_job['id']/'preparation.json'
    prep = read(prep_path)
    meta = None if pretrained else read(RUNS/job['id']/'finetune_config.json')
    if not pretrained:
        if (meta['status'] != 'completed' or meta['class_names'] != cfg['class_names']
                or meta['class_protocol'] != cfg['class_protocol']
                or meta['preparation_sha256'] != sha256(prep_path)
                or meta['input_sha256']['data'] != sha256(ROOT/cfg['data'])):
            raise ValueError('Training contract changed or training is incomplete.')
    identity = meta or dict(test_manifest_sha256=prep['excluded_test_manifest_sha256'])
    eval_cfg = {**cfg, 'class_names': ALL}
    manifest_path, manifest, items = load_test(eval_cfg, identity, prep)
    full_dev_path = ROOT/'data/raw/chache_val100/manifest.json'
    dev_subset = read(ROOT/cfg['dev_manifest'])
    if sha256(full_dev_path) != dev_subset['source_manifest_sha256']:
        raise ValueError('Original 100-image dev manifest changed.')
    full_dev = read(full_dev_path)
    dev_hashes = {s['image_sha256'] for s in full_dev['samples']}
    test_hashes = {s['image_sha256'] for s in items}
    train_hashes = {s['image_sha256'] for s in prep['samples'] if s['split'] == 'train'}
    novel_hashes = set()
    supports, sources, seen = {}, [], set()
    for label in ALL:
        root = ROOT/'data/references/chache'/label
        frozen_path = root/f'{job["shot"]}shot/seed_0.json'
        frozen = read(frozen_path)
        if frozen['positive_count'] != job['shot'] or frozen['selection_seed'] != 0:
            raise ValueError('Support count/seed mismatch.')
        index = {r['image']: r for r in read(root/'manifest.json')['images']}
        sources.append(dict(label=label, path=str(frozen_path), sha256=sha256(frozen_path)))
        supports[label] = []
        for chosen in frozen['positive_samples']:
            row = index[chosen['sample_id']]
            image = safe_child(root, row['image'])
            annotation = safe_child(root, row['annotation'])
            checksum = sha256(image)
            if checksum != chosen['image_sha256'] or sha256(annotation) != chosen['annotation_sha256']:
                raise ValueError('Frozen support bytes changed.')
            if checksum in seen | dev_hashes | test_hashes:
                raise ValueError('Support overlap/duplicate.')
            if label in NOVEL:
                novel_hashes.add(checksum)
            if job['partition'] == 'base' and label in NOVEL:
                if checksum in train_hashes:
                    raise ValueError('Novel reference leaked into base training.')
            elif checksum not in train_hashes:
                raise ValueError('Expected training-pool support missing.')
            seen.add(checksum)
            with Image.open(image) as opened:
                truth = load_ground_truth(annotation, *opened.size, {label})
            if not truth:
                raise ValueError('Empty visual support.')
            supports[label].append(dict(sample_id=chosen['sample_id'], image_path=str(image),
                json_path=str(annotation), image_sha256=checksum,
                annotation_sha256=chosen['annotation_sha256'], bboxes=[v['bbox_pixels_xyxy'] for v in truth]))
        if len(supports[label]) != job['shot']:
            raise ValueError('Support count mismatch.')
    if job['partition'] == 'base':
        if novel_hashes & {s['image_sha256'] for s in prep['samples']}:
            raise ValueError('Novel support leaked into base preparation.')
        if any(label in NOVEL for s in prep['samples'] for label in s['box_counts']):
            raise ValueError('Novel annotations leaked into base preparation.')
    return cfg, meta, prep, manifest_path, manifest, items, supports, sources


def evaluate(job):
    from evaluate import infer_visual, check_capability
    from industrial_fewshot.evaluation.evaluator import evaluate_records
    from industrial_fewshot.evaluation.reporting import format_metrics
    cfg, meta, prep, manifest_path, manifest, items, supports, sources = load_contract(job)
    destination = RUNS/job['id']/'test_val300'
    if destination.exists():
        saved = read(destination/'run.json')
        if saved['status'] == 'completed':
            return
        raise FileExistsError(f'Incomplete output retained, choose a new run: {destination}')
    pretrained = job['partition'] == 'pretrained'
    weights = ROOT/cfg['weights'] if pretrained else RUNS/job['id']/'weights/best.pt'
    sys.path.insert(0, str(ROOT/cfg['source_dir']))
    import torch
    from ultralytics import YOLOE
    from industrial_fewshot.detectors.yoloe_visual import source_fingerprint
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = YOLOE(str(weights))
    head = model.model.model[-1]
    if not hasattr(head, 'savpe') or getattr(head, 'is_fused', False):
        raise ValueError('No unfused SAVPE available.')
    verification = dict(savpe_present=True, is_fused=False, has_new_class_gradient_supervision=job['partition']=='all')
    if not pretrained:
        verification.update(check_capability(model, 'yoloe-vp', cfg['class_names']))
        original = YOLOE(str(ROOT/cfg['weights']))
        verification['backbone_weights_changed'] = not torch.equal(
            model.model.model[0].conv.weight, original.model.model[0].conv.weight)
        before = dict(original.model.model[-1].savpe.named_parameters())
        verification['savpe_changed_parameters'] = [name for name, p in head.savpe.named_parameters()
            if not torch.equal(p, before[name])]
        if not verification['backbone_weights_changed'] or not verification['savpe_changed_parameters']:
            raise ValueError('Backbone/SAVPE did not actually update.')
        del original
    del model
    settings = dict(source_dir=str(ROOT/cfg['source_dir']), weights=str(weights), device='0',
        imgsz=cfg['training']['imgsz'], confidence=.001, nms_iou=.7, max_detections=100, torch_threads=2)
    metric_settings = dict(iou_threshold=.5, score_threshold=.5, score_thresholds=[.5,.7],
                           max_detections_per_image=100)
    destination.mkdir(parents=True)
    state = dict(schema='industrial_fewshot.novel_class_test.v1', status='running', **job,
        scene_name='chache_val300', label_scope='all_five_with_explicit_novel_projection',
        protocol=protocol(), weights=str(weights), weights_sha256=sha256(weights),
        dataset_manifest=str(manifest_path), dataset_manifest_sha256=sha256(manifest_path),
        dataset_sha256=manifest.get('dataset_sha256'), source_sha256=source_fingerprint(ROOT/cfg['source_dir']),
        config_sha256=sha256(config_path(job)), class_names=ALL, references=supports, support_manifests=sources,
        verification=verification, training_run=None if pretrained else str(RUNS/job['id']),
        training_classes=[] if pretrained else cfg['class_names'], training_shot_per_class=0 if pretrained else job['shot'],
        novel_training_shot=job['shot'] if job['partition']=='all' else 0, prompt_shot_per_class=job['shot'],
        image_count=len(items), expected_requests=len(items)*len(ALL), evaluation=metric_settings, inference=settings)
    write(destination/'run.json', state)
    started = time.perf_counter()
    try:
        with (destination/'records.jsonl').open('w', encoding='utf-8') as stream:
            records = infer_visual(settings, supports, items, stream)
        if (len(records) != 1500 or len({(r['relative_path'], r['target_id']) for r in records}) != 1500
                or any(r['status'] != 'success' for r in records)):
            raise ValueError('Incomplete prediction records.')
        for scope, labels in [('novel', NOVEL), ('base', BASE), ('all', ALL)]:
            scope_items = [{**r, 'ground_truth': [g for g in r['ground_truth'] if g['label'] in labels]} for r in items]
            scope_records = [r for r in records if r['target_id'] in labels]
            metrics = evaluate_records(scope_items, scope_records, labels, metric_settings)
            target = destination/scope
            target.mkdir()
            write(target/'metrics.json', metrics)
            (target/'evaluation.txt').write_text(format_metrics(metrics)+'\n', encoding='utf-8')
        state.update(status='completed', success_request_count=len(records), failed_request_count=0,
                     elapsed_seconds=time.perf_counter()-started)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write(destination/'run.json', state)


def worker(job):
    statefile = RUNTIME/'jobs'/(job['id']+'.json')
    state = dict(**job, pid=os.getpid(), status='running', started=time.time(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(statefile, state)
    try:
        if job['partition'] != 'pretrained':
            metadata = RUNS/job['id']/'finetune_config.json'
            if not metadata.exists() or read(metadata)['status'] != 'completed':
                command = [sys.executable, str(COMMON/'train.py'), '--config', str(config_path(job)),
                    '--method', 'yoloe-vp', '--name', job['id'], '--device', '0', '--batch', '4', '--execute']
                state.update(stage='training', argv=command)
                write(statefile, state)
                print(json.dumps(dict(argv=command)), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
        state.update(stage='testing_all_five')
        write(statefile, state)
        evaluate(job)
        state.update(status='completed', stage='complete', finished=time.time())
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), finished=time.time())
        traceback.print_exc()
        raise
    finally:
        write(statefile, state)


def launch():
    if not (REPORT/'protocol.json').exists():
        raise ValueError('Run --prepare first.')
    for job in JOBS:
        statefile = RUNTIME/'jobs'/(job['id']+'.json')
        if statefile.exists():
            state = read(statefile)
            if state['status'] == 'completed':
                continue
            if state['status'] == 'running':
                try:
                    os.kill(state['pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f'Already running: {job["id"]}')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', YOLO_AUTOINSTALL='false',
            YOLO_CONFIG_DIR=str(RUNTIME/'settings'/job['id']))
        font = Path.home()/'.config/Ultralytics/Arial.ttf'
        settings_dir = Path(env['YOLO_CONFIG_DIR'])
        settings_dir.mkdir(parents=True, exist_ok=True)
        if font.is_file() and not (settings_dir/font.name).exists():
            shutil.copy2(font, settings_dir/font.name)
        log = RUNTIME/'logs'/(job['id']+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', job['id']],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(job['id'], 'physical GPU', job['gpu'], 'PID', proc.pid, flush=True)


def summarize():
    states = [read(RUNTIME/'jobs'/(j['id']+'.json')) for j in JOBS]
    if any(s['status'] != 'completed' for s in states):
        raise ValueError('All six conditions must finish before final summary.')
    verify_results()
    rows = []
    snapshots = REPORT/'configs'
    snapshots.mkdir(exist_ok=True)
    for job in JOBS:
        shutil.copy2(config_path(job), snapshots/config_path(job).name)
    for shot in [5, 10]:
        for partition in ['pretrained', 'base', 'all']:
            job = next(j for j in JOBS if j['shot']==shot and j['partition']==partition)
            destination = RUNS/job['id']/'test_val300'
            run = read(destination/'run.json')
            if run['status'] != 'completed' or run['success_request_count'] != 1500:
                raise ValueError('Incomplete saved evaluation.')
            row = dict(condition=partition, label=NAMES[partition], shot=shot,
                training_image_count=0 if partition=='pretrained' else shot*(4 if partition=='base' else 5),
                novel_training_shot=shot if partition=='all' else 0,
                weights_sha256=run['weights_sha256'], source=str(destination.relative_to(ROOT)),
                config_sha256=run['config_sha256'],
                training_epochs=0, best_base_dev_epoch=None)
            if partition != 'pretrained':
                with (RUNS/job['id']/'results.csv').open(newline='') as stream:
                    epochs = list(csv.DictReader(stream))
                row['training_epochs'] = len(epochs)
                best = max(epochs, key=lambda r: .1*float(r['vp_metrics/mAP50(B)'])
                           + .9*float(r['vp_metrics/mAP50-95(B)']))
                row['best_base_dev_epoch'] = int(best['epoch'])
            for scope in ['novel', 'base', 'all']:
                m = read(destination/scope/'metrics.json')
                if m['evaluation_status'] != 'complete' or m['evaluated_image_count'] != 300:
                    raise ValueError('Incomplete metrics.')
                for key in ['mAP50','mAP75','mAP50_95','precision','recall','f2','tp','fp','fn',
                            'negative_error_rate','negative_image_count','negative_error_image_count',
                            'image_accuracy','mean_matched_iou']:
                    row[f'{scope}_{key}'] = m[key]
                row[f'{scope}_f1'] = 2*m['precision']*m['recall']/(m['precision']+m['recall']) if m['precision']+m['recall'] else 0.
                row[f'{scope}_metrics_sha256'] = sha256(destination/scope/'metrics.json')
            rows.append(row)
    write(REPORT/'results.json', dict(protocol=read(REPORT/'protocol.json'), results=rows))
    with (REPORT/'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['## 完整测试结果', '', '指标均为百分数。新类为Chache4；基础mAP是其余四类的平均。P/R/F1/F2取score≥0.5。', '',
        '| 提示K | 条件 | 训练图数 | 新类训练K | 新类AP50 | 新类AP50:95 | 新类P | 新类R | 新类F1 | 新类F2 | 新类TP/FP/FN | 基础AP50 | 全五类AP50 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for row in rows:
        link = os.path.relpath(ROOT/row['source']/'novel/evaluation.txt', REPORT)
        values = [f'{100*row[f"novel_{key}"]:.2f}' for key in ['mAP50','mAP50_95','precision','recall','f1','f2']]
        counts = '/'.join(str(row[f'novel_{key}']) for key in ['tp','fp','fn'])
        lines.append(f'| {row["shot"]} | [{row["label"]}]({link}) | {row["training_image_count"]} | {row["novel_training_shot"]} | '
            + ' | '.join(values) + f' | {counts} | {100*row["base_mAP50"]:.2f} | {100*row["all_mAP50"]:.2f} |')
    lines += ['', '[完整CSV](results.csv) · [JSON与来源哈希](results.json) · [完整性与隔离检查](verification.json)', '',
        '新类单类指标与现有总表的全五类mAP不是同一范围，因此单独汇总；全部原始预测和三种范围指标均保留。', '']
    for shot in [5, 10]:
        group = {r['condition']:r for r in rows if r['shot']==shot}
        base, original, all_classes = (group[c] for c in ['base','pretrained','all'])
        delta = 100*(base['novel_mAP50']-original['novel_mAP50'])
        gap = 100*(base['novel_mAP50']-all_classes['novel_mAP50'])
        lines += [f'{shot}shot：四类微调后新增提示的Chache4 AP50较原始权重变化{delta:+.2f}个百分点，'
            f'较五类监督微调变化{gap:+.2f}个百分点。其score≥0.5的召回为{100*base["novel_recall"]:.2f}%，'
            'AP改善与部署阈值下的召回改善应分开看。', '']
    lines += ['实际完成轮数与基础验证集fitness最佳轮数见CSV；不同训练池和早停使优化步数不同，未按测试集挑轮数。', '',
        '配置快照：'+' · '.join(f'[{j["id"]}](configs/{config_path(j).name})' for j in JOBS), '',
        '可视化选冻结测试清单按文件名排序的首张Chache4正图，未按检测好坏挑图；绿色是真值，红色是score≥0.5预测。', '',
        '![5shot三条件同图对照](preview_5shot.jpg)', '',
        '![10shot三条件同图对照](preview_10shot.jpg)', '']
    render_previews(rows)
    (REPORT/'README.md').write_text(documentation()+'\n'.join(lines), encoding='utf-8')
    for index in [REPORT.parent/'README.md', ROOT/'outputs/YOLOE/finetune/README.md', ROOT/'outputs/README.md']:
        entry = f'[Chache4新类别：四类微调后加提示 / 五类微调 / 预训练，5与10shot]({os.path.relpath(REPORT/"README.md",index.parent)})'
        content = index.read_text(encoding='utf-8')
        if entry not in content:
            index.write_text(content.rstrip()+'\n\n'+entry+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def verify_results():
    """Check saved artifact identities, paired support equality and label scopes."""
    paired = {}
    checks = []
    for job in JOBS:
        cfg, meta, prep, manifest_path, manifest, items, supports, sources = load_contract(job)
        destination = RUNS/job['id']/'test_val300'
        state = read(destination/'run.json')
        if state['status'] != 'completed' or state['failed_request_count'] != 0:
            raise ValueError('Incomplete test run.')
        if sha256(Path(state['weights'])) != state['weights_sha256']:
            raise ValueError('Checkpoint changed after inference.')
        if sha256(config_path(job)) != state['config_sha256'] or state['references'] != supports:
            raise ValueError('Configuration/reference contract changed after inference.')
        if job['shot'] in paired and paired[job['shot']] != state['references']:
            raise ValueError('Conditions do not use identical inference references.')
        paired[job['shot']] = state['references']
        records = [json.loads(line) for line in (destination/'records.jsonl').read_text().splitlines() if line.strip()]
        expected = {(i['relative_path'], label) for i in items for label in ALL}
        if len(records) != 1500 or {(r['relative_path'],r['target_id']) for r in records} != expected:
            raise ValueError('Prediction requests missing/duplicated or label IDs differ.')
        image_hashes = {i['relative_path']:i['image_sha256'] for i in items}
        if any(r['status'] != 'success' or r['image_sha256'] != image_hashes[r['relative_path']]
               or any(o['label'] != r['target_id'] for o in r['objects']) for r in records):
            raise ValueError('Prediction record identity/class differs.')
        scopes = {scope:read(destination/scope/'metrics.json') for scope in ['novel','base','all']}
        for scope, metric in scopes.items():
            if metric['evaluated_image_count'] != 300 or metric['evaluation_status'] != 'complete':
                raise ValueError('Metrics exclude test images or requests.')
        novel = scopes['novel']
        if novel['tp']+novel['fn'] != 30 or set(novel['per_class']) != {'Chache4'}:
            raise ValueError('Novel GT projection differs.')
        if set(scopes['base']['per_class']) != set(BASE) or set(scopes['all']['per_class']) != set(ALL):
            raise ValueError('Base/all GT projection differs.')
        for key in ['tp','fp','fn']:
            if scopes['all']['per_class']['Chache4'][key] != novel[key]:
                raise ValueError('Novel operating point differs between label scopes.')
        for key in ['mAP50','mAP75','mAP50_95']:
            if abs(scopes['all'][key]-(4*scopes['base'][key]+novel[key])/5) > 1e-8:
                raise ValueError('Class-average AP inconsistent across scopes.')
        if meta:
            initialization = read(RUNS/job['id']/'initialization_verification.json')
            if not initialization['pretrained_tensor_equal']:
                raise ValueError('Training did not start from the declared pretrained backbone.')
            if not state['verification']['backbone_weights_changed'] or not state['verification']['savpe_changed_parameters']:
                raise ValueError('Trained backbone/SAVPE update verification missing.')
        checks.append(dict(id=job['id'], verified_requests=len(records),
            records_sha256=sha256(destination/'records.jsonl'),
            weights_sha256=state['weights_sha256'], novel_gt_count=30,
            train_class_count=0 if meta is None else len(meta['class_names']),
            excluded_novel_training_supervision=job['partition']!='all'))
    write(REPORT/'verification.json', dict(status='passed', conditions=len(checks),
        successful_requests=sum(r['verified_requests'] for r in checks),
        identical_paired_references=True, train_dev_test_hash_disjoint=True,
        label_projection_consistent=True, checks=checks))
    print('Verified six conditions, 9000 successful requests and novel/base/all label scopes.', flush=True)


def render_previews(rows):
    from PIL import Image, ImageDraw, ImageFont
    manifest = read(ROOT/'data/raw/chache_val300/manifest.json')
    sample = next(s for s in sorted(manifest['samples'], key=lambda r:r['relative_path'])
                  if any(g['label'] in NOVEL for g in s['ground_truth']))
    source = ROOT/'data/raw/chache_val300/images'/sample['relative_path']
    font_path = Path.home()/'.config/Ultralytics/Arial.ttf'
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    for shot in [5, 10]:
        panels = []
        for partition in ['pretrained','base','all']:
            row = next(r for r in rows if r['shot']==shot and r['condition']==partition)
            with Image.open(source) as original:
                image = original.convert('RGB')
            width, height = image.size
            image.thumbnail((960, 600))
            sx, sy = image.width/width, image.height/height
            predictions = []
            with (ROOT/row['source']/'records.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line)
                    if record['relative_path']==sample['relative_path'] and record['target_id']=='Chache4':
                        predictions = [p for p in record['objects'] if p['confidence'] >= .5]
            draw = ImageDraw.Draw(image)
            for objects, color in [([g for g in sample['ground_truth'] if g['label'] in NOVEL], '#00ff66'),
                                    (predictions, '#ff3333')]:
                for obj in objects:
                    x1,y1,x2,y2 = obj['bbox_pixels_xyxy']
                    box = (x1*sx,y1*sy,x2*sx,y2*sy)
                    draw.rectangle(box, outline=color, width=3)
                    label = 'GT Chache4' if 'confidence' not in obj else f'Chache4 {obj["confidence"]:.2f}'
                    draw.text((box[0],max(54,box[1]-21)),label,fill=color,font=font)
            draw.rectangle((0,0,image.width,52), fill='black')
            draw.text((6,3),f'{partition} | prompt {shot}shot | Green GT, red predictions >= 0.5',fill='white',font=font)
            draw.text((6,27),sample['relative_path'],fill='white',font=font)
            panels.append(image)
        canvas = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)), 'white')
        x = 0
        for panel in panels:
            canvas.paste(panel,(x,0));x += panel.width
        canvas.save(REPORT/f'preview_{shot}shot.jpg', quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--launch', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--summarize', action='store_true')
    group.add_argument('--verify', action='store_true')
    group.add_argument('--worker', choices=[j['id'] for j in JOBS], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        launch()
    elif args.summarize:
        summarize()
    elif args.verify:
        verify_results()
    elif args.worker:
        worker(next(j for j in JOBS if j['id']==args.worker))
    else:
        for job in JOBS:
            statefile = RUNTIME/'jobs'/(job['id']+'.json')
            print(json.dumps(read(statefile) if statefile.exists() else dict(job, status='not_started'), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Compare WeDetect prompts on five special-vehicle classes via single-class queries.

No training or API calls. Each class uses the existing adapter, then predictions
are merged for class-aware evaluation. Reports use --output; model results use
outputs/runs/<dataset>/WeDetect/<model-shot-prompt>, with auxiliary files in runtime.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.configuration import load_document
from industrial_fewshot.data.labelme import load_evaluation_truth, read_json, write_json
from industrial_fewshot.evaluation.evaluator import evaluate_records
from industrial_fewshot.evaluation.reporting import format_metrics

LABELS = ['Chache1', 'Chache2', 'Chache3', 'Chache4', 'Qianyinche']
POINTS = [.001, .002, .005, .01, .02, .05, .1, .2, .3, .5, .7]
GROUP_NAMES = {'short':'简短类别词', 'names':'具体类别名', 'phrases':'短描述', 'long':'长描述'}


def runtime_dir(output):
    return ROOT / 'outputs/runtime' / output.name


def read_plan(output):
    path = runtime_dir(output) / 'plan.json'
    return read_json(path if path.exists() else output / 'plan.json')


def combined_dir(output, group):
    plan = read_plan(output)
    if plan.get('runs'):
        return Path(plan['runs'][group]['run_dir'])
    return output / group / 'combined'


def reserve_runs(output, plan):
    from industrial_fewshot.result_layout import namespace_for_report
    manifest_path = ROOT / 'data/raw/chache_val300/manifest.linux.local.json'
    local = read_json(Path(plan['source_manifest']))
    local.update(image_root=str(ROOT / 'data/raw/chache_val300/images'),
                 ground_truth_root=str(ROOT / 'data/raw/chache_val300/images'))
    if manifest_path.exists():
        assert read_json(manifest_path) == local, f'Existing local manifest differs: {manifest_path}'
    else:
        write_json(manifest_path, local)
    canonical = digest({'images': [dict(relative_path=r['relative_path'], image_sha256=r['image_sha256'],
                                        ground_truth=r['ground_truth']) for r in plan['items']]})
    plan['legacy_dataset_sha256'] = plan.get('legacy_dataset_sha256', plan['dataset_sha256'])
    plan['dataset_sha256'] = canonical
    plan['dataset_manifest'] = str(manifest_path)
    runs = {}
    for group, prompts in plan['groups'].items():
        report = dict(scene_name='chache_val300', image_count=plan['image_count'],
                      image_dir=local['image_root'], ground_truth_dir=local['ground_truth_root'],
                      dataset_manifest=str(manifest_path), dataset_sha256=canonical,
                      targets=[dict(backend='wedetect', prompt_snapshot=dict(kind='text', text=prompts[label],
                                                                            review_template='')) for label in LABELS])
        namespace, layout = namespace_for_report(ROOT / 'outputs/runs', report, create=True)
        marker = Path(layout['prompt_registry_dir']) / 'prompt_version.json'
        version = read_json(marker)
        version.update(description=f'WeDetect 五类特种车辆：{GROUP_NAMES[group]}；每类独立查询后合并评估。',
                       prompt_group=group, class_prompts=prompts)
        write_json(marker, version)
        runs[group] = dict(run_dir=str(namespace / ('wedetect_base_0shot' + layout['model_name_suffix'])),
                           result_layout=layout)
    plan['runs'] = runs
    plan['comparison_dir'] = str(output)
    plan['runtime_dir'] = str(runtime_dir(output))
    return plan


def prompt_groups(labels):
    return {
        'short': dict(zip(LABELS, ['叉车', '搬运车', '堆高车', '叉车', '牵引车'])),
        'names': dict(zip(LABELS, ['平衡重叉车', '电动托盘搬运车', '站驾式堆高车', '带属具的叉车', '电动牵引车'])),
        'phrases': dict(zip(LABELS, ['有护顶架和普通货叉的座驾叉车', '无门架的低位双叉搬运车',
                                  '有高门架的站驾式堆高车', '装有黄色大型属具的叉车',
                                  '没有门架和货叉的座驾牵引车'])),
        'long': {row['name']: row['description'] for row in labels},
    }


def cohort(name):
    from PIL import Image
    raw, _ = load_document(ROOT / f'configs/common/{name}.yaml')
    manifest_path = Path(raw['dataset']['manifest_file'])
    manifest = read_json(manifest_path)
    directory = Path(raw['dataset']['image_dir'])
    names = {r['name'] for r in raw['labels']}
    items = []
    for row in sorted(manifest['samples'], key=lambda r: r['relative_path']):
        path = directory / row['relative_path']
        with Image.open(path) as im:
            size = im.size
        assert size == (row['width'], row['height']), path
        assert image_sha256(path) == row['image_sha256'], path
        annotation = path.with_suffix('.json')
        if row.get('annotation_sha256'):
            assert image_sha256(annotation) == row['annotation_sha256'], annotation
        truth, origin = load_evaluation_truth(annotation, *size, names, 'error')
        assert truth == row['ground_truth'], f'Frozen GT differs: {path}'
        items.append(dict(path=str(path), relative_path=row['relative_path'], width=size[0], height=size[1],
                          image_sha256=row['image_sha256'], ground_truth=truth, ground_truth_source=origin))
    return raw, manifest, items


def prepare(output):
    raw, source, items = cohort('chache_val300')
    assert len(items) == 300
    fingerprint = digest([dict(relative_path=r['relative_path'], image_sha256=r['image_sha256'],
                               ground_truth=r['ground_truth']) for r in items])
    plan = dict(schema='wedetect.prompt_comparison.v1', labels=raw['labels'], groups=prompt_groups(raw['labels']),
                image_count=len(items), dataset_sha256=fingerprint,
                source_manifest=str(ROOT / 'data/raw/chache_val300/manifest.json'),
                source_manifest_sha256=image_sha256(ROOT / 'data/raw/chache_val300/manifest.json'),
                ground_truth_counts=dict(Counter(g['label'] for row in items for g in row['ground_truth'])),
                negative_image_count=sum(not row['ground_truth'] for row in items),
                protocol='Five independent single-class queries per image, combined class-aware evaluation. '
                         'No cross-class NMS; 100 candidates/class. short intentionally shares 叉车 for Chache1/4.',
                evaluation=dict(iou_threshold=.5, score_threshold=.5, score_thresholds=POINTS,
                                max_detections_per_image=100), items=items)
    runtime = runtime_dir(output)
    if (runtime / 'plan.json').exists() or (output / 'plan.json').exists():
        old = read_plan(output)
        assert fingerprint in {old['dataset_sha256'], old.get('legacy_dataset_sha256')} and old['groups'] == plan['groups'], 'Existing plan differs'
    plan = reserve_runs(output, plan)
    write_json(runtime / 'plan.json', plan)
    for label in raw['labels']:
        name = label['name']
        local = {**source, 'image_root': raw['dataset']['image_dir'],
                 'ground_truth_root': raw['dataset']['ground_truth_dir'],
                 'samples': [{**row, 'ground_truth': [g for g in row['ground_truth'] if g['label'] == name]}
                             for row in source['samples']]}
        manifest_path = runtime / 'manifests' / f'{name}.json'
        write_json(manifest_path, local)
        for group, prompts in plan['groups'].items():
            config = dict(extends=str(ROOT / 'configs/common/chache_val300.yaml'), labels=[label],
                          scene_name=f'chache_wedetect_{group}_{name}', task_id=name,
                          backend='wedetect', mode='train',
                          dataset=dict(manifest_file=str(manifest_path)),
                          reference_image_dir=None, support_manifest_dir=None,
                          max_reference_examples=0, max_reference_crops=0, max_reference_negatives=0,
                          output_dir=str(ROOT / 'outputs/runs'), results_layout=False, reuse_predictions=True,
                          prediction_cache_dir=str(runtime / 'cache'),
                          cache_search_dirs=[str(runtime / 'cache_legacy'), str(ROOT / 'outputs/runs/chache_val300')],
                          evaluation=dict(iou_threshold=.5, score_threshold=.5,
                                          score_thresholds=[.5, .7], max_detections_per_image=100),
                          wedetect=dict(source_dir=str(ROOT / 'third_party/wedetect'),
                              weights=str(ROOT / 'weights/wedetect/wedetect_base.pth'), variant='base',
                              worker_python=str(Path.home() / 'miniconda3/envs/industrial-wedetect/bin/python'),
                              language_model_dir=str(ROOT / 'weights/wedetect/xlm-roberta-base'),
                              text_prompt=prompts[name], device='cuda:0', torch_threads=4,
                              confidence=.001, nms_iou=.7, max_detections=100, timeout_seconds=600,
                              exclude_reference_overlap=True))
            write_json(runtime / 'configs' / f'{group}_{name}.local.json', config)
    print(json.dumps({k: v for k, v in plan.items() if k not in {'items', 'labels'}}, ensure_ascii=False, indent=2))


def run_group(output, group, device):
    from industrial_fewshot.config import load_config
    from industrial_fewshot.runner import run
    plan = read_plan(output)
    if group not in plan['groups']:
        raise ValueError(group)
    group_dir = combined_dir(output, group)
    index = runtime_dir(output) / 'indices' / f'{group}.json'
    completed = read_json(index) if index.exists() else {}
    if (group_dir / 'metrics.json').exists() and (group_dir / 'summary.json').exists():
        assert len(completed) == len(LABELS)
        for path in completed.values():
            assert read_json(Path(path) / 'summary.json')['failed_request_count'] == 0
        print(f'GROUP REUSE {group}: {group_dir}', flush=True)
        return
    reports = []
    for label in LABELS:
        saved = Path(completed[label]) if label in completed else group_dir / label
        if (saved / 'metrics.json').is_file() and (saved / 'summary.json').is_file():
            report = read_json(saved / 'summary.json')
            assert report['image_count'] == plan['image_count'] and report['failed_request_count'] == 0
            print(f'REUSE {group}/{label}: {saved}', flush=True)
        else:
            config = load_config(runtime_dir(output) / 'configs' / f'{group}_{label}.local.json')
            config.update(prediction_cache_dir=runtime_dir(output) / 'cache',
                          cache_search_dirs=[*config['cache_search_dirs'], runtime_dir(output) / 'cache_legacy',
                                             ROOT / 'outputs/runs/chache_val300'])
            config['targets'][0]['config']['wedetect']['device'] = device
            config['resolved_config']['wedetect']['device'] = device
            destination = group_dir / label
            attempt = 2
            while destination.exists():
                destination = group_dir / f'{label}_attempt{attempt}'
                attempt += 1
            report = asyncio.run(run(config, result_dir=destination))
            assert report['failed_request_count'] == 0, report['run_dir']
        reports.append(report)
        completed[label] = report['run_dir']
        write_json(index, completed)
        print(f'CLASS COMPLETE {group}/{label}: AP50={report["metrics"]["mAP50"]:.6f}', flush=True)
    records = []
    targets = []
    for report in reports:
        records.extend(json.loads(line) for line in (Path(report['run_dir']) / 'records.jsonl').read_text().splitlines() if line)
        targets.extend(report['targets'])
    assert len(records) == plan['image_count'] * len(LABELS)
    assert len({(r['relative_path'], r['target_id']) for r in records}) == len(records)
    item_map = {r['relative_path']: r for r in plan['items']}
    for record in records:
        assert record['image_sha256'] == item_map[record['relative_path']]['image_sha256']
    metrics = evaluate_records(plan['items'], records, LABELS, plan['evaluation'])
    merged = group_dir
    report = {**reports[0], 'run_dir': str(merged), 'targets': targets, 'metrics': metrics,
              'scene_name': f'chache_val300_wedetect_{group}', 'evaluation': plan['evaluation'],
              'success_request_count': len(records), 'failed_request_count': 0,
              'prompt_group': group, 'class_prompts': plan['groups'][group],
              'dataset_sha256': plan['dataset_sha256'], 'dataset_manifest': plan['dataset_manifest'],
              'dataset_manifest_sha256': image_sha256(Path(plan['dataset_manifest'])),
              'result_layout': plan['runs'][group]['result_layout'], 'output_dir': str(merged.parent),
              'prompt_index_root': str(ROOT / 'outputs/runs'), 'protocol': plan['protocol']}
    write_json(merged / 'run.json', {k: v for k, v in report.items() if k != 'metrics'})
    write_json(merged / 'metrics.json', metrics)
    write_json(merged / 'summary.json', report)
    (merged / 'records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    (merged / 'prompt.txt').write_text('\n'.join(f'{k}: {v}' for k, v in plan['groups'][group].items()) + '\n', encoding='utf-8')
    (merged / 'evaluation.txt').write_text(format_metrics(metrics) + '\n', encoding='utf-8')
    version = Path(plan['runs'][group]['result_layout']['prompt_registry_dir']) / 'prompt_version.json'
    write_json(merged / 'prompt_version.json', read_json(version))
    write_json(merged / 'prompt.json', dict(schema='wedetect.multi_query_prompt.v1',
        prompts=plan['groups'][group], snapshots=[target['prompt_snapshot'] for target in targets]))
    write_json(merged / 'experiment.json', dict(group=group, description=GROUP_NAMES[group],
        protocol=plan['protocol'], comparison_dir=str(output),
        class_configs={label:str(runtime_dir(output)/'configs'/f'{group}_{label}.local.json') for label in LABELS},
        class_run_dirs=completed))
    from industrial_fewshot.prompt_records import refresh_prompt_index
    for root in (ROOT / 'outputs/runs', merged.parent.parent, merged.parent):
        refresh_prompt_index(root)
    refresh_method_results(merged.parent)
    print(f'GROUP COMPLETE {group}: mAP50={metrics["mAP50"]:.6f}, mAP50_95={metrics["mAP50_95"]:.6f}', flush=True)


def row_for(group, metrics):
    row = dict(group=group, mAP50=metrics['mAP50'], mAP50_95=metrics['mAP50_95'],
               precision_05=metrics['precision'], recall_05=metrics['recall'], f1_05=metrics['f1'])
    for key, point in metrics['score_threshold_results'].items():
        p, r = point['precision'], point['recall']
        row.update({f'precision_{key}': p, f'recall_{key}': r, f'f1_{key}': 2*p*r/(p+r) if p+r else 0})
    for label, cat in metrics['per_class'].items():
        row[f'{label}_AP50'] = cat['AP50']
    return row


def refresh_method_results(method_dir):
    """One row per five-class experiment, rather than repeating its mAP per class."""
    rows = []
    for path in sorted(method_dir.glob('*/run.json')):
        directory = path.parent
        if not (directory / 'metrics.json').is_file():
            continue
        report = read_json(path)
        metrics = read_json(directory / 'metrics.json')
        group = report.get('prompt_group')
        rows.append(dict(run_dir=directory.name, prompt_group=group,
            description=GROUP_NAMES.get(group, report.get('result_layout', {}).get('prompt_version', '')),
            image_count=report['image_count'], dataset_sha256=report['dataset_sha256'],
            **row_for(group, metrics)))
    write_json(method_dir / 'experiment_results.json', dict(method='WeDetect', runs=rows))
    lines = ['# WeDetect 实验结果', '',
        '每行是一次完整实验的合并指标；五类提示词、单类指标及预测位于对应实验目录。', '',
        '| 实验 | 提示词 | 图片数 | mAP50 | mAP50:95 | P@0.5 | R@0.5 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| [{row["run_dir"]}]({row["run_dir"]}/evaluation.txt) | {row["description"]} | '
            f'{row["image_count"]} | {100*row["mAP50"]:.2f}% | {100*row["mAP50_95"]:.2f}% | '
            f'{100*row["precision_05"]:.2f}% | {100*row["recall_05"]:.2f}% |')
    (method_dir / 'experiment_results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def diagnose_diaowu(output):
    _, _, items = cohort('diaowu_val300')
    item_map = {r['relative_path']: r for r in items}
    rows = []
    for directory in sorted((ROOT / 'outputs/runs/diaowu_val300').glob('*/wedetect_base_0shot')):
        records = [json.loads(line) for line in (directory / 'records.jsonl').read_text().splitlines() if line]
        assert len(records) == len(items)
        assert all(r['image_sha256'] == item_map[r['relative_path']]['image_sha256'] for r in records)
        scores = [obj['confidence'] for record in records for obj in record.get('objects', [])]
        metrics = evaluate_records(items, records, ['hanging object'],
                                   dict(iou_threshold=.5, score_threshold=.5, score_thresholds=POINTS))
        from industrial_fewshot.evaluation.detection_metrics import box_iou
        by_path = {r['relative_path']: r.get('objects', []) for r in records}
        best_ious = [max((box_iou(g['bbox_pixels_xyxy'], p['bbox_pixels_xyxy'])
                          for p in by_path[item['relative_path']]), default=0)
                     for item in items for g in item['ground_truth']]
        rows.append(dict(run_dir=str(directory), prompt=(directory / 'prompt.txt').read_text().strip(),
                         maximum_score=max(scores, default=0), candidate_count=len(scores),
                         candidate_gt_coverage_iou50=sum(v >= .5 for v in best_ious) / len(best_ious),
                         candidate_gt_coverage_policy='Diagnostic upper bound only: each GT has any saved candidate '
                             'at IoU>=0.5; ignores one-to-one matching and ranking.',
                         metrics=metrics))
    write_json(output / 'diaowu_diagnosis.json', rows)
    return rows


def previews(output, plan, results):
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path('/usr/share/fonts/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    selected = []
    for label in LABELS:
        candidates = [row for row in plan['items'] if any(g['label'] == label for g in row['ground_truth'])]
        # Include both small targets and large targets for each class.
        candidates.sort(key=lambda row: max((g['bbox_pixels_xyxy'][2]-g['bbox_pixels_xyxy'][0]) *
                                            (g['bbox_pixels_xyxy'][3]-g['bbox_pixels_xyxy'][1])
                                            for g in row['ground_truth'] if g['label'] == label))
        for row in (candidates[0], candidates[-1]):
            if row['relative_path'] not in {r['relative_path'] for r in selected}:
                selected.append(row)
    selected.extend([row for row in plan['items'] if not row['ground_truth']][:2])
    write_json(output / 'previews/selected_images.json', selected)
    for group, metrics in results.items():
        records = [json.loads(line) for line in (combined_dir(output, group) / 'records.jsonl').read_text().splitlines() if line]
        by_path = {}
        for record in records:
            by_path.setdefault(record['relative_path'], []).extend(record.get('objects', []))
        best = max(metrics['score_threshold_results'].values(), key=lambda p: (
            2*p['precision']*p['recall']/(p['precision']+p['recall']) if p['precision']+p['recall'] else 0))
        threshold = best['settings']['score_threshold']
        for score in sorted({.5, threshold}):
            cells = []
            for item in selected:
                image = Image.open(item['path']).convert('RGB')
                ratio = min(640/image.width, 360/image.height)
                image = image.resize((round(image.width*ratio), round(image.height*ratio)))
                cell = Image.new('RGB', (640, 400), '#181818')
                cell.paste(image, (0, 35))
                draw = ImageDraw.Draw(cell)
                draw.text((6, 3), f'{group} score>={score:g} {Path(item["relative_path"]).stem}', font=font, fill='white')
                for objs, color in ((item['ground_truth'], 'lime'), (by_path[item['relative_path']], 'red')):
                    for obj in objs:
                        if color == 'red' and obj['confidence'] < score:
                            continue
                        box = [v*ratio + (35 if i%2 else 0) for i, v in enumerate(obj['bbox_pixels_xyxy'])]
                        draw.rectangle(box, outline=color, width=2)
                        text = obj['label'] + (f' {obj["confidence"]:.3f}' if color == 'red' else ' GT')
                        draw.text((box[0], max(35, box[1]-18)), text, font=font, fill=color)
                cells.append(cell)
            sheet = Image.new('RGB', (640*3, 400*((len(cells)+2)//3)), '#181818')
            for index, cell in enumerate(cells):
                sheet.paste(cell, ((index%3)*640, (index//3)*400))
            sheet.save(output / 'previews' / f'{group}_score_{score:g}.jpg', quality=90)


def summarize(output):
    plan = read_plan(output)
    results = {group: read_json(combined_dir(output, group) / 'metrics.json') for group in plan['groups']}
    rows = [row_for(group, metrics) for group, metrics in results.items()]
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    diagnosis = diagnose_diaowu(output)
    from industrial_fewshot.postprocess.boxes import nms
    localization = {}
    for group in results:
        by_path = {item['relative_path']: [] for item in plan['items']}
        for line in (combined_dir(output, group) / 'records.jsonl').read_text().splitlines():
            if line:
                record = json.loads(line)
                by_path[record['relative_path']].extend({**obj, 'label':'vehicle'} for obj in record.get('objects', []))
        items = [{**item, 'ground_truth':[{**g, 'label':'vehicle'} for g in item['ground_truth']]} for item in plan['items']]
        records = [dict(relative_path=path, target_id='vehicle', status='success', objects=nms(objs, .7, 100))
                   for path, objs in by_path.items()]
        localization[group] = evaluate_records(items, records, ['vehicle'], plan['evaluation'])
    write_json(output / 'localization_diagnostic.json',
               dict(protocol='Diagnostic only: collapse five GT/prediction labels to vehicle, NMS=0.7/maxDets=100; '
                             'not a five-class identification metric.', metrics=localization))
    write_json(output / 'comparison.json', dict(plan={k:v for k,v in plan.items() if k != 'items'}, rows=rows,
              groups=[dict(id=group, run_dir=str(combined_dir(output, group))) for group in results],
              metrics=results, localization_diagnostic=localization, diaowu=diagnosis))
    lines = ['# WeDetect 五类特种车辆提示词对比', '',
             '数据目录沿用 chache_val300；类别为 Chache1、Chache2、Chache3、Chache4、Qianyinche。', '',
             f'固定 {plan["image_count"]} 张图片，五类共 {sum(plan["ground_truth_counts"].values())} 个 GT 框；'
             f'{plan["negative_image_count"]} 张负例。输入分辨率 640，候选分数下限 0.001，NMS 0.7。',
             '每类独立查询，再合并为五类结果；没有按 GT 选择查询类别，不提供参考图片。',
             'short 组对 Chache1 和 Chache4 均使用“叉车”，因此无法区分这两个细类，这是粗类别基线。', '',
             '| group | mAP50 (%) | mAP50:95 (%) | P@0.5 (%) | R@0.5 (%) |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["group"]} | {100*row["mAP50"]:.3f} | {100*row["mAP50_95"]:.3f} | '
                     f'{100*row["precision_05"]:.2f} | {100*row["recall_05"]:.2f} |')
    lines += ['', '## 实际提示词', '', '| 类别 | short | names | phrases | long |', '|---|---|---|---|---|']
    for label in LABELS:
        lines.append('| ' + label + ' | ' + ' | '.join(plan['groups'][g][label] for g in results) + ' |')
    lines += ['', '## 分类 AP50 (%)', '', '| group | ' + ' | '.join(LABELS) + ' |', '|---|' + '---:|'*len(LABELS)]
    for group, metrics in results.items():
        lines.append('| ' + group + ' | ' + ' | '.join(f'{100*metrics["per_class"][label]["AP50"]:.3f}' for label in LABELS) + ' |')
    lines += ['', '## 阈值分析', '', '以下阈值来自同一验证集的探索性比较，不是独立测试集上验证过的部署阈值。', '',
              '| group | score | P (%) | R (%) | F1 (%) | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, metrics in results.items():
        for point in metrics['score_threshold_results'].values():
            p, r = point['precision'], point['recall']
            f1 = 2*p*r/(p+r) if p+r else 0
            lines.append(f'| {group} | {point["settings"]["score_threshold"]:g} | {100*p:.2f} | {100*r:.2f} | '
                         f'{100*f1:.2f} | {point["tp"]} | {point["fp"]} | {point["fn"]} |')
    lines += ['', '## diaowu 已有结果复核', '']
    for row in diagnosis:
        m = row['metrics']
        lines += [f'- `{row["prompt"]}`：AP50={100*m["mAP50"]:.4f}%，最高分={row["maximum_score"]:.6f}，'
                  f'候选框={row["candidate_count"]}，P/R@0.5={100*m["precision"]:.2f}%/{100*m["recall"]:.2f}%。']
        point = m['score_threshold_results']['0.001']
        lines += [f'  候选下限 0.001 时，P/R={100*point["precision"]:.2f}%/{100*point["recall"]:.2f}%；'
                  f'忽略排序和一对一匹配后的候选 GT 覆盖上限为 {100*row["candidate_gt_coverage_iou50"]:.2f}%（IoU>=0.5，仅诊断）。']
    lines += ['', '## 粗定位诊断', '',
              '将五类 GT 与预测标签均折叠为 vehicle，再做 NMS=0.7/maxDets=100；只衡量粗定位，不能与五类 mAP 混用。', '',
              '| group | coarse vehicle AP50 (%) |', '|---|---:|']
    for group, metrics in localization.items():
        lines.append(f'| {group} | {100*metrics["mAP50"]:.3f} |')
    lines += ['', '提示词长度和措辞同时变化，因此不能将差异全部归因于长度；原始长描述还包含颜色、姿态、否定条件及不同类别名。']
    lines += ['', 'AP 使用全部已保存候选分数，P/R 取指定阈值。降阈值只能恢复低分框，不能直接修复很低的 AP。',
              '画框对比：`previews/`，绿色为 GT，红色为预测。每组均有固定 0.5 和该组验证集最佳 F1 阈值的相同图片。',
              '完整结果位于 `outputs/runs/chache_val300/WeDetect/wedetect_base_0shot_prompt_vN/`，'
              '其下 Chache1～Qianyinche 为分类明细；准备文件、缓存和日志位于 `outputs/runtime/`。', '',
              '## 运行目录', '']
    for group in results:
        relative = combined_dir(output, group).relative_to(ROOT)
        lines.append(f'- {group}（{GROUP_NAMES[group]}）：[{relative}]({combined_dir(output, group)})')
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    previews(output, plan, results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(LABELS)); width=.19
    for index, (group, metrics) in enumerate(results.items()):
        axes[0].bar(x+(index-1.5)*width, [100*metrics['per_class'][label]['AP50'] for label in LABELS], width, label=group)
        points = list(metrics['score_threshold_results'].values())
        axes[1].plot([p['settings']['score_threshold'] for p in points],
                     [2*p['precision']*p['recall']/(p['precision']+p['recall'])*100 if p['precision']+p['recall'] else 0 for p in points],
                     marker='o', label=group)
    axes[0].set_xticks(x, LABELS); axes[0].set_ylabel('AP50 (%)'); axes[0].set_title('Class-aware AP50')
    axes[1].set_xscale('log'); axes[1].set_xlabel('Score threshold'); axes[1].set_ylabel('F1 (%)'); axes[1].set_title('Validation operating points')
    for ax in axes:
        ax.legend(); ax.grid(axis='y', alpha=.25)
    fig.tight_layout(); fig.savefig(output / 'comparison.png', dpi=160); plt.close(fig)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f'REPORT: {output / "report.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'group', 'run-all', 'complete', 'summarize', 'status'], required=True)
    parser.add_argument('--group', choices=['short', 'names', 'phrases', 'long'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/comparisons/wedetect_chache_prompts_20261027',
                        help='Comparison report directory; model results are stored under outputs/runs')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_dir(output)
    runtime.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.phase == 'status':
        status = runtime / 'workflow_status.json'
        print(json.dumps(read_json(status), ensure_ascii=False) if status.exists() else 'No workflow status')
        for group in ('short', 'names', 'phrases', 'long'):
            saved = runtime / 'indices' / f'{group}.json'
            complete = read_json(saved) if saved.exists() else {}
            log = runtime / 'logs' / f'{group}.log'
            lines = log.read_text(errors='replace').splitlines() if log.exists() else []
            print(f'{group}: {len(complete)}/5 classes complete; {lines[-1] if lines else "not started"}')
    elif args.phase == 'prepare':
        prepare(output)
    elif args.phase == 'group':
        run_group(output, args.group, args.device)
    elif args.phase == 'complete':
        status_path = runtime / 'workflow_status.json'
        write_json(status_path, dict(status='running', phase='inference'))
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--phase', 'run-all',
                            '--output', str(output)], cwd=ROOT, check=True)
            write_json(status_path, dict(status='running', phase='report'))
            summarize(output)
        except BaseException as exc:
            write_json(status_path, dict(status='failed', error=f'{type(exc).__name__}: {exc}'))
            raise
        write_json(status_path, dict(status='complete', report=str(output / 'report.md')))
    elif args.phase == 'run-all':
        children = []
        try:
            (runtime / 'logs').mkdir(parents=True, exist_ok=True)
            for index, group in enumerate(read_plan(output)['groups']):
                log = (runtime / 'logs' / f'{group}.log').open('a', encoding='utf-8')
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--phase', 'group',
                    '--group', group, '--device', f'cuda:{index}', '--output', str(output)],
                    cwd=ROOT, env={**os.environ, 'PYTHONNOUSERSITE':'1', 'OMP_NUM_THREADS':'4',
                                   'OPENBLAS_NUM_THREADS':'4', 'PYTHONUNBUFFERED':'1'},
                    stdout=log, stderr=subprocess.STDOUT)
                children.append((group, process, log))
                print(f'START {group} on cuda:{index}, PID {process.pid}', flush=True)
            failed = []
            for group, process, log in children:
                code = process.wait()
                print(f'FINISH {group}: exit={code}', flush=True)
                if code:
                    failed.append(group)
            if failed:
                raise RuntimeError(f'Groups failed: {failed}; inspect group logs')
        except KeyboardInterrupt:
            for _, process, _ in children:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            raise
        finally:
            for _, _, log in children:
                log.close()
    else:
        summarize(output)


if __name__ == '__main__':
    main()

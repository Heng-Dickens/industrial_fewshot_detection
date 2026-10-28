"""Offline AP85/90/95 and GT-relative 20% box-tolerance reports from saved runs.

Reads consolidated results_summary.json indexes; never invokes a model or rewrites
source metrics. Missing raw coordinates remain unavailable, rather than zero.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from urllib.parse import quote
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from industrial_fewshot.evaluation.detection_metrics import _curve, box_iou

IOU_LEVELS = (0.85, 0.90, 0.95)
SCORE_LEVELS = tuple(i / 10 for i in range(11))
OVD_METHODS = frozenset({'GroundingDINO', 'PET-DINO', 'WeDetect', 'YOLOE'})


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def sha256(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def saved_ap(metrics):
    """Macro AP across classes with GT, preserving unavailable class values."""
    classes = {k: v for k, v in metrics.get('per_class', {}).items()
               if v.get('ground_truth_count', 0) > 0}
    per_class, result = {}, {}
    for label, values in classes.items():
        grid = {float(k): v for k, v in values.get('ap_by_iou', {}).items()}
        per_class[label] = {f'AP{round(t * 100)}': grid.get(t) for t in IOU_LEVELS}
    for t in IOU_LEVELS:
        key = f'AP{round(t * 100)}'
        values = [v[key] for v in per_class.values()]
        result[key] = sum(values) / len(values) if values and all(v is not None for v in values) else None
    result['AP85_95'] = (sum(result[f'AP{round(t * 100)}'] for t in IOU_LEVELS) / 3
                         if all(result[f'AP{round(t * 100)}'] is not None for t in IOU_LEVELS) else None)
    result['class_AP'] = per_class
    return result


def select_ovd_best(indexed_rows, root):
    """One eligible AP85 winner per OVD method; exclude MLLM and failed runs."""
    best = {}
    for index, row in indexed_rows:
        method = row['method']
        if method not in OVD_METHODS or not row.get('eligible_for_comparison'):
            continue
        aps = saved_ap(read(root / row['run_dir'] / 'metrics.json'))
        if aps['AP85'] is None:
            continue
        key = (aps['AP85'], aps['AP85_95'] if aps['AP85_95'] is not None else -1)
        if method not in best or key > best[method][0]:
            best[method] = (key, index, row)
    return sorted((index, row) for _, index, row in best.values())


def box_errors(prediction, truth):
    px1, py1, px2, py2 = prediction
    gx1, gy1, gx2, gy2 = truth
    gw, gh = gx2 - gx1, gy2 - gy1
    if gw <= 0 or gh <= 0:
        raise ValueError('GT box must have positive width and height')
    return dict(width=abs((px2 - px1) - gw) / gw,
                height=abs((py2 - py1) - gh) / gh,
                center_x=abs((px1 + px2 - gx1 - gx2) / 2) / gw,
                center_y=abs((py1 + py2 - gy1 - gy2) / 2) / gh)


def tolerance_ok(prediction, truth, tolerance=0.2):
    return all(v <= tolerance + 1e-12 for v in box_errors(prediction, truth).values())


def ranked_predictions(sample, label, lower, upper, max_dets):
    # Apply the same score-ranked per-image/category cap as AP, then slice scores.
    ranked = sorted((p for p in sample['predictions'] if p['label'] == label),
                    key=lambda p: -p['confidence'])[:max_dets]
    return [p for p in ranked if p['confidence'] >= lower
            and (upper is None or p['confidence'] < upper)]


def match_counts(samples, labels, lower, upper=None, max_dets=100, tolerance=0.2):
    """Independent class-aware greedy associations for IoU and box tolerances.

    A prediction can claim at most one unclaimed GT; highest confidence goes
    first, with maximum IoU among *eligible* GT and existing AP tie-breaking.
    No additional IoU gate is imposed on the 20% geometric acceptance rule.
    """
    total = Counter(prediction_count=0, ground_truth_count=0, tp85=0, tp50=0,
                    joint20_count=0, joint20_iou85_count=0,
                    size20_on_iou50_count=0, offset20_on_iou50_count=0)
    per_class = {}
    for label in labels:
        counts = Counter({k: 0 for k in total})
        for sample in samples:
            gt = [g for g in sample['ground_truth'] if g['label'] == label]
            predictions = ranked_predictions(sample, label, lower, upper, max_dets)
            counts['prediction_count'] += len(predictions)
            counts['ground_truth_count'] += len(gt)
            claimed = {kind: set() for kind in ('iou50', 'iou85', 'tol20', 'both')}
            for prediction in predictions:
                box = prediction['bbox_pixels_xyxy']
                candidates = [(box_iou(box, g['bbox_pixels_xyxy']), j,
                               box_errors(box, g['bbox_pixels_xyxy'])) for j, g in enumerate(gt)]
                for kind in claimed:
                    valid = []
                    for overlap, j, errors in candidates:
                        if j in claimed[kind]:
                            continue
                        good20 = all(v <= tolerance + 1e-12 for v in errors.values())
                        accepted = (overlap >= 0.5 if kind == 'iou50' else
                                    overlap >= 0.85 if kind == 'iou85' else
                                    good20 if kind == 'tol20' else good20 and overlap >= 0.85)
                        if accepted:
                            valid.append((overlap, j, errors))
                    if not valid:
                        continue
                    _, j, errors = max(valid, key=lambda entry: (entry[0], entry[1]))
                    claimed[kind].add(j)
                    field = dict(iou50='tp50', iou85='tp85', tol20='joint20_count',
                                 both='joint20_iou85_count')[kind]
                    counts[field] += 1
                    if kind == 'iou50':
                        counts['size20_on_iou50_count'] += (errors['width'] <= tolerance + 1e-12
                                                           and errors['height'] <= tolerance + 1e-12)
                        counts['offset20_on_iou50_count'] += (errors['center_x'] <= tolerance + 1e-12
                                                             and errors['center_y'] <= tolerance + 1e-12)
        total.update(counts)
        per_class[label] = dict(counts)
    result = dict(total)
    for prefix, count in (('iou85', total['tp85']), ('joint20', total['joint20_count'])):
        result[prefix + '_prediction_rate'] = (count / total['prediction_count']
                                               if total['prediction_count'] else None)
        result[prefix + '_gt_coverage'] = (count / total['ground_truth_count']
                                           if total['ground_truth_count'] else None)
    result['fp85'] = total['prediction_count'] - total['tp85']
    result['fn85'] = total['ground_truth_count'] - total['tp85']
    result['per_class'] = per_class
    return result


def portable_file(value, project):
    if not value:
        return None
    normalized = str(value).replace('\\', '/')
    path = Path(normalized)
    if path.is_file():
        return path
    for prefix in ('data/', 'outputs/'):
        position = normalized.find('/' + prefix)
        relative = normalized[position + 1:] if position >= 0 else normalized
        if relative.startswith(prefix) and (project / relative).is_file():
            return project / relative
    return None


def load_manifest(marker, project):
    name = marker['name']
    candidates = [portable_file(marker.get('dataset_manifest'), project),
                  project / f'data/raw/{name}/manifest.json']
    if name == 'diaowu_99':
        candidates.append(project / 'data/splits/diaowu/test.json')
    for path in candidates:
        if path is None or not path.is_file():
            continue
        manifest = read(path)
        samples = sorted(manifest.get('samples', []), key=lambda s: s['relative_path'])
        if not samples:
            continue
        if len({s['relative_path'] for s in samples}) != len(samples):
            raise ValueError('GT manifest contains duplicate paths')
        fingerprints = []
        for as_float in (False, True):
            images = []
            for sample in samples:
                truth = sample['ground_truth']
                if as_float:
                    truth = [dict(g, bbox_pixels_xyxy=list(map(float, g['bbox_pixels_xyxy']))) for g in truth]
                images.append(dict(relative_path=sample['relative_path'],
                                   image_sha256=sample['image_sha256'], ground_truth=truth))
            fingerprints.append(digest(dict(images=images)))
        if marker['dataset_sha256'] in fingerprints:
            return samples, path
    raise ValueError('未找到与数据指纹一致的 GT 清单（已兼容整数/浮点坐标序列化）')


def load_records(directory, run, components, project):
    path = directory / 'records.jsonl'
    if path.is_file():
        records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        return records, [path]
    summary_path = directory / 'summary.json'
    summary = read(summary_path)
    if isinstance(summary.get('annotations'), list):
        return summary['annotations'], [summary_path]
    if components:
        records, files = [], []
        dataset_root = project / 'outputs/runs' / directory.relative_to(project / 'outputs/runs').parts[0]
        for relative in components:
            child_dir = dataset_root / relative
            child_run = read(child_dir / 'run.json')
            child_records, child_files = load_records(child_dir, child_run, [], project)
            records.extend(child_records)
            files.extend(child_files)
        return records, files
    raise ValueError('缺少预测坐标：本机无 records.jsonl/含 annotations 的 summary.json；需在原运行机器计算')


def build_samples(manifest, records, run, labels):
    expected = {s['relative_path']: s for s in manifest}
    paths = run.get('image_paths') or sorted(expected)
    if len(paths) != len(set(paths)) or set(paths) != set(expected):
        raise ValueError('运行图片清单与 GT 清单不同，不能混用数据版本')
    targets = [t['id'] for t in run.get('targets', [])]
    if not targets:
        raise ValueError('运行元数据缺少 target IDs')
    latest, duplicates = {}, 0
    for record in records:
        name = record.get('relative_path')
        target = record.get('target_id') or (targets[0] if len(targets) == 1 else None)
        if name not in expected or target not in targets:
            raise ValueError('预测记录包含未知图片或 target ID')
        if record.get('status') not in ('success', 'failed'):
            raise ValueError('预测记录状态无效')
        if record.get('image_sha256') and record['image_sha256'] != expected[name]['image_sha256']:
            raise ValueError('预测图片哈希与 GT 清单不一致')
        key = (name, target)
        duplicates += key in latest
        latest[key] = record
    if set(latest) != {(name, target) for name in expected for target in targets}:
        raise ValueError(f'逐图记录不完整：{len(latest)}/{len(expected) * len(targets)}；不把缺失图片当成空预测')
    samples = []
    for name in sorted(expected):
        sample = expected[name]
        predictions = []
        for target in targets:
            record = latest[name, target]
            if record['status'] != 'success':
                continue
            for prediction in record.get('objects', []):
                if prediction.get('label') not in labels:
                    raise ValueError('预测类别不在保存的评估类别内')
                confidence = prediction.get('confidence')
                box = prediction.get('bbox_pixels_xyxy')
                if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                        or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                    raise ValueError('预测置信度无效或缺失')
                if (not isinstance(box, list) or len(box) != 4
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in box)
                        or not 0 <= box[0] < box[2] <= sample['width']
                        or not 0 <= box[1] < box[3] <= sample['height']):
                    raise ValueError('预测像素坐标缺失、退化或越界')
                predictions.append(prediction)
        samples.append(dict(relative_path=name, width=sample['width'], height=sample['height'],
                            ground_truth=[g for g in sample['ground_truth'] if g['label'] in labels],
                            predictions=predictions))
    return samples, dict(records=len(latest), duplicate_records_using_last=duplicates,
                         successful_requests=sum(r['status'] == 'success' for r in latest.values()),
                         failed_requests=sum(r['status'] == 'failed' for r in latest.values()))


def csv_data(rows):
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    columns = list(dict.fromkeys(k for row in flat for k in row))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(flat)
    return buffer.getvalue().encode('utf-8-sig')


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}'


def counts_cell(row, score):
    point = row.get('score_thresholds', {}).get(f'{score:g}')
    if point is None:
        return '—'
    return (f"{point['joint20_count']}/{point['prediction_count']}"
            f"（{pct(point['joint20_prediction_rate'])}%）；GT {pct(point['joint20_gt_coverage'])}%")


def markdown(data):
    title = '# OVD 最佳实验：AP85 与框几何误差 20%' if data.get('ovd_best') else '# AP85 与框几何误差 20% 统计'
    lines = [title, '',
             f"生成时间：{data['generated_at']}（北京时间）。读取 {len(data['rows'])} 组独立实验，保留各数据集和原状态。", '',
             '**口径**', '',
             '- AP85/90/95：同类别预测按置信度排序，一对一匹配，101 点插值 AP；多类取有 GT 类别的宏平均。直接读取原指标中的 `ap_by_iou`，不先按 IoU 筛掉误检。',
             '- AP85:95 = (AP85 + AP90 + AP95) / 3。AP 使用全部已保存候选分数；置信度阈值统计是另外的工作点，不是 AP。候选在模型输出前被删掉的部分无法恢复。',
             '- 宽误差 = |预测宽 − GT宽| / GT宽；高误差同理。横向中心偏移 = |预测中心x − GT中心x| / GT宽；纵向偏移以 GT高归一化。四项均 ≤ 20% 才合格；未额外要求 IoU ≥ 0.85。',
             '- 同类预测按置信度降序，一张图内每个 GT 最多使用一次；从满足四项误差约束且未匹配的 GT 中选最大 IoU。误检、重复框、错分类不重复计数。',
             '- 数量表为“合格框/选中预测框数（预测合格率）；GT覆盖率”。漏检保留在 GT 分母。无预测时合格数为 0、预测合格率未定义，GT覆盖率为 0。每图每类沿用保存的 maxDets 上限。',
             '- `confidence_thresholds.csv` 给出 score≥0.0/0.1/…/1.0 的累计统计；`confidence_bins.csv` 给出独立分数区间（左闭右开，最后区间含1.0）。分箱各自重新一对一匹配，不可将分箱合格数相加当成累计结果。',
             '- CSV 中带 `_pct` 的字段为百分数，其余率字段和 JSON 使用 0–1。缺原始坐标显示“—”，不能从 AP、TP/FP/FN 或平均 IoU 反推误差合格数。',
             '- `size20_on_iou50_count` / `offset20_on_iou50_count` 是 IoU≥0.5 匹配框中分别满足宽高/中心偏移约束的辅助数量；`joint20_count` 独立按四项联合约束匹配。', '',
             '[AP及主统计 CSV](ap_summary.csv) · [置信度累计 CSV](confidence_thresholds.csv) · '
             '[置信度分箱 CSV](confidence_bins.csv) · [各类别AP CSV](class_ap.csv) · [完整汇总JSON](analysis.json)', '']
    if data.get('ovd_best'):
        lines += ['仅比较 GroundingDINO、PET-DINO、WeDetect、YOLOE；每个数据集、每种方法选取允许全数据集比较的完整实验中 AP85 最高的一组。'
                  'AP85 并列时比较 AP85:95，再并列保持原总表顺序。这里的最佳配置按 AP85 选择，可能与 AP50 最佳配置不同。', '']
    for dataset in data['datasets']:
        rows = [r for r in data['rows'] if r['dataset'] == dataset]
        lines += [f'## {dataset}', '',
                  f"共 {len(rows)} 组；保存 AP85 的 {sum(r['AP85'] is not None for r in rows)} 组；"
                  f"完成框误差统计的 {sum(r['geometry_status'] == '可计算' for r in rows)} 组。", '',
                  '| # | 方法/模型 | prompt / shot / 输入 / 负例 | AP85% | AP90% | AP95% | AP85:95% | score≥0.5 四项≤20% | score≥0.7 四项≤20% | 状态 |',
                  '|---:|---|---|---:|---:|---:|---:|---|---|---|']
        for row in rows:
            path = quote('../../runs/' + dataset + '/' + row['run_dir'] + '/' + row['detail_file'], safe='/')
            description = f"{row['prompt_version']} / {row['shot_per_class']}shot / {row['input']} / {row['negative_references']}负"
            status = row['original_status'] + ('；缺坐标' if row['geometry_status'] != '可计算' else '')
            lines.append('| ' + ' | '.join([row['id'], f"{row['method']} / {row['model']}",
                         f'[{description}]({path})', *[pct(row[k]) for k in ('AP85', 'AP90', 'AP95', 'AP85_95')],
                         counts_cell(row, .5), counts_cell(row, .7), status]) + ' |')
        lines += ['', '**各方法在本数据集的最高 AP85（仅原总表允许比较的完整实验）**', '',
                  '| 方法 | AP85% | AP90% | AP95% | shot / 输入 / 负例 | 实验ID |',
                  '|---|---:|---:|---:|---|---|']
        for method in sorted({r['method'] for r in rows}):
            eligible = [r for r in rows if r['method'] == method and r['eligible_for_comparison'] and r['AP85'] is not None]
            if not eligible:
                continue
            best = max(eligible, key=lambda r: r['AP85'])
            lines.append('| ' + ' | '.join([method, *[pct(best[k]) for k in ('AP85', 'AP90', 'AP95')],
                         f"{best['shot_per_class']} / {best['input']} / {best['negative_references']}", best['id']]) + ' |')
        unavailable = [r for r in rows if r['geometry_status'] != '可计算']
        if unavailable:
            lines += ['', '**无法计算框误差的实验**', '']
            lines += [f"- {r['id']} `{r['run_dir']}`：{r['geometry_status']}。" for r in unavailable]
        inconsistent = [r for r in rows if r.get('source_consistency') not in (None, '一致')]
        if inconsistent:
            lines += ['', '**原始记录与保存指标校验**', '']
            lines += [f"- {r['id']}：{r['source_consistency']}。" for r in inconsistent]
        lines.append('')
    command = 'python scripts/analyze_box_precision.py'
    if data.get('ovd_best'):
        command += ' --ovd-best' + ''.join(' --dataset ' + name for name in data['datasets'])
    lines += ['## 复算与数据来源', '',
              '`' + command + '`；只分析一个数据集可使用 `--dataset diaowu_val300`。', '',
              '原始预测、GT、metrics 和已有 results_summary 均保持原字节。数据来源和 SHA256 记录在 `analysis.json`。'
              '框误差复算要求 GT 清单与运行时数据指纹相符，并验证逐图记录完整性及图像哈希；'
              '没有复核远端机器上尚未同步的原始预测。', '']
    return '\n'.join(lines)


def analyze(outputs, destination, datasets=None, tolerance=0.2, ovd_best=False):
    outputs, destination = Path(outputs).resolve(), Path(destination).resolve()
    project = outputs.parent
    sources, rows, operating_rows, bin_rows, class_rows = {}, [], [], [], []

    def remember(path):
        if path and path.is_file():
            key = path.relative_to(project).as_posix() if path.is_relative_to(project) else str(path)
            sources[key] = sha256(path)

    summaries = sorted((outputs / 'runs').glob('*/results_summary.json'))
    known = {p.parent.name for p in summaries}
    if set(datasets or []) - known:
        raise ValueError('Unknown datasets: ' + ', '.join(sorted(set(datasets) - known)))
    for summary_path in summaries:
        dataset = summary_path.parent.name
        if datasets and dataset not in datasets:
            continue
        summary = read(summary_path)
        remember(summary_path)
        manifest_error = None
        try:
            manifest, manifest_path = load_manifest(summary['dataset'], project)
            remember(manifest_path)
        except (ValueError, OSError, KeyError) as exc:
            manifest, manifest_error = None, str(exc)
        indexed = list(enumerate(summary['rows'], 1))
        if ovd_best:
            for _, candidate in indexed:
                if candidate['method'] in OVD_METHODS:
                    remember(summary_path.parent / candidate['run_dir'] / 'metrics.json')
            indexed = select_ovd_best(indexed, summary_path.parent)
        for index, original in indexed:
            directory = summary_path.parent / original['run_dir']
            metric_path = directory / 'metrics.json'
            metrics, run = read(metric_path), read(directory / 'run.json')
            remember(metric_path)
            remember(directory / 'run.json')
            identity = {k: original.get(k) for k in ('dataset', 'method', 'model', 'prompt_version',
                        'shot_per_class', 'input', 'negative_references', 'run_dir')}
            identity['id'] = f'{dataset}-{index:02}'
            row = dict(identity, **saved_ap(metrics), original_status=original['status'],
                       eligible_for_comparison=original['eligible_for_comparison'],
                       ground_truth_count=original['ground_truth_count'],
                       geometry_status='未计算', score_thresholds={}, source_consistency=None)
            row['detail_file'] = 'metrics.json' if metric_path.is_file() else 'run.json'
            try:
                if manifest_error:
                    raise ValueError(manifest_error)
                if run.get('dataset_sha256') != summary['dataset']['dataset_sha256']:
                    raise ValueError('运行数据指纹与当前数据集不同，未混用 GT')
                labels = list(metrics.get('per_class', {}))
                if not labels:
                    raise ValueError('缺少保存的评估指标/类别范围')
                records, record_files = load_records(directory, run, original.get('component_runs', []), project)
                for path in record_files:
                    remember(path)
                samples, evidence = build_samples(manifest, records, run, labels)
                actual_gt = Counter(g['label'] for s in samples for g in s['ground_truth'])
                saved_gt = {k: v['ground_truth_count'] for k, v in metrics['per_class'].items()}
                if any(actual_gt[k] != v for k, v in saved_gt.items()):
                    raise ValueError('当前 GT 类别框数与原评估不同')
                limit = metrics.get('settings', {}).get('max_detections_per_image', 100)
                raw_aps, discrepancies = {}, []
                for t in (0.5, *IOU_LEVELS):
                    values = [_curve(samples, label, t, limit)['ap'] for label in labels]
                    values = [v for v in values if v is not None]
                    key = f'AP{round(t * 100)}'
                    raw_aps[key] = sum(values) / len(values) if values else None
                    saved_value = metrics.get('mAP50') if t == 0.5 else row[key]
                    if saved_value is not None and (raw_aps[key] is None or abs(raw_aps[key] - saved_value) > 1e-9):
                        discrepancies.append(f'{key}: 原{saved_value:.8f}，本地复算{raw_aps[key]}')
                row.update(raw_recomputed_AP=raw_aps, source_consistency='；'.join(discrepancies) if discrepancies else '一致',
                           source_evidence=evidence)
                if discrepancies:
                    raise ValueError('本地预测与保存 AP 不一致，未发布混用来源的框误差；详见校验说明')
                row['geometry_status'] = '可计算'
                for score in SCORE_LEVELS:
                    point = match_counts(samples, labels, score, max_dets=limit, tolerance=tolerance)
                    row['score_thresholds'][f'{score:g}'] = point
                    operating_rows.append(dict(identity, score_threshold=score,
                                               geometry_status='可计算', original_status=row['original_status'],
                                               eligible_for_comparison=row['eligible_for_comparison'],
                                               source_failed_requests=evidence['failed_requests'], **point))
                for i in range(10):
                    lower = i / 10
                    upper = (i + 1) / 10 if i < 9 else None
                    point = match_counts(samples, labels, lower, upper, limit, tolerance)
                    bin_rows.append(dict(identity, geometry_status='可计算', original_status=row['original_status'],
                                         eligible_for_comparison=row['eligible_for_comparison'],
                                         confidence_lower=lower, confidence_upper=upper if upper else 1.0,
                                         upper_inclusive=i == 9, **point))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                row['geometry_status'] = str(exc)
                # Retain every experiment in confidence CSVs; missing data is blank.
                for score in SCORE_LEVELS:
                    operating_rows.append(dict(identity, score_threshold=score, geometry_status=str(exc),
                                               original_status=row['original_status'],
                                               eligible_for_comparison=row['eligible_for_comparison']))
                for i in range(10):
                    bin_rows.append(dict(identity, confidence_lower=i / 10, confidence_upper=(i + 1) / 10,
                                         upper_inclusive=i == 9, geometry_status=str(exc),
                                         original_status=row['original_status'],
                                         eligible_for_comparison=row['eligible_for_comparison']))
            rows.append(row)
            for label, aps in row['class_AP'].items():
                class_rows.append(dict(identity, label=label, **aps))
        print(dataset + ': ' + str(sum(r['dataset'] == dataset and r['geometry_status'] == '可计算' for r in rows))
              + '/' + str(len(indexed)) + ' geometry runs available', flush=True)
    data = dict(schema='industrial_fewshot.box_precision.v1',
                generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
                tolerance=tolerance, iou_thresholds=IOU_LEVELS, score_thresholds=SCORE_LEVELS,
                ovd_best=ovd_best,
                datasets=sorted({r['dataset'] for r in rows}), rows=rows,
                source_sha256=sources,
                source_policy='Saved AP grid; geometry only on complete source records with matching manifest identity and AP checks.',
                box_error_policy='abs(dw)/GTw, abs(dh)/GTh, abs(dcx)/GTw, abs(dcy)/GTh; all <= tolerance; class-aware score-ranked one-to-one matching.',
                confidence_bins=bin_rows)
    # Assert source files stayed byte-identical throughout analysis.
    for key, expected in sources.items():
        path = project / key
        if sha256(path) != expected:
            raise RuntimeError('Source changed during analysis: ' + key)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'analysis.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    ap_rows = []
    for row in rows:
        item = dict(row)
        for key in ('AP85', 'AP90', 'AP95', 'AP85_95'):
            item[key + '_pct'] = item[key] * 100 if item[key] is not None else None
        for score in (0.5, 0.7):
            point = row['score_thresholds'].get(f'{score:g}', {})
            for key in ('prediction_count', 'joint20_count', 'joint20_prediction_rate', 'joint20_gt_coverage', 'tp85'):
                item[f'{key}_score_{score:g}'] = point.get(key)
        ap_rows.append(item)
    for filename, records in (('ap_summary.csv', ap_rows), ('class_ap.csv', class_rows),
                              ('confidence_thresholds.csv', operating_rows), ('confidence_bins.csv', bin_rows)):
        (destination / filename).write_bytes(csv_data(records))
    (destination / 'README.md').write_text(markdown(data), encoding='utf-8')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=PROJECT / 'outputs')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--dataset', action='append')
    parser.add_argument('--ovd-best', action='store_true', help='Keep the best eligible AP85 run per OVD method/dataset.')
    args = parser.parse_args()
    destination = args.destination or PROJECT / ('outputs/comparisons/ovd_best_iou85_box20_20261028'
                                                if args.ovd_best else 'outputs/comparisons/iou85_box20_20261028')
    result = analyze(args.outputs, destination, args.dataset, ovd_best=args.ovd_best)
    print('Wrote ' + str(destination) + '; ' + str(len(result['rows'])) + ' experiments')

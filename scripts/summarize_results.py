"""Build one saved-experiment table per dataset, without inference or reevaluation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
import os
import copy
import sys
from pathlib import Path
import re
from urllib.parse import quote
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from industrial_fewshot.paths import resolve_location
EXCLUDED = {'runtime', 'legacy', 'logs', 'feedback', 'visualizations', 'archive'}
METHODS = {'pet_dino': 'PET-DINO', 'wedetect': 'WeDetect', 'yoloe': 'YOLOE',
           'yoloe_text': 'YOLOE', 'yoloe_fusion': 'YOLOE', 'groundingdino': 'GroundingDINO',
           'yolo_onnx': 'JSTY-YOLO（监督）', 'ovd_mllm': 'OVD-MLLM'}
RATE_FIELDS = ['AP50', 'AP75', 'AP50_95', *[f'{key}_{suffix}' for suffix in ('0_5', '0_7')
                                           for key in ('precision', 'recall', 'f1', 'f2')]]


def read(path):
    if not path.is_file():
        return {}
    with path.open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def artifacts(root):
    """Visit metadata directories, including the single-class children of merged runs."""
    pending = [root]
    while pending:
        directory = pending.pop()
        if (directory / 'run.json').is_file() or (directory / 'metrics.json').is_file():
            yield directory
        for child in sorted(directory.iterdir(), reverse=True):
            if (child.name.startswith('.') or child.name.startswith('predictions')
                    or child.name in EXCLUDED or child.is_symlink()):
                continue
            if child.is_dir():
                pending.append(child)


def portable_path(value, project):
    """Rebase saved Windows/other-machine paths onto the current checkout."""
    if not value:
        return None
    normalized = re.sub('/+', '/', str(value).replace('\\', '/'))
    candidate = resolve_location(Path(normalized))
    if candidate.is_file():
        return candidate
    for prefix in ('data/', 'outputs/', 'configs/'):
        position = normalized.find('/' + prefix)
        relative = normalized[position + 1:] if position >= 0 else normalized
        if relative.startswith(prefix) and (project / relative).is_file():
            return project / relative
    return None


def dataset_manifest(marker, project):
    path = portable_path(marker.get('dataset_manifest'), project)
    manifest = read(path) if path else {}
    # A relocated manifest must still describe this exact dataset version.
    if marker.get('dataset_sha256') not in manifest_identities(manifest):
        return {}
    return manifest


def manifest_identities(manifest):
    """Recognize the manifest and benchmark fingerprints of the same cohort."""
    identities = {manifest.get('dataset_sha256')} - {None, ''}
    samples = manifest.get('samples', [])
    if not samples:
        return identities
    try:
        images = [dict(relative_path=s['relative_path'], image_sha256=s['image_sha256'],
            ground_truth=[dict(g, bbox_pixels_xyxy=[float(v) for v in g['bbox_pixels_xyxy']])
                          for g in s['ground_truth']])
            for s in sorted(samples, key=lambda s: s['relative_path'])]
    except (KeyError, TypeError, ValueError):
        return identities
    identities.add(hashlib.sha256(json.dumps(dict(images=images), ensure_ascii=False,
        sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest())
    return identities


def record_evidence(directory, targets, expected, review_enabled):
    path = directory / 'records.jsonl'
    if not path.is_file():
        return None
    seen, successes, failures = set(), defaultdict(set), set()
    duplicate = invalid = mismatch = review_fallbacks = count = 0
    errors = set()
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            count += 1
            try:
                item = json.loads(line)
                name = item['relative_path']
                target = item.get('target_id') or (targets[0] if len(targets) == 1 else '')
                key = (target, name)
                if key in seen:
                    duplicate += 1
                seen.add(key)
                if targets and target not in targets:
                    invalid += 1
                if expected and (name not in expected or item.get('image_sha256') != expected[name]):
                    mismatch += 1
                if item.get('status') == 'success':
                    successes[target].add(name)
                    if review_enabled and (item.get('review_error') or not item.get('review_performed')):
                        review_fallbacks += 1
                else:
                    failures.add(name)
                    errors.add(str(item.get('error', ''))[:240])
            except (json.JSONDecodeError, KeyError, TypeError):
                invalid += 1
    complete_images = set.intersection(*(successes[t] for t in targets)) if targets else set()
    expected_pairs = {(t, p) for t in targets for p in expected}
    return dict(record_count=count, success_images=len(complete_images), failed_images=len(failures),
                success_requests=sum(len(v) for v in successes.values()), duplicate_records=duplicate,
                invalid_records=invalid, hash_mismatches=mismatch, review_fallback_count=review_fallbacks,
                manifest_matches=bool(expected) and seen == expected_pairs and not mismatch,
                errors=sorted(errors))


def model_name(targets):
    names = []
    for target in targets:
        name = str(target.get('model', '')).replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]
        if Path(name).suffix in {'.pt', '.pth', '.onnx', '.bin', '.safetensors'}:
            name = Path(name).stem
        if name and name not in names:
            names.append(name)
    return '+'.join(names) or '未知模型'


def collect_dataset(root, project):
    marker = read(root / 'dataset.json')
    manifest = dataset_manifest(marker, project)
    expected = {s['relative_path']: s.get('image_sha256') for s in manifest.get('samples', [])}
    expected_classes = dict(Counter(g['label'] for s in manifest.get('samples', []) for g in s.get('ground_truth', [])))
    sources = sorted((project / 'outputs').glob(f'*/experiments/{root.name}')) if root.parent.name == 'datasets' else [root]
    directories = sorted(p for source in sources for p in artifacts(source))
    relative = lambda p: Path(os.path.relpath(p, root)).as_posix()
    loaded = {p: (read(p / 'run.json'), read(p / 'metrics.json')) for p in directories}
    if not expected_classes:
        # With reports only, use the fullest saved class scope; mark verification as metadata only.
        fullest = max((m.get('per_class', {}) for _, m in loaded.values()),
                      key=lambda c: (len(c), sum(v.get('ground_truth_count', 0) for v in c.values())), default={})
        expected_classes = {k: v.get('ground_truth_count') for k, v in fullest.items()}
    primary, children = [], defaultdict(list)
    for directory in directories:
        owner = next((parent for parent in directory.parents if parent in loaded), None)
        if owner:
            children[owner].append(directory)
        else:
            primary.append(directory)
    rows = []
    for directory in primary:
        run, metrics = loaded[directory]
        extra = read(directory / 'experiment.json')
        targets = run.get('targets', [])
        target_ids = [t['id'] for t in targets if 'id' in t]
        backend = targets[0].get('backend', '') if targets else extra.get('backend', '')
        method = METHODS.get(backend, 'MLLM' if backend in {'generic', 'diaowu', 'openai', 'anthropic'}
                             or Path(relative(directory)).parts[0].startswith('MLLM') else backend or '未知算法')
        exp = run.get('experiment', {})
        shot = exp.get('positive_count', extra.get('shot_per_class'))
        if shot is None:
            match = re.search(r'_(\d+)shot', directory.name)
            shot = int(match[1]) if match else None
        references = [t.get('reference_count', {}) for t in targets]
        negatives = exp.get('negative_count', sum(r.get('reference_negatives', 0) for r in references))
        variant = exp.get('variant')
        if not variant:
            variant = ('视觉参考' if backend in {'pet_dino', 'yoloe'} else '视觉+文字' if backend == 'yoloe_fusion'
                       else '监督检测' if backend == 'yolo_onnx' else 'text_only' if shot == 0 else '未记录')
        layout = run.get('result_layout', {})
        prompt = read(directory / 'prompt.json')
        snapshots = prompt.get('snapshots', []) or [prompt]
        hashes = [p.get('text_sha256') for p in snapshots if p.get('text_sha256')]
        if not hashes:
            hashes = [t['prompt_sha256'] for t in targets if t.get('prompt_sha256')]
        # A single-class empty generic prompt hash is not the actual detector text hash.
        prompt_hash = hashes[0] if len(hashes) == 1 else hashlib.sha256(
            json.dumps(hashes, separators=(',', ':')).encode()).hexdigest() if hashes else None
        version = layout.get('prompt_version') or next((p for p in Path(relative(directory)).parts
                                                       if re.search(r'_v\d+$', p)), '')
        review = any(t.get('generation_settings', {}).get('enable_completeness_review', False) for t in targets)
        evidence = record_evidence(directory, target_ids, expected, review)
        same_dataset = bool(run.get('dataset_sha256')) and run.get('dataset_sha256') == marker.get('dataset_sha256')
        reconciled_identity = False
        if not same_dataset and run.get('dataset_sha256'):
            run_manifest = dataset_manifest(dict(dataset_manifest=run.get('dataset_manifest'),
                dataset_sha256=marker.get('dataset_sha256')), project)
            reconciled_identity = bool(run_manifest) and run['dataset_sha256'] == run_manifest.get('dataset_sha256')
            same_dataset = reconciled_identity
        classes = {k: v.get('ground_truth_count') for k, v in metrics.get('per_class', {}).items()}
        full_scope = bool(classes) and classes == expected_classes
        count = run.get('image_count', marker.get('image_count'))
        evaluated = metrics.get('evaluated_image_count')
        failed = evidence['failed_images'] if evidence else metrics.get('failed_image_count')
        notes = []
        if reconciled_identity:
            notes.append('清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集')
        complete = (bool(metrics) and metrics.get('evaluation_status') == 'complete'
                    and evaluated == count == marker.get('image_count') and failed == 0)
        if evidence:
            complete = complete and evidence['success_images'] == count and not any(
                evidence[k] for k in ('invalid_records', 'duplicate_records', 'hash_mismatches'))
            if expected:
                complete = complete and evidence['manifest_matches']
            verification = '逐图内容核验' if expected and evidence['manifest_matches'] else '逐图记录核验'
            if evidence['hash_mismatches']:
                notes.append('逐图内容与当前清单不一致')
            if evidence['duplicate_records'] or evidence['invalid_records']:
                notes.append('存在重复或损坏的逐图记录')
            if any('与请求模型' in error for error in evidence['errors']):
                notes.append('上游模型路由不符')
            if evidence['review_fallback_count']:
                notes.append(f"二次复核回退{evidence['review_fallback_count']}次，保留首轮成功预测")
        else:
            verification = '仅保存的指标与元数据'
            notes.append('本机无records.jsonl，未进行逐图内容核验')
        if not same_dataset:
            notes.append('运行数据指纹缺失或与当前数据集不同')
        if metrics and not full_scope:
            notes.append('类别范围或真值数量不同，不参与全数据集比较')
        if not metrics:
            notes.append('本机缺少metrics.json，无法判断评测结果和类别覆盖范围')
        status = '完成' if complete else '未完成/失败' if failed else '待核验'
        if not metrics:
            status = '未产出评估指标'
        if complete and not evidence:
            status = '完成（元数据）'
        if not same_dataset:
            status = '其他数据版本/指纹缺失'
        settings = metrics.get('settings', {})
        row = dict(dataset=root.name, method=method, model=model_name(targets), prompt_version=version,
                   prompt_sha256=prompt_hash, shot_per_class=shot, input=variant, negative_references=negatives,
                   reference_images=exp.get('reference_image_count', sum(
                       r.get(k, 0) for r in references for k in (
                           'reference_originals', 'reference_crops', 'reference_boxed_images', 'reference_negatives'))),
                   seed=exp.get('selection_seed', extra.get('seed')),
                   protocol=extra.get('protocol') or run.get('protocol', ''),
                   repeat=int(re.search(r'_第(\d+)次', str(directory))[1]) if re.search(r'_第(\d+)次', str(directory)) else 1,
                   AP50=metrics.get('mAP50'), AP75=metrics.get('mAP75'), AP50_95=metrics.get('mAP50_95'))
        for suffix, threshold in [('0_5', .5), ('0_7', .7)]:
            point = metrics.get('score_threshold_results', {}).get(str(threshold)) or (
                metrics if settings.get('score_threshold') == threshold else {})
            for key in ('precision', 'recall', 'f1', 'f2', 'tp', 'fp', 'fn'):
                value = point.get(key)
                if key == 'f2' and value is None and point.get('precision') is not None and point.get('recall') is not None:
                    p, r = point['precision'], point['recall']
                    value = 5 * p * r / (4 * p + r) if p + r else 0.
                row[f'{key}_{suffix}'] = value
        report_name = next((n for n in ('evaluation.txt', 'metrics.json', 'run.json') if (directory / n).is_file()), '')
        row.update(total_images=count, evaluated_images=evaluated,
                   success_images=evidence['success_images'] if evidence else None, failed_images=failed,
                   ground_truth_count=sum(v or 0 for v in classes.values()), classes=classes,
                   class_AP50={k: v.get('AP50') for k, v in metrics.get('per_class', {}).items()},
                   status=status, complete=complete, same_dataset=same_dataset, full_class_scope=full_scope,
                   eligible_for_comparison=complete and same_dataset and full_scope, verification=verification,
                   dataset_sha256=run.get('dataset_sha256'), iou_threshold=settings.get('iou_threshold'),
                   max_detections_per_image=settings.get('max_detections_per_image'),
                   default_score_threshold=settings.get('score_threshold'), review_enabled=review,
                   review_fallback_count=evidence['review_fallback_count'] if evidence else None,
                   run_dir=relative(directory),
                   report=relative(directory / report_name) if report_name else '',
                   prompt_file=relative(directory / 'prompt.txt') if (directory / 'prompt.txt').is_file() else '',
                   metric_file=relative(directory / 'metrics.json') if metrics else '',
                   component_runs=[relative(p) for p in children[directory]],
                   record_evidence=evidence, notes='；'.join(notes))
        rows.append(row)
    rows.sort(key=lambda r: (r['method'], r['model'], r['prompt_version'], r['shot_per_class'] or 0,
                             r['input'], r['negative_references'], r['repeat'], r['run_dir']))
    return dict(schema='industrial_fewshot.dataset_summary.v1', dataset=marker,
                manifest_available=bool(manifest), class_ground_truth_counts=expected_classes,
                row_count=len(rows), counts=dict(Counter(r['status'] for r in rows)),
                methods=dict(Counter(r['method'] for r in rows)), rows=rows,
                coverage=dict(primary_runs=len(rows), component_runs=sum(len(v) for v in children.values()),
                              metadata_directories=len(directories)))


def pct(value):
    return '—' if value is None else f'{value * 100:.2f}'


def link(label, target):
    return f'[{label}]({quote(target, safe="/._-")})' if target else '—'


def render_markdown(summary):
    name = summary['dataset'].get('name', summary['rows'][0]['dataset'] if summary['rows'] else '数据集')
    title = f"{summary['method_folder']} / {name}" if summary.get('method_folder') else name
    rows = summary['rows']
    classes = summary['class_ground_truth_counts']
    lines = [f'# {title}：实验结果总表', '',
             f"更新时间：{summary['generated_at']}（北京时间）。共 {len(rows)} 组独立实验；算法分布：" +
             '、'.join(f'{k} {v}组' for k, v in summary['methods'].items()) + '。', '',
             f"当前数据集：{summary['dataset'].get('image_count', '—')} 张；真值框：" +
             '、'.join(f'{k}={v}' for k, v in classes.items()) + '。',
             f"数据指纹：`{summary['dataset'].get('dataset_sha256', '')}`。", '',
             '- 每个独立实验一行；五类合并实验使用合并指标，单类子运行作为来源保留，不重复当成五组实验。',
             '- 所有率指标为百分数；P@0.5、R@0.5、F2@0.5分别单列，表示置信度≥0.5、按表中IoU阈值匹配；另保留分数阈值0.7的P/R/F2。AP读取完整候选评估值。缺失值为“—”，不是0。',
             '- O=原图，C=目标裁剪，B=带框原图，J=标注JSON；shot读取实际实验设置，纯文字为0shot。',
             '- 不同数据指纹、类别范围、不完整运行保留在表内，但不参与全数据集比较。仅元数据完成的运行不代表已核验逐图内容。',
             '- 不同模型的预处理、候选下限、参考样本和提示词可能不同；比较前查看实际prompt、IoU/maxDets和原始配置。', '',
             link('CSV表格', 'results_summary.csv') + ' · ' + link('JSON完整字段', 'results_summary.json'), '',
             '| # | 算法 | 模型 | prompt | shot/类 | 输入 | 负例 | AP50 | AP50:95 | P@0.5 | R@0.5 | F2@0.5 | P/R/F2≥0.7 | TP/FP/FN≥0.5 | 成功/总图 | 状态 | IoU/maxDets | 详情 |',
             '|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|']
    if summary.get('finetuning_report'):
        lines[2:2] = [link('微调模型的300张测试完整成绩（含P@0.5、R@0.5）', summary['finetuning_report']), '']
    for number, row in enumerate(rows, 1):
        triple = lambda suffix: '/'.join(pct(row[f'{k}_{suffix}']) for k in ('precision', 'recall', 'f2'))
        counts = '/'.join(str(row[f'{k}_0_5']) if row[f'{k}_0_5'] is not None else '—' for k in ('tp', 'fp', 'fn'))
        values = [number, row['method'], row['model'], link(row['prompt_version'] or 'prompt', row['prompt_file']),
                  row['shot_per_class'] if row['shot_per_class'] is not None else '—', row['input'],
                  row['negative_references'], pct(row['AP50']), pct(row['AP50_95']),
                  pct(row['precision_0_5']), pct(row['recall_0_5']), pct(row['f2_0_5']), triple('0_7'),
                  counts, f"{row['success_images'] if row['success_images'] is not None else '—'}/{row['total_images']}",
                  row['status'], f"{row['iou_threshold']}/{row['max_detections_per_image']}", link('查看', row['report'])]
        lines.append('| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in values) + ' |')
    lines += ['', '## 核验与特殊情况', '']
    for number, row in enumerate(rows, 1):
        if row['notes']:
            lines.append(f"- 第{number}组 {link(row['model'] + ' / ' + row['input'], row['report'])}：{row['notes']}。")
    lines += ['', 'CSV另含每类AP50、指纹、评估范围、复核开关、来源路径等字段；JSON包含单类子运行路径和记录核验统计。', '',
              f'更新：`python scripts/summarize_results.py --dataset {name}`（在项目根目录执行）。',
              '生成过程只读取已保存结果，不调用模型、不重算AP、不改写原始指标。', '']
    return '\n'.join(lines)


def csv_bytes(rows, labels):
    exported = []
    excluded = {'class_AP50', 'classes', 'component_runs', 'record_evidence'}
    for row in rows:
        item = {}
        for key, value in row.items():
            if key in excluded:
                continue
            item[key + ('_pct' if key in RATE_FIELDS else '')] = (
                round(value * 100, 4) if value is not None else '') if key in RATE_FIELDS else value
        item['classes'] = ','.join(row['classes'])
        item['component_run_count'] = len(row['component_runs'])
        for label in labels:
            value = row['class_AP50'].get(label)
            item[f'{label}_AP50_pct'] = round(value * 100, 4) if value is not None else ''
        exported.append(item)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(exported[0]) if exported else ['dataset', 'method', 'model'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(exported)
    return stream.getvalue().encode('utf-8-sig')


def write_summary(outputs, dataset_root, summary):
    generated = {'results_summary.md': render_markdown(summary).encode('utf-8'),
                 'results_summary.csv': csv_bytes(summary['rows'], summary['class_ground_truth_counts']),
                 'results_summary.json': (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8')}
    for name, data in generated.items():
        path = dataset_root / name
        if path.is_file() and path.read_bytes() != data:
            original = path.read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            backup = outputs / '_shared/archive/history/summary_snapshots' / dataset_root.name / digest / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(original)
        path.write_bytes(data)


def write_method_indexes(outputs, summaries):
    """Keep each method's tables beside its runs; only cross-method tables are shared."""
    method_dirs = sorted(p for p in outputs.iterdir() if p.is_dir() and not p.name.startswith(('_', '.')))
    lines = ['# 实验结果导航', '',
        '按方法查看实验；每个方法的结果、比较文档和复现脚本放在同一目录。微调位于该方法的 finetune/。', '',
        '| 方法 | 实验结果与比较 | 微调 |', '|---|---|---|']
    for method in method_dirs:
        experiments = method / 'experiments'
        comparison = method / 'comparisons'
        comparison.mkdir(exist_ok=True)
        method_lines = [f'# {method.name} 实验', '', '[返回全部方法](../README.md)', '',
            '| 数据集 | 原始实验 | 结果对照 |', '|---|---|---|']
        if method.name == 'JSTY-YOLO':
            method_lines[2:2] = ['自定义 YOLOv26_v8 监督检测网络的ONNX评估，按叉车、吊物及历史数据队列分别保存。',
                'ONNX是推理文件格式；监督训练预算与视觉提示shot分别记录。两场景FP32 ONNX的300张测试结果已加入全部算法总表。', '',
                '[模型来源和结构说明](../../docs/JSTY_MODEL_GUIDE.md) · [原始运行配置](../../configs/specific/jsty_hanging_object.yaml)', '']
        for root, summary in summaries:
            source = experiments / root.name
            if not source.is_dir():
                continue
            destination = comparison / root.name
            destination.mkdir(parents=True, exist_ok=True)
            subset = copy.deepcopy(summary)
            subset['method_folder'] = method.name
            tuning_report = method / 'finetune' / root.name / 'comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md'
            if tuning_report.is_file():
                subset['finetuning_report'] = os.path.relpath(tuning_report, destination)
            selected = [r for r in subset['rows'] if (root / r['run_dir']).resolve().is_relative_to(source)]
            subset.update(rows=selected, row_count=len(selected),
                methods=dict(Counter(r['method'] for r in selected)), counts=dict(Counter(r['status'] for r in selected)))
            component_count = sum(len(r['component_runs']) for r in selected)
            subset['coverage'] = dict(primary_runs=len(selected), component_runs=component_count,
                                      metadata_directories=len(selected) + component_count)
            for row in selected:
                for key in ('run_dir', 'report', 'prompt_file', 'metric_file'):
                    if row.get(key):row[key] = os.path.relpath((root / row[key]).resolve(), destination)
                row['component_runs'] = [os.path.relpath((root / p).resolve(), destination) for p in row['component_runs']]
            write_summary(outputs, destination, subset)
            method_lines.append(f'| {root.name} | [实验目录](experiments/{root.name}/) | [统一表](comparisons/{root.name}/results_summary.md) |')
        method_lines += ['', '## 专题比较', '']
        reports = sorted(p for p in comparison.rglob('*.md') if p.name not in {'results_summary.md', 'README.md'})
        reports += sorted(p for p in comparison.rglob('README.md') if p.parent != comparison)
        for report in reports:
            method_lines.append(f'- [{report.parent.name} / {report.stem}]({report.relative_to(method).as_posix()})')
        if (method / 'scripts').is_dir():
            method_lines += ['', '[实验流程与脚本](scripts/)']
        if (method / 'finetune').is_dir():
            method_lines += ['', '[微调配置、运行结果与比较](finetune/README.md)']
        method_lines += ['', '跨方法核查：[对齐基线](../_shared/comparisons/aligned_baselines_20261027/report.md) · '
            '[定位精度与四项误差≤20%](../_shared/comparisons/iou85_box20_20261028/README.md)', '']
        (method / 'README.md').write_text('\n'.join(method_lines), encoding='utf-8')
        (comparison / 'README.md').write_text(f'# {method.name} 比较文档\n\n[全部实验与专题入口](../README.md)\n', encoding='utf-8')
        tuning = '[微调](%s/finetune/README.md)' % method.name if (method/'finetune').is_dir() else '—'
        lines.append(f'| {method.name} | [{method.name}]({method.name}/README.md) | {tuning} |')
    lines += ['', '## 跨方法总表', '', '| 数据集 | 全方法结果 |', '|---|---|']
    for root, summary in summaries:
        lines.append(f'| {root.name} | [总表]({root.relative_to(outputs).as_posix()}/results_summary.md) |')
    if (outputs / 'ALL_RESULTS.md').is_file():
        lines += ['', '[两个300张测试集：全部算法、提示与微调设置总表](ALL_RESULTS.md) · [完整CSV指标](ALL_RESULTS.csv)']
    if (outputs / 'YOLOE/finetune/5SHOT_COMPARISON.md').is_file():
        lines += ['', '[YOLOv8与YOLOE：两场景5shot训练及测试推理对照](YOLOE/finetune/5SHOT_COMPARISON.md)']
    lines += ['', '```text', 'outputs/', '  <方法>/', '    README.md',
        '    experiments/<数据集>/<实验>/', '    comparisons/<数据集或专题>/',
        '    scripts/', '    finetune/<数据集>/runs/与comparisons/',
        '  _shared/  # 跨方法数据身份、总表、缓存、日志、环境与历史归档', '```', '',
        '原始指标、逐图预测、权重及请求回答保留；历史路径通过迁移清单解析。',
        '更新导航与各方法总表：`python scripts/summarize_results.py`。', '']
    (outputs / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def refresh(outputs=PROJECT / 'outputs', datasets=None):
    outputs = Path(outputs).resolve()
    project = outputs.parent
    roots = sorted(p.parent for p in (outputs / '_shared/datasets').glob('*/dataset.json'))
    requested = set(datasets or [])
    unknown = requested - {p.name for p in roots}
    if unknown:
        raise ValueError('Unknown dataset(s): ' + ', '.join(sorted(unknown)))
    summaries = []
    for root in roots:
        if not requested or root.name in requested:
            summary = collect_dataset(root, project)
            # Preserve byte-identical exports when inputs have not changed.
            previous = read(root / 'results_summary.json')
            unchanged = {k: v for k, v in previous.items() if k != 'generated_at'} == summary
            summary['generated_at'] = previous['generated_at'] if unchanged else datetime.now(
                ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
            write_summary(outputs, root, summary)
            print(f"{root.name}: {summary['row_count']} experiments; {summary['methods']}; {summary['coverage']}")
        else:
            summary = read(root / 'results_summary.json')
        summaries.append((root, summary))
    write_method_indexes(outputs, summaries)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=PROJECT / 'outputs')
    parser.add_argument('--dataset', action='append', help='Dataset folder name; repeat to select several.')
    args = parser.parse_args()
    refresh(args.outputs, args.dataset)

"""Build one saved-experiment table per dataset, without inference or reevaluation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import quote
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
EXCLUDED = {'runtime', 'legacy', 'logs', 'feedback', 'visualizations', 'archive'}
METHODS = {'pet_dino': 'PET-DINO', 'wedetect': 'WeDetect', 'yoloe': 'YOLOE',
           'yoloe_text': 'YOLOE', 'yoloe_fusion': 'YOLOE', 'groundingdino': 'GroundingDINO',
           'yolo_onnx': 'ONNX监督检测'}
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
    candidate = Path(normalized)
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
    if manifest.get('dataset_sha256') != marker.get('dataset_sha256'):
        return {}
    return manifest


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
    directories = sorted(artifacts(root))
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
                             or directory.relative_to(root).parts[0].startswith('MLLM') else backend or '未知算法')
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
        version = layout.get('prompt_version') or next((p for p in directory.relative_to(root).parts
                                                       if re.search(r'_v\d+$', p)), '')
        review = any(t.get('generation_settings', {}).get('enable_completeness_review', False) for t in targets)
        evidence = record_evidence(directory, target_ids, expected, review)
        same_dataset = bool(run.get('dataset_sha256')) and run.get('dataset_sha256') == marker.get('dataset_sha256')
        classes = {k: v.get('ground_truth_count') for k, v in metrics.get('per_class', {}).items()}
        full_scope = bool(classes) and classes == expected_classes
        count = run.get('image_count', marker.get('image_count'))
        evaluated = metrics.get('evaluated_image_count')
        failed = evidence['failed_images'] if evidence else metrics.get('failed_image_count')
        notes = []
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
        if not full_scope:
            notes.append('类别范围或真值数量不同，不参与全数据集比较')
        status = '完成' if complete else '未完成/失败' if failed else '待核验'
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
                   run_dir=directory.relative_to(root).as_posix(),
                   report=(directory / report_name).relative_to(root).as_posix() if report_name else '',
                   prompt_file=(directory / 'prompt.txt').relative_to(root).as_posix() if (directory / 'prompt.txt').is_file() else '',
                   metric_file=(directory / 'metrics.json').relative_to(root).as_posix() if metrics else '',
                   component_runs=[p.relative_to(root).as_posix() for p in children[directory]],
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
    rows = summary['rows']
    classes = summary['class_ground_truth_counts']
    lines = [f'# {name}：全部实验结果总表', '',
             f"更新时间：{summary['generated_at']}（北京时间）。共 {len(rows)} 组独立实验；算法分布：" +
             '、'.join(f'{k} {v}组' for k, v in summary['methods'].items()) + '。', '',
             f"当前数据集：{summary['dataset'].get('image_count', '—')} 张；真值框：" +
             '、'.join(f'{k}={v}' for k, v in classes.items()) + '。',
             f"数据指纹：`{summary['dataset'].get('dataset_sha256', '')}`。", '',
             '- 每个独立实验一行；五类合并实验使用合并指标，单类子运行作为来源保留，不重复当成五组实验。',
             '- 所有率指标为百分数；AP读取已保存的完整候选评估值，P/R/F2分别列出分数阈值0.5和0.7。缺失值为“—”，不是0。',
             '- O=原图，C=目标裁剪，B=带框原图，J=标注JSON；shot读取实际实验设置，纯文字为0shot。',
             '- 不同数据指纹、类别范围、不完整运行保留在表内，但不参与全数据集比较。仅元数据完成的运行不代表已核验逐图内容。',
             '- 不同模型的预处理、候选下限、参考样本和提示词可能不同；比较前查看实际prompt、IoU/maxDets和原始配置。', '',
             link('CSV表格', 'results_summary.csv') + ' · ' + link('JSON完整字段', 'results_summary.json'), '',
             '| # | 算法 | 模型 | prompt | shot/类 | 输入 | 负例 | AP50 | AP50:95 | P/R/F2≥0.5 | P/R/F2≥0.7 | TP/FP/FN≥0.5 | 成功/总图 | 状态 | IoU/maxDets | 详情 |',
             '|---:|---|---|---|---:|---|---:|---:|---:|---|---|---|---|---|---|---|']
    for number, row in enumerate(rows, 1):
        triple = lambda suffix: '/'.join(pct(row[f'{k}_{suffix}']) for k in ('precision', 'recall', 'f2'))
        counts = '/'.join(str(row[f'{k}_0_5']) if row[f'{k}_0_5'] is not None else '—' for k in ('tp', 'fp', 'fn'))
        values = [number, row['method'], row['model'], link(row['prompt_version'] or 'prompt', row['prompt_file']),
                  row['shot_per_class'] if row['shot_per_class'] is not None else '—', row['input'],
                  row['negative_references'], pct(row['AP50']), pct(row['AP50_95']), triple('0_5'), triple('0_7'),
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
            backup = outputs / 'archive/summary_snapshots' / dataset_root.name / digest / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(original)
        path.write_bytes(data)


def refresh(outputs=PROJECT / 'outputs', datasets=None):
    outputs = Path(outputs).resolve()
    project = outputs.parent
    roots = sorted(p.parent for p in (outputs / 'runs').glob('*/dataset.json'))
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
    lines = ['# 实验结果导航', '',
             '按数据集打开总表。各算法、shot、提示词、参考输入组合和重复运行均在同一张表中；CSV适合筛选与跨机器比较。', '',
             '| 数据集 | 图片数 | 实验数 | 算法 | 总表 | CSV |', '|---|---:|---:|---|---|---|']
    for root, summary in summaries:
        relative = root.relative_to(outputs).as_posix()
        lines.append('| ' + ' | '.join([root.name, str(read(root / 'dataset.json').get('image_count', '—')),
                     str(summary.get('row_count', '待更新')), '、'.join(summary.get('methods', {})),
                     link('打开', relative + '/results_summary.md'), link('下载', relative + '/results_summary.csv')]) + ' |')
    lines += ['', '```text', 'outputs/', '  README.md                         # 当前导航',
              '  runs/<数据集>/', '    dataset.json                    # 数据版本身份',
              '    results_summary.md/csv/json      # 该数据集全部实验的统一总表',
              '    <算法或提示词版本>/<实验>/        # 原始运行、指标、预测与prompt',
              '  comparisons/<专题>/               # 跨算法/shot/提示词的专题分析及对比图',
              '  runtime/                          # 运行配置、清单副本、缓存和日志',
              '  environment_setup/                # 环境验证日志',
              '  archive/                          # 中断、旧布局及总表覆盖前的原文件快照', '```', '',
              '原始运行路径保持稳定，专题分析继续保留；总表通过相对链接组织它们。历史总表覆盖前的原字节保存在 '
              '`archive/summary_snapshots/<数据集>/<SHA256>/`。不把旧总表或单类子运行当成新实验重复统计。', '',
              '更新全部数据集：`python scripts/summarize_results.py`。',
              '仅更新一个数据集：`python scripts/summarize_results.py --dataset chache_val300`。',
              '即使其他机器只有已同步的报告和元数据，也能生成表格；缺少逐图记录时会明确标注核验限制。', '']
    (outputs / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=PROJECT / 'outputs')
    parser.add_argument('--dataset', action='append', help='Dataset folder name; repeat to select several.')
    args = parser.parse_args()
    refresh(args.outputs, args.dataset)

"""Complete all existing 5shot input modes, preserving the two finished baseline groups."""
import argparse
import asyncio
import base64
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.cache import image_sha256
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.data.supports import reference_pool, group_specs
from industrial_fewshot.evaluation.reporting import save_text_report
from industrial_fewshot.experiments import run_ablation, comparison_row
from industrial_fewshot.outputs import group_name
from industrial_fewshot.prompts.references import configure_reference_inputs, prepare_reference_examples
from industrial_fewshot.result_layout import apply_result_layout
from industrial_fewshot.runner import inspect_dataset, run
from industrial_fewshot.runtime import run_async

CONFIG = ROOT / 'configs/specific/diaowu_qwen27b_all_5shot.yaml'
RUNTIME = ROOT / 'outputs/runtime/diaowu_qwen27b_all_5shot'
STATE = RUNTIME / 'status.json'
STOP = RUNTIME / 'STOP'
BASELINE = ROOT / 'outputs/runtime/diaowu_qwen27b_prompt_5shot/status.json'
DATASET = '470b98531e4a34810e7ee42bf250e39c97480e42427c3d84152e29e4d368f3fc'


def now():
    return datetime.now(timezone.utc).isoformat()


def verify(saved, planned):
    assert saved['dataset_sha256'] == planned['dataset_sha256'] == DATASET
    assert saved['evaluation'] == planned['evaluation']
    assert saved['targets'][0]['request_signature'] == planned['targets'][0]['request_signature']
    assert saved['targets'][0]['model'] == 'qwen3.8-27b'
    assert saved['evaluation_status'] == 'complete' and saved['failed_request_count'] == 0
    assert saved['image_count'] == 300
    assert sum(c['ground_truth_count'] for c in saved['metrics']['per_class'].values()) == 350
    records = [json.loads(x) for x in (Path(saved['run_dir']) / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == len({r['relative_path'] for r in records}) == 300
    assert {r['relative_path'] for r in records} == set(saved['image_paths'])
    assert all(r['status'] == 'success' for r in records)
    assert all(o.get('reason') for r in records for o in r['objects'])
    assert set(saved['metrics']['score_threshold_results']) == {'0.5', '0.7'}
    return saved


def save_reports(state, directory, template):
    groups = [read_json(Path(j['run_dir']) / 'summary.json') for j in state['jobs'] if j['status'] == 'complete']
    rows = [comparison_row(g['experiment']['id'], g) for g in groups]
    report = {**template, 'dry_run': False, 'comparison_dir': str(directory), 'status': state['status'],
              'groups': groups, 'rows': rows, 'group_count': 15,
              'failed_request_count': sum(g['failed_request_count'] for g in groups)}
    write_json(directory / 'comparison_all_modes.json', report)
    save_text_report(directory, report, 'comparison_all_modes.txt')
    if rows:
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        (directory / 'comparison_all_modes.csv').write_text(stream.getvalue(), encoding='utf-8-sig')
    lines = ['# 新版吊物300张：Qwen3.8-27B完整5shot组合', '',
             f"状态：{state['status']}；完成{len(groups)}/15组。已有两组保留，本次补跑13组、3900张次。", '',
             '同一300张图片（210正例、90负例、350GT）、当前少量遮挡提示、seed0固定5shot；开启完整性复核。含负例时为5张负例原图。', '',
             'O=原图，C=目标裁剪图，B=带框原图，J=参考标注JSON；所有组合均包含相同文字提示。', '',
             '| 输入 | 状态 | AP50 | AP50:95 | Precision@0.5 | Recall@0.5 | F2@0.5 | TP/FP/FN |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    by_dir = {g['run_dir']: g for g in groups}
    for job in state['jobs']:
        g = by_dir.get(job['run_dir'])
        if g:
            m = g['metrics']; vals = [f"{m[k]*100:.2f}%" for k in ['mAP50','mAP50_95','precision','recall','f2']]
            lines.append('| ' + ' | '.join([job['name'], '完成（已有）' if job.get('existing') else '完成', *vals,
                         f"{m['tp']}/{m['fp']}/{m['fn']}"]) + ' |')
        else:
            lines.append('| ' + ' | '.join([job['name'], job['status'], *['-']*6]) + ' |')
    lines.extend(['', '各组直接保存prompt.txt、prompt.json、records.jsonl、metrics.json、evaluation.txt、feedback与0.5/0.7预测。',
                  '原两组comparison.txt/json与单组结果保持原样；全组合汇总见comparison_all_modes.txt/json/csv。',
                  'AP使用全部分数排序；同分排序敏感性仍存在，不由单次小幅AP差值推断稳定优势。'])
    if state.get('error'):
        lines.extend(['', '未完成原因：' + state['error']])
    (directory / 'comparison_all_modes.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def archive_partial(source, directory):
    """Use native PowerShell, validating both absolute boundaries before a directory move."""
    parent = ROOT / 'outputs/archive/interrupted_runs/diaowu_val300/all_5shot'
    destination = parent / (source.name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    script = r'''
$source = (Resolve-Path -LiteralPath $env:IFS_SOURCE -ErrorAction Stop).Path
$boundary = (Resolve-Path -LiteralPath $env:IFS_BOUNDARY -ErrorAction Stop).Path
$archive = [IO.Path]::GetFullPath($env:IFS_ARCHIVE)
$archiveBoundary = [IO.Path]::GetFullPath($env:IFS_ARCHIVE_BOUNDARY)
if (-not $source.StartsWith($boundary + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source escaped experiment boundary' }
if (-not $archive.StartsWith($archiveBoundary + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Archive escaped boundary' }
$nodes = @((Get-Item -LiteralPath $source -Force)) + @(Get-ChildItem -LiteralPath $source -Recurse -Force -ErrorAction Stop)
if (@($nodes | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Source contains reparse points' }
$ancestor = Split-Path -Parent $archive
while (-not (Test-Path -LiteralPath $ancestor)) { $ancestor = Split-Path -Parent $ancestor }
while ($ancestor) {
 if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Archive ancestor is a reparse point' }
 $ancestor = Split-Path -Parent $ancestor
}
if (Test-Path -LiteralPath $archive) { throw 'Archive destination exists' }
New-Item -ItemType Directory -Path (Split-Path -Parent $archive) -Force | Out-Null
Move-Item -LiteralPath $source -Destination $archive -ErrorAction Stop
'''
    env = dict(os.environ, IFS_SOURCE=str(source), IFS_BOUNDARY=str(directory),
               IFS_ARCHIVE=str(destination), IFS_ARCHIVE_BOUNDARY=str(parent))
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded], env=env, check=True)
    logging.info('Preserved interrupted attempt: %s', destination)


async def execute(dry_run=False):
    baseline = read_json(BASELINE)
    assert baseline['status'] == 'complete' and baseline['dataset_sha256'] == DATASET
    directory = Path(baseline['comparison_dir']).resolve()
    config = load_config(CONFIG)
    plan = await run_ablation(config, dry_run=True)
    assert plan['image_count'] == 300 and plan['group_count'] == 15 and plan['dataset_sha256'] == DATASET
    assert not plan['excluded_reference_overlap']
    assert Path(plan['output_dir']).resolve() == directory.parent
    assert all(t['model'] == 'qwen3.8-27b' for g in plan['groups'] for t in g['targets'])
    assert len({g['targets'][0]['prompt_sha256'] for g in plan['groups']}) == 1
    items = inspect_dataset(config)
    for item in items:
        item['image_sha256'] = image_sha256(item['path'])
    config = apply_result_layout(config, items, create=False)
    pools = [reference_pool(t['config'], 0) for t in config['targets']]
    jobs, variants = [], []
    for spec, planned in zip(group_specs(config['experiment_suite']), plan['groups']):
        destination = directory / group_name(spec)
        variant = copy.deepcopy(config)
        variant['output_dir'] = destination
        metadata = {**spec, 'name': destination.name, 'selection_seed': 0, 'source_reference_count': 0,
                    'reference_image_count': 0, 'reference_sets': []}
        for target, pool in zip(variant['targets'], pools):
            if spec['positive_count']:
                selection = pool.prepare(target['config'], spec)
                metadata['reference_sets'].append({'target_id': target['id'], **selection})
                for key in ('source_reference_count', 'reference_image_count'):
                    metadata[key] += selection[key]
            else:
                configure_reference_inputs(target['config'], 'text_only')
                prepare_reference_examples(target['config'])
        variant['experiment'] = metadata
        job = dict(id=spec['id'], name=destination.name, run_dir=str(destination), status='pending',
                   request_signature=planned['targets'][0]['request_signature'], existing=False)
        saved_file = destination / 'summary.json'
        if saved_file.is_file():
            saved = read_json(saved_file)
            if saved.get('evaluation_status') == 'complete' and not saved.get('failed_request_count'):
                verify(saved, planned)
                job.update(status='complete', existing=True)
        jobs.append(job); variants.append((variant, planned))
    previous = read_json(STATE) if STATE.is_file() else {}
    identity = dict(dataset_sha256=DATASET, signatures=[j['request_signature'] for j in jobs])
    if previous:
        assert all(previous.get(k) == v for k,v in identity.items()), 'Saved experiment identity changed'
        for job, old in zip(jobs, previous['jobs']):
            job['existing'] = old.get('existing', False)
    state = dict(**identity, status='planned' if dry_run else 'running', started=previous.get('started', now()),
                 updated=now(), pid=os.getpid(), comparison_dir=str(directory), jobs=jobs,
                 total_image_tasks=4500, completed_groups=sum(j['status']=='complete' for j in jobs),
                 pending_image_tasks=300*sum(j['status']!='complete' for j in jobs))
    RUNTIME.mkdir(parents=True, exist_ok=True)
    write_json(RUNTIME / 'preflight.json', state)
    if dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2)); return
    if STOP.exists():
        STOP.unlink()
    write_json(STATE, state); save_reports(state, directory, plan)
    for job, (variant, planned) in zip(jobs, variants):
        if job['status'] == 'complete':
            continue
        destination = Path(job['run_dir'])
        try:
            if STOP.exists():
                raise asyncio.CancelledError('STOP requested')
            if destination.exists():
                saved = read_json(destination / 'run.json')
                assert saved['dataset_sha256'] == DATASET
                assert saved['targets'][0]['request_signature'] == job['request_signature']
                archive_partial(destination, directory)
            job.update(status='running', started=now())
            state.update(updated=now(), current_group=job['name'])
            write_json(STATE, state); save_reports(state, directory, plan)
            task = asyncio.create_task(run(variant, shared_items=copy.deepcopy(items), references_prepared=True,
                                           result_dir=destination))
            while not task.done():
                await asyncio.wait({task}, timeout=1)
                if STOP.exists():
                    task.cancel()
            result = verify(await task, planned)
            job.update(status='complete', finished=now(), executed_inference_count=result['executed_inference_count'],
                       reused_prediction_count=result['reused_prediction_count'])
            state.update(completed_groups=sum(j['status']=='complete' for j in jobs), updated=now())
        except BaseException as exc:
            task = locals().get('task')
            if task is not None and not task.done():
                task.cancel(); await asyncio.gather(task, return_exceptions=True)
            status = 'interrupted' if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else 'failed'
            job.update(status=status, error=str(exc))
            state.update(status=status, error=str(exc), updated=now())
            write_json(STATE, state); save_reports(state, directory, plan)
            raise
        write_json(STATE, state); save_reports(state, directory, plan)
    state.update(status='complete', updated=now(), pending_image_tasks=0)
    write_json(STATE, state); save_reports(state, directory, plan)
    print('COMPLETE: ' + str(directory / 'comparison_all_modes.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop', action='store_true', help='Gracefully stop this batch; successful predictions remain resumable.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stop:
        RUNTIME.mkdir(parents=True, exist_ok=True)
        STOP.write_text('stop requested\n', encoding='utf-8')
        print('STOP requested')
    else:
        try:
            run_async(execute(args.dry_run))
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise SystemExit(130)
        except Exception as exc:
            logging.exception('Batch stopped: %s', exc)
            raise SystemExit(1)

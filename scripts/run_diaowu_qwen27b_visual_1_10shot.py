"""Run/resume 1shot and 10shot O+C/O+B, each with and without equal-count negatives."""
import argparse
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.data.supports import reference_pool
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.runtime import run_async
from run_diaowu_qwen27b_all_5shot import verify

CONFIG = ROOT / 'configs/specific/diaowu_qwen27b_visual_1_10shot.yaml'
RUNTIME = ROOT / 'outputs/runtime/diaowu_qwen27b_visual_1_10shot'
STATE = RUNTIME / 'status.json'
STOP = RUNTIME / 'STOP'
DATASET = '470b98531e4a34810e7ee42bf250e39c97480e42427c3d84152e29e4d368f3fc'
BASELINE_STATE = ROOT / 'outputs/runtime/diaowu_qwen27b_all_5shot/status.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def report(state, baseline):
    dataset = Path(baseline['comparison_dir']).parents[1]
    rows, groups = [], []
    for shot in [1, 5, 10]:
        job = next((j for j in state['jobs'] if j['shot'] == shot), None)
        directory = Path(baseline['comparison_dir']) if shot == 5 else Path(job['comparison_dir'])
        for variant in ['原图+裁剪图', '原图+带框图']:
            for neg in [False, True]:
                name = variant + (f'_含{shot}张负例' if neg else '_无负例')
                file = directory / name / 'summary.json'
                group = read_json(file) if file.is_file() else None
                if group and group.get('evaluation_status') == 'complete' and not group.get('failed_request_count'):
                    groups.append(group)
                    m = group['metrics']
                    values = [f'{m[k]*100:.2f}%' for k in ['mAP50','mAP50_95','precision','recall','f2']]
                    rows.append([str(shot), name, '完成（已有）' if shot == 5 else '完成', *values,
                                 f"{m['tp']}/{m['fp']}/{m['fn']}"])
                else:
                    rows.append([str(shot), name, job['status'], *['-']*6])
    lines = ['# 吊物：Qwen3.8-27B视觉提示1/5/10shot对比', '',
             f"本次状态：{state['status']}；新增1shot、10shot共8组/2400张次，5shot四组为已完成对照。", '',
             '固定新版300张（210正例、90负例、350GT），相同少量遮挡提示词、seed0嵌套参考集；保留完整性复核和逐框理由。',
             '负例数与shot一致：1shot含1张负例、5shot含5张、10shot含10张；无负例版本为0张。', '',
             '| shot | 输入 | 状态 | AP50 | AP50:95 | Precision@0.5 | Recall@0.5 | F2@0.5 | TP/FP/FN |',
             '|---:|---|---|---:|---:|---:|---:|---:|---|']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    lines.extend(['', '各组直接保存prompt.txt/json、records.jsonl、metrics.json、evaluation.txt、feedback和0.5/0.7预测。',
                  'AP使用全部置信度排序。小幅AP差异须结合同分排序敏感性和重复实验解释。'])
    if state.get('error'): lines.extend(['', '未完成原因：' + state['error']])
    output = dataset / 'qwen27b_visual_shot_comparison.md'
    output.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output.with_suffix('.json'), dict(status=state['status'], jobs=state['jobs'], groups=groups))


async def execute(dry_run=False):
    baseline = read_json(BASELINE_STATE)
    assert baseline['status'] == 'complete' and baseline['dataset_sha256'] == DATASET
    parent = Path(baseline['comparison_dir']).parent
    old = read_json(STATE) if STATE.is_file() else {}
    plans, jobs = [], []
    selections = {}
    for shot in [1, 5, 10]:
        pool = reference_pool(load_config(CONFIG)['targets'][0]['config'], 0)
        positives, negatives = pool.select(shot, shot)
        selections[shot] = dict(positive=[r['sample_id'] for r in positives], negative=[r['sample_id'] for r in negatives])
    for kind in ['positive','negative']:
        assert set(selections[1][kind]) <= set(selections[5][kind]) <= set(selections[10][kind])
    for shot in [1, 10]:
        config = load_config(CONFIG)
        config['experiment_suite']['positive_counts'] = [shot]
        config['resolved_config']['experiment_suite']['positive_counts'] = [shot]
        previous = next((j for j in old.get('jobs',[]) if j['shot'] == shot), None)
        if previous and (Path(previous['comparison_dir'])/'comparison.json').is_file():
            config['resume_comparison_dir'] = Path(previous['comparison_dir'])
        plan = await run_ablation(config, dry_run=True)
        assert plan['dataset_sha256'] == DATASET and plan['image_count'] == 300 and plan['group_count'] == 4
        assert Path(plan['output_dir']) == parent and not plan['excluded_reference_overlap']
        assert plan['modes'] == [f'O_C_k{shot:02d}_pos', f'O_C_k{shot:02d}_posneg',
                                 f'O_B_k{shot:02d}_pos', f'O_B_k{shot:02d}_posneg']
        for group in plan['groups']:
            t = group['targets'][0]; refs = t['reference_count']; exp = group['experiment']
            assert t['model'] == 'qwen3.8-27b' and refs['reference_originals'] == shot
            assert refs['reference_negatives'] == exp['negative_count']
            assert refs['reference_boxed_images'] == (shot if exp['variant'] == 'O+B' else 0)
            selected = exp['reference_sets'][0]
            assert selected['positive_sample_ids'] == selections[shot]['positive']
            assert selected['negative_sample_ids'] == (selections[shot]['negative'] if exp['negative_count'] else [])
        signatures = [g['targets'][0]['request_signature'] for g in plan['groups']]
        if previous: assert previous['signatures'] == signatures
        job = dict(shot=shot, status='pending', comparison_dir=plan['comparison_dir'], signatures=signatures)
        if previous: job = {**previous, **job}
        plans.append((config,plan)); jobs.append(job)
    assert len({g['targets'][0]['prompt_sha256'] for _,p in plans for g in p['groups']}) == 1
    state = dict(status='planned' if dry_run else 'running', dataset_sha256=DATASET, jobs=jobs,
                 image_tasks=2400, started=old.get('started', now()), updated=now(), selections=selections)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    write_json(RUNTIME/'preflight.json', state)
    if dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2)); return
    if STOP.exists(): STOP.unlink()
    write_json(STATE,state); report(state,baseline)
    for job,(config,plan) in zip(jobs,plans):
        try:
            if STOP.exists(): raise asyncio.CancelledError('STOP requested')
            job.update(status='running', started=now()); state['updated']=now()
            write_json(STATE,state); report(state,baseline)
            task = asyncio.create_task(run_ablation(config))
            while not task.done():
                await asyncio.wait({task}, timeout=1)
                if STOP.exists(): task.cancel()
            result = await task
            job['comparison_dir'] = result['comparison_dir']
            assert result['status'] == 'complete' and not result['failed_request_count']
            for actual,planned in zip(result['groups'],plan['groups']): verify(actual,planned)
            job.update(status='complete', finished=now(), executed_inference_count=result['executed_inference_count'],
                       reused_prediction_count=result['reused_prediction_count'])
        except BaseException as exc:
            task = locals().get('task')
            if task is not None and not task.done():
                task.cancel(); await asyncio.gather(task, return_exceptions=True)
            status = 'interrupted' if isinstance(exc,(KeyboardInterrupt,asyncio.CancelledError)) else 'failed'
            job.update(status=status,error=str(exc)); state.update(status=status,error=str(exc),updated=now())
            write_json(STATE,state); report(state,baseline); raise
        state['updated']=now(); write_json(STATE,state); report(state,baseline)
    state.update(status='complete',updated=now()); write_json(STATE,state); report(state,baseline)
    print('COMPLETE: '+str(parent.parent/'qwen27b_visual_shot_comparison.md'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--stop',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stop:
        RUNTIME.mkdir(parents=True,exist_ok=True); STOP.write_text('stop requested\n',encoding='utf-8')
    else:
        try: run_async(execute(args.dry_run))
        except (KeyboardInterrupt,asyncio.CancelledError): raise SystemExit(130)
        except Exception as exc:
            logging.exception('Batch stopped: %s',exc); raise SystemExit(1)

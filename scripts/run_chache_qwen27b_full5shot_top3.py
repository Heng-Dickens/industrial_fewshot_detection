"""Complete positive-only expanded-v2 5shot modes, then run the frozen top three at 1/10shot."""
import argparse
import asyncio
import base64
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from industrial_fewshot.cache import image_sha256
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.data.supports import DEFAULT_VARIANTS, group_specs, reference_pool
from industrial_fewshot.evaluation.reporting import save_text_report
from industrial_fewshot.experiments import comparison_row, run_ablation
from industrial_fewshot.outputs import group_name
from industrial_fewshot.result_layout import apply_result_layout
from industrial_fewshot.run_discovery import saved_runs
from industrial_fewshot.runner import inspect_dataset, run
from industrial_fewshot.runtime import run_async
from run_chache_expanded_shots import verify_group as verify_existing

CONFIG = ROOT/'configs/specific/chache_qwen27b_full5shot_top3.yaml'
RUNTIME = ROOT/'outputs/runtime/chache_qwen27b_full5shot_top3'
STATE = RUNTIME/'status.json'
STOP = RUNTIME/'STOP'
DATASET = 'bf04a484ca78255b5d8a8208dac79a6ac350c0dae1b4aa696bb5d921140b8d2b'
MODEL = 'qwen3.8-27b'


def now():
    return datetime.now(timezone.utc).isoformat()


def rank_top3(groups):
    """Macro AP50 wins; freeze deterministic tie breaks and exclude non-5shot/text baselines."""
    candidates = [g for g in groups if g['experiment']['positive_count'] == 5
                  and g['experiment']['negative_count'] == 0
                  and g['experiment']['variant'] in DEFAULT_VARIANTS]
    if len(candidates) != 7 or len({g['experiment']['variant'] for g in candidates}) != 7:
        raise ValueError('Rank only after all seven distinct positive-only 5shot modes complete')
    if any(g['evaluation_status'] != 'complete' or g['failed_request_count'] for g in candidates):
        raise ValueError('Incomplete/failed groups cannot participate in top-three selection')
    return sorted(candidates, key=lambda g: (
        *[-g['metrics'][key] for key in ('mAP50','mAP50_95','recall','f2')], g['experiment']['variant']))[:3]


def verify(saved, planned, shot):
    target = planned['targets'][0]
    if saved.get('evaluation_status')!='complete' or saved.get('failed_request_count'):
        raise RuntimeError(f"{shot}shot {saved.get('experiment',{}).get('variant')}: "
                           f"evaluation={saved.get('evaluation_status')}, failed_images={saved.get('failed_request_count')}; "
                           'successful predictions are preserved; rerun to retry failures only')
    verify_existing(saved, dict(dataset_sha256=DATASET, request_signature=target['request_signature'],
                                model=MODEL, shot=shot))
    assert saved['evaluation'] == planned['evaluation']
    assert saved['experiment']['variant'] == planned['experiment']['variant']
    assert saved['experiment']['negative_count'] == 0
    assert not saved['targets'][0]['generation_settings']['enable_completeness_review']
    refs = saved['targets'][0]['reference_count']
    assert refs == target['reference_count'] and refs['reference_negatives'] == 0
    assert refs['reference_originals'] == 5*shot
    return saved


@contextmanager
def batch_lock():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME/'batch.lock').open('a+b') as stream:
        if not stream.tell(): stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name == 'nt': msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream, fcntl.LOCK_UN)


def archive_partial(source, directory):
    parent = ROOT/'outputs/archive/interrupted_runs/chache_val300/full5shot_top3'/directory.name
    destination = parent/(source.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    assert source.resolve().is_relative_to(directory.resolve())
    assert destination.resolve().is_relative_to((ROOT/'outputs/archive/interrupted_runs/chache_val300').resolve())
    if os.name != 'nt':
        if any(p.is_symlink() for p in [source,*source.rglob('*')]): raise ValueError('Reparse/symlink in partial run')
        parent.mkdir(parents=True,exist_ok=True); source.rename(destination); return
    script = r'''
$source=(Resolve-Path -LiteralPath $env:IFS_SOURCE -ErrorAction Stop).Path
$boundary=(Resolve-Path -LiteralPath $env:IFS_BOUNDARY -ErrorAction Stop).Path
$archive=[IO.Path]::GetFullPath($env:IFS_ARCHIVE)
$archiveBoundary=[IO.Path]::GetFullPath($env:IFS_ARCHIVE_BOUNDARY)
if (-not $source.StartsWith($boundary+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid source boundary' }
if (-not $archive.StartsWith($archiveBoundary+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid archive boundary' }
$nodes=@((Get-Item -LiteralPath $source -Force))+@(Get-ChildItem -LiteralPath $source -Recurse -Force -ErrorAction Stop)
if (@($nodes | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count) { throw 'Reparse point in source' }
$ancestor=Split-Path -Parent $archive
while (-not (Test-Path -LiteralPath $ancestor)) {$ancestor=Split-Path -Parent $ancestor}
while ($ancestor) {
 if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Archive ancestor reparse point'}
 $ancestor=Split-Path -Parent $ancestor
}
if (Test-Path -LiteralPath $archive) {throw 'Archive exists'}
New-Item -ItemType Directory -Path (Split-Path -Parent $archive) -Force | Out-Null
Move-Item -LiteralPath $source -Destination $archive -ErrorAction Stop
'''
    env = dict(os.environ, IFS_SOURCE=str(source), IFS_BOUNDARY=str(directory),
               IFS_ARCHIVE=str(destination), IFS_ARCHIVE_BOUNDARY=str(parent))
    subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',
                    base64.b64encode(script.encode('utf-16-le')).decode()],env=env,check=True)


def prepare(config, items, shot, variants):
    settings = copy.deepcopy(config)
    settings['experiment_suite'].update(positive_counts=[shot],positive_variants=variants)
    settings['resolved_config']['experiment_suite'].update(positive_counts=[shot],positive_variants=variants)
    settings = apply_result_layout(settings, items, create=False)
    pools = [reference_pool(t['config'],0) for t in settings['targets']]
    directory = Path(settings['output_dir'])/f'{MODEL}_{shot}shot'
    prepared = []
    for spec in group_specs(settings['experiment_suite']):
        assert spec['positive_count'] == shot and spec['negative_count'] == 0
        variant = copy.deepcopy(settings)
        variant['output_dir'] = directory/group_name(spec)
        metadata = {**spec,'name':group_name(spec),'selection_seed':0,
                    'source_reference_count':0,'reference_image_count':0,'reference_sets':[]}
        for target,pool in zip(variant['targets'],pools):
            selection=pool.prepare(target['config'],spec)
            metadata['reference_sets'].append({'target_id':target['id'],**selection})
            for key in ('source_reference_count','reference_image_count'): metadata[key]+=selection[key]
        variant['experiment']=metadata
        prepared.append((spec,variant))
    return prepared


def scan_complete(namespace, shot, planned):
    for directory in sorted(namespace.glob(f'{MODEL}_{shot}shot*')):
        file=directory/planned['experiment']['name']/'summary.json'
        if not file.is_file(): continue
        saved=read_json(file)
        if saved.get('dataset_sha256') != DATASET: continue
        if saved.get('targets',[{}])[0].get('request_signature') != planned['targets'][0]['request_signature']: continue
        if saved.get('evaluation_status') != 'complete' or saved.get('failed_request_count'): continue
        verify(saved,planned,shot)
        return saved
    return None


def protected_files(namespace):
    files=set()
    for file in saved_runs(namespace):
        for name in ('run.json','summary.json','records.jsonl','metrics.json','evaluation.txt','prompt.txt','prompt.json'):
            if (file.parent/name).is_file(): files.add(file.parent/name)
        files.update(p for p in file.parent.parent.glob('comparison*.*') if p.is_file())
    files.update(p for p in namespace.parent.glob('*.md') if p.is_file() and p.name!='prompt_results.md')
    files.update(p for p in (ROOT/'data/references/chache').rglob('*') if p.is_file())
    files.add(ROOT/'src/industrial_fewshot/prompts/templates/chache_v2_expanded.txt')
    files.add(ROOT/'data/raw/chache_val300/manifest.json')
    return {str(p):image_sha256(p) for p in sorted(files)}


def groups_for(state):
    return [{k:v for k,v in read_json(Path(j['run_dir'])/'summary.json').items() if k!='annotations'}
            for j in state['jobs'] if j['status']=='complete']


def save_reports(state):
    groups=groups_for(state)
    namespace=Path(state['namespace'])
    report=dict(status=state['status'],stage=state['stage'],dataset_sha256=DATASET,
                group_count=13,completed_group_count=len(groups),groups=groups,jobs=state['jobs'],
                top3=state.get('top3',[]),failed_request_count=sum(g['failed_request_count'] for g in groups))
    root=namespace.parent
    write_json(root/'qwen27b_expanded_full5shot_top3.json',report)
    rows=[dict(shot=g['experiment']['positive_count'],**comparison_row(g['experiment']['id'],g)) for g in groups]
    if rows:
        stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        (root/'qwen27b_expanded_full5shot_top3.csv').write_text(stream.getvalue(),encoding='utf-8-sig')
    five=[g for g in groups if g['experiment']['positive_count']==5]
    five_report=dict(status='complete' if len(five)==7 else 'running',groups=five,group_count=7,
                     comparison_dir=str(namespace/f'{MODEL}_5shot'),failed_request_count=0)
    five_dir=namespace/f'{MODEL}_5shot'
    five_dir.mkdir(parents=True,exist_ok=True)
    write_json(five_dir/'comparison_all_positive_modes.json',five_report)
    save_text_report(five_dir,five_report,'comparison_all_positive_modes.txt')
    lines=['# 特种车辆：Qwen3.8-27B扩展提示词全5shot及前三名1/10shot', '',
           f"状态：{state['status']}；阶段：{state['stage']}；已完成{len(groups)}/13组（匹配的旧结果经核验复用）。", '',
           '固定扩展版v2、同300张图及188GT（40/30/40/30/48）；shot按每类计，仅正例，无负例，无二次复核。',
           '5shot比较7种材料组合；按五类宏平均AP50降序选前三，平局依次按AP50:95、Recall、F2、组合名称确定。纯文字基线不参与前三排序。', '',
           '| 5shot输入 | 状态 | AP50 | AP50:95 | Precision@0.5 | Recall@0.5 | F2@0.5 | TP/FP/FN |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    by_dir={g['run_dir']:g for g in groups}
    for job in state['jobs']:
        if job['shot']!=5: continue
        g=by_dir.get(job['run_dir'])
        if not g: lines.append('| '+' | '.join([job['name'],job['status'],*['-']*6])+' |'); continue
        m=g['metrics']; values=[f'{m[k]*100:.2f}%' for k in ('mAP50','mAP50_95','precision','recall','f2')]
        lines.append('| '+' | '.join([job['name'],'完成（已有）' if job['existing'] else '完成',*values,
                                      f"{m['tp']}/{m['fp']}/{m['fn']}"])+' |')
    pure=namespace/f'{MODEL}_5shot'/'仅文字提示/summary.json'
    if pure.is_file():
        p=read_json(pure)
        lines.extend(['',f"已有同v2纯文字基线：AP50={p['metrics']['mAP50']*100:.2f}%，仅作对照，不重跑、不参与前三筛选。"])
    if state.get('top3'):
        lines.extend(['','## 冻结的前三名及shot对照','','| 排名 | 输入组合 | shot/类 | 状态 | AP50 | AP50:95 | Precision | Recall | F2 |',
                      '|---:|---|---:|---|---:|---:|---:|---:|---:|'])
        for rank,selected in enumerate(state['top3'],1):
            for shot in (1,5,10):
                job=next((j for j in state['jobs'] if j['shot']==shot and j['variant']==selected['variant']),None)
                g=by_dir.get(job['run_dir']) if job else None
                values=[f"{g['metrics'][k]*100:.2f}%" for k in ('mAP50','mAP50_95','precision','recall','f2')] if g else ['-']*5
                lines.append('| '+' | '.join([str(rank),selected['variant'],str(shot),job['status'] if job else 'pending',*values])+' |')
    if state.get('error'): lines.extend(['','未完成原因：'+state['error']])
    lines.extend(['','原有模型/shot结果与comparison文件保持原样，新组直接保存在对应模型_shot目录下。各组包含完整prompt、逐图反馈、COCO/VOC及0.5/0.7输出。',
                  '前三按本次固定验证集结果选择；单次小幅AP差值受模型采样和同分排序影响。'])
    (root/'qwen27b_expanded_full5shot_top3.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def finish_verification(state):
    protected=read_json(RUNTIME/'baseline_hashes.json')
    assert all(image_sha256(Path(p))==sha for p,sha in protected.items()), 'An original result/reference/prompt changed'
    assert len(state['jobs'])==13 and all(j['status']=='complete' for j in state['jobs'])
    groups=groups_for(state)
    assert len({g['targets'][0]['prompt_sha256'] for g in groups})==1
    assert state['top3']==[dict(variant=g['experiment']['variant'],run_dir=g['run_dir'],
        request_signature=g['targets'][0]['request_signature'],
        **{k:g['metrics'][k] for k in ('mAP50','mAP50_95','recall','f2')})
        for g in rank_top3([g for g in groups if g['experiment']['positive_count']==5])]
    cohorts=[]; selections={}; executed=0; reused=0
    for job,g in zip([j for j in state['jobs'] if j['status']=='complete'],groups):
        records=[json.loads(x) for x in (Path(g['run_dir'])/'records.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records)==len({r['relative_path'] for r in records})==300
        assert all(r['status']=='success' and not r.get('review_performed') for r in records)
        assert all(o.get('reason') for r in records for o in r['objects'])
        assert all(Path(r['feedback_json']).is_file() and Path(r['feedback_text']).is_file() for r in records)
        assert all(set(r['labelme_json_by_score'])=={'0.5','0.7'} for r in records)
        assert all(Path(p).is_file() for r in records for p in r['labelme_json_by_score'].values())
        assert g['targets'][0]['reference_count']['reference_negatives']==0
        assert len(g['support_manifests'])==5 and all(s['negative_count']==0 for s in g['support_manifests'])
        cohorts.append({(r['relative_path'],r['image_sha256']) for r in records})
        selection=g['experiment']['reference_sets'][0]['class_selections']
        ids={label:rows['positive_sample_ids'] for label,rows in selection.items()}
        shot=job['shot']
        if shot in selections: assert selections[shot]==ids
        selections[shot]=ids
        if not job['existing']:
            executed+=job.get('executed_inference_count',0); reused+=job.get('reused_prediction_count',0)
    assert all(c==cohorts[0] for c in cohorts)
    for label in selections[5]:
        assert selections[1][label]==selections[5][label][:1]
        assert selections[5][label]==selections[10][label][:5]
    routing=[]
    database=Path.home()/'.cc-switch/cc-switch.db'
    if database.is_file():
        with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
            routing=db.execute("SELECT request_model,model,status_code,count(*) FROM proxy_request_logs WHERE app_type='claude' AND request_model=? AND created_at>=? GROUP BY request_model,model,status_code",
                (MODEL,datetime.fromisoformat(state['started']).timestamp())).fetchall()
        assert all(r[1]==MODEL for r in routing if r[2]==200)
    write_json(RUNTIME/'verification.json',dict(status='verified',verified_at=now(),dataset_sha256=DATASET,
        group_count=13,image_tasks=3900,negative_reference_count=0,prompt_sha256=groups[0]['targets'][0]['prompt_sha256'],
        preserved_files=len(protected),new_groups=sum(not j['existing'] for j in state['jobs']),
        executed_inference_count=executed,reused_prediction_count=reused,actual_model_requests=routing,top3=state['top3']))


async def execute(dry_run=False):
    config=load_config(CONFIG)
    assert config['targets'][0]['config']['model']==MODEL
    assert not config['targets'][0]['config']['enable_completeness_review']
    assert config['experiment_suite']['negative_variants']==[False]
    plan=await run_ablation(config,dry_run=True)
    assert plan['dataset_sha256']==DATASET and plan['image_count']==300 and plan['group_count']==7
    assert not plan['excluded_reference_overlap']
    namespace=Path(plan['output_dir'])
    items=inspect_dataset(config)
    for item in items: item['image_sha256']=image_sha256(item['path'])
    previous=read_json(STATE) if STATE.is_file() else {}
    identity=dict(dataset_sha256=DATASET,five_signatures=[g['targets'][0]['request_signature'] for g in plan['groups']])
    if previous: assert all(previous.get(k)==v for k,v in identity.items()), 'Saved request identity changed'
    state=dict(**identity,status='planned' if dry_run else 'running',stage='5shot',namespace=str(namespace),
               started=previous.get('started',now()),updated=now(),jobs=[],top3=previous.get('top3',[]),
               total_group_count=13,total_image_tasks=3900,pid=os.getpid())
    prepared=prepare(config,items,5,DEFAULT_VARIANTS)

    async def attach(spec,variant,planned,shot):
        name=group_name(spec)
        saved=scan_complete(namespace,shot,planned)
        directory=namespace/f'{MODEL}_{shot}shot'
        job=dict(id=f'{shot}:{spec["variant"]}',shot=shot,variant=spec['variant'],name=name,
                 run_dir=saved['run_dir'] if saved else str(directory/name),status='complete' if saved else 'pending',
                 request_signature=planned['targets'][0]['request_signature'],existing=bool(saved))
        old=next((j for j in previous.get('jobs',[]) if j['id']==job['id']),None)
        if old:
            assert old['request_signature']==job['request_signature']
            job['existing']=old['existing']
            if not saved: job['run_dir']=old['run_dir']
        variant['output_dir']=Path(job['run_dir'])
        state['jobs'].append(job)
        return job,variant,planned

    five_jobs=[await attach(spec,variant,planned,5) for (spec,variant),planned in zip(prepared,plan['groups'])]
    RUNTIME.mkdir(parents=True,exist_ok=True)
    write_json(RUNTIME/'preflight.json',dict(**state,existing_5shot_groups=sum(j['status']=='complete' for j,_,_ in five_jobs),
        new_5shot_image_tasks=300*sum(j['status']!='complete' for j,_,_ in five_jobs),max_followup_image_tasks=1800))
    if dry_run:
        print(json.dumps(read_json(RUNTIME/'preflight.json'),ensure_ascii=False,indent=2)); return
    baseline=RUNTIME/'baseline_hashes.json'
    if not baseline.is_file(): write_json(baseline,protected_files(namespace))
    if STOP.exists(): STOP.unlink()

    def persist():
        state['updated']=now(); write_json(STATE,state); save_reports(state)

    async def process(job,variant,planned):
        if job['status']=='complete':
            logging.info('Preserve complete group: %s',job['run_dir']); return
        destination=Path(job['run_dir'])
        if STOP.exists(): raise asyncio.CancelledError('STOP requested')
        if destination.exists():
            saved=read_json(destination/'run.json')
            assert saved['dataset_sha256']==DATASET and saved['targets'][0]['request_signature']==job['request_signature']
            archive_partial(destination,destination.parent)
        job.update(status='running',started=now()); state['current_group']=job['id']; persist()
        task=asyncio.create_task(run(variant,shared_items=copy.deepcopy(items),references_prepared=True,result_dir=destination))
        try:
            while not task.done():
                await asyncio.wait({task},timeout=1)
                if STOP.exists(): task.cancel()
            result=verify(await task,planned,job['shot'])
        finally:
            if not task.done(): task.cancel(); await asyncio.gather(task,return_exceptions=True)
        job.update(status='complete',finished=now(),executed_inference_count=result['executed_inference_count'],
                   reused_prediction_count=result['reused_prediction_count'])
        persist()

    persist()
    try:
        for job,variant,planned in five_jobs: await process(job,variant,planned)
        chosen=rank_top3([g for g in groups_for(state) if g['experiment']['positive_count']==5])
        selected=[dict(variant=g['experiment']['variant'],run_dir=g['run_dir'],
                       request_signature=g['targets'][0]['request_signature'],
                       **{k:g['metrics'][k] for k in ('mAP50','mAP50_95','recall','f2')}) for g in chosen]
        if state['top3']: assert state['top3']==selected, 'Frozen top-three ranking differs'
        state.update(top3=selected,stage='top3_1shot_10shot')
        ranking_file=RUNTIME/'top3.json'
        if ranking_file.is_file():
            frozen=read_json(ranking_file)
            assert frozen['groups']==selected and frozen['dataset_sha256']==DATASET
        else:
            write_json(ranking_file,dict(selected_at=now(),dataset_sha256=DATASET,
                       metric='macro mAP50; ties mAP50_95/recall/f2 descending then variant ascending',groups=selected))
        followups=[]
        for shot in (1,10):
            for spec,variant in prepare(config,items,shot,[g['variant'] for g in selected]):
                planned=await run(variant,dry_run=True,shared_items=copy.deepcopy(items),references_prepared=True,
                                  verify_shared_images=False,result_dir=Path(variant['output_dir']))
                assert planned['dataset_sha256']==DATASET
                assert planned['targets'][0]['prompt_sha256']==plan['groups'][0]['targets'][0]['prompt_sha256']
                followups.append(await attach(spec,variant,planned,shot))
        persist()
        for job,variant,planned in followups: await process(job,variant,planned)
        finish_verification(state)
    except BaseException as exc:
        status='interrupted' if isinstance(exc,(KeyboardInterrupt,asyncio.CancelledError)) else 'failed'
        for job in state['jobs']:
            if job['status']=='running': job.update(status=status,error=str(exc))
        state.update(status=status,error=str(exc)); persist(); raise
    state.update(status='complete',stage='complete',current_group=None); persist()
    print('COMPLETE: '+str(namespace.parent/'qwen27b_expanded_full5shot_top3.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--stop',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stop:
        RUNTIME.mkdir(parents=True,exist_ok=True); STOP.write_text('stop requested\n',encoding='utf-8')
    else:
        try:
            with batch_lock(): run_async(execute(args.dry_run))
        except (KeyboardInterrupt,asyncio.CancelledError): raise SystemExit(130)
        except Exception as exc: logging.exception('Batch stopped: %s',exc); raise SystemExit(1)

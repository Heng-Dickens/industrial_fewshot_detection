"""Complete saved Qwen 27B input-ablation gaps, then refresh the final tables.

Plans only missing/incomplete first-stage conditions. Four detached workers,
two image tasks each; successful requests are reused on resume. Does not change
proxy routing, prompts, generation parameters or frozen support/test data.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
sys.path.insert(0,str(ROOT/'src'))
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json,write_json
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.outputs import group_name
from industrial_fewshot.runtime import run_async

RUNTIME=ROOT/'outputs/_shared/runtime/qwen27b_input_supplement_20261103'
REPORT=ROOT/'outputs/Qwen/comparisons/input_supplement_20261103'
CONFIGS={scene:ROOT/f'configs/specific/{scene}_qwen27b_input_supplement.yaml'
         for scene in ('chache','diaowu')}
GT={'chache':dict(zip(['Chache1','Chache2','Chache3','Chache4','Qianyinche'],[40,30,40,30,48])),
    'diaowu':{'hanging object':350}}
MODEL='qwen3.8-27b'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs():return read_json(RUNTIME/'plan.json')['jobs']


def state_path(job):return RUNTIME/'jobs'/(job['id']+'.json')


def live(pid):
    try:return Path(f'/proc/{pid or 0}/cmdline').exists() and SCRIPT.name in Path(f'/proc/{pid}/cmdline').read_text()
    except OSError:return False


def isolate_loopback():
    for path in CONFIGS.values():
        if urlparse(load_config(path)['targets'][0]['config']['base_url']).hostname not in {'127.0.0.1','localhost'}:
            raise ValueError('Review proxy isolation for a non-loopback endpoint.')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(key,None)


def configuration(job):
    cfg=load_config(CONFIGS[job['scene']])
    cfg['experiment_suite'].update(positive_counts=[job['shot']],positive_variants=[job['variant']],
        negative_variants=[False],include_prompt_only=False)
    cfg['resolved_config']['experiment_suite']=copy.deepcopy(cfg['experiment_suite'])
    return cfg


def variant_name(job):
    names={'O':'original','C':'crop','B':'boxed','J':'json'}
    return group_name(dict(positive_count=job['shot'],negative_count=0,
        positive_inputs=[names[token] for token in job['variant'].split('+')]))


def matching_comparison(state):
    matches=[]
    namespace=Path(state['comparison_dir']).parent
    for directory in namespace.glob(f'{MODEL}_{state["shot"]}shot*'):
        metadata=directory/variant_name(state)/'run.json'
        if not metadata.is_file() or not (directory/'comparison.json').is_file():continue
        saved=read_json(metadata)
        if (saved.get('dataset_sha256')==state['dataset_sha256']
                and saved.get('targets',[{}])[0].get('request_signature')==state['request_signature']
                and read_json(directory/'comparison.json').get('experiment_suite')==state['experiment_suite']):
            matches.append((metadata.stat().st_mtime_ns,directory))
    return max(matches,key=lambda x:x[0])[1] if matches else None


def successful(state):
    if state['status']=='completed':return 300
    directory=matching_comparison(state)
    path=directory/variant_name(state)/'records.jsonl' if directory else None
    seen=set()
    if path and path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:record=json.loads(line)
            except ValueError:continue
            if record.get('status')=='success':seen.add((record['relative_path'],record['target_id']))
    return len(seen)


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    states=[read_json(state_path(job)) for job in jobs()]
    lines=['# Qwen3.8-27B输入消融补齐','',
        '仅补第一部分缺失/未完成条件；已有5shot与10shot成功实验保留。',
        '叉车沿用v2，吊物沿用v1及完整性复核；无负例，seed0，各自固定300张测试图。',
        '叉车1个worker、吊物3个worker，每个并发2张图；保留成功缓存，失败可恢复，不改代理全局路由。','',
        '| 场景 | shot/类 | 输入 | 状态 | 成功/总图 | AP50 | AP50:95 | 结果 |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | --- |']
    total=0
    for state in states:
        count=successful(state);total+=count
        status=state['status']
        if status=='running' and not live(state.get('pid')):status='interrupted_process_exited'
        metrics=state.get('metrics',{})
        path=Path(state['run_dir'])/'evaluation.txt' if state.get('run_dir') else None
        result=f'[查看]({os.path.relpath(path,REPORT)})' if path and path.is_file() else '—'
        cells=[state['scene'],str(state['shot']),state['variant'],status,f'{count}/300',
            *[f'{100*metrics[k]:.2f}' if k in metrics else '—' for k in ('mAP50','mAP50_95')],result]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['',f'累计成功：{total}/{300*len(states)}张次。',
        '', '[按输入→shot→模型阅读最终Qwen表](../../../_shared/comparisons/final_results/01_qwen_inputs.md)。',
        '每组完成自动刷新完整明细、Qwen子表与全局代表方案；若最佳输入改变，后两阶段采用新输入并明确显示缺项。',
        '', '命令：`python scripts/run_qwen27b_input_supplement.py --prepare`、`--probe`、`--background`；',
        '查看/刷新进度：`python scripts/run_qwen27b_input_supplement.py --status`。','']
    (REPORT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


async def prepare():
    if (RUNTIME/'plan.json').exists():raise FileExistsError('Plan exists; use --probe/--background/--status.')
    document=read_json(ROOT/'outputs/FINAL_COMPARISON.json')
    raw={r['run_dir']:r for r in document['primary_rows']}
    planned=[]
    for stage in document['qwen_analysis']['stages']:
        if stage['stage']!='inputs':continue
        for condition in stage['conditions']:
            if condition['run_dir'] and raw[condition['run_dir']]['eligible_for_comparison']:continue
            scene=condition['dataset'].removesuffix('_val300')
            job=dict(scene=scene,shot=condition['inference_shot_per_class'],variant=condition['input_variant'])
            job['id']=f'{scene}_{job["shot"]}shot_{job["variant"].replace("+", "")}'
            if state_path(job).exists():
                saved=read_json(state_path(job))
                if saved['status']=='prepared' and saved['config_sha256']==sha(CONFIGS[scene]):
                    planned.append(job)
                    print('Retain prepared plan:',job['id'],flush=True)
                    continue
            cfg=configuration(job)
            result=await run_ablation(cfg,dry_run=True)
            if result['image_count']!=300 or result['group_count']!=1 or result['excluded_reference_overlap']:
                raise ValueError('Unexpected dataset/group/overlap.')
            group=result['groups'][0];target=group['targets'][0]
            existing=next(r for r in raw.values() if r['algorithm']=='Qwen' and r['dataset']==condition['dataset']
                and r['model']==MODEL and r['prompt_version']==condition['prompt_version']
                and r['input_variant']==job['variant'] and r['inference_shot_per_class']==5
                and r['negative_references']==0 and r['eligible_for_comparison'])
            if (target['model']!=MODEL or target.get('labels',[target['label']])!=list(GT[scene])
                    or group['dataset_sha256']!=existing['dataset_sha256']
                    or target['prompt_snapshot']['text_sha256']!=existing['prompt_sha256']):
                raise ValueError('Model, labels, prompt or cohort identity differs from the existing comparison.')
            references=target['reference_count']
            if references['reference_originals']!=job['shot']*len(GT[scene]) or references['reference_negatives']:
                raise ValueError('Wrong frozen support budget.')
            state=dict(job,status='prepared',config_sha256=sha(CONFIGS[scene]),
                dataset_sha256=group['dataset_sha256'],prompt_sha256=target['prompt_sha256'],
                request_signature=target['request_signature'],reference_count=references,
                experiment_suite=result['experiment_suite'],comparison_dir=result['comparison_dir'],
                previous_run=condition['run_dir'])
            write_json(state_path(job),state);planned.append(job)
            print('Prepared:',job['id'],references,flush=True)
    if not planned:raise ValueError('No missing first-stage conditions.')
    write_json(RUNTIME/'plan.json',dict(model=MODEL,image_tasks=300*len(planned),jobs=planned))
    render()


async def probe():
    isolate_loopback()
    for scene in CONFIGS:
        job=next(j for j in jobs() if j['scene']==scene)
        cfg=configuration(job)
        cfg.update(results_layout=False,output_dir=RUNTIME/'probes'/scene)
        cfg['resolved_config'].update(results_layout=False,output_dir=str(cfg['output_dir']))
        result=await run_ablation(cfg,limit=1)
        if result['status']!='complete' or result['failed_request_count']:
            raise RuntimeError(f'{scene}: image probe failed; inspect preserved records.')
        write_json(RUNTIME/f'probe_{scene}.json',dict(status='passed',run_dir=result['groups'][0]['run_dir']))
        print('Image probe passed:',scene,flush=True)


def verify(group,state):
    target=group['targets'][0]
    if (group['image_count']!=300 or group['dataset_sha256']!=state['dataset_sha256']
            or target['model']!=MODEL or target['request_signature']!=state['request_signature']
            or group['evaluation_status']!='complete' or group['failed_request_count']):
        raise ValueError('Incomplete or changed run identity.')
    metrics=group['metrics']
    if {k:v['ground_truth_count'] for k,v in metrics['per_class'].items()}!=GT[state['scene']]:
        raise ValueError('Wrong class scope.')
    path=Path(group['run_dir'])
    records=[json.loads(line) for line in (path/'records.jsonl').read_text().splitlines() if line.strip()]
    if (len(records)!=300 or len({r['relative_path'] for r in records})!=300
            or any(r['status']!='success' for r in records)
            or {r['relative_path'] for r in records}!=set(group['image_paths'])):
        raise ValueError('Incomplete prediction records.')
    for manifest in group['support_manifests']:
        if sha(manifest['path'])!=manifest['sha256'] or manifest['positive_count']!=state['shot']:
            raise ValueError('Frozen support identity changed.')
    state.update(status='completed',run_dir=group['run_dir'],comparison_dir=str(path.parent),
        metrics={k:metrics[k] for k in ('mAP50','mAP50_95','precision','recall','f2')},
        metrics_sha256=sha(path/'metrics.json'),records_sha256=sha(path/'records.jsonl'),
        executed_inference_count=group['executed_inference_count'],
        reused_prediction_count=group['reused_prediction_count'],finished=time.time())


async def worker(index):
    isolate_loopback()
    assigned=([j for j in jobs() if j['scene']=='chache'] if index==0 else
              [j for j in jobs() if j['scene']=='diaowu'][index-1::3])
    for job in assigned:
        state=read_json(state_path(job))
        if state['status']=='completed':continue
        if state['status']=='running' and live(state.get('pid')):raise RuntimeError('Job already has a live owner.')
        if sha(CONFIGS[job['scene']])!=state['config_sha256']:raise ValueError('Config changed since preparation.')
        cfg=configuration(job)
        directory=matching_comparison(state)
        if directory:cfg['resume_comparison_dir']=directory
        state.pop('error',None)
        state.update(status='running',pid=os.getpid(),started=time.time())
        write_json(state_path(job),state)
        try:
            for attempt in range(3):
                result=await run_ablation(cfg)
                state['comparison_dir']=result['comparison_dir']
                write_json(state_path(job),state)
                if result['failed_request_count'] and attempt<2:
                    cfg['resume_comparison_dir']=Path(result['comparison_dir'])
                    continue
                verify(result['groups'][0],state)
                break
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,asyncio.CancelledError)) else 'failed',
                error=str(exc),finished=time.time())
            raise
        finally:write_json(state_path(job),state)
        print('Completed:',job['id'],state['metrics'],flush=True)


def refresh():
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_all_algorithms.py')],cwd=ROOT,check=True)


def launch():
    for scene in CONFIGS:
        if not any(job['scene']==scene for job in jobs()):continue
        if read_json(RUNTIME/f'probe_{scene}.json')['status']!='passed':raise ValueError('Run --probe first.')
    previous=read_json(RUNTIME/'controller.json') if (RUNTIME/'controller.json').exists() else {}
    if previous.get('status')=='running' and live(previous.get('pid')):raise RuntimeError('Controller already running.')
    processes=[]
    write_json(RUNTIME/'controller.json',dict(status='running',pid=os.getpid(),started=time.time()))
    completed={j['id'] for j in jobs() if read_json(state_path(j))['status']=='completed'}
    try:
        worker_indices=([0] if any(j['scene']=='chache' for j in jobs()) else [])
        worker_indices+=list(range(1,1+min(3,sum(j['scene']=='diaowu' for j in jobs()))))
        for index in worker_indices:
            with (RUNTIME/f'worker_{index}.log').open('a') as stream:
                processes.append(subprocess.Popen([sys.executable,str(SCRIPT),'--worker',str(index)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONUNBUFFERED='1')))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Worker failed; caches retained.')
            render()
            current={j['id'] for j in jobs() if read_json(state_path(j))['status']=='completed'}
            if current!=completed:refresh();completed=current
            time.sleep(5)
        if any(p.returncode for p in processes):raise RuntimeError('Worker failed; caches retained.')
        refresh();render()
        write_json(RUNTIME/'controller.json',dict(status='completed',pid=os.getpid(),finished=time.time()))
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:p.terminate()
        write_json(RUNTIME/'controller.json',dict(status='interrupted_or_failed',pid=os.getpid(),error=str(exc)))
        render()
        raise


def background():
    if not (RUNTIME/'plan.json').exists():raise ValueError('Run --prepare first.')
    previous=read_json(RUNTIME/'controller.json') if (RUNTIME/'controller.json').exists() else {}
    if previous.get('status')=='running' and live(previous.get('pid')):raise RuntimeError('Controller already running.')
    with (RUNTIME/'controller.log').open('a') as stream:
        p=subprocess.Popen([sys.executable,str(SCRIPT),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONUNBUFFERED='1'),start_new_session=True,close_fds=True)
    print('Detached controller PID:',p.pid,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for option in ('prepare','probe','background','run','status'):group.add_argument('--'+option,action='store_true')
    group.add_argument('--worker',type=int,choices=range(4),help=argparse.SUPPRESS)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.prepare:run_async(prepare())
    elif args.probe:run_async(probe())
    elif args.worker is not None:run_async(worker(args.worker))
    elif args.background:background()
    elif args.run:launch()
    else:
        render()
        for job in jobs():
            state=read_json(state_path(job))
            print(job['id'],state['status'],f'{successful(state)}/300',state.get('metrics',''))


if __name__=='__main__':main()

"""Fill the agreed Flash matrix on frozen 300-image cohorts, retaining success caches.

Four workers, two concurrent image tasks each. Hanging-object settings retain
the 27B completeness review. The proxy response must identify qwen3.8-flash.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
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
sys.path.insert(0,str(ROOT/'src'))
from industrial_fewshot.config import load_config
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.runtime import run_async
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.paths import resolve_location

RUNTIME=ROOT/'outputs/_shared/runtime/qwenflash_supplement_20261103'
REPORT=ROOT/'outputs/Qwen/comparisons/flash_supplement_20261103'
CONFIGS={
    'chache':ROOT/'configs/specific/chache_qwenflash_oc_shots.yaml',
    'diaowu':ROOT/'configs/specific/diaowu_qwenflash_visual_shots.yaml'}
JOBS=[dict(id=f'{scene}_{shot}shot_{variant.replace("+", "")}',scene=scene,shot=shot,variant=variant)
      for scene in CONFIGS for shot in [1,5,10]
      for variant in (['O+C'] if scene=='chache' else ['O+B','O+C'])]
LABELS={'chache':['Chache1','Chache2','Chache3','Chache4','Qianyinche'], 'diaowu':['hanging object']}
GT={'chache':dict(zip(LABELS['chache'],[40,30,40,30,48])), 'diaowu':{'hanging object':350}}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(job):
    cfg=load_config(CONFIGS[job['scene']])
    cfg['experiment_suite'].update(positive_counts=[job['shot']], positive_variants=[job['variant']],
                                   include_prompt_only=False, negative_variants=[False])
    cfg['resolved_config']['experiment_suite']=copy.deepcopy(cfg['experiment_suite'])
    return cfg


def local_client_environment():
    # Only the child process environment changes. The configured proxy itself
    # controls upstream access; localhost must not be sent to shell SOCKS proxies.
    for path in CONFIGS.values():
        if urlparse(load_config(path)['targets'][0]['config']['base_url']).hostname not in {'127.0.0.1','localhost'}:
            raise ValueError('Review proxy-environment isolation for a non-loopback endpoint.')
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
        os.environ.pop(key,None)


def state_path(job):return RUNTIME/'jobs'/(job['id']+'.json')


def live(pid):
    command=Path(f'/proc/{pid or 0}/cmdline')
    try:return command.exists() and 'run_qwenflash_supplement.py' in command.read_text()
    except OSError:return False


def matching_comparison(state):
    """Find this variant by frozen request identity, not a shared model/shot name."""
    namespace=Path(state['comparison_dir']).parent
    variant={'O+B':'原图+带框图_无负例','O+C':'原图+裁剪图_无负例'}[state['variant']]
    matches=[]
    for directory in namespace.glob(f'qwen3.8-flash_{state["shot"]}shot*'):
        metadata=directory/variant/'run.json'
        if not metadata.is_file():continue
        saved=read_json(metadata)
        if (saved.get('dataset_sha256')==state['dataset_sha256']
                and saved.get('targets',[{}])[0].get('request_signature')==state['request_signature']
                and (directory/'comparison.json').is_file()):
            matches.append((metadata.stat().st_mtime_ns,directory))
    return max(matches,key=lambda p:p[0])[1] if matches else None


def successful_images(state):
    if state['status']=='completed':return 300
    comparison=matching_comparison(state)
    if comparison is None:return 0
    variant={'O+B':'原图+带框图_无负例','O+C':'原图+裁剪图_无负例'}[state['variant']]
    records=comparison/variant/'records.jsonl'
    seen=set()
    if records.is_file():
        for line in records.read_text(encoding='utf-8').splitlines():
            try:row=json.loads(line)
            except ValueError:continue  # A concurrent writer may be appending its last line.
            if row.get('status')=='success':seen.add((row['relative_path'],row['target_id']))
    return len(seen)


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    lines=['# Flash补齐：叉车O+C与吊物O+B/O+C','',
        '固定两个300张测试集、seed0、各1/5/10shot、无负例。共9组、2700张图像任务。',
        '叉车沿用加长v2五类联合提示；吊物沿用原27B的提示、离地判定、逐框理由及完整性复核。',
        '使用现有Claude Messages转发别名，响应模型必须严格为qwen3.8-flash；不修改服务全局路由。',
        '叉车按shot顺序运行、并发2；吊物按三个shot并行、各并发2，总并发最多8个图像任务；成功记录缓存保留。', '',
        '| 场景 | shot/类 | 输入 | 状态 | 成功/总图 | AP50 | AP50:95 | P | R | F2 | 结果 |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|']
    total=0
    errors=[]
    for job in JOBS:
        state=read_json(state_path(job)) if state_path(job).exists() else dict(status='not_started')
        metrics=state.get('metrics',{})
        values=[f'{100*metrics[k]:.2f}' if k in metrics else '—' for k in ['mAP50','mAP50_95','precision','recall','f2']]
        path=Path(state['run_dir'])/'evaluation.txt' if state.get('run_dir') else None
        result=f'[查看]({os.path.relpath(path,REPORT)})' if path and path.is_file() else '—'
        count=successful_images(state) if 'comparison_dir' in state else 0
        total+=count
        status=state['status']
        if status=='running' and not live(state.get('pid')):status='interrupted_process_exited'
        lines.append('| '+' | '.join([job['scene'],str(job['shot']),job['variant'],status,f'{count}/300',*values,result])+' |')
        if state.get('error'):
            errors.append(job['id'])
    lines+=['',f'累计成功：{total}/2700张次（{100*total/2700:.1f}%）；成功数按逐图记录去重，恢复时不重复计数。']
    if errors:lines+=['','存在错误记录的组：'+', '.join(errors)+'；完整错误保留在runtime状态和日志中。']
    lines+=['','运行：`python scripts/run_qwenflash_supplement.py --prepare` → `--probe` → `--background`。',
            '查看/刷新进度：`python scripts/run_qwenflash_supplement.py --status`。',
            '所有组完成后自动刷新Qwen子表、原始明细及全局比较；不完整结果不参与最优值排名。','']
    (REPORT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


async def prepare():
    if (RUNTIME/'plan.json').exists():raise FileExistsError('Plan exists; use --probe/--run/--status.')
    plans=[]
    for job in JOBS:
        cfg=configuration(job)
        plan=await run_ablation(cfg,dry_run=True)
        if plan['group_count']!=1 or plan['image_count']!=300 or plan['excluded_reference_overlap']:
            raise ValueError('Unexpected dataset/group/overlap in dry run.')
        group=plan['groups'][0];target=group['targets'][0]
        names=target.get('labels',[target['label']])
        if names!=LABELS[job['scene']] or target['model']!='qwen3.8-flash':
            raise ValueError('Model/class contract differs.')
        refs=target['reference_count'];n=job['shot']*len(names)
        expected_crops=0
        if job['variant']=='O+C':
            for label in names:
                reference=ROOT/'data/references'/job['scene']
                if job['scene']=='chache':reference=reference/label
                frozen=read_json(reference/f'{job["shot"]}shot/seed_0.json')
                for chosen in frozen['positive_samples']:
                    annotation=read_json(resolve_location(Path(chosen['json_path'])))
                    expected_crops+=sum(s['label'] in names for s in annotation['shapes'])
        expected_boxed=n if job['variant']=='O+B' else 0
        if (refs['reference_originals']!=n or refs['reference_crops']!=expected_crops
                or refs['reference_boxed_images']!=expected_boxed or refs['reference_negatives']):
            raise ValueError('Visual input budget differs.')
        state=dict(job,status='prepared',config=str(CONFIGS[job['scene']]),
            config_sha256=sha(CONFIGS[job['scene']]),dataset_sha256=group['dataset_sha256'],
            request_signature=target['request_signature'],prompt_sha256=target['prompt_sha256'],
            reference_manifest_sha256=target['reference_manifest_sha256'],reference_count=refs,
            comparison_dir=plan['comparison_dir'])
        write_json(state_path(job),state);plans.append(state)
        print(job['id'],refs,flush=True)
    write_json(RUNTIME/'plan.json',dict(image_tasks=2700,required_response_model='qwen3.8-flash',
        proxy_request_model='claude-sonnet-4-6',workers=4,concurrency_per_worker=2,jobs=plans))
    render()


async def probe():
    local_client_environment()
    for scene in CONFIGS:
        job=next(j for j in JOBS if j['scene']==scene and j['shot']==1 and j['variant']=='O+C')
        cfg=configuration(job)
        cfg.update(results_layout=False,output_dir=RUNTIME/'probes'/scene)
        cfg['resolved_config'].update(results_layout=False,output_dir=str(cfg['output_dir']))
        result=await run_ablation(cfg,limit=1)
        if result['failed_request_count'] or result['status']!='complete':
            raise RuntimeError(f'{scene} Flash image probe failed; inspect preserved feedback.')
        group=result['groups'][0]
        write_json(RUNTIME/f'probe_{scene}.json',dict(status='passed',run_dir=group['run_dir'],
            request_signature=group['targets'][0]['request_signature'],required_response_model='qwen3.8-flash',
            transport_validation='Strict response.model validation in claude_messages.read_message'))
        print('Flash image probe passed:',scene,flush=True)


def verify(group,state):
    if (group['dataset_sha256']!=state['dataset_sha256'] or group['image_count']!=300
            or group['targets'][0]['request_signature']!=state['request_signature']
            or group['targets'][0]['model']!='qwen3.8-flash' or group['failed_request_count']
            or group['evaluation_status']!='complete'):
        raise ValueError('Incomplete or changed run identity.')
    metric=group['metrics']
    if {c:v['ground_truth_count'] for c,v in metric['per_class'].items()}!=GT[state['scene']]:
        raise ValueError('Wrong class scope/test cohort.')
    path=Path(group['run_dir'])
    rows=[json.loads(line) for line in (path/'records.jsonl').read_text().splitlines() if line.strip()]
    if len(rows)!=300 or len({r['relative_path'] for r in rows})!=300 or any(r['status']!='success' for r in rows):
        raise ValueError('Missing/failed prediction records.')
    if {r['relative_path'] for r in rows}!=set(group['image_paths']):
        raise ValueError('Prediction/test identity differs.')
    for m in group['support_manifests']:
        if sha(m['path'])!=m['sha256'] or m['positive_count']!=state['shot']:
            raise ValueError('Frozen support identity differs.')
    state.update(status='completed',run_dir=group['run_dir'],comparison_dir=str(path.parent),
        metrics={k:metric[k] for k in ['mAP50','mAP50_95','precision','recall','f2']},
        executed_inference_count=group['executed_inference_count'],reused_prediction_count=group['reused_prediction_count'],
        metrics_sha256=sha(path/'metrics.json'),records_sha256=sha(path/'records.jsonl'),finished=time.time())


async def worker(scene,shot=None):
    local_client_environment()
    for job in [j for j in JOBS if j['scene']==scene and (shot is None or j['shot']==shot)]:
        state=read_json(state_path(job))
        if state['status']=='completed':continue
        cfg=configuration(job)
        if sha(CONFIGS[scene])!=state['config_sha256']:raise ValueError('Config changed since preparation.')
        if state['status']=='running' and live(state.get('pid')):
            raise RuntimeError(f'Another live worker owns {job["id"]}.')
        comparison=matching_comparison(state)
        if comparison is not None:
            cfg['resume_comparison_dir']=comparison
            state['comparison_dir']=str(comparison)
        if state['status'] in {'running','failed','interrupted'}:
            state.setdefault('resume_history',[]).append(dict(previous_pid=state.get('pid'),
                previous_status=state['status'],resumed_at=time.time(),successful_images=successful_images(state)))
        state.pop('error',None)
        state.update(status='running',pid=os.getpid(),last_started=time.time())
        state.setdefault('started',time.time())
        write_json(state_path(job),state)
        try:
            for attempt in range(3):
                result=await run_ablation(cfg)
                state['comparison_dir']=result['comparison_dir']
                write_json(state_path(job),state)
                if result['failed_request_count'] and attempt<2:
                    logging.warning('Retry only failed predictions for %s; successful caches retained.',job['id'])
                    cfg['resume_comparison_dir']=Path(result['comparison_dir'])
                    continue
                verify(result['groups'][0],state)
                break
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,asyncio.CancelledError)) else 'failed',
                         error=str(exc),finished=time.time())
            raise
        finally:
            write_json(state_path(job),state)
        print('Completed:',job['id'],state['metrics'],flush=True)


def launch():
    if not (RUNTIME/'plan.json').exists():raise ValueError('Run --prepare first.')
    controller=RUNTIME/'controller.json'
    if controller.exists():
        previous=read_json(controller)
        command=Path(f'/proc/{previous.get("pid",0)}/cmdline')
        if previous['status']=='running' and command.exists() and 'run_qwenflash_supplement.py' in command.read_text():
            raise RuntimeError('This matrix already has a live controller; use --status.')
    for scene in CONFIGS:
        if not (RUNTIME/f'probe_{scene}.json').exists() or read_json(RUNTIME/f'probe_{scene}.json')['status']!='passed':
            raise ValueError('Run --probe before the full matrix.')
    processes=[]
    for scene,shot in [('chache',None),('diaowu',1),('diaowu',5),('diaowu',10)]:
        log=RUNTIME/(f'{scene}_{shot}shot.log' if shot else f'{scene}.log')
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',scene]
        if shot:command+=['--shot',str(shot)]
        with log.open('a') as stream:
            p=subprocess.Popen(command,cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        processes.append(p)
        print(scene,shot or 'all shots','PID',p.pid,'log',log,flush=True)
    write_json(RUNTIME/'controller.json',dict(status='running',pid=os.getpid(),child_pids=[p.pid for p in processes]))
    try:
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in {None,0} for p in processes):
                raise RuntimeError('One Flash worker failed; stopping this batch and retaining successful caches.')
            render();time.sleep(2)
        codes=[p.returncode for p in processes]
        render()
        if any(codes):raise RuntimeError(f'Scene worker failed: {codes}; successful predictions retained.')
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_all_algorithms.py')],cwd=ROOT,check=True)
        write_json(RUNTIME/'controller.json',dict(status='completed',pid=os.getpid(),child_pids=[p.pid for p in processes]))
    except BaseException as exc:
        # A user stop cancels the entire batch; never silently restart workers.
        for p in processes:
            if p.poll() is None:p.terminate()
        write_json(RUNTIME/'controller.json',dict(status='interrupted_or_failed',error=str(exc),pid=os.getpid()))
        raise


def background():
    if not (RUNTIME/'plan.json').exists():raise ValueError('Run --prepare first.')
    controller=RUNTIME/'controller.json'
    if controller.exists():
        previous=read_json(controller)
        if previous['status']=='running' and live(previous.get('pid')):
            raise RuntimeError('A live controller already exists; use --status.')
    with (RUNTIME/'controller.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONUNBUFFERED='1'),start_new_session=True,close_fds=True)
    write_json(RUNTIME/'background_launcher.json',dict(pid=process.pid,started=time.time(),
        detached_session=True,stdio='independent log files, stdin=/dev/null'))
    print('Detached Flash controller PID:',process.pid,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--probe',action='store_true')
    group.add_argument('--run',action='store_true');group.add_argument('--status',action='store_true')
    group.add_argument('--background',action='store_true')
    group.add_argument('--worker',choices=list(CONFIGS),help=argparse.SUPPRESS)
    parser.add_argument('--shot',type=int,choices=[1,5,10],help=argparse.SUPPRESS)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.prepare:run_async(prepare())
    elif args.probe:run_async(probe())
    elif args.worker:run_async(worker(args.worker,args.shot))
    elif args.run:launch()
    elif args.background:background()
    else:
        render()
        for job in JOBS:
            s=read_json(state_path(job)) if state_path(job).exists() else dict(status='not_started')
            status=s['status']
            if status=='running' and not live(s.get('pid')):status='interrupted_process_exited'
            print(job['id'],status,f'{successful_images(s) if "comparison_dir" in s else 0}/300',s.get('metrics',''))


if __name__=='__main__':main()

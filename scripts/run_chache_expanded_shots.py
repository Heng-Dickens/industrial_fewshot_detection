"""Run the expanded-v2 27B/Flash x 1/5/10shot matrix with validated success reuse."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json,write_json
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.labels import all_label_names
from industrial_fewshot.runtime import run_async
from industrial_fewshot.paths import resolve_location

RUNTIME=ROOT/'outputs/runtime/chache_expanded_shots'
STATE=RUNTIME/'status.json'
REPORT=ROOT/'outputs/runs/chache_val300/qwen_expanded_shot_comparison.md'
CONFIGS={
    'qwen3.8-27b':ROOT/'configs/specific/chache_qwen27b_expanded_shots.yaml',
    'qwen3.8-flash':ROOT/'configs/specific/chache_qwenflash_expanded_shots.yaml',
}
LABELS=['Chache1','Chache2','Chache3','Chache4','Qianyinche']
# Reuse known 5shot immediately; check Flash on actual image inference early.
ORDER=[('qwen3.8-27b',5),('qwen3.8-flash',1),('qwen3.8-27b',1),
       ('qwen3.8-27b',10),('qwen3.8-flash',5),('qwen3.8-flash',10)]


def now():return datetime.now(timezone.utc).isoformat()


def identity(plan):
    return dict(dataset_sha256=plan['dataset_sha256'],request_signature=plan['groups'][0]['targets'][0]['request_signature'])


def verify_group(group,job):
    assert group['dataset_sha256']==job['dataset_sha256'] and group['image_count']==300
    assert group['targets'][0]['request_signature']==job['request_signature']
    assert group['targets'][0]['model']==job['model'] and group['targets'][0]['labels']==LABELS
    assert group.get('evaluation_status')=='complete' and group['failed_request_count']==0
    assert {label:row['ground_truth_count'] for label,row in group['metrics']['per_class'].items()}==dict(zip(LABELS,[40,30,40,30,48]))
    rows=[json.loads(line) for line in (Path(group['run_dir'])/'records.jsonl').read_text(encoding='utf-8').splitlines() if line]
    assert len(rows)==len({r['relative_path'] for r in rows})==300
    assert all(r['status']=='success' for r in rows)
    assert {r['relative_path'] for r in rows}==set(group['image_paths'])
    assert len(group['support_manifests'])==5
    assert set(group['metrics']['score_threshold_results'])=={'0.5','0.7'}
    for manifest in group['support_manifests']:
        assert manifest['positive_count']==job['shot']
        path=resolve_location(Path(manifest['path']))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
        for row in read_json(path)['positive_samples']:
            assert hashlib.sha256(resolve_location(Path(row['image_path'])).read_bytes()).hexdigest()==row['image_sha256']
            assert hashlib.sha256(resolve_location(Path(row['json_path'])).read_bytes()).hexdigest()==row['annotation_sha256']
    return group


def find_complete(plan,job):
    # Look only in this dataset/prompt namespace; never copy or overwrite old runs.
    namespace=Path(plan['output_dir'])
    for file in sorted(namespace.glob('*/**/summary.json')):
        group=read_json(file)
        if (group.get('targets',[{}])[0].get('request_signature')!=job['request_signature']
                or group.get('dataset_sha256')!=job['dataset_sha256']):continue
        if group.get('evaluation')!=plan['groups'][0]['evaluation']:continue
        try:return verify_group(group,job)
        except (AssertionError,KeyError,OSError,ValueError):continue
    return None


def report(state):
    lines=['# 叉车加长版：Qwen3.8-27B / Flash 的1、5、10shot对比','',
        f"状态：{state['status']}。共{len(state['jobs'])}组，每组300张，{state['image_tasks']}张图像任务；已有完整结果经验证复用。",'',
        '固定加长版v2、同300张验证图与五类共188个GT框。shot按每类计，原图＋带框图，无负例、无额外复核；参考按seed0嵌套选择。','',
        '| 模型 | shot/类 | 状态 | AP50 | AP50:95 | Precision@0.5 | Recall@0.5 | F2@0.5 | 结果 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---|']
    percentage=lambda x:f'{x*100:.2f}'
    complete=[]
    for model in CONFIGS:
        for shot in [1,5,10]:
            job=next((j for j in state['jobs'] if j['model']==model and j['shot']==shot),None)
            if job is None:continue
            if job['status']=='complete':
                group=read_json(Path(job['run_dir'])/'summary.json');complete.append((job,group))
                m=group['metrics'];evaluation=Path(job['run_dir'])/'evaluation.txt'
                values=[percentage(m[k]) for k in ['mAP50','mAP50_95','precision','recall','f2']]
                lines.append('| '+' | '.join([model,str(shot),'完成（复用）' if job.get('reused_existing') else '完成',*values,f'[evaluation]({evaluation.as_posix()})'])+' |')
            else:lines.append(f"| {model} | {shot} | {job['status']} | - | - | - | - | - | - |")
    for job,group in complete:
        lines.extend(['',f"## {job['model']} / {job['shot']}shot",'',
            '| 类别 | GT | TP | FP | FN | AP50 | Recall@0.5 | F2@0.5 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for label,row in group['metrics']['per_class'].items():
            lines.append('| '+' | '.join(map(str,[label,row['ground_truth_count'],row['tp'],row['fp'],row['fn'],
                percentage(row['AP50']),percentage(row['recall']),percentage(row['f2'])]))+' |')
    errors=[j for j in state['jobs'] if j.get('error')]
    if errors:
        lines.extend(['','## 未完成原因',''])
        lines.extend(f"- {j['model']} / {j['shot']}shot：{j['error']}" for j in errors)
    lines.extend(['','各组同时保留score阈值0.5、0.7的结果；AP使用完整置信度排序。模型名称不符的响应不作为成功预测，也不计作该模型的实验结果。'])
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')


async def execute(dry_run=False,model_filter=None,concurrency=None):
    plans=[]
    for model,shot in ORDER:
        if model_filter and model!=model_filter:continue
        config=load_config(CONFIGS[model])
        if concurrency is not None:
            config['concurrency']=concurrency
            config['targets'][0]['config']['concurrency']=concurrency
            config['resolved_config']['concurrency']=concurrency
        config['experiment_suite']['positive_counts']=[shot]
        config['resolved_config']['experiment_suite']['positive_counts']=[shot]
        assert all_label_names(config['targets'])==LABELS and len(config['targets'])==1
        assert config['targets'][0]['config']['model']==model
        assert config['targets'][0]['prompt_file'].endswith('chache_v2_expanded.txt')
        assert not config['targets'][0]['config']['enable_completeness_review']
        plan=await run_ablation(config,dry_run=True)
        assert plan['image_count']==300 and plan['group_count']==1
        assert plan['modes']==[f'O_B_k{shot:02d}_pos']
        target=plan['groups'][0]['targets'][0]
        assert target['reference_count']['reference_originals']==shot*5
        assert target['reference_count']['reference_boxed_images']==shot*5
        assert not plan['excluded_reference_overlap']
        job=dict(model=model,shot=shot,status='pending',config=str(CONFIGS[model]),
            comparison_dir=plan['comparison_dir'],**identity(plan))
        existing=find_complete(plan,job)
        if existing:job.update(status='complete',run_dir=existing['run_dir'],
            comparison_dir=str(Path(existing['run_dir']).parent),reused_existing=True)
        plans.append((job,config,plan))
    assert len({p['dataset_sha256'] for _,_,p in plans})==1
    for shot in [1,5,10]:
        matching=[p['groups'][0]['targets'][0] for j,_,p in plans if j['shot']==shot]
        if len(matching)>1:
            assert matching[0]['prompt_sha256']==matching[1]['prompt_sha256']
            assert matching[0]['reference_manifest_sha256']==matching[1]['reference_manifest_sha256']
    if dry_run:
        print(json.dumps(dict(image_tasks=300*len(plans),existing_success_images=300*sum(j['status']=='complete' for j,_,_ in plans),
            jobs=[j for j,_,_ in plans]),ensure_ascii=False,indent=2));return
    previous=read_json(STATE) if STATE.is_file() else {}
    state=dict(status='running',started=previous.get('started',now()),updated=now(),image_tasks=300*len(plans),jobs=[])
    for job,_,_ in plans:
        old=next((j for j in previous.get('jobs',[]) if (j['model'],j['shot'])==(job['model'],job['shot'])),None)
        if old:
            assert old['dataset_sha256']==job['dataset_sha256']
            if old.get('previous_attempts'):job['previous_attempts']=old['previous_attempts']
            if old['request_signature']!=job['request_signature']:
                assert old['status']!='complete', 'A completed job identity changed; retain it as a separate experiment'
                job['previous_attempts']=[*old.get('previous_attempts',[]),old]
            elif job['status']!='complete':
                job['comparison_dir']=old['comparison_dir']
        state['jobs'].append(job)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    write_json(STATE,state);report(state)
    unavailable={}
    for job,(_,config,plan) in zip(state['jobs'],plans):
        if job['status']=='complete':continue
        if job['model'] in unavailable:
            job.update(status='not_run',error=unavailable[job['model']]);write_json(STATE,state);report(state);continue
        old=Path(job['comparison_dir'])/'comparison.json'
        if old.is_file():config['resume_comparison_dir']=old.parent
        job.update(status='running',started=now());state['updated']=now();write_json(STATE,state);report(state)
        try:
            result=await run_ablation(config)
            group=verify_group(result['groups'][0],job)
            job.update(status='complete',comparison_dir=result['comparison_dir'],run_dir=group['run_dir'],
                executed_inference_count=group['executed_inference_count'],reused_prediction_count=group['reused_prediction_count'],finished=now())
        except (asyncio.CancelledError,KeyboardInterrupt):
            job.update(status='interrupted');state.update(status='interrupted',updated=now());write_json(STATE,state);report(state);raise
        except Exception as exc:
            job.update(status='failed',error=str(exc),finished=now())
            logging.error('%s / %sshot failed: %s',job['model'],job['shot'],exc)
            if '与请求模型' in str(exc) or any(code in str(exc) for code in ['HTTP 401','HTTP 403','HTTP 404']):
                unavailable[job['model']]=str(exc)
        state['updated']=now();write_json(STATE,state);report(state)
    state.update(status='complete' if all(j['status']=='complete' for j in state['jobs']) else 'incomplete',updated=now())
    write_json(STATE,state);report(state)
    print('STATUS',state['status'],'REPORT',REPORT)
    return state['status']=='complete'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--model',choices=list(CONFIGS),help='Run one model using an independent state/report')
    parser.add_argument('--concurrency',type=int,choices=[1,2],help='Bound per-process concurrent requests')
    args=parser.parse_args()
    if args.model:
        RUNTIME=RUNTIME/args.model
        STATE=RUNTIME/'status.json'
        REPORT=ROOT/f'outputs/runs/chache_val300/{args.model}_expanded_shot_progress.md'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if not args.dry_run:
        RUNTIME.mkdir(parents=True,exist_ok=True)
        handler=logging.FileHandler(RUNTIME/'launcher.log',encoding='utf-8');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logging.getLogger().addHandler(handler)
    try:
        result=run_async(execute(args.dry_run,args.model,args.concurrency));raise SystemExit(0 if args.dry_run or result else 2)
    except KeyboardInterrupt:raise SystemExit(130)

"""Run exactly short/expanded x text-only/5shot O+B on the same 300-image cohort."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.labels import all_label_names
from industrial_fewshot.runtime import run_async

RUNTIME = ROOT/'outputs/runtime/chache_prompt_comparison'
STATE = RUNTIME/'status.json'
JOBS = [('v1_short', ROOT/'configs/specific/chache_qwen_prompt_comparison.yaml'),
        ('v2_expanded', ROOT/'configs/specific/chache_qwen_prompt_comparison_expanded.yaml')]
LABELS = ['Chache1','Chache2','Chache3','Chache4','Qianyinche']


def now():
    return datetime.now(timezone.utc).isoformat()


def plan_identity(plan):
    return dict(dataset_sha256=plan['dataset_sha256'],
        signatures=[g['targets'][0]['request_signature'] for g in plan['groups']])


def comparison_document(state):
    text = ['# 叉车 Qwen3.8-27B 提示词对比', '',
        '两轮各600张，共1200张图像任务；每张图在一次请求中检测五类。仅比较纯文字与每类5shot原图＋带框图，无负例、无二次复核。', '',
        '同一300张验证图、188个GT框、同一批每类5个参考。v0原始模板保留；实验比较用户修改后的v1短版与v2加长版。', '',
        '| 版本 | 输入 | 已处理图 | 失败 | AP50 | AP50:95 | Precision@0.5 | Recall@0.5 | F2@0.5 | 结果 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    comparisons = {}
    def fmt(value):return '-' if value is None else f'{value*100:.2f}'
    for job in state['jobs']:
        if job['status'] == 'pending':continue
        file = Path(job['comparison_dir'])/'comparison.json'
        if not file.is_file():continue
        report = read_json(file)
        comparisons[job['name']] = report
        for row,group in zip(report.get('rows',[]),report.get('groups',[])):
            m=group['metrics']
            evaluation=Path(group['run_dir'])/'evaluation.txt'
            text.append('| '+' | '.join(map(str,[job['name'], row['experiment_name'],group['image_count'],
                group['failed_request_count'],fmt(m['mAP50']),fmt(m['mAP50_95']),fmt(m['precision']),fmt(m['recall']),fmt(m['f2']),
                f'[evaluation]({evaluation.as_posix()})']))+' |')
    for name,report in comparisons.items():
        for group in report.get('groups',[]):
            m=group['metrics']
            text.extend(['',f"## {name} / {group['experiment']['name']}",'',
                '| 类别 | GT | TP | FP | FN | AP50 | Recall@0.5 | F2@0.5 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|'])
            for label in LABELS:
                row=m['per_class'][label]
                text.append('| '+' | '.join(map(str,[label,row['ground_truth_count'],row['tp'],row['fp'],row['fn'],
                    fmt(row['AP50']),fmt(row['recall']),fmt(row['f2'])]))+' |')
    if all(name in comparisons and comparisons[name]['status']=='complete' for name,_ in JOBS):
        text.extend(['','## 加长版相对短版的变化','',
            '| 输入 | ΔAP50（百分点） | ΔAP50:95 | ΔPrecision@0.5 | ΔRecall@0.5 | ΔF2@0.5 |',
            '|---|---:|---:|---:|---:|---:|'])
        for old,new in zip(comparisons['v1_short']['groups'],comparisons['v2_expanded']['groups']):
            before,after=old['metrics'],new['metrics']
            delta=[f"{(after[key]-before[key])*100:+.2f}" for key in ['mAP50','mAP50_95','precision','recall','f2']]
            text.append('| '+' | '.join([old['experiment']['name'],*delta])+' |')
        text.extend(['','分数阈值0.7的精确率、召回率、F2及类别表也保存在各组evaluation.txt与metrics.json；AP使用完整置信度排序。',
            '此比较每组仅运行一次，变化反映本次固定样本和随机采样条件下的结果，不能单凭一轮认定统计显著性。'])
    if state.get('error'):text.extend(['',f"当前未完成：{state['error']}"])
    output=ROOT/'outputs/runs/chache_val300/prompt_comparison.md'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(text)+'\n',encoding='utf-8')


async def execute(dry_run=False):
    plans=[]
    for name,path in JOBS:
        config=load_config(path)
        assert all_label_names(config['targets'])==LABELS and len(config['targets'])==1
        target=config['targets'][0]
        assert target['config']['model']=='qwen3.8-27b'
        assert not target['config']['enable_completeness_review']
        plan=await run_ablation(config,dry_run=True)
        assert plan['image_count']==300 and plan['group_count']==2
        assert plan['modes']==['prompt_only','O_B_k05_pos']
        assert all(len(g['targets'])==1 and g['image_count']==300 for g in plan['groups'])
        visual=plan['groups'][1]
        assert visual['targets'][0]['reference_count']['reference_originals']==25
        assert visual['targets'][0]['reference_count']['reference_boxed_images']==25
        assert len(visual['support_manifests'])==5
        assert not plan['excluded_reference_overlap']
        plans.append((name,path,config,plan))
    assert plans[0][3]['dataset_sha256']==plans[1][3]['dataset_sha256']
    assert plans[0][3]['groups'][1]['targets'][0]['reference_manifest_sha256']==plans[1][3]['groups'][1]['targets'][0]['reference_manifest_sha256']
    if dry_run:
        print(json.dumps(dict(total_image_tasks=1200,classes=LABELS,model='qwen3.8-27b',
            jobs=[dict(name=name,group_count=2,image_tasks=600,comparison_dir=plan['comparison_dir'],
                       **plan_identity(plan)) for name,_,_,plan in plans]),ensure_ascii=False,indent=2))
        return
    previous=read_json(STATE) if STATE.is_file() else {}
    state=dict(status='running',started=previous.get('started',now()),updated=now(),total_image_tasks=1200,jobs=[])
    for name,path,config,plan in plans:
        old=next((j for j in previous.get('jobs',[]) if j['name']==name),None)
        job=dict(name=name,config=str(path),status='pending',comparison_dir=plan['comparison_dir'],**plan_identity(plan))
        if old:
            if {k:old.get(k) for k in plan_identity(plan)}!=plan_identity(plan):
                raise ValueError('Saved experiment identity differs; preserve old state and start an explicit new comparison')
            job=old
        state['jobs'].append(job)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    write_json(STATE,state)
    for job,(_,path,config,plan) in zip(state['jobs'],plans):
        if job['status']=='pending':
            # Earlier dry runs cannot reserve a distinct namespace. After round1
            # exists, resolve round2 again before recording its resume location.
            refreshed=await run_ablation(config,dry_run=True)
            assert plan_identity(refreshed)==plan_identity(plan)
            job['comparison_dir']=refreshed['comparison_dir']
        comparison=Path(job['comparison_dir'])/'comparison.json'
        if comparison.is_file():config['resume_comparison_dir']=comparison.parent
        job.update(status='running',started=now());state['updated']=now();write_json(STATE,state)
        try:
            result=await run_ablation(config)
            job.update(status=result['status'],comparison_dir=result['comparison_dir'],
                executed_inference_count=result['executed_inference_count'],
                reused_prediction_count=result['reused_prediction_count'],failed_request_count=result['failed_request_count'],finished=now())
            job.pop('error',None)
            if result['status']!='complete':raise RuntimeError('该轮有失败图像，已保留成功缓存；修复后重运行原命令')
        except BaseException as exc:
            job.update(status='interrupted' if isinstance(exc,(asyncio.CancelledError,KeyboardInterrupt)) else 'failed',error=str(exc))
            state.update(status=job['status'],error=str(exc),updated=now());write_json(STATE,state);comparison_document(state)
            raise
        write_json(STATE,state);comparison_document(state)
    state.update(status='complete',updated=now());state.pop('error',None);write_json(STATE,state);comparison_document(state)
    print('COMPLETE: '+str(ROOT/'outputs/runs/chache_val300/prompt_comparison.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    try:
        run_async(execute(args.dry_run))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        logging.error('Experiment stopped: %s',exc)
        raise SystemExit(1)

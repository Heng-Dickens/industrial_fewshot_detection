"""Fresh YOLOE visual and GroundingDINO text runs aligned to PET-DINO/WeDetect."""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from run_wedetect_chache_prompts import ROOT, GROUP_NAMES, POINTS, cohort
from run_pet_dino_shots import atomic_json
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.evaluation.evaluator import evaluate_records
from industrial_fewshot.evaluation.reporting import format_metrics
from industrial_fewshot.outputs import result_directory
from industrial_fewshot.prompt_records import refresh_prompt_index
from industrial_fewshot.result_layout import namespace_for_report
from industrial_fewshot.runner import run

DEFAULT_RUNTIME=ROOT/'outputs/runtime/aligned_baselines_20261027'
PET_RUNTIME=ROOT/'outputs/runtime/pet_dino_shots_20261027'
WE_RUNTIME=ROOT/'outputs/runtime/wedetect_chache_prompts_20261027'
PROTOCOL=('Frozen 300-image cohorts. YOLOE uses exactly the PET-DINO 1/5/10-shot supports, '
    'boxes and seed=0. GroundingDINO uses verbatim WeDetect Chinese prompts. Every class '
    'queries every image; class-labelled predictions are merged without cross-class NMS. '
    'Candidate floor=0.001, NMS IoU=0.7, maxDets=100/class, evaluation IoU=0.5 and score=0.5. '
    'Model-native preprocessing and feature aggregation are retained. Fresh inference; no training/API calls.')
MODEL_NAMES={'yoloe':'yoloe-v8l-seg','groundingdino':'grounding-dino-base'}


def load_job(runtime,job,device=None):
    config=load_config(Path(job['config_file']))
    config.update(prediction_cache_dir=runtime/'cache',prompt_index_root=ROOT/'outputs/runs',
        result_layout=job['result_layout'],cache_search_dirs=[runtime/'cache'],reuse_predictions=False)
    if device:
        config['targets'][0]['config'][job['backend']]['device']=device
        config['resolved_config'][job['backend']]['device']=device
    return config


def prepare(runtime):
    if (runtime/'plan.json').exists():
        raise RuntimeError('Plan already exists; use --phase complete to resume or choose a new --runtime.')
    pet=read_json(PET_RUNTIME/'plan.json'); we=read_json(WE_RUNTIME/'plan.json')
    plan=dict(schema='aligned_baselines.v1',protocol=PROTOCOL,datasets={},experiments=[],jobs=[])
    for dataset in ('diaowu_val300','chache_val300'):
        raw,manifest,items=cohort(dataset)
        saved=pet['datasets'][dataset]
        identity=digest({'images':[dict(relative_path=i['relative_path'],image_sha256=i['image_sha256'],
                                       ground_truth=i['ground_truth']) for i in items]})
        assert identity==saved['dataset_sha256'] and len(items)==300
        plan['datasets'][dataset]=dict(items=items,labels=raw['labels'],dataset_sha256=identity,
            dataset_manifest=saved['dataset_manifest'],image_dir=raw['dataset']['image_dir'],
            ground_truth_dir=raw['dataset']['ground_truth_dir'])
        for shot in (1,5,10):
            baseline=Path(saved['runs'][str(shot)])
            baseline_report=read_json(baseline/'run.json')
            sources=[]
            for oldjob in pet['jobs']:
                if oldjob['dataset']!=dataset or oldjob['shot']!=shot: continue
                config=read_json(Path(oldjob['config_file']))
                config.pop('pet_dino')
                config.update(backend='yoloe',reuse_predictions=False)
                config['yoloe']=dict(source_dir=str(ROOT/'third_party/yoloe'),
                    weights=str(ROOT/'weights/yoloe/yoloe-v8l-seg.pt'),positive_count=shot,
                    selection_seed=0,aggregation='normalized_mean',device='cuda:0',torch_threads=4,
                    imgsz=640,confidence=.001,nms_iou=.7,max_detections=100,exclude_reference_overlap=True)
                target=next(t for t in baseline_report['targets'] if t['id']==oldjob['target_id'])
                sources.append(dict(config=config,target_id=oldjob['target_id'],label=oldjob['label'],
                    expected_prompt=target['prompt_snapshot'],support_manifest=oldjob['support_manifest']))
            add_experiment(runtime,plan,dataset,'yoloe',f'{shot}shot',f'{shot}-shot',sources,
                baseline,shot=shot,evaluation=saved['evaluation'])
        if dataset=='chache_val300':
            text_groups=[(group,GROUP_NAMES[group],Path(we['runs'][group]['run_dir'])) for group in we['groups']]
        else:
            text_groups=[]
            for number,directory in enumerate(sorted((ROOT/'outputs/runs/diaowu_val300').glob('*/wedetect_base_0shot')),1):
                if (directory/'run.json').is_file():
                    text_groups.append((f'prompt{number}',read_json(directory/'run.json')['targets'][0]['prompt_snapshot']['text'],directory))
            assert len(text_groups)==2, 'Expected the two saved diaowu WeDetect prompts'
        for group,description,baseline in text_groups:
            old=read_json(baseline/'run.json'); sources=[]
            for target in old['targets']:
                label=next(row for row in raw['labels'] if row['name']==target['label'])
                single=len(old['targets'])==1
                config=dict(extends=str(ROOT/f'configs/common/{dataset}.yaml'),backend='groundingdino',
                    mode='train',scene_name=dataset,task_id=target['id'],labels=[label],
                    dataset=dict(manifest_file=saved['dataset_manifest'] if single else
                        str(WE_RUNTIME/'manifests'/f'{target["id"]}.json')),
                    reference_image_dir=None,support_manifest_dir=None,max_reference_examples=0,
                    max_reference_crops=0,max_reference_negatives=0,reuse_predictions=False,results_layout=False,
                    groundingdino=dict(checkpoint=str(ROOT/'weights/grounding-dino-base'),device='cuda:0',
                        text_prompt=target['prompt_snapshot']['text'],box_threshold=.001,text_threshold=.25,
                        nms_iou=.7,max_detections=100,torch_threads=4,exclude_reference_overlap=True))
                sources.append(dict(config=config,target_id=target['id'],label=target['label'],
                    expected_prompt=target['prompt_snapshot']))
            evaluation=read_json(baseline/'metrics.json')['settings']
            add_experiment(runtime,plan,dataset,'groundingdino',group,description,sources,
                baseline,shot=0,evaluation=evaluation)
    assert len(plan['jobs'])==40 and len(plan['experiments'])==12
    atomic_json(runtime/'plan.json',plan)
    for job in plan['jobs']:
        report=asyncio.run(run(load_job(runtime,job),dry_run=True,result_dir=Path(job['run_dir'])))
        assert report['image_count']==300 and not report['excluded_reference_overlap']
        target=report['targets'][0]; expected=job['expected_prompt']; actual=target['prompt_snapshot']
        assert target['local_model']['runtime_ready'], target['local_model']
        if job['backend']=='yoloe':
            assert actual['references']==expected['references'], job['id']
        else:
            assert actual['text']==expected['text'] and not target['reference_sample_ids']
        print('PREFLIGHT',job['id'],flush=True)
    atomic_json(runtime/'preflight.json',dict(status='passed',jobs=40,experiments=12,requests=12000,
        aligned_inputs='Exact reference images/boxes or verbatim prompt snapshots checked.'))


def add_experiment(runtime,plan,dataset,backend,variant,description,sources,baseline,shot,evaluation):
    scene=plan['datasets'][dataset]
    method,layout=namespace_for_report(ROOT/'outputs/runs',dict(scene_name=dataset,image_count=300,
        image_dir=scene['image_dir'],ground_truth_dir=scene['ground_truth_dir'],
        dataset_manifest=scene['dataset_manifest'],dataset_sha256=scene['dataset_sha256'],
        targets=[dict(backend=backend,prompt_snapshot=s['expected_prompt']) for s in sources]),create=True)
    name=MODEL_NAMES[backend]+(f'_{shot}shot' if backend=='yoloe' else '_0shot'+layout['model_name_suffix'])
    directory=result_directory(method,name)
    experiment_id=f'{dataset}_{backend}_{variant}'
    experiment=dict(id=experiment_id,dataset=dataset,backend=backend,variant=variant,description=description,
        shot_per_class=shot,run_dir=str(directory),baseline_run_dir=str(baseline),evaluation=evaluation,
        result_layout=layout,jobs=[])
    for source in sources:
        job_id=experiment_id+'_'+source['target_id']
        config=source['config']; config.update(output_dir=str(method),results_layout=False,evaluation=evaluation)
        path=runtime/'configs'/f'{job_id}.local.json'; write_json(path,config)
        job=dict(id=job_id,experiment_id=experiment_id,dataset=dataset,backend=backend,
            target_id=source['target_id'],label=source['label'],shot=shot,config_file=str(path),
            run_dir=str(directory if len(sources)==1 else directory/source['target_id']),result_layout=layout,
            expected_prompt=source['expected_prompt'])
        if source.get('support_manifest'): job['support_manifest']=source['support_manifest']
        plan['jobs'].append(job); experiment['jobs'].append(job_id)
    plan['experiments'].append(experiment)


def completed(job):
    path=Path(job['run_dir'])
    if not (path/'summary.json').is_file() or not (path/'metrics.json').is_file(): return False
    report=read_json(path/'summary.json')
    if report['image_count']!=300 or report['failed_request_count'] or report['targets'][0]['backend']!=job['backend']:
        return False
    actual=report['targets'][0]['prompt_snapshot']; expected=job['expected_prompt']
    return (actual['references']==expected['references'] if job['backend']=='yoloe' else actual['text']==expected['text'])


def run_class(runtime,job_id,device):
    plan=read_json(runtime/'plan.json'); job=next(j for j in plan['jobs'] if j['id']==job_id)
    if completed(job):
        print('COMPLETE REUSE',job_id,flush=True); return
    destination=Path(job['run_dir'])
    if destination.exists():
        archive=result_directory(ROOT/'outputs/archive/interrupted_runs'/runtime.name,job_id)
        archive.parent.mkdir(parents=True,exist_ok=True); destination.rename(archive)
    report=asyncio.run(run(load_job(runtime,job,device),result_dir=destination,force=True))
    assert report['image_count']==300 and report['failed_request_count']==0 and report['reused_prediction_count']==0
    assert completed(job)
    write_json(destination/'experiment.json',dict(protocol=PROTOCOL,job=job))
    print('COMPLETE',job_id,'AP50=',report['metrics']['mAP50'],flush=True)


def merge_and_verify(runtime,plan):
    rows=[]; total=0
    for experiment in plan['experiments']:
        jobs=[j for j in plan['jobs'] if j['id'] in experiment['jobs']]
        assert all(completed(j) for j in jobs)
        reports=[read_json(Path(j['run_dir'])/'summary.json') for j in jobs]
        directory=Path(experiment['run_dir']); scene=plan['datasets'][experiment['dataset']]
        records=[]
        for job in jobs:
            records.extend(json.loads(line) for line in (Path(job['run_dir'])/'records.jsonl').read_text().splitlines() if line)
        expected=300*len(jobs)
        assert len(records)==expected and len({(r['relative_path'],r['target_id']) for r in records})==expected
        hashes={i['relative_path']:i['image_sha256'] for i in scene['items']}
        for record in records:
            assert record['status']=='success' and record['image_sha256']==hashes[record['relative_path']]
            for field in ('labelme_json','feedback_json','feedback_text'):
                assert Path(record[field]).is_file(),record[field]
            assert all(Path(path).is_file() for path in record['labelme_json_by_score'].values())
        total+=len(records)
        if len(jobs)>1:
            metrics=evaluate_records(scene['items'],records,[r['name'] for r in scene['labels']],experiment['evaluation'])
            report={k:v for k,v in reports[0].items() if k not in {'annotations','metrics'}}
            report.update(run_dir=str(directory),output_dir=str(directory.parent),scene_name=experiment['dataset'],
                dataset_sha256=scene['dataset_sha256'],dataset_manifest=scene['dataset_manifest'],
                dataset_manifest_sha256=image_sha256(Path(scene['dataset_manifest'])),
                targets=[t for r in reports for t in r['targets']],metrics=metrics,
                config_files={j['target_id']:j['config_file'] for j in jobs},
                support_manifests=[s for r in reports for s in r['support_manifests']],
                local_model_setup={k:v for r in reports for k,v in r['local_model_setup'].items()},
                success_request_count=expected,failed_request_count=0,reused_prediction_count=0,
                scheduled_inference_count=expected,protocol=PROTOCOL)
            write_json(directory/'run.json',{k:v for k,v in report.items() if k!='metrics'})
            write_json(directory/'summary.json',report); write_json(directory/'metrics.json',metrics)
            (directory/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
            (directory/'evaluation.txt').write_text(format_metrics(metrics)+'\n',encoding='utf-8')
            (directory/'prompt.txt').write_text('\n'.join(f'{t["id"]}: '+(t['prompt_snapshot']['text'] or
                f'{experiment["shot_per_class"]}-shot visual supports; see class prompt.json') for t in report['targets'])+'\n',encoding='utf-8')
            write_json(directory/'prompt.json',dict(snapshots=[t['prompt_snapshot'] for t in report['targets']]))
        else:
            report=reports[0]; metrics=report['metrics']
        assert report['dataset_sha256']==scene['dataset_sha256']
        write_json(directory/'experiment.json',experiment)
        baseline=read_json(Path(experiment['baseline_run_dir'])/'metrics.json')
        rows.append(dict(dataset=experiment['dataset'],method=experiment['result_layout']['method'],
            variant=experiment['description'],shot_per_class=experiment['shot_per_class'],
            mAP50=metrics['mAP50'],mAP50_95=metrics['mAP50_95'],precision=metrics['precision'],recall=metrics['recall'],
            baseline_AP50=baseline['mAP50'],baseline_AP50_95=baseline['mAP50_95'],
            run_dir=str(directory),baseline_run_dir=experiment['baseline_run_dir'],failed=0,
            per_class=metrics['per_class']))
    assert total==12000
    return rows


def summarize(runtime,plan,rows):
    comparison=ROOT/'outputs/comparisons'/runtime.name; comparison.mkdir(parents=True,exist_ok=True)
    write_json(comparison/'comparison.json',dict(protocol=PROTOCOL,runs=rows))
    for method_dir in sorted({Path(e['run_dir']).parent for e in plan['experiments']}):
        selected=[row for row in rows if Path(row['run_dir']).parent==method_dir]
        write_json(method_dir/'experiment_results.json',dict(protocol=PROTOCOL,runs=selected))
        lines=[f'# {method_dir.name} 对齐实验','',PROTOCOL,'',
            '| 实验 | mAP50 | mAP50:95 | P@0.5 | R@0.5 | 对齐基线 AP50 | 结果 |',
            '|---|---:|---:|---:|---:|---:|---|']
        for row in selected:
            name=Path(row['run_dir']).name
            lines.append(f'| {row["variant"]} | {100*row["mAP50"]:.3f}% | {100*row["mAP50_95"]:.3f}% | '
                f'{100*row["precision"]:.2f}% | {100*row["recall"]:.2f}% | {100*row["baseline_AP50"]:.3f}% | '
                f'[{name}]({name}/evaluation.txt) |')
        lines+=['','YOLOE 对齐 PET-DINO；GroundingDINO 对齐 WeDetect。每次实验使用完整300张图片；chache覆盖五类。',
            'YOLOE 使用640方形输入及 normalized_mean 聚合；GroundingDINO保留本地800/1333预处理和text_threshold=0.25。',
            '提示词逐字对齐，不作翻译。模型架构、预处理、分词器及特征聚合仍各自独立。']
        (method_dir/'experiment_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        refresh_prompt_index(method_dir); refresh_prompt_index(method_dir.parent)
    refresh_prompt_index(ROOT/'outputs/runs')
    fields=['dataset','method','variant','mAP50','mAP50_95','precision','recall','baseline_AP50','run_dir','baseline_run_dir']
    with (comparison/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader();writer.writerows(rows)
    lines=['# 对齐条件的算法对比','','每组的图片、类别、支持集/提示词和共享评估参数均经过核验。','',
        '| 数据集 | 方法 | 条件 | AP50 | 基线AP50 | 差值（百分点） |', '|---|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["dataset"]} | {row["method"]} | {row["variant"]} | {100*row["mAP50"]:.3f}% | '
            f'{100*row["baseline_AP50"]:.3f}% | {100*(row["mAP50"]-row["baseline_AP50"]):+.3f} |')
    (comparison/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(runtime/'verification.json',dict(status='passed',successful_records=12000,experiments=12,class_jobs=40,
        aligned_inputs=True,all_output_links_verified=True,failures=0,reused_predictions=0))


def complete(runtime,devices):
    plan=read_json(runtime/'plan.json'); assert read_json(runtime/'preflight.json')['status']=='passed'
    state=dict(status='running',phase='inference',pid=os.getpid(),jobs={}); pending=list(plan['jobs']); active={};failed=[]
    (runtime/'logs').mkdir(exist_ok=True); atomic_json(runtime/'workflow_status.json',state)
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending: continue
                job=pending.pop(0)
                python=(Path.home()/'miniconda3/envs'/('industrial-groundingdino' if job['backend']=='groundingdino'
                    else 'industrial-fewshot')/'bin/python')
                log=(runtime/'logs'/f'{job["id"]}.log').open('a',encoding='utf-8')
                process=subprocess.Popen([str(python),str(Path(__file__).resolve()),'--phase','class',
                    '--runtime',str(runtime),'--job',job['id'],'--device',device],cwd=ROOT,
                    env={**os.environ,'PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1',
                        'YOLO_CONFIG_DIR':str(runtime/'yoloe_settings'),'YOLO_AUTOINSTALL':'false'},
                    stdout=log,stderr=subprocess.STDOUT)
                active[device]=(job,process,log);state['jobs'][job['id']]=dict(status='running',device=device,pid=process.pid)
                atomic_json(runtime/'workflow_status.json',state);print('START',job['id'],device,flush=True)
            for device,(job,process,log) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close(); del active[device]
                state['jobs'][job['id']].update(status='complete' if code==0 else 'failed',exit_code=code)
                if code:failed.append(job['id'])
                atomic_json(runtime/'workflow_status.json',state);print('FINISH',job['id'],code,flush=True)
            if active:time.sleep(1)
        if failed:raise RuntimeError(f'Failed jobs: {failed}; see {runtime}/logs')
        state['phase']='evaluation';atomic_json(runtime/'workflow_status.json',state)
        rows=merge_and_verify(runtime,plan); summarize(runtime,plan,rows)
        state.update(status='complete',phase='complete',successful_records=12000,results=rows)
        atomic_json(runtime/'workflow_status.json',state)
    except BaseException as exc:
        for _,process,log in active.values():process.terminate();log.close()
        state.update(status='failed',error=str(exc));atomic_json(runtime/'workflow_status.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['prepare','class','complete','summarize','status'])
    parser.add_argument('--runtime',type=Path,default=DEFAULT_RUNTIME)
    parser.add_argument('--devices',default=','.join(f'cuda:{i}' for i in range(8)))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--job')
    args=parser.parse_args();runtime=args.runtime.resolve();runtime.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.phase=='prepare':prepare(runtime)
    elif args.phase=='class':run_class(runtime,args.job,args.device)
    elif args.phase=='complete':complete(runtime,args.devices.split(','))
    elif args.phase=='summarize':
        plan=read_json(runtime/'plan.json');summarize(runtime,plan,merge_and_verify(runtime,plan))
    else:print(json.dumps(read_json(runtime/'workflow_status.json'),ensure_ascii=False,indent=2))

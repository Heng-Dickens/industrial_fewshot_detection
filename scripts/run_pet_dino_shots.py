"""Run frozen 1/5/10-shot Visual-G on diaowu and all five special-vehicle classes."""
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

from run_wedetect_chache_prompts import ROOT, cohort, POINTS
from industrial_fewshot.cache import digest, image_sha256
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.evaluation.evaluator import evaluate_records
from industrial_fewshot.evaluation.reporting import format_metrics
from industrial_fewshot.prompt_records import refresh_prompt_index
from industrial_fewshot.result_layout import namespace_for_report
from industrial_fewshot.runner import run

DATASETS = ('diaowu_val300', 'chache_val300')
SHOTS = (1, 5, 10)
DEFAULT_RUNTIME = ROOT/'outputs/runtime/pet_dino_shots_20261027'
PROTOCOL = ('PET-DINO Swin-T Visual-G, seed=0, frozen nested supports per class; '
    'average the official per-support visual embeddings. Each class independently queries '
    'all 300 images, including negatives. Merge class-labelled predictions without cross-class NMS. '
    'No query GT is used for support encoding or class routing. No training.')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    write_json(temporary,value)
    temporary.replace(path)


def key(dataset, shot, label):
    return f'{dataset}_{shot}shot_{label}'


def configuration(runtime, job):
    config=load_config(Path(job['config_file']))
    config.update(prediction_cache_dir=runtime/'cache', prompt_index_root=ROOT/'outputs/runs',
        result_layout=job['result_layout'],
        cache_search_dirs=[runtime/'cache', ROOT/'outputs/runs',
                           ROOT/'outputs/runtime/pet_dino_smoke_20261027'])
    return config


def prepare(runtime):
    old=read_json(runtime/'plan.json') if (runtime/'plan.json').exists() else None
    plan=dict(schema='pet_dino.shots.v1', protocol=PROTOCOL, datasets={}, jobs=[])
    for dataset in DATASETS:
        raw,source,items=cohort(dataset)
        assert len(items)==300
        identity=digest({'images':[dict(relative_path=i['relative_path'],image_sha256=i['image_sha256'],
                                        ground_truth=i['ground_truth']) for i in items]})
        full=copy.deepcopy(source)
        full.update(image_root=raw['dataset']['image_dir'],ground_truth_root=raw['dataset']['ground_truth_dir'])
        manifest=ROOT/f'data/raw/{dataset}/manifest.linux.local.json'
        if manifest.exists():
            assert read_json(manifest)==full, manifest
        else:
            write_json(manifest,full)
        labels=raw['labels']
        method,layout=namespace_for_report(ROOT/'outputs/runs',dict(scene_name=dataset,
            image_count=300,image_dir=raw['dataset']['image_dir'],ground_truth_dir=raw['dataset']['ground_truth_dir'],
            dataset_manifest=str(manifest),dataset_sha256=identity,
            targets=[dict(backend='pet_dino',prompt_snapshot=dict(kind='visual',text=None)) for _ in labels]),create=True)
        scene=dict(labels=labels,items=items,dataset_sha256=identity,dataset_manifest=str(manifest),
            method_dir=str(method),result_layout=layout,evaluation=dict(iou_threshold=.5,
                score_threshold=.5,score_thresholds=POINTS,max_detections_per_image=100),runs={})
        query_hashes={i['image_sha256'] for i in items}
        for label in labels:
            name=label['name']
            target='diaowu' if dataset=='diaowu_val300' else name
            reference=ROOT/'data/references'/('diaowu' if dataset=='diaowu_val300' else f'chache/{name}')
            support_root=runtime/'supports'/dataset/target
            selected=[]
            for shot in SHOTS:
                frozen=read_json(reference/f'{shot}shot/seed_0.json')
                samples=frozen['positive_samples']
                assert len(samples)==shot and [r['sample_id'] for r in samples[:len(selected)]]==selected
                selected=[r['sample_id'] for r in samples]
                prefix=frozen['reference_root'].replace('\\','/').rstrip('/')
                def localize(value):
                    if isinstance(value,dict): return {k:localize(v) for k,v in value.items()}
                    if isinstance(value,list): return [localize(v) for v in value]
                    if isinstance(value,str):
                        path=value.replace('\\','/')
                        if path==prefix or path.startswith(prefix+'/'):
                            return str(reference)+path[len(prefix):]
                    return value
                local=localize(frozen)
                for row in local['positive_samples']:
                    assert image_sha256(Path(row['image_path']))==row['image_sha256']
                    assert image_sha256(Path(row['json_path']))==row['annotation_sha256']
                    assert row['image_sha256'] not in query_hashes, row['sample_id']
                write_json(support_root/f'{shot}shot/seed_0.json',local)
                class_manifest=runtime/'manifests'/dataset/f'{target}.json'
                subset={**full,'samples':[{**r,'ground_truth':[g for g in r['ground_truth'] if g['label']==name]}
                                         for r in full['samples']]}
                write_json(class_manifest,subset)
                directory=method/f'pet_dino_swin_t_{shot}shot'
                destination=directory if len(labels)==1 else directory/target
                config_path=runtime/'configs'/f'{key(dataset,shot,target)}.local.json'
                config=dict(extends=str(ROOT/f'configs/common/{dataset}.yaml'),backend='pet_dino',mode='train',
                    scene_name=dataset,task_id=target,labels=[label],
                    dataset=dict(manifest_file=str(manifest if len(labels)==1 else class_manifest)),
                    reference_image_dir=str(reference),support_manifest_dir=str(support_root),
                    max_reference_examples=0,max_reference_crops=0,max_reference_negatives=0,
                    output_dir=str(method),results_layout=False,reuse_predictions=True,
                    pet_dino=dict(source_dir=str(ROOT/'third_party/pet_dino'),
                        worker_python=str(Path.home()/'miniconda3/envs/industrial-pet-dino/bin/python'),
                        model_config=str(ROOT/'third_party/pet_dino/configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py'),
                        weights=str(ROOT/'weights/pet_dino/pet_dino_swin_t.pth'),
                        language_model_dir=str(ROOT/'weights/pet_dino/bert-base-uncased'),
                        prompt_mode='visual',positive_count=shot,selection_seed=0,device='cuda:0',
                        confidence=.001,nms_iou=.7,max_detections=100,torch_threads=4,
                        timeout_seconds=600,exclude_reference_overlap=True),evaluation=scene['evaluation'])
                write_json(config_path,config)
                job=dict(id=key(dataset,shot,target),dataset=dataset,shot=shot,label=name,target_id=target,
                    config_file=str(config_path),run_dir=str(destination),result_layout=layout,
                    support_manifest=str(support_root/f'{shot}shot/seed_0.json'))
                if destination.exists() and old is None:
                    raise RuntimeError(f'Refusing to overwrite existing experiment: {destination}')
                plan['jobs'].append(job)
                scene['runs'][str(shot)]=str(directory)
        plan['datasets'][dataset]=scene
    if old:
        assert old==plan, 'Existing plan differs; choose a new --runtime'
    atomic_json(runtime/'plan.json',plan)
    for job in plan['jobs']:
        config=configuration(runtime,job)
        report=asyncio.run(run(config,dry_run=True,result_dir=Path(job['run_dir'])))
        assert report['image_count']==300 and not report['excluded_reference_overlap']
        assert len(report['targets'][0]['reference_sample_ids'])==job['shot']
        print('PREFLIGHT',job['id'],'300 queries, supports=',job['shot'],flush=True)
    atomic_json(runtime/'preflight.json',dict(status='passed',jobs=len(plan['jobs']),queries=5400,
        supports='Frozen, nested 1/5/10 per class, seed=0; image and annotation hashes verified.'))


def completed(job):
    path=Path(job['run_dir'])
    if not (path/'summary.json').is_file() or not (path/'metrics.json').is_file(): return False
    report=read_json(path/'summary.json')
    return (report['image_count']==300 and report['failed_request_count']==0
            and report['targets'][0]['reference_count']['reference_examples']==job['shot'])


def run_class(runtime, job_id, device):
    plan=read_json(runtime/'plan.json')
    job=next(j for j in plan['jobs'] if j['id']==job_id)
    if completed(job):
        print('REUSE',job_id,flush=True)
        return
    destination=Path(job['run_dir'])
    if destination.exists():
        # Successful records are also stored in the fingerprinted runtime cache.
        archive=ROOT/'outputs/archive/interrupted_runs'/runtime.name/job_id
        count=1
        while archive.exists():
            count+=1
            archive=archive.with_name(f'{job_id}_attempt{count}')
        archive.parent.mkdir(parents=True,exist_ok=True)
        destination.rename(archive)
    config=configuration(runtime,job)
    config['targets'][0]['config']['pet_dino']['device']=device
    config['resolved_config']['pet_dino']['device']=device
    report=asyncio.run(run(config,result_dir=destination))
    assert report['image_count']==300 and report['failed_request_count']==0, destination
    write_json(destination/'experiment.json',dict(dataset=job['dataset'],shot_per_class=job['shot'],
        seed=0,protocol=PROTOCOL,support_manifest=job['support_manifest'],runtime=str(runtime)))
    print('COMPLETE',job_id,'AP50=',report['metrics']['mAP50'],flush=True)


def merge(runtime, plan):
    for dataset,scene in plan['datasets'].items():
        for shot in SHOTS:
            jobs=[j for j in plan['jobs'] if j['dataset']==dataset and j['shot']==shot]
            assert all(completed(j) for j in jobs)
            directory=Path(scene['runs'][str(shot)])
            if len(jobs)==1: continue
            reports=[read_json(Path(j['run_dir'])/'summary.json') for j in jobs]
            records=[]
            for job in jobs:
                records.extend(json.loads(line) for line in (Path(job['run_dir'])/'records.jsonl').read_text().splitlines() if line)
            assert len(records)==1500 and len({(r['relative_path'],r['target_id']) for r in records})==1500
            assert all(r['status']=='success' for r in records)
            hashes={i['relative_path']:i['image_sha256'] for i in scene['items']}
            assert all(r['image_sha256']==hashes[r['relative_path']] for r in records)
            labels=[r['name'] for r in scene['labels']]
            metrics=evaluate_records(scene['items'],records,labels,scene['evaluation'])
            report={k:v for k,v in reports[0].items() if k not in {'annotations','metrics'}}
            report.update(scene_name=dataset,run_dir=str(directory),output_dir=str(directory.parent),
                dataset_sha256=scene['dataset_sha256'],dataset_manifest=scene['dataset_manifest'],
                dataset_manifest_sha256=image_sha256(Path(scene['dataset_manifest'])),
                result_layout=scene['result_layout'],config_files={j['target_id']:j['config_file'] for j in jobs},
                targets=[t for r in reports for t in r['targets']],metrics=metrics,
                support_manifests=[s for r in reports for s in r['support_manifests']],
                success_request_count=1500,failed_request_count=0,protocol=PROTOCOL,
                reused_prediction_count=sum(r['reused_prediction_count'] for r in reports),
                scheduled_inference_count=sum(r['scheduled_inference_count'] for r in reports),
                local_model_setup={k:v for r in reports for k,v in r['local_model_setup'].items()})
            write_json(directory/'run.json',{k:v for k,v in report.items() if k!='metrics'})
            write_json(directory/'summary.json',report)
            write_json(directory/'metrics.json',metrics)
            (directory/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            (directory/'evaluation.txt').write_text(format_metrics(metrics)+'\n',encoding='utf-8')
            (directory/'prompt.txt').write_text(f'PET-DINO Visual-G: {shot} independent supports per class, seed=0.\n'
                +'\n'.join(f'{j["target_id"]}: {j["support_manifest"]}' for j in jobs)+'\n',encoding='utf-8')
            write_json(directory/'prompt.json',dict(schema='pet_dino.multi_class_visual_prompt.v1',
                snapshots=[t['prompt_snapshot'] for t in report['targets']]))
            write_json(directory/'experiment.json',dict(protocol=PROTOCOL,shot_per_class=shot,seed=0,
                class_runs={j['target_id']:j['run_dir'] for j in jobs},runtime=str(runtime)))


def summarize(runtime, plan):
    rows=[]
    for dataset,scene in plan['datasets'].items():
        method=Path(scene['method_dir'])
        for shot in SHOTS:
            directory=Path(scene['runs'][str(shot)])
            r=read_json(directory/'summary.json'); m=r['metrics']
            rows.append(dict(dataset=dataset,shot_per_class=shot,image_count=300,
                mAP50=m['mAP50'],mAP50_95=m['mAP50_95'],precision=m['precision'],recall=m['recall'],
                failures=r['failed_request_count'],run_dir=str(directory),per_class=m['per_class']))
        selected=[r for r in rows if r['dataset']==dataset]
        write_json(method/'experiment_results.json',dict(protocol=PROTOCOL,runs=selected))
        lines=['# PET-DINO 视觉提示实验','',PROTOCOL,'',
            '| shot / 类 | 图片数 | mAP50 | mAP50:95 | P@0.5 | R@0.5 | 失败 | 结果 |',
            '|---:|---:|---:|---:|---:|---:|---:|---|']
        for row in selected:
            name=Path(row['run_dir']).name
            lines.append(f'| {row["shot_per_class"]} | 300 | {100*row["mAP50"]:.3f}% | '
                f'{100*row["mAP50_95"]:.3f}% | {100*row["precision"]:.2f}% | {100*row["recall"]:.2f}% | '
                f'{row["failures"]} | [{name}]({name}/evaluation.txt) |')
        lines+=['','AP 使用全部保存候选分数；P/R 使用 0.5 阈值。分数下限 0.001，NMS=0.7，每类最多100框。',
            '1/5/10-shot 为每类参考图数量；固定 seed=0，支持集逐级包含，且与验证图片内容不重合。']
        (method/'experiment_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        refresh_prompt_index(method)
        refresh_prompt_index(method.parent)
    refresh_prompt_index(ROOT/'outputs/runs')
    comparison=ROOT/'outputs/comparisons'/runtime.name
    comparison.mkdir(parents=True,exist_ok=True)
    write_json(comparison/'comparison.json',dict(protocol=PROTOCOL,runs=rows))
    fields=['dataset','shot_per_class','image_count','mAP50','mAP50_95','precision','recall','failures','run_dir']
    with (comparison/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    return rows


def complete(runtime, devices):
    plan=read_json(runtime/'plan.json')
    state=dict(status='running',pid=os.getpid(),phase='inference',jobs={})
    atomic_json(runtime/'workflow_status.json',state)
    pending=list(plan['jobs']); active={}; failed=[]
    (runtime/'logs').mkdir(parents=True,exist_ok=True)
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending: continue
                job=pending.pop(0)
                log=(runtime/'logs'/f'{job["id"]}.log').open('a',encoding='utf-8')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','class',
                    '--runtime',str(runtime),'--job',job['id'],'--device',device],cwd=ROOT,
                    env={**os.environ,'PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'},
                    stdout=log,stderr=subprocess.STDOUT)
                active[device]=(job,process,log)
                state['jobs'][job['id']]=dict(status='running',device=device,pid=process.pid)
                atomic_json(runtime/'workflow_status.json',state)
                print('START',job['id'],device,flush=True)
            for device,(job,process,log) in list(active.items()):
                code=process.poll()
                if code is None: continue
                log.close(); del active[device]
                state['jobs'][job['id']].update(status='complete' if code==0 else 'failed',exit_code=code)
                if code: failed.append(job['id'])
                atomic_json(runtime/'workflow_status.json',state)
                print('FINISH',job['id'],'exit=',code,flush=True)
            if active: time.sleep(1)
        if failed: raise RuntimeError(f'Jobs failed: {failed}; inspect {runtime}/logs')
        state.update(phase='evaluation'); atomic_json(runtime/'workflow_status.json',state)
        merge(runtime,plan)
        rows=summarize(runtime,plan)
        state.update(status='complete',phase='complete',successful_records=5400,results=rows)
        atomic_json(runtime/'workflow_status.json',state)
        print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        for _,process,log in active.values():
            process.terminate(); log.close()
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        atomic_json(runtime/'workflow_status.json',state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['prepare','class','complete','summarize','status'])
    parser.add_argument('--runtime',type=Path,default=DEFAULT_RUNTIME)
    parser.add_argument('--job')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--devices',default=','.join(f'cuda:{i}' for i in range(8)))
    args=parser.parse_args(); runtime=args.runtime.resolve(); runtime.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.phase=='prepare': prepare(runtime)
    elif args.phase=='class': run_class(runtime,args.job,args.device)
    elif args.phase=='complete': complete(runtime,args.devices.split(','))
    elif args.phase=='summarize':
        plan=read_json(runtime/'plan.json'); merge(runtime,plan); summarize(runtime,plan)
    else:
        path=runtime/'workflow_status.json'
        print(json.dumps(read_json(path),ensure_ascii=False,indent=2) if path.exists() else 'Not started')

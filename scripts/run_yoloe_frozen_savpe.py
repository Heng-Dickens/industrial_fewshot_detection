"""Add paired frozen-SAVPE experiments without changing completed searches.

Each scene clones all seven existing VP head/neck/full configurations. Only
freeze_savpe changes. SAVPE-only configurations cannot learn after this freeze.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import statistics
import sys
import time

import run_yoloe_optimization_sweep as sweep
from industrial_fewshot.comparison_layout import optimization_report_dir
from industrial_fewshot.finetuning_methods import method_policy,metadata_policy,verify_checkpoint_policy,write_method_document
from summarize_yoloe_optimization import csv_write,FIELDS

ROOT = sweep.ROOT
EXTENSION = 'frozen_savpe_20261030'


def original_stamp(scene):
    return 'yoloe_search_20261029_r2' if scene=='chache' else 'yoloe_diaowu_search_20261029'


def paired_job(job, config, project, runtime):
    if job['method']!='yoloe-vp' or config.get('adaptation',{}).get('scope','savpe')=='savpe':
        return None
    result=deepcopy(job);cfg=deepcopy(config)
    result['paired_trial_id']=job['id']
    result['id']=job['id']+'_frozen_savpe'
    result['adaptation']=dict(cfg.get('adaptation',{}),freeze_savpe=True,freeze_savpe_bn_stats=True)
    result['role']='frozen_savpe_search'
    result['config']=str(runtime/'configs'/f"{result['id']}.json")
    result['run_dir']=str(project/result['id'])
    cfg['project']=str(project);cfg['adaptation']=result['adaptation']
    return result,cfg


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig'))) if path.exists() else []


def pct(value):
    return '—' if value is None or value=='' else f'{100*float(value):.2f}'


def archive_json(source, target):
    if source.is_file():
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)


def read_metadata(path):
    # Training writes its metadata at initialization and completion. Retry a
    # transient partial write without mistaking it for an experiment failure.
    for attempt in range(5):
        try:return sweep.read(path)
        except json.JSONDecodeError:
            if attempt==4:raise
            time.sleep(.05)


def manager_alive(runtime):
    launch=runtime/'launch.json'
    if not launch.exists():return False
    pid=sweep.read(launch)['pid']
    status=Path(f'/proc/{pid}/stat')
    if not status.exists():return False
    return status.read_text().split(') ',1)[1].split()[0] not in {'Z','X'}


def summarize():
    report=sweep.REPORT;parent=report.parent
    plan=sweep.read(report/'plan.json')
    state=sweep.STATE
    original={r['id']:r for r in read_csv(parent/'all_trials.csv')}
    jobs=list(plan['jobs'])
    jobs.extend(e['job'] for e in state.get('jobs',{}).values() if e['job'].get('role')=='seed_repeat')
    rows=[];tests=[];methods=[];pairs=[]
    for job in jobs:
        cfg=sweep.read(job['config']);run=Path(job['run_dir'])
        entry=state.get('jobs',{}).get(job['id'],{})
        policy=method_policy('yoloe-vp',cfg['adaptation']['scope'],freeze_savpe=True)
        row=dict(id=job['id'],method='yoloe-vp',scope=cfg['adaptation']['scope'],
            protocol=cfg['adaptation'].get('protocol','same_image'),loss=cfg['adaptation'].get('loss','classification'),
            paired_trial_id=job['paired_trial_id'],role=job.get('role'),status=entry.get('status','planned'),
            training_seed=cfg['training']['seed'],training_shot=cfg['shot'],prompt_shots=10,
            imgsz=cfg['training']['imgsz'],batch=job['batch'],max_epochs=cfg['training']['epochs'],
            weights=str((run/'weights/best.pt').relative_to(ROOT)),config=str((report/'configs'/f"{job['id']}.json").relative_to(ROOT)),
            **policy)
        if (run/'finetune_config.json').exists():
            meta=read_metadata(run/'finetune_config.json')
            row.update(metadata_policy(meta,run))
            row['trainable_parameters']=meta.get('trainable_parameter_count')
            for name in ['finetune_config.json','trainable_parameters.json','frozen_savpe_verification.json','results.csv']:
                archive_json(run/name,report/'trials'/job['id']/name)
            archive_json(Path(job['config']),report/'trials'/job['id']/'config.json')
            if (run/'trainable_parameters.json').exists():methods.append(row)
            if (run/'results.csv').exists():row['actual_epochs']=len(read_csv(run/'results.csv'))
        dev=Path(entry.get('dev_output',str(run/'dev_val100')))
        for path in dev.glob('*/metrics.json'):
            metric=sweep.read(path);evidence=sweep.read(path.parent/'run.json')
            if metric['evaluated_image_count']!=100 or evidence['split']!='dev':raise ValueError('Invalid dev set.')
            verify_checkpoint_policy(policy,evidence)
            row.update({f'dev_{key}':metric[key] for key in FIELDS})
            row['weights_sha256']=evidence['weights_sha256']
            for name in ['metrics.json','run.json']:
                archive_json(path.parent/name,report/'metrics'/job['id']/'dev'/name)
        rows.append(row)
        if row['training_seed']==0:
            old=original[job['paired_trial_id']]
            pairs.append(dict(id=job['id'],paired_trial_id=job['paired_trial_id'],scope=row['scope'],
                protocol=row['protocol'],lr0=cfg['training']['lr0'],original_dev_mAP50_95=old['dev_mAP50_95'],
                frozen_dev_mAP50_95=row.get('dev_mAP50_95'),
                delta_dev_mAP50_95=row['dev_mAP50_95']-float(old['dev_mAP50_95']) if 'dev_mAP50_95' in row else None))
        for path in sorted((run/'test_val300').glob('*/metrics.json')):
            metric=sweep.read(path);evidence=sweep.read(path.parent/'run.json')
            if metric['evaluated_image_count']!=300 or evidence['split']!='test':raise ValueError('Invalid test set.')
            verify_checkpoint_policy(policy,evidence)
            test=dict(row)
            test.update({key:metric[key] for key in FIELDS})
            test['prompt_shots']=evidence['prompt_shot_per_class']
            tests.append(test)
            for name in ['metrics.json','run.json']:
                archive_json(path.parent/name,report/'metrics'/job['id']/path.parent.name/name)
    csv_write(report/'all_trials.csv',rows)
    csv_write(report/'paired_dev_results.csv',pairs)
    if tests:csv_write(report/'all_test_results.csv',tests)
    if methods:
        csv_write(report/'SAVPE_METHODS.csv',methods);write_method_document(report,methods,ROOT)
    combined=[]
    for row in read_csv(parent/'all_trials.csv'):
        row.update(experiment_group='original_search');combined.append(row)
    combined.extend(dict(row,experiment_group='frozen_savpe_extension') for row in rows)
    csv_write(parent/'all_trials_with_frozen_savpe.csv',combined)
    combined_tests=read_csv(parent/'all_test_results.csv')+[dict(row,experiment_group='frozen_savpe_extension') for row in tests]
    csv_write(parent/'all_test_results_with_frozen_savpe.csv',combined_tests)
    overview=['# 全部微调实验：含冻结SAVPE条件','',
        f'原搜索与新增条件共{len(combined)}项训练记录、{len(combined_tests)}项测试设置。未完成行没有分数。',
        '[原搜索及基础对照](README.md)、[新增条件与成对差值](frozen_savpe_20261030/README.md)、'
        '[训练CSV](all_trials_with_frozen_savpe.csv)、[测试CSV](all_test_results_with_frozen_savpe.csv)。','',
        '| 配置 | 方法/范围 | SAVPE | 训练seed | 状态 | dev AP50 | dev AP50:95 |',
        '|---|---|---|---:|---|---:|---:|']
    for row in combined:
        frozen=row.get('savpe_parameters_frozen') in {True,'True'}
        status='保留并冻结' if frozen else '保留并训练' if row.get('savpe_state')=='retained' else '移除' if row.get('savpe_state')=='removed' else '原结构无SAVPE'
        overview.append(f'| {row["id"]} | {row["method"]}/{row["scope"]} | {status} | {row["training_seed"]} | {row["status"]} | {pct(row.get("dev_mAP50"))} | {pct(row.get("dev_mAP50_95"))} |')
    overview+=['','## 300张测试','',
        '| 配置 | 训练seed | 推理shot | AP50 | AP50:95 | P@0.5 | R@0.5 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in combined_tests:
        overview.append(f'| {row["id"]} | {row["training_seed"]} | {row["prompt_shots"] or "不适用"} | '+' | '.join(pct(row[k]) for k in FIELDS[:4])+' |')
    (parent/'ALL_EXPERIMENTS.md').write_text('\n'.join(overview)+'\n')
    completed=sum(r['status']=='completed' for r in rows)
    lines=[f'# {"叉车" if plan["scene"]=="chache" else "吊物"}：冻结SAVPE、保留视觉提示','',
        f'阶段：{state.get("stage","prepared")}；训练/dev完成{completed}/{len(rows)}，已有{len(tests)}项300张测试设置。',
        '每组从相同原始预训练权重开始，只新增freeze_savpe=true；SAVPE参数和BN统计固定，仍参与前向运算并向上游传递梯度。',
        '覆盖原搜索全部7组head/neck/full视觉提示配置。仅SAVPE的19组配置冻结后无可学习参数，不重复训练；原始未微调基线已覆盖不训练的情况。',
        '每类10张训练、独立100张dev、300张test；dev使用冻结10shot支持，test分别5/10shot。','',
        '[全部配置与dev结果](all_trials.csv)、[成对dev差值](paired_dev_results.csv)、[配置快照](configs/)、[冻结证据与训练曲线](trials/)。',
        '[场景全部训练总表](../all_trials_with_frozen_savpe.csv)、[场景全部测试总表](../all_test_results_with_frozen_savpe.csv)、[返回原搜索报告](../README.md)。','',
        '| 冻结SAVPE配置 | 原配对配置 | 范围/协议 | lr0 | 原dev AP50:95 | 冻结后dev AP50:95 | 差值/百分点 |',
        '|---|---|---|---:|---:|---:|---:|']
    for pair in pairs:
        lines.append(f'| {pair["id"]} | {pair["paired_trial_id"]} | {pair["scope"]}/{pair["protocol"]} | {pair["lr0"]} | {pct(pair["original_dev_mAP50_95"])} | {pct(pair["frozen_dev_mAP50_95"])} | {pct(pair["delta_dev_mAP50_95"])} |')
    lines+=['','候选只按本轮dev AP50:95选前三名，第一名补训练seed=1、2；冻结候选后测试，不能按test重选。原搜索候选和结果保持原记录。',
        '原配对dev是同一配置seed=0的比较；方法优胜配置的多种子test比较同时包含选参差异，不是单因素因果证明。']
    statistics_rows=[]
    if tests:
        csv_write(report/'all_test_results.csv',tests)
        lines+=['','## 300张测试结果','',
            '| 实验 | 训练seed | 推理shot | AP50 | AP50:95 | P@0.5 | R@0.5 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in tests:
            lines.append(f'| {row["id"]} | {row["training_seed"]} | {row["prompt_shots"]} | '+ ' | '.join(pct(row[k]) for k in FIELDS[:4])+' |')
        selected=report/'test_selection.json'
        if selected.exists():
            winner=sweep.read(selected)['winner']
            for shot in [5,10]:
                group=[r for r in tests if r['id'].split('_seed')[0]==winner and r['prompt_shots']==shot]
                if sorted(r['training_seed'] for r in group)==[0,1,2]:
                    summary=dict(method='yoloe-vp-frozen-savpe',dev_selected_config=winner,prompt_shots=shot,n=3)
                    for key in FIELDS[:5]:
                        summary[key+'_mean']=statistics.mean(r[key] for r in group)
                        summary[key+'_std']=statistics.stdev(r[key] for r in group)
                    statistics_rows.append(summary)
            if statistics_rows:
                csv_write(report/'seed_statistics.csv',statistics_rows)
                lines+=['','## dev优胜配置：三个训练种子均值±样本标准差','',
                    '| 推理shot | test AP50 | test AP50:95 |','|---:|---:|---:|']
                for row in statistics_rows:
                    lines.append(f'| {row["prompt_shots"]} | {pct(row["mAP50_mean"])} ± {pct(row["mAP50_std"])} | {pct(row["mAP50_95_mean"])} ± {pct(row["mAP50_95_std"])} |')
    lines+=['','正式训练FP32、单卡独立进程。原生best按0.1 AP50+0.9 AP50:95选取；跨配置按重新统一推理的dev AP50:95排序。',
        '检查frozen_savpe_verification.json中的参数及缓冲区SHA256相等、其他参数更新和非零梯度；EMA中的SAVPE也保持同步，避免舍入漂移。',
        '原生checkpoint遵循既有上游FP16序列化，加载后FP32推理；本轮未更改原搜索的保存协议。',
        '已有测试集此前已查看；此轮仍为探索实验，不能称为全新盲测。','',
        f'命令和日志：`outputs/_shared/runtime/{plan["stamp"]}/`；权重与预测：`outputs/YOLOE/finetune/{plan["scene"]}_val300/runs/{plan["stamp"]}/`。','']
    (report/'README.md').write_text('\n'.join(lines))
    sweep.write(report/'progress.json',dict(stage=state.get('stage'),train_dev_completed=completed,
        train_dev_total=len(rows),test_settings=len(tests),seed_statistics=statistics_rows,updated=time.time()))
    marker='<!-- frozen-savpe:start -->'
    section='\n'.join([marker,'','## 新增：冻结SAVPE、保留视觉提示','',
        f'状态：{state.get("stage")}；新增训练/dev完成{completed}/{len(rows)}，测试设置{len(tests)}项。',
        f'[全部微调实验总表（含新增条件）](ALL_EXPERIMENTS.md)、[新增对照报告]({EXTENSION}/README.md)、[全部训练CSV](all_trials_with_frozen_savpe.csv)、[全部测试CSV](all_test_results_with_frozen_savpe.csv)。',
        '原搜索各表保留原有实验协议与成绩，新条件独立记录并合并到上述总表。','','<!-- frozen-savpe:end -->'])
    document=(parent/'README.md').read_text()
    if marker in document:document=re.sub(r'<!-- frozen-savpe:start -->.*?<!-- frozen-savpe:end -->',section,document,flags=re.S)
    else:
        heading,rest=document.split('\n',1);document=heading+'\n\n'+section+'\n'+rest
    (parent/'README.md').write_text(document)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=['chache','diaowu'],required=True)
    parser.add_argument('--gpus',nargs='+',type=int,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--wait',action='store_true',help='With --summarize-only, refresh reports after the manager exits.')
    args=parser.parse_args()
    if len(args.gpus)!=len(set(args.gpus)):raise ValueError('Duplicate GPU allocation.')
    stamp=f'yoloe_{args.scene}_frozen_savpe_20261030'
    parent=optimization_report_dir(original_stamp(args.scene))
    from summarize_finetuning_best import is_compact,process_scene
    if is_compact(parent):
        if args.execute:raise ValueError('This completed experiment is archived; preserve it and use a new experiment ID for new training.')
        process_scene(args.scene)
        print(f'Updated compact report: {parent / "ALL_EXPERIMENTS.md"}')
        return
    sweep.configure(args.scene,stamp)
    if args.wait:
        if not args.summarize_only:raise ValueError('--wait requires --summarize-only.')
        while manager_alive(sweep.RUNTIME):
            sweep.STATE.clear()
            sweep.STATE.update(sweep.read(sweep.RUNTIME/'state.json'))
            try:summarize()
            except (json.JSONDecodeError,FileNotFoundError) as error:
                print(f'Report refresh deferred during artifact write: {error}',flush=True)
            time.sleep(30)
    if args.resume or args.summarize_only:
        sweep.STATE.update(sweep.read(sweep.RUNTIME/'state.json'))
        if args.wait and sweep.STATE['stage']!='complete':
            sweep.STATE['stage']='failed_or_interrupted'
            sweep.write(sweep.RUNTIME/'state.json',sweep.STATE)
        for pid in sweep.STATE.get('processes',{}):
            try:os.kill(int(pid),0)
            except ProcessLookupError:continue
            if args.summarize_only:break
            raise RuntimeError(f'Prior worker may still be active: {pid}')
        plan=sweep.read(sweep.REPORT/'plan.json');jobs=plan['jobs']
    else:
        if (sweep.REPORT/'plan.json').exists():raise FileExistsError('Use --resume; preserving prepared plan.')
        originals=sweep.read(parent/'plan.json')['jobs'];jobs=[]
        for job in originals:
            cfg=sweep.read(job['config'])
            pair=paired_job(job,cfg,sweep.PROJECT,sweep.RUNTIME)
            if pair is None:continue
            paired,cfg=pair;jobs.append(paired)
            sweep.write(paired['config'],cfg);sweep.write(sweep.REPORT/'configs'/f"{paired['id']}.json",cfg)
        if len(jobs)!=7:raise ValueError(f'Expected seven paired VP configurations, got {len(jobs)}.')
        plan=dict(scene=args.scene,stamp=stamp,jobs=jobs,gpus=args.gpus,
            source_report=str(parent.relative_to(ROOT)),selection='dev AP50:95 top3; winner seeds 1/2; test5/10shot',
            excluded='SAVPE-only would leave zero trainable parameters; fixed-head/YOLOv8 have no SAVPE.')
        sweep.write(sweep.REPORT/'plan.json',plan)
        sweep.STATE.update(stage='prepared',scene=args.scene,jobs={},processes={})
        sweep.write(sweep.RUNTIME/'state.json',sweep.STATE)
    original_publish=sweep.publish
    def publish():
        original_publish();summarize()
    sweep.publish=publish
    if args.summarize_only:summarize();return
    sweep.publish()
    if not args.execute:
        print(f'{len(jobs)} paired jobs prepared at {sweep.REPORT}');return
    smoke_path=sweep.REPORT/'smoke_results.json'
    if not smoke_path.exists():
        smoke=[job for job in jobs if job['paired_trial_id'] in {'vp_head_lr1e-05','vp_neck_lr1e-05','vp_full_lr1e-05','vp_cross_neck'}]
        slots=queue.Queue()
        for gpu in args.gpus:slots.put(gpu)
        def smoke_one(job):
            gpu=slots.get()
            try:
                code=sweep.run_command(sweep.train_command(job,gpu,True),sweep.RUNTIME/'logs'/f"{job['id']}_smoke.log")
                if code:raise RuntimeError(f'Smoke failed: {job["id"]}; see retained log.')
                proof=sweep.read(Path(job['run_dir']+'_smoke')/'frozen_savpe_verification.json')
                return dict(id=job['id'],gpu=gpu,exit_code=code,verification=proof)
            finally:slots.put(gpu)
        sweep.STATE['stage']='smoke';sweep.publish()
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:outcomes=list(pool.map(smoke_one,smoke))
        sweep.write(smoke_path,dict(outcomes=outcomes))
    if args.smoke_only:return
    sweep.STATE['stage']='training';sweep.publish()
    sweep.dispatch([j for j in jobs if sweep.STATE['jobs'].get(j['id'],{}).get('status')!='completed'],args.gpus)
    if any(sweep.STATE['jobs'][j['id']]['status']!='completed' for j in jobs):raise RuntimeError('A training/dev job failed; inspect logs before selection.')
    ranked=sorted(jobs,key=lambda j:sweep.read(sweep.dev_metric_path(j))['mAP50_95'],reverse=True)
    winner=ranked[0];repeats=[]
    for seed in [1,2]:
        job=deepcopy(winner);job.update(id=winner['id']+f'_seed{seed}',role='seed_repeat')
        cfg=sweep.read(winner['config']);cfg['training']['seed']=seed
        job['config']=str(sweep.RUNTIME/'configs'/f"{job['id']}.json");job['run_dir']=str(sweep.PROJECT/job['id'])
        sweep.write(job['config'],cfg);sweep.write(sweep.REPORT/'configs'/f"{job['id']}.json",cfg)
        repeats.append(job)
    selected=ranked[:3]+repeats
    selection=sweep.REPORT/'test_selection.json'
    if selection.exists():
        if [j['id'] for j in sweep.read(selection)['selected']]!=[j['id'] for j in selected]:raise ValueError('Frozen candidates differ.')
    else:sweep.write(selection,dict(winner=winner['id'],selected=selected,criterion='dev AP50:95 only',created=time.time()))
    sweep.STATE['stage']='seed_repeats';sweep.publish()
    sweep.dispatch([j for j in repeats if sweep.STATE['jobs'].get(j['id'],{}).get('status')!='completed'],args.gpus)
    if any(sweep.STATE['jobs'][j['id']]['status']!='completed' for j in repeats):raise RuntimeError('Seed repeat failed.')
    sweep.STATE['stage']='final_tests';sweep.publish()
    slots=queue.Queue()
    for gpu in args.gpus:slots.put(gpu)
    def test_one(job):
        gpu=slots.get()
        try:
            output=Path(job['run_dir'])/'test_val300'
            if all((output/name/'metrics.json').exists() and sweep.read(output/name/'run.json').get('status')=='completed' for name in ['visual_5shot','visual_10shot']):return 0
            if output.exists():raise RuntimeError(f'Partial test output retained: {output}')
            code=sweep.run_command(sweep.eval_command(job,gpu,'test'),sweep.RUNTIME/'logs'/f"{job['id']}_test.log")
            with sweep.LOCK:summarize()
            return code
        finally:slots.put(gpu)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:codes=list(pool.map(test_one,selected))
    if any(codes):raise RuntimeError('Test failed; inspect logs.')
    sweep.STATE['stage']='complete';sweep.publish();sweep.write(sweep.RUNTIME/'state.json',sweep.STATE)
    print(f'COMPLETE {sweep.REPORT}',flush=True)


if __name__=='__main__':
    main()

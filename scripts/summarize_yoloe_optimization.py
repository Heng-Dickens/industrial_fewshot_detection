"""Archive and compare a finished YOLOE search without launching any model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from industrial_fewshot.comparison_layout import optimization_report_dir
from industrial_fewshot.finetuning_methods import metadata_policy, method_policy, verify_checkpoint_policy, write_method_document
FIELDS = ['mAP50', 'mAP50_95', 'precision', 'recall', 'f2', 'tp', 'fp', 'fn']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(rows, label='id'):
    lines = ['| 实验 | SAVPE | 推理shot | AP50 | AP50:95 | P@0.5 | R@0.5 | TP/FP/FN |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r[label]} | {r['savpe_state']} | {r.get('prompt_shots') or '不适用'} | "
            + ' | '.join(f'{100*r[k]:.2f}' for k in FIELDS[:4])
            + f" | {r['tp']}/{r['fp']}/{r['fn']} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp', default='yoloe_search_20261029_r2')
    parser.add_argument('--partial', action='store_true', help='Archive progress without claiming completion.')
    args = parser.parse_args()
    runtime = ROOT/'outputs/_shared/runtime'/args.stamp
    report = optimization_report_dir(args.stamp, project_root=ROOT)
    from summarize_finetuning_best import is_compact,process_scene
    if is_compact(report):
        process_scene('diaowu' if report.parent.parent.name=='diaowu_val300' else 'chache')
        print(f'Updated compact report: {report / "ALL_EXPERIMENTS.md"}')
        return
    state = read(runtime/'state.json')
    plan = read(report/'plan.json')
    scene = plan.get('scene','chache')
    config = read(report/'base_config.json') if (report/'base_config.json').exists() else read(ROOT/f'outputs/YOLOE/finetune/scripts/{scene}/config.json')
    train_images = len(config['class_names'])*config['shot']
    dev_boxes = sum(len(r['ground_truth']) for r in read(ROOT/config['dev_manifest'])['samples'])
    test_boxes = sum(len(r['ground_truth']) for r in read(ROOT/config['test_manifest'])['samples'])
    complete = state['stage'] == 'complete'
    if not complete and not args.partial:
        raise RuntimeError(f"Search is not complete: {state['stage']}")
    jobs = state['jobs']
    dev_rows, test_rows, per_class, artifacts = [], [], [], []
    for job_id, entry in jobs.items():
        job = entry['job']
        run = Path(job['run_dir'])
        meta_path = run/'finetune_config.json'
        if not meta_path.exists():
            continue
        meta = read(meta_path)
        trial = report/'trials'/job_id
        write(trial/'config.json', read(job['config']))
        write(trial/'finetune_config.json', meta)
        policy = metadata_policy(meta, run)
        write(trial/'trainable_parameters.json', read(run/'trainable_parameters.json'))
        result_path = run/'results.csv'
        epochs = []
        if result_path.exists():
            epochs = list(csv.DictReader(result_path.open(encoding='utf-8-sig')))
            csv_write(trial/'results.csv', epochs)
        native_best = None
        if epochs:
            ap50_key = next(k for k in epochs[0] if k.endswith('/mAP50(B)'))
            ap95_key = next(k for k in epochs[0] if k.endswith('/mAP50-95(B)'))
            # Upstream fitness: 0.1 AP50 + 0.9 AP50:95; equality overwrites best.
            native_best = max(epochs, key=lambda r: (.1*float(r[ap50_key])+.9*float(r[ap95_key]), int(r['epoch'])))
        dev = Path(entry.get('dev_output', str(run/'dev_val100')))
        dev_paths = list(dev.glob('*/metrics.json'))
        row = dict(id=job_id, method=job['method'], scope=meta['strategy'],role=job.get('role','search'),
            protocol=meta.get('adaptation', {}).get('protocol', 'same_image') if job['method']=='yoloe-vp' else 'fixed_head',
            loss=meta.get('adaptation', {}).get('loss', 'classification') if job['method']=='yoloe-vp' else 'detection',
            status=entry['status'], evaluation_repaired=any(h.get('kind')!='training_restart' for h in entry.get('history',[])),
            training_retried=any(h.get('kind')=='training_restart' for h in entry.get('history',[])),
            gpu=entry.get('gpu'), cuda_visible_devices=meta.get('cuda_visible_devices'),
            training_seed=meta['training']['seed'], support_seed=0,
            training_shot=meta['shot_per_class'], prompt_shots=10 if job['method']=='yoloe-vp' else None,
            batch=meta['batch'], max_epochs=meta['training']['epochs'], actual_epochs=len(epochs),
            native_best_epoch_estimate=int(native_best['epoch']) if native_best else None,
            trainable_parameters=meta.get('trainable_parameter_count'),
            resumed=bool(meta.get('resume_history')), weights=rel(run/'weights/best.pt'), config=rel(trial/'config.json'),
            **{k:v for k,v in meta['training'].items() if k != 'epochs'},
            **{k:v for k,v in meta.get('adaptation',{}).items() if k not in {'scope','protocol','loss'}})
        row.update(policy)
        if len(dev_paths)==1:
            metric = read(dev_paths[0]); evidence = read(dev_paths[0].parent/'run.json')
            verify_checkpoint_policy(policy, evidence)
            if metric['evaluated_image_count'] != 100 or evidence['split'] != 'dev':
                raise ValueError(f'Invalid dev split: {job_id}')
            row.update({f'dev_{k}':metric[k] for k in FIELDS})
            row['weights_sha256'] = evidence['weights_sha256']
            row['dev_dataset_sha256'] = evidence['dataset_sha256']
            for filename in ['metrics.json','run.json']:
                write(report/'metrics'/job_id/'dev'/filename,read(dev_paths[0].parent/filename))
        dev_rows.append(row)
        for metric_path in sorted((run/'test_val300').glob('*/metrics.json')):
            metric = read(metric_path); evidence = read(metric_path.parent/'run.json')
            if metric['evaluated_image_count']!=300 or evidence['split']!='test':
                raise ValueError(f'Invalid test split: {job_id}')
            test = dict(id=job_id, method=job['method'], scope=meta['strategy'],
                training_seed=meta['training']['seed'], training_shot=meta['shot_per_class'],
                prompt_shots=evidence['prompt_shot_per_class'], dev_mAP50_95=row.get('dev_mAP50_95'),
                **{k:metric[k] for k in FIELDS}, weights=rel(run/'weights/best.pt'),
                weights_sha256=evidence['weights_sha256'], test_dataset_sha256=evidence['dataset_sha256'])
            verify_checkpoint_policy(policy, evidence)
            test.update(policy)
            test_rows.append(test)
            for label, values in metric['per_class'].items():
                per_class.append(dict(id=job_id,training_seed=meta['training']['seed'],
                    prompt_shots=test['prompt_shots'],class_name=label,
                    **{k:values[k] for k in ['AP50','AP50_95','precision','recall','tp','fp','fn','ground_truth_count']}))
            for filename in ['metrics.json','run.json']:
                write(report/'metrics'/job_id/metric_path.parent.name/filename,read(metric_path.parent/filename))
        artifacts.append(dict(id=job_id,run_dir=rel(run),training_status=meta['status'],
            original_job_history=entry.get('history',[]),config_sha256=digest(job['config']),
            metadata_sha256=digest(meta_path),input_sha256=meta['input_sha256'],
            preparation_sha256=meta['preparation_sha256'],dev_manifest_sha256=meta['dev_manifest_sha256'],
            test_manifest_sha256=meta['test_manifest_sha256'],source=meta['source'],
            resume_history=meta.get('resume_history',[])))
    dev_rows.sort(key=lambda r:r.get('dev_mAP50_95',-1),reverse=True)
    csv_write(report/'all_trials.csv',dev_rows)
    csv_write(report/'SAVPE_METHODS.csv',dev_rows)
    write_method_document(report,dev_rows,ROOT)
    csv_write(report/'all_test_results.csv',test_rows)
    csv_write(report/'per_class_test.csv',per_class)
    write(report/'artifact_index.json',dict(stage=state['stage'],trials=artifacts))

    base_dir=ROOT/f'outputs/YOLOE/finetune/{scene}_val300/runs'
    baselines=[('原始YOLOE视觉提示5shot',ROOT/f'outputs/YOLOE/experiments/{scene}_val300/yoloe-v8l-seg_5shot/metrics.json',5),
               ('原始YOLOE视觉提示10shot',ROOT/f'outputs/YOLOE/experiments/{scene}_val300/yoloe-v8l-seg_10shot/metrics.json',10)]
    if scene=='chache':
        baselines += [
               ('旧YOLOE固定头v1',base_dir/'yoloe_head_val100_v1/test_val300/fixed_head/metrics.json',None),
               ('旧YOLOE视觉提示v1-5shot',base_dir/'yoloe_vp_10shot_val100_v1/test_val300/visual_5shot/metrics.json',5),
               ('旧YOLOE视觉提示v1-10shot',base_dir/'yoloe_vp_10shot_val100_v1/test_val300/visual_10shot/metrics.json',10),
               ('YOLOv8-L微调v1',base_dir/'yolov8_val100_v1/test_val300/fixed_head/metrics.json',None)]
    else:
        base_dir=base_dir/args.stamp
        baselines += [('YOLOE基础固定头',base_dir/'baseline_head/test_val300/fixed_head/metrics.json',None),
                      ('YOLOE基础视觉提示5shot',base_dir/'baseline_vp/test_val300/visual_5shot/metrics.json',5),
                      ('YOLOE基础视觉提示10shot',base_dir/'baseline_vp/test_val300/visual_10shot/metrics.json',10),
                      ('YOLOv8-L基础微调',base_dir/'baseline_yolov8/test_val300/fixed_head/metrics.json',None)]
    baseline_rows=[]
    for index,(label,path,shot) in enumerate(baselines):
        if not path.exists() and args.partial:continue
        metric=read(path)
        baseline_policy = method_policy('yoloe-vp' if shot else 'yolov8' if 'YOLOv8' in label else 'yoloe-head', pretrained=index<2)
        baseline_rows.append(dict(id=label,prompt_shots=shot,**baseline_policy,**{k:metric[k] for k in FIELDS},source=rel(path)))
        write(report/'baselines'/f'test_{index}_metrics.json',metric)
        if (path.parent/'run.json').is_file():
            write(report/'baselines'/f'test_{index}_run.json',read(path.parent/'run.json'))
    y8_dev=runtime/'yolov8_dev100/fixed_head' if scene=='chache' else base_dir/'baseline_yolov8/dev_val100/fixed_head'
    for name,directory in [('raw_yoloe_dev',runtime/'baseline_dev100/visual_10shot'),('yolov8_dev',y8_dev)]:
        for filename in ['metrics.json','run.json']:
            if not (directory/filename).exists() and args.partial:continue
            write(report/'baselines'/name/filename,read(directory/filename))
    csv_write(report/'baselines.csv',baseline_rows)
    if scene=='diaowu' and (runtime/'baseline_dev100/visual_10shot/run.json').exists():
        visual=read(runtime/'baseline_dev100/visual_10shot/run.json')['references']
        reference_pool={r['image_sha256']:r for values in visual.values() for r in values}
        checks=[]
        for shot in [5,10]:
            chosen=read(ROOT/f'data/references/{scene}/{shot}shot/seed_0.json')['positive_samples']
            hashes={r['image_sha256'] for r in chosen}
            expected=sorted([dict(image_sha256=h,bboxes=reference_pool[h]['bboxes']) for h in hashes],key=lambda r:r['image_sha256'])
            historical=read(ROOT/f'outputs/YOLOE/experiments/{scene}_val300/yoloe-v8l-seg_{shot}shot/run.json')
            original=sorted([dict(image_sha256=r['image_sha256'],bboxes=r['bboxes'])
                for target in historical['targets'] for r in target['prompt_snapshot']['references']],key=lambda r:r['image_sha256'])
            if original!=expected:
                raise ValueError(f'Historical {shot}-shot baseline used different reference images or boxes.')
            checks.append(dict(prompt_shots=shot,equal=True,image_count=len(expected),
                reference_sha256=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()))
        write(report/'baseline_reference_equivalence.json',dict(checks=checks,
            policy='Compare image content hashes and all pixel prompt boxes, using the same frozen seed-0 support sets.'))
    winners={family:next((r for r in dev_rows if r['method']==family and '_seed' not in r['id'] and r['role']!='baseline' and 'dev_mAP50_95' in r),None)
             for family in ['yoloe-head','yoloe-vp']}
    summary=[]
    for family,winner in winners.items():
        if not winner:
            continue
        shots=[None] if family=='yoloe-head' else [5,10]
        for shot in shots:
            group=[r for r in test_rows if r['id'].split('_seed')[0]==winner['id'] and r['prompt_shots']==shot]
            if complete and sorted(r['training_seed'] for r in group)!=[0,1,2]:
                raise ValueError(f'Missing seed results: {winner["id"]}, prompt={shot}')
            if group:
                item=dict(method=family,dev_selected_config=winner['id'],prompt_shots=shot,n=len(group),
                          seeds=sorted(r['training_seed'] for r in group),std_definition='sample SD, ddof=1; training seeds only')
                for key in FIELDS[:5]:
                    item[key+'_mean']=statistics.mean(r[key] for r in group)
                    item[key+'_std']=statistics.stdev(r[key] for r in group) if len(group)>1 else None
                summary.append(item)
    write(report/'seed_statistics.json',dict(rows=summary))
    csv_write(report/'seed_statistics.csv',summary)
    if complete:
        if len(dev_rows)!=len(plan['jobs'])+4 or any(r['status']!='completed' or 'dev_mAP50_95' not in r for r in dev_rows):
            raise ValueError('Expected all planned trials + 4 seed repeats with complete dev evaluation.')
        frozen=read(report/'test_selection.json')['selected']
        selected_ids={j['id'] for j in frozen}
        expected_test_count=sum(2 if j['method']=='yoloe-vp' else 1 for j in frozen)
        if selected_ids!={r['id'] for r in test_rows} or len(test_rows)!=expected_test_count:
            raise ValueError('Test results do not match frozen candidates and inference settings.')
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    write(report/'summary.json',dict(status='complete' if complete else 'partial',updated=now,
        scene=scene,main_trials=sum('_seed' not in r['id'] and r['role']!='baseline' for r in dev_rows),
        baseline_trials=sum(r['role']=='baseline' for r in dev_rows),seed_repeats=sum('_seed' in r['id'] for r in dev_rows),
        test_settings=len(test_rows),dev_winners={k:v['id'] if v else None for k,v in winners.items()},seed_statistics=summary))
    lines=[f'# YOLOE优化结果：{"叉车" if scene=="chache" else "吊物"}场景','',f'状态：{state["stage"]}。更新时间：{now}。',
        '',f'共{sum("_seed" not in r["id"] and r["role"]!="baseline" for r in dev_rows)}组主配置、{sum(r["role"]=="baseline" for r in dev_rows)}组基础对照、{sum("_seed" in r["id"] for r in dev_rows)}组训练种子复跑。',
        f'固定{train_images}张训练图片（每类10张，support-seed=0）、独立100张dev（{dev_boxes}框）、300张test（{test_boxes}框）。',
        '配置只按dev统一推理AP50:95排序。每种方法取seed=0前三名，加各自第一名的seed=1、2，再做test；',
        '候选冻结记录见[test_selection.json](test_selection.json)。推理5/10shot使用训练池冻结子集，不改变训练预算。','',
        '## SAVPE分类与阅读入口','',
        '固定头yoloe-head：训练前移除SAVPE，推理不需要参考图。原搜索的视觉提示yoloe-vp未冻结SAVPE，各scope均保留并微调它；新增freeze_savpe条件另表记录。两种VP都用SAVPE编码5/10shot参考图。',
        '原生YOLOv8从未包含SAVPE。表中removed / retained / not_applicable分别对应这三类。',
        '[逐实验方法表](SAVPE_METHODS.md)、[筛选用CSV](SAVPE_METHODS.csv)、'
        +f'[完整机制说明]({os.path.relpath(ROOT/"docs/guides/YOLOE_FINETUNING_METHODS.md",report)})。','',
        '## 按dev选出的配置及三个训练种子的测试结果','',
        '均值±样本标准差，单位为百分比/百分点；固定支持图，只改变训练随机种子。',
        '不能把三个种子的平均值称为支持集抽样方差，也不以测试集成绩更高的种子替代dev选择。','',
        '| 方法 | dev选定配置 | 推理shot | n | test AP50 | test AP50:95 |','|---|---|---:|---:|---:|---:|']
    for r in summary:
        def fmt(key):
            sd=r[key+'_std']
            return f'{100*r[key+"_mean"]:.2f} ± {100*sd:.2f}' if sd is not None else f'{100*r[key+"_mean"]:.2f}'
        lines.append(f'| {r["method"]} | {r["dev_selected_config"]} | {r["prompt_shots"] or "不适用"} | {r["n"]} | {fmt("mAP50")} | {fmt("mAP50_95")} |')
    if complete:
        vp=next(r for r in summary if r['method']=='yoloe-vp' and r['prompt_shots']==10)
        old=baseline_rows[4]
        lines += ['',f'视觉提示10shot相对基础视觉微调：三个训练种子的平均AP50变化{100*(vp["mAP50_mean"]-old["mAP50"]):+.2f}个百分点，'
            f'AP50:95变化{100*(vp["mAP50_95_mean"]-old["mAP50_95"]):+.2f}个百分点。',
            '与YOLOv8-L基线比较见下表；单次差异不代表统计显著。',
            '各配置涉及不同训练范围、损失和预算，不能把全部收益单独归因为其中某一项。']
    if (report/'test_comparison.png').exists():
        lines += ['', '![300张test：dev选定配置，三个训练种子均值与样本标准差](test_comparison.png)']
    lines+=['','## 基础对照与原始权重（300张test）','']+table(baseline_rows)
    lines+=['','## 冻结候选全部测试结果','']+table(test_rows)
    lines+=['','## 全部dev结果','',
        'AP和P/R均以百分比展示。原生训练器的best按0.1 AP50+0.9 AP50:95选取；表中dev分数是在该best上重新统一推理的成绩，',
        '与原生训练日志的分数不混用。`native_best_epoch_estimate`是按results.csv回算的原生fitness优胜轮次。','',
        '| 配置 | 方法/范围 | SAVPE | 协议/损失 | seed | 实际/最大轮次 | batch/尺寸 | dev AP50 | dev AP50:95 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in dev_rows:
        values=f'{100*r["dev_mAP50"]:.2f} | {100*r["dev_mAP50_95"]:.2f}' if 'dev_mAP50' in r else '— | —'
        lines.append(f'| {r["id"]} | {r["method"]}/{r["scope"]} | {r["savpe_state"]} | {r["protocol"]}/{r["loss"]} | {r["training_seed"]} | {r["actual_epochs"]}/{r["max_epochs"]} | {r["batch"]}/{r["imgsz"]} | {values} |')
    lines+=['','## 搜索范围与复现','',
        '48组覆盖：固定头linear/neck/full、学习率、AdamW/SGD、weight decay、余弦调度、增强、batch、640/960/1024、文本描述；',
        '视觉提示覆盖SAVPE/head/neck/full、分类/检测损失、SAVPE BN统计固定、跨图支持训练、分辨率和增强。',
        '这是有限且有记录的搜索，未穷举所有参数组合。每个GPU独立一项实验；使用过物理GPU 0–7；不使用DDP。FP32，amp/half=false。',
        '每轮dev验证，搜索patience=12，基础对照patience=10。实际epoch含早停差异，训练参数量与完整配置见[all_trials.csv](all_trials.csv)、[trials/](trials/)。',
        '', '```bash',
        f'python scripts/summarize_yoloe_optimization.py --stamp {args.stamp}',
        '# 仅在此搜索被中断、并确认原进程已经退出时继续：',
        f'python scripts/run_yoloe_optimization_sweep.py --scene {scene} --stamp {args.stamp} --execute --resume --gpus 0 1 2 3 4 5 6 7',
        '```','',
        '保留原始权重重新微调；只有意外中断的同一VP运行从自身last.pt恢复。恢复记录保存checkpoint哈希、已完成epoch及原文件备份。',
        f'发生恢复的配置：{", ".join(r["id"] for r in dev_rows if r["resumed"]) or "无"}。上游恢复不保存RNG和早停器完整历史。',
        f'补评恢复的配置：{", ".join(r["id"] for r in dev_rows if r["evaluation_repaired"]) or "无"}。执行经过见RUN_LOG.md。',
        f'从原始权重重新执行的配置：{", ".join(r["id"] for r in dev_rows if r["training_retried"]) or "无"}；原失败尝试与原因保留。',
        '', '## 指标口径和适用范围','',
        '主表为项目class-aware 101点AP；P/R为confidence≥0.5、IoU≥0.5。AP使用全部保留置信分数，不是该阈值下的P。',
        '各metrics.json另含pycocotools官方COCO AP、0.5/0.7阈值和逐类结果；两种AP的数值不混写。',
        '固定头一次多类查询，视觉提示按类查询后合并；conf=0.001、NMS IoU=0.7、FP32，最终每图每类上限100。',
        '各实验按自己的训练尺寸推理；高分辨率行不是相同推理成本。clip后零面积框被丢弃并记录，非有限值报错。',
        '原始YOLOE视觉基线来自既有seg checkpoint的bbox输出；微调视觉提示用detect结构保留SAVPE。',
        f'YOLOE使用48组搜索，YOLOv8仅有一次基础训练，计算预算不同；结论仅针对当前已见{len(config["class_names"])}类，未验证开放词汇/新类别能力。',
        '旧300张测试结果此前已被查看；本轮按独立dev选参，但仍属于探索结果，不应宣称全新盲测。',
        '训练/dev/test内容哈希无交集；不等于已证明不同视频序列独立。','',
        '## 文件入口','',
        '- [全部参数与dev表](all_trials.csv)、[全部候选test表](all_test_results.csv)、[逐类test表](per_class_test.csv)、[种子统计](seed_statistics.csv)。',
        '- [基线表](baselines.csv)、[原始指标及推理身份](metrics/)、[配置和训练曲线CSV](trials/)、[数据/源码/权重输入指纹](artifact_index.json)。',
        '- [执行与修复日志](RUN_LOG.md)。报告中的JSON/CSV/Markdown可按现有规则随Git同步；未自动add/commit/push。',
        f'- 权重与逐图预测在`outputs/YOLOE/finetune/{scene}_val300/runs/{args.stamp}/`；命令与控制台日志在`outputs/_shared/runtime/{args.stamp}/`，均留本机。','']
    if (report/'dev_ranking.png').exists():
        lines += ['[48组dev排名图](dev_ranking.png)、[原生训练曲线](training_curves.png)。','']
    if (report/'thresholds/README.md').exists():
        lines += ['[仅用dev选置信度阈值后的P/R/F2对照](thresholds/README.md)。该调整不改变AP。','']
    if scene=='diaowu':
        lines += ['基线一致性核验：[测试图片和真值](baseline_dataset_equivalence.json)、[5/10shot参考原图及框](baseline_reference_equivalence.json)。','']
    if (report/'frozen_savpe_20261030/plan.json').exists():
        lines += ['','<!-- frozen-savpe:start -->','','## 新增：冻结SAVPE、保留视觉提示','',
            '[全部微调实验总表（含新增条件）](ALL_EXPERIMENTS.md)、[新增对照报告](frozen_savpe_20261030/README.md)、'
            '[全部训练CSV](all_trials_with_frozen_savpe.csv)、[全部测试CSV](all_test_results_with_frozen_savpe.csv)。','',
            '<!-- frozen-savpe:end -->','']
    target=report/('README.md' if complete else 'live_progress.md')
    target.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status=state['stage'],trials=len(dev_rows),test_rows=len(test_rows),report=rel(target)),ensure_ascii=False))


if __name__=='__main__':
    main()

"""Keep compact, traceable fine-tuning maxima; --compact archives/removes old report details.

No training, inference, checkpoint or dataset mutations. Existing compact layouts
are refreshed from their consolidated CSVs without recreating process folders.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from industrial_fewshot.comparison_layout import optimization_report_dir

LABELS={'yoloe-head':'YOLOE固定头微调','yoloe-vp':'YOLOE视觉提示·训练SAVPE',
        'yoloe-vp-frozen':'YOLOE视觉提示·冻结SAVPE','yolov8':'YOLOv8-L预训练微调','raw':'原始YOLOE·未微调'}
KEEP={'README.md','ALL_EXPERIMENTS.md','BEST_RESULTS.csv','all_trials.csv','all_test_results.csv','best_evidence.json','SAVPE_METHODS.md'}
METRICS=('mAP50','mAP50_95')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_read(path):
    return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))


def csv_write(path, rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_report(scene):
    return optimization_report_dir('yoloe_search_20261029_r2' if scene=='chache' else 'yoloe_diaowu_search_20261029')


def is_compact(report):
    marker=Path(report)/'best_evidence.json'
    return marker.exists() and read(marker).get('layout_version')=='compact_v1'


def family(row):
    if row.get('method')=='yoloe-vp' and str(row.get('savpe_parameters_frozen',row.get('freeze_savpe',''))).lower()=='true':
        return 'yoloe-vp-frozen'
    return row['method']


def checkpoint_identity(row):
    return row.get('weights_sha256') or row.get('weights') or row.get('source') or row['id']


def deduplicate(rows):
    unique={}
    for row in rows:
        key=(checkpoint_identity(row),str(row.get('prompt_shots') or ''))
        unique.setdefault(key,row)
    return list(unique.values())


def select_maxima(rows):
    groups={}
    for row in deduplicate(rows):
        groups.setdefault((family(row),str(row.get('prompt_shots') or '')),[]).append(row)
    result=[]
    for (method,shot),candidates in sorted(groups.items(),key=lambda x:(list(LABELS).index(x[0][0]),int(x[0][1] or 0))):
        for metric in METRICS:
            # Retain one real row per maximum. Never combine P/R or other scores
            # from another checkpoint into a synthetic winning experiment.
            best=max(candidates,key=lambda row:float(row[metric]))
            result.append(dict(best,method_family=method,method_display=LABELS[method],
                               selection_metric=metric,selection_value=float(best[metric]),tested_settings=len(candidates)))
    return result


def display_maxima(winners):
    """Show a winning test once when it leads both AP metrics."""
    merged={}
    for row in winners:
        key=(row['method_family'],checkpoint_identity(row),str(row.get('prompt_shots') or ''))
        item=merged.setdefault(key,dict(row,selection_metrics=[]))
        item['selection_metrics'].append(row['selection_metric'])
    for row in merged.values():
        row['selection_label']=('两项AP均最高' if len(row['selection_metrics'])==2 else
            '最高AP50' if row['selection_metrics']==['mAP50'] else '最高AP50:95')
    return list(merged.values())


def metric_path(row,report):
    if row.get('source') and (ROOT/row['source']).is_file():return ROOT/row['source']
    mode='visual_'+str(row['prompt_shots'])+'shot' if row.get('prompt_shots') else 'fixed_head'
    if row.get('weights'):
        candidate=(ROOT/row['weights']).parent.parent/'test_val300'/mode/'metrics.json'
        if candidate.exists():return candidate
    for base in [report,report/'frozen_savpe_20261030']:
        candidate=base/'metrics'/row['id']/mode/'metrics.json'
        if candidate.exists():return candidate
    raise FileNotFoundError(f'Missing recorded test metrics: {row["id"]}')


def training_directory(metric):
    return next((p for p in metric.parents if (p/'finetune_config.json').is_file()),None)


def load_original(report,scene):
    trials=csv_read(report/'all_trials_with_frozen_savpe.csv')
    tests=csv_read(report/'all_test_results_with_frozen_savpe.csv')
    baselines=csv_read(report/'baselines.csv')
    for row in baselines:
        source=ROOT/row['source'];identity=read(source.parent/'run.json')
        run=training_directory(source);meta=read(run/'finetune_config.json') if run else {}
        row['method']=identity.get('method') or ('raw' if row['id'].startswith('原始') else 'yolov8' if 'YOLOv8' in row['id'] else 'yoloe-vp' if row['prompt_shots'] else 'yoloe-head')
        row['training_seed']=meta.get('training',{}).get('seed','')
        row['scope']=meta.get('strategy','')
        row['savpe_parameters_frozen']=False
        row['training_shot']=meta.get('shot_per_class',0)
        if identity.get('weights_sha256'):row['weights_sha256']=identity['weights_sha256']
        if run:
            row['weights']=str((run/'weights/best.pt').relative_to(ROOT))
            if not any(checkpoint_identity(r)==checkpoint_identity(row) for r in trials):
                trial=dict(row,id=run.name,status=meta['status'],role='historical_baseline',
                    protocol=meta.get('adaptation',{}).get('protocol','same_image') if row['method']=='yoloe-vp' else 'fixed_head',
                    imgsz=meta['training']['imgsz'],batch=meta['batch'],max_epochs=meta['training']['epochs'])
                for key in ['mAP50','mAP50_95','precision','recall','f2','tp','fp','fn']:trial.pop(key,None)
                if row['method']=='yolov8':
                    metric=report/'baselines/yolov8_dev/metrics.json'
                    if metric.exists():trial.update({f'dev_{k}':v for k,v in read(metric).items() if k in METRICS})
                trials.append(trial)
        tests.append(row)
    tests=deduplicate(tests)
    for row in tests:
        source=metric_path(row,report)
        row['source']=str(source.relative_to(ROOT))
        row['record_key']=hashlib.sha256((checkpoint_identity(row)+'|'+str(row.get('prompt_shots') or '')).encode()).hexdigest()[:20]
    for row in trials+tests:
        if row.get('config') and (ROOT/row['config']).is_relative_to(report):
            row['archived_config_member']='search_report/'+str((ROOT/row['config']).relative_to(report))
            if row.get('weights'):
                run=(ROOT/row['weights']).parent.parent
                candidate=ROOT/'outputs/_shared/runtime'/run.parent.name/'configs'/f'{run.name}.json'
                row['config']=str((candidate if candidate.exists() else run/'finetune_config.json').relative_to(ROOT))
    return trials,tests


def archive_and_verify(sources,destination):
    if destination.exists():raise FileExistsError(f'Preserving prior archive: {destination}')
    files={}
    for directory,prefix in sources:
        for file in sorted(directory.rglob('*')):
            if file.is_file():files[prefix+'/'+str(file.relative_to(directory))]=dict(bytes=file.stat().st_size,sha256=sha(file))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    with tarfile.open(temporary,'w:gz') as archive:
        for directory,prefix in sources:
            archive.add(directory,arcname=prefix)
    with tarfile.open(temporary,'r:gz') as archive:
        found={}
        for member in archive.getmembers():
            if member.isfile():found[member.name]=dict(bytes=member.size,sha256=hashlib.sha256(archive.extractfile(member).read()).hexdigest())
    if found!=files:raise RuntimeError('Archive content verification failed; no files removed.')
    temporary.rename(destination)
    return dict(path=str(destination.relative_to(ROOT)),sha256=sha(destination),file_count=len(files),verified_all_file_sha256=True)


def collect_evidence(report,winners,trials):
    evidence={}
    chosen={row['record_key']:row for row in winners}
    for key,row in chosen.items():
        source=metric_path(row,report)
        item=dict(source=str(source.relative_to(ROOT)),metrics_sha256=sha(source),metrics=read(source),run=read(source.parent/'run.json'))
        run=training_directory(source)
        if run:
            item['finetune_config']=read(run/'finetune_config.json')
            for name in ['trainable_parameters.json','frozen_savpe_verification.json']:
                if (run/name).exists():item[name]=read(run/name)
        evidence[key]=item
    stats=[]
    for path in [report/'seed_statistics.csv',report/'frozen_savpe_20261030/seed_statistics.csv']:
        if path.exists():stats.extend(csv_read(path))
    return dict(layout_version='compact_v1',scene=report.parent.parent.name.removesuffix('_val300'),
        all_completed=True,training_record_count=len(trials),test_setting_count=len(deduplicate(csv_read(report/'all_test_results_with_frozen_savpe.csv'))),
        policy='Observed test maxima separately for AP50 and AP50:95. Full real rows, no synthetic metric mixing. Not dev-selected unbiased estimates.',
        winners=evidence,dev_selected_seed_statistics=stats)


def percentage(value):return '—' if value in {None,''} else f'{100*float(value):.2f}'


def render(report,trials,tests,bundle):
    winners=select_maxima(tests)
    csv_write(report/'BEST_RESULTS.csv',winners)
    name='叉车' if bundle['scene']=='chache' else '吊物'
    lines=[f'# {name}：全部测试集推理结果','',
        f'全部实验已完成。合并{len(trials)}项训练记录、{len(tests)}项测试设置；测试集固定300张。',
        '**下表保留全部已测试设置，每个模型权重与推理shot组合只列一次；所有指标均为300张test上的推理结果，不是训练或dev成绩。**','',
        '末列标注同一方法、同一推理shot下的历史最高值；两项AP的最高值可能来自不同实验。最高值不是按dev选定配置的多种子平均，也不是盲测成绩。','',
        '| 方法 | 推理shot | 对应实验 / seed | test AP50 % | test AP50:95 % | P@0.5 % | R@0.5 % | F2@0.5 % | TP/FP/FN | 最高值标记 |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|---|']
    if (report.parent/'yolo_5shot_20261102/README.md').is_file():
        lines[2:2]=['本表为每类10张训练的历史搜索及其原始基线；[每类5张训练的新对照](../yolo_5shot_20261102/README.md)单独汇总，避免混淆训练预算与推理参考数量。','']
    selections={(row['method_family'],checkpoint_identity(row),str(row.get('prompt_shots') or '')):row['selection_label']
                for row in display_maxima(winners)}
    ordered=sorted(tests,key=lambda row:(list(LABELS).index(family(row)),int(row.get('prompt_shots') or 0),-float(row['mAP50'])))
    for row in ordered:
        selection=selections.get((family(row),checkpoint_identity(row),str(row.get('prompt_shots') or '')),'—')
        label=f'`{row["id"]}` / {row["training_seed"] if row.get("training_seed") not in {None,""} else "—"}'
        values=' | '.join(percentage(row.get(k)) for k in ['mAP50','mAP50_95','precision','recall','f2'])
        counts='/'.join(str(row.get(k) or '—') if row.get(k) not in {0,'0'} else '0' for k in ['tp','fp','fn'])
        lines.append(f'| {LABELS[family(row)]} | {row.get("prompt_shots") or "不适用"} | {label} | {values} | {counts} | {selection} |')
    lines+=['','每个最高值的完整同次实验AP、P/R、TP/FP/FN、配置、权重路径和哈希见[BEST_RESULTS.csv](BEST_RESULTS.csv)；原始获胜指标及身份信息见[best_evidence.json](best_evidence.json)。',
        'P/R为confidence≥0.5、IoU≥0.5；AP使用完整保留候选。每行AP、P/R和TP/FP/FN均来自同一次实际测试，不跨配置拼接。训练每类10张；VP的5/10shot只表示推理参考数量。']
    lines+=['','## 文件与历史明细','',
        '- [全部训练与dev CSV](all_trials.csv)：保留各范围、参数和seed记录；只有dev成绩的配置不参与上表test比较。',
        '- [全部300张test CSV](all_test_results.csv)：包含原始视觉基线、YOLOv8及冻结/训练SAVPE；同权重同shot去重。',
        '- [方法说明](SAVPE_METHODS.md)：哪些模块更新、哪些使用SAVPE。',
        '- `best_evidence.json`也保留原dev选定配置的多种子统计，以及旧报告完整压缩档案的位置和SHA256。',
        '旧过程报告、重复CSV/JSON、逐实验配置副本和曲线图已从主目录清理；完整历史在本机压缩归档，所有训练权重、数据和runtime原记录保持原位。',
        '不同方法的尺寸、训练范围和调参预算不同；当前成绩只描述已见类别。','',
        '更新本表：`python scripts/summarize_finetuning_best.py`。不会启动训练、推理，也不会重新生成过程文件。','']
    (report/'ALL_EXPERIMENTS.md').write_text('\n'.join(lines))
    (report/'README.md').write_text(f'# {name}微调结果\n\n直接查看[全部测试集推理结果（含最高值标记）](ALL_EXPERIMENTS.md)或[最高值CSV](BEST_RESULTS.csv)。\n\n全部逐项记录保存在[训练与dev CSV](all_trials.csv)和[测试推理CSV](all_test_results.csv)，最佳结果原始证据在[best_evidence.json](best_evidence.json)。\n')
    (report/'SAVPE_METHODS.md').write_text('''# 微调方法速查

| 方法 / 范围 | Backbone | Neck | 检测头 | SAVPE | 推理参考图 |
|---|---|---|---|---|---|
| 固定头head | 冻结 | 冻结 | 仅分类输出层训练 | 移除 | 无 |
| 固定头neck | 冻结 | 训练 | 训练 | 移除 | 无 |
| 固定头full | 训练 | 训练 | 训练 | 移除 | 无 |
| VP savpe | 冻结 | 冻结 | 其余部分冻结 | 训练 | 5/10shot |
| VP head | 冻结 | 冻结 | 训练 | 训练 | 5/10shot |
| VP neck | 冻结 | 训练 | 训练 | 训练 | 5/10shot |
| VP full | 训练 | 训练 | 训练 | 训练 | 5/10shot |
| VP head + freeze_savpe | 冻结 | 冻结 | 其余部分训练 | 保留并冻结 | 5/10shot |
| VP neck + freeze_savpe | 冻结 | 训练 | 其余部分训练 | 保留并冻结 | 5/10shot |
| VP full + freeze_savpe | 训练 | 训练 | 其余部分训练 | 保留并冻结 | 5/10shot |
| YOLOv8-L全量 | 训练 | 训练 | 训练 | 原架构没有 | 无 |

same_image/cross_image表示训练参考来自当前图/另一训练图，不改变上述范围。仅固定BN统计不等于冻结SAVPE权重；freeze_savpe同时固定全部参数和BN统计。full遵守DFL固定投影等上游例外。

[全部测试推理结果与最高值](ALL_EXPERIMENTS.md)、[逐实验参数和dev记录](all_trials.csv)、[详细机制](../../../../../../docs/guides/YOLOE_FINETUNING_METHODS.md)。
''')
    bundle['training_record_count']=len(trials);bundle['test_setting_count']=len(tests)
    (report/'best_evidence.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return winners


def process_scene(scene,compact=False):
    report=scene_report(scene)
    if is_compact(report):
        trials=csv_read(report/'all_trials.csv');tests=csv_read(report/'all_test_results.csv');bundle=read(report/'best_evidence.json')
    else:
        for stamp in ['yoloe_search_20261029_r2' if scene=='chache' else 'yoloe_diaowu_search_20261029',f'yoloe_{scene}_frozen_savpe_20261030']:
            state=read(ROOT/'outputs/_shared/runtime'/stamp/'state.json')
            if state['stage']!='complete' or state.get('processes'):raise RuntimeError(f'Experiment not complete: {stamp}')
        trials,tests=load_original(report,scene)
        winners=select_maxima(tests);bundle=collect_evidence(report,winners,trials)
        if not compact:raise ValueError('Initial cleanup requires explicit --compact; no files changed.')
        sources=[(report,'search_report')]
        legacy=report.parent/'yoloe_fixed_head_baseline_20261029'
        if legacy.exists():sources.append((legacy,'legacy_fixed_head_report'))
        bundle['history_archive']=archive_and_verify(sources,ROOT/f'outputs/_shared/archive/history/finetuning_details_20261030/{scene}_val300.tar.gz')
        # The archive has been verified byte for byte before removing anything.
        for item in report.iterdir():
            if item.is_dir():shutil.rmtree(item)
            elif item.name not in KEEP:item.unlink()
        if legacy.exists():shutil.rmtree(legacy)
        csv_write(report/'all_trials.csv',trials);csv_write(report/'all_test_results.csv',tests)
    winners=render(report,trials,tests,bundle)
    return report,winners


def cross_dataset_index(results):
    base=ROOT/'outputs/YOLOE/finetune'
    lines=['# 两个场景：各微调方式的测试推理最高成绩','',
        '本页汇总20261029搜索及其冻结SAVPE对照，微调训练预算均为每类10张；[每类5张训练的新对照](5SHOT_COMPARISON.md)另表保存。','',
        '此页是300张测试集推理成绩的最高值摘要，不是训练成绩。全部训练和测试已完成；两个场景各31项测试设置的完整推理结果见文末链接。AP50与AP50:95分别取已测试结果最高值；同一实验同时获得两项最高值时合并为一行，不同实验分别列出。','',
        '| 场景 | 方法 | 推理shot | 选取依据 | AP50 % | AP50:95 % | P@0.5 % | R@0.5 % | 实验 / seed |',
        '|---|---|---:|---|---:|---:|---:|---:|---|']
    for report,winners in results:
        scene='叉车' if report.parent.parent.name=='chache_val300' else '吊物'
        for row in display_maxima(winners):
            if family(row)=='raw':continue
            selection=row['selection_label']
            values=' | '.join(percentage(row.get(k)) for k in ['mAP50','mAP50_95','precision','recall'])
            lines.append(f'| {scene} | {row["method_display"]} | {row.get("prompt_shots") or "不适用"} | {selection} | {values} | `{row["id"]}` / {row.get("training_seed", "—")} |')
    lines+=['','[叉车完整推理表](chache_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md) · [吊物完整推理表](diaowu_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md) · [全部专题](../README.md)','']
    (base/'BEST_RESULTS.md').write_text('\n'.join(lines))
    for report,_ in results:
        content=f'# {report.parent.parent.name}微调\n\n[全部测试推理结果与最高值](yoloe_search_20261029/ALL_EXPERIMENTS.md) · [两个场景最高成绩对照](../../BEST_RESULTS.md)\n'
        if (report.parent/'yolo_5shot_20261102/README.md').is_file():
            content+='\n[5shot训练与完整测试推理对照](yolo_5shot_20261102/README.md)\n'
        (report.parent/'README.md').write_text(content)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=['chache','diaowu'],action='append')
    parser.add_argument('--compact',action='store_true',help='Archive and remove redundant process report files.')
    args=parser.parse_args()
    results=[process_scene(scene,args.compact) for scene in args.scene or ['chache','diaowu']]
    if len(results)==2:cross_dataset_index(results)
    print(json.dumps([dict(report=str(p.relative_to(ROOT)),kept_files=len(list(p.iterdir())),best_rows=len(r)) for p,r in results],ensure_ascii=False))


if __name__=='__main__':main()

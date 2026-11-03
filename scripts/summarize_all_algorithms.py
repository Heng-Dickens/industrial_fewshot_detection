"""One traceable table for every saved experiment on the two frozen 300-image test cohorts.

Reads saved metrics only. Training/dev/historical 99-image scores are not test rows.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import csv
import hashlib
import json
import os
import sys
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from industrial_fewshot.paths import resolve_location
from summarize_results import refresh, manifest_identities, portable_path, record_evidence

DATASETS=['chache_val300','diaowu_val300']
METHODS=['GroundingDINO','PET-DINO','WeDetect','YOLOE','YOLOv8','Qwen','OVD-MLLM','JSTY-YOLO']
RATES=['AP50','AP75','AP50_95','AP85','AP90','AP95','AP_box20','COCO_AR100',
       'COCO_AP_small','COCO_AP_medium','COCO_AP_large','mean_matched_iou',
       'precision_0_5','recall_0_5','f1_0_5','f2_0_5','precision_0_7','recall_0_7','f1_0_7','f2_0_7',
       'image_accuracy','negative_false_positive_rate','negative_error_rate']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).is_file() else {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def link(label,path):
    return f'[{label}]({quote(path,safe="/._-")})' if path else '—'


def pct(value):
    return '—' if value is None else f'{100*value:.2f}'


def geometry_index():
    result={}
    for row in read(ROOT/'outputs/_shared/comparisons/iou85_box20_20261028/analysis.json').get('rows',[]):
        dataset=row['dataset']
        if dataset not in DATASETS:
            continue
        value=row['run_dir']
        candidates=[ROOT/'outputs/_shared/datasets'/dataset/value,ROOT/'outputs/runs'/dataset/value,ROOT/value]
        source=next((resolve_location(p) for p in candidates if resolve_location(p).is_dir()),None)
        if source:
            result[source.resolve()]=row.get('AP_box20')
    return result


def training_metadata(source):
    for directory in source.parents:
        p=directory/'finetune_config.json'
        if p.is_file():
            return read(p)
        if directory==ROOT:
            break
    return {}


def collect_one(source,dataset,marker,expected,geometry,indexed=None):
    indexed=indexed or {}
    run=read(source/'run.json');metrics=read(source/'metrics.json')
    metadata=training_metadata(source)
    cascade=run.get('cascade',{})
    if cascade.get('source_run'):
        inherited=portable_path(cascade['source_run'],ROOT)
        metadata=training_metadata(inherited) if inherited else {}
    method=run.get('method','')
    folder=source.relative_to(ROOT/'outputs').parts[0]
    algorithm='YOLOv8' if method=='yolov8' else folder
    if algorithm not in METHODS:
        raise ValueError(f'Unknown method: {source}')
    targets=run.get('targets',[])
    model=indexed.get('model') or Path(run.get('checkpoint') or run.get('model') or metadata.get('paths',{}).get('weights','')).stem
    if metadata:
        model=Path(metadata['paths']['weights']).stem
    if not model:
        model='+'.join(dict.fromkeys(Path(str(t.get('model',''))).stem for t in targets)) or source.name
    budget=run.get('training_shot_per_class')
    regime='任务预训练权重/无任务微调'
    savpe='—'
    scope=metadata.get('strategy','')
    protocol=metadata.get('adaptation',{}).get('protocol','')
    if metadata:
        budget=metadata.get('shot_per_class') if budget is None else budget
        regime='fewshot微调'
        if metadata.get('status')=='initialized_no_training':
            budget=0;regime='目标分类输出初始化/未训练'
        if method=='yoloe-head':
            savpe='移除'
        elif method=='yoloe-vp':
            savpe='冻结' if metadata.get('adaptation',{}).get('freeze_savpe') else '训练'
    elif algorithm=='JSTY-YOLO':
        budget=None;regime='自定义网络监督训练（训练量未绑定快照）'
    else:
        budget=0
        if algorithm=='YOLOE':savpe='预训练/冻结'
    shot=run.get('prompt_shot_per_class',indexed.get('shot_per_class'))
    if shot is None:
        shot=run.get('experiment',{}).get('positive_count',run.get('inference_shot_per_class',0))
    if algorithm in {'YOLOv8','JSTY-YOLO'} or method=='yoloe-head':
        shot=None
    stage=run.get('stage','')
    mllm_shot=None
    if algorithm=='OVD-MLLM':
        stage=stage or source.name
        mllm_shot=cascade.get('mllm_reference_shot_per_class',1) if stage in {'filtered','deduplicated','refined'} else 0
        savpe=cascade.get('savpe','预训练/冻结')
        shot=cascade.get('prompt_shot_per_class',10)
        if cascade:
            budget=cascade['training_shot_per_class'];regime='fewshot微调→Qwen候选复核'
    version=indexed.get('prompt_version','')
    variant=indexed.get('input') or run.get('experiment',{}).get('variant') or stage
    seed=metadata.get('training',{}).get('seed',indexed.get('seed'))
    support_seed=indexed.get('seed')
    if metadata:
        preparation=portable_path(metadata.get('paths',{}).get('data'),ROOT)
        support_seed=read(preparation.with_name('preparation.json')).get('support_seed') if preparation else None
    negatives=indexed.get('negative_references',0)
    inference=run.get('inference',{})
    local_settings={}
    for target in targets:
        local_settings=target.get('local_model',{}).get('settings',{})
        if local_settings:break
    if support_seed is None:
        support_seed=local_settings.get('selection_seed')
    size=inference.get('input_size',inference.get('imgsz',metadata.get('training',{}).get('imgsz')))
    if size is None:
        for target in targets:
            settings=target.get('local_model',{}).get('settings',{})
            size=settings.get('imgsz',settings.get('input_size'))
            if size is not None:break
    input_size='×'.join(map(str,size)) if isinstance(size,list) else str(size) if size is not None else ''
    model_note=model
    experiment=source.parent.parent.name+'/'+source.parent.name if metadata else source.name
    if version or variant:
        experiment+=' / '+' / '.join(str(v) for v in [version,variant] if v)
    if algorithm=='Qwen':experiment+=f' / 负例{negatives}'
    if seed is not None:experiment+=f' / seed{seed}'
    settings=metrics.get('settings',run.get('evaluation',{}))
    same=run.get('dataset_sha256')==marker['dataset_sha256']
    if not same:
        manifest=portable_path(run.get('dataset_manifest'),ROOT)
        if manifest:
            same=marker['dataset_sha256'] in manifest_identities(read(manifest))
    expected_classes=dict(Counter(g['label'] for s in expected.values() for g in s['ground_truth']))
    actual_classes={k:v.get('ground_truth_count') for k,v in metrics.get('per_class',{}).items()}
    full_scope=actual_classes==expected_classes
    if indexed:
        complete=indexed['complete'];verification=indexed['verification'];status=indexed['status']
        success=indexed.get('success_images');notes=indexed.get('notes','')
    else:
        target_ids=[t['id'] for t in targets]
        evidence=record_evidence(source,target_ids,{k:v['image_sha256'] for k,v in expected.items()},False)
        complete=(metrics.get('evaluation_status')=='complete' and metrics.get('evaluated_image_count')==300
                  and metrics.get('failed_image_count',0)==0 and run.get('status')=='completed')
        if evidence:
            complete=complete and evidence['success_images']==300 and evidence['manifest_matches'] and not any(
                evidence[k] for k in ['invalid_records','duplicate_records','hash_mismatches'])
        success=evidence['success_images'] if evidence else None
        verification='逐图内容核验' if evidence and evidence['manifest_matches'] else '仅保存的指标与元数据'
        status='完成' if complete else '未完成/待核验'
        notes='' if evidence else '本机无逐图记录'
    if not same:status+='；队列不符'
    if not metrics:status='未产出评估指标'
    elif not full_scope:status+='；类别范围不全'
    if algorithm=='JSTY-YOLO':
        notes+='；监督训练预算与训练/测试重叠未绑定原始训练快照'
    source_metric=source/'metrics.json'
    detail=next((source/n for n in ['evaluation.txt','metrics.json','run.json'] if (source/n).is_file()))
    row=dict(dataset=dataset,algorithm=algorithm,model=model_note,experiment=experiment,
        training_regime=regime,training_shot_per_class=budget,training_image_count=(budget*(5 if dataset.startswith('chache') else 1)) if budget is not None else None,
        inference_shot_per_class=shot,mllm_reference_shot_per_class=mllm_shot,negative_references=negatives,
        savpe=savpe,training_scope=scope,training_protocol=protocol,seed=seed,support_seed=support_seed,
        prompt_version=version,input_variant=variant,input_size=input_size,
        precision=run.get('precision') or ('FP32' if metadata and metadata.get('training',{}).get('amp') is False
                  and metadata.get('training',{}).get('half') is False else None),
        AP50=metrics.get('mAP50'),AP75=metrics.get('mAP75'),AP50_95=metrics.get('mAP50_95'),
        AP_box20=metrics.get('AP_box20',geometry.get(source.resolve())),
        mean_matched_iou=metrics.get('mean_matched_iou'),
        image_accuracy=metrics.get('image_accuracy'),negative_false_positive_rate=metrics.get('negative_false_positive_rate'),
        negative_error_rate=metrics.get('negative_error_rate'),correct_images=metrics.get('correct_image_count'),
        negative_images=metrics.get('negative_image_count'),negative_fp_images=metrics.get('negative_false_positive_image_count'),
        negative_failed_images=metrics.get('negative_failed_image_count'),total_images=300,success_images=success,
        failed_images=metrics.get('failed_image_count'),ground_truth_count=sum(v or 0 for v in actual_classes.values()),
        status=status,complete=complete,same_dataset=same,full_class_scope=full_scope,
        eligible_for_comparison=complete and same and full_scope,verification=verification,
        iou_threshold=settings.get('iou_threshold'),operating_score=.5,max_detections=settings.get('max_detections_per_image'),
        candidate_floor=inference.get('confidence',local_settings.get('confidence',local_settings.get('box_threshold'))),
        nms_iou=inference.get('nms_iou',local_settings.get('nms_iou')),
        execution_device=inference.get('device') or ','.join(run.get('providers',[])),
        elapsed_seconds=run.get('elapsed_seconds'),
        run_dir=relative(source),metric_file=relative(source_metric) if source_metric.exists() else '',
        report=relative(detail),metrics_sha256=sha(source_metric) if source_metric.exists() else None,
        weights_sha256=run.get('weights_sha256'),dataset_sha256=run.get('dataset_sha256'),notes=notes.strip('；'),
        prompt_file=relative((ROOT/'outputs/_shared/datasets'/dataset/indexed['prompt_file']).resolve()) if indexed.get('prompt_file') else '',
        prompt_sha256=indexed.get('prompt_sha256'),training_seed=metadata.get('training',{}).get('seed'),
        cascade_method=cascade.get('method'),cascade_source_run=cascade.get('source_run'),
        per_class=metrics.get('per_class',{}))
    for threshold in [.85,.9,.95]:
        values=[c.get('ap_by_iou',{}).get(str(threshold)) for c in metrics.get('per_class',{}).values()]
        row[f'AP{round(threshold*100)}']=sum(values)/len(values) if values and all(v is not None for v in values) else None
    coco=metrics.get('coco',{}).get('stats',{})
    for field,key in [('COCO_AR100','AR100'),('COCO_AP_small','AP_small'),('COCO_AP_medium','AP_medium'),('COCO_AP_large','AP_large')]:
        value=coco.get(key);row[field]=value if value is not None and value>=0 else None
    for suffix,threshold in [('0_5',.5),('0_7',.7)]:
        point=metrics.get('score_threshold_results',{}).get(str(threshold),{})
        if not point and settings.get('score_threshold')==threshold:point=metrics
        for key in ['precision','recall','f1','f2','tp','fp','fn']:
            row[f'{key}_{suffix}']=point.get(key)
        p,r=point.get('precision'),point.get('recall')
        if p is not None and r is not None:
            if row[f'f1_{suffix}'] is None:row[f'f1_{suffix}']=2*p*r/(p+r) if p+r else 0
            if row[f'f2_{suffix}'] is None:row[f'f2_{suffix}']=5*p*r/(4*p+r) if p+r else 0
    return row


def collect():
    rows=[];seen=set();geometry=geometry_index()
    for dataset in DATASETS:
        base=ROOT/'outputs/_shared/datasets'/dataset
        marker=read(base/'dataset.json');manifest=read(ROOT/f'data/raw/{dataset}/manifest.json')
        expected={s['relative_path']:s for s in manifest['samples']}
        for indexed in read(base/'results_summary.json')['rows']:
            source=(base/indexed['run_dir']).resolve()
            if source in seen:continue
            rows.append(collect_one(source,dataset,marker,expected,geometry,indexed));seen.add(source)
        for metrics in sorted((ROOT/'outputs/YOLOE/finetune'/dataset/'runs').rglob('metrics.json')):
            source=metrics.parent
            run=read(source/'run.json')
            if run.get('image_count')!=300 or run.get('scene_name')!=dataset or run.get('split','test')!='test':
                continue
            if source.resolve() in seen:continue
            rows.append(collect_one(source,dataset,marker,expected,geometry));seen.add(source.resolve())
    rows.sort(key=lambda r:(DATASETS.index(r['dataset']),METHODS.index(r['algorithm']),
        r['training_shot_per_class'] if r['training_shot_per_class'] is not None else -1,
        r['savpe'],str(r['inference_shot_per_class']),r['experiment']))
    for number,row in enumerate(rows,1):row['id']=number
    return rows


def export(rows):
    path=ROOT/'outputs/ALL_RESULTS.md'
    counts=Counter(r['dataset'] for r in rows)
    lines=['# 全部运行明细（两个300张测试集）','',
        f'更新时间：{datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")}（北京时间）。共{len(rows)}项独立运行：'+
        '、'.join(f'{k} {v}项' for k,v in counts.items())+'。','',
        '[CSV完整指标与筛选字段](ALL_RESULTS.csv) · [JSON及逐类指标](ALL_RESULTS.json)','',
        '- 每个实际测试设置一行；同一来源只引用一次。保留全部提示、shot、种子和微调范围，最高值不替代完整结果。',
        '- 训练shot表示本任务的权重微调预算；提示shot表示推理参考数。自定义监督模型的原始训练快照未随权重绑定，训练量标为“监督”，不计作0shot。OVD→Qwen提示列为“YOLOE参考数+Qwen参考数”。',
        '- 率指标均为百分数。AP使用该次运行保留的全部候选；P/R/F1/F2及TP/FP/FN取score≥0.5、IoU≥0.5、每图每类最多100框。缺失值为“—”。',
        '- 图准确率要求整图全部GT匹配且零FP/FN，失败图片计错；负图错误率为（负图误报数+负图失败数）/负图总数。',
        '- 当前测试队列、GT已统一；输入尺寸、预训练来源、候选下限、查询方式及训练预算仍可能不同，完整参数见CSV和原始报告。新JSTY模型为960×1280 FP32 ONNX；YOLOv8换头未训练是随机目标分类输出负对照。',
        '- 旧99张队列及100张dev不混入本表。未完成、类别范围不全和仅元数据完成的记录明确标注；仅元数据完成不代表已核验逐图内容。','',
        '| # | 数据集 | 算法 | 模型 / 实验 | 训练/类 | 提示/类 | SAVPE | AP50 | AP75 | AP50:95 | P@0.5 | R@0.5 | F1@0.5 | F2@0.5 | TP/FP/FN | 图准确率 | 负图错误率 | 状态 | 详情 |',
        '|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|']
    for row in rows:
        hint=row['inference_shot_per_class']
        if row['mllm_reference_shot_per_class'] is not None:hint=f'{hint}+{row["mllm_reference_shot_per_class"]}'
        if hint is None:hint='不适用'
        budget=row['training_shot_per_class'] if row['training_shot_per_class'] is not None else '监督'
        counts='/'.join(str(row[f'{k}_0_5']) if row[f'{k}_0_5'] is not None else '—' for k in ['tp','fp','fn'])
        name=row['model']+' / '+row['experiment']
        values=[row['id'],row['dataset'],row['algorithm'],name,budget,hint,row['savpe'],
                *[pct(row[k]) for k in ['AP50','AP75','AP50_95','precision_0_5','recall_0_5','f1_0_5','f2_0_5']],
                counts,pct(row['image_accuracy']),pct(row['negative_error_rate']),row['status'],
                link('查看',os.path.relpath(ROOT/row['report'],path.parent))]
        lines.append('| '+' | '.join(str(v).replace('|','\\|').replace('\n',' ') for v in values)+' |')
    lines+=['','CSV另含AP85/90/95、AP_box20、COCO AR100、按目标大小划分的AP、0.7工作点、平均匹配IoU、负图误报率、成功/失败数量、逐类指标、输入尺寸、训练范围、种子、来源路径与SHA256。',
        'AP_box20只读取已有定位分析；缺少已存统计时留空。COCO AR100使用IoU=0.50:0.95，与R@0.5含义不同。','',
        '更新总表：`python scripts/summarize_all_algorithms.py`；只读取保存的运行和指标，不调用模型或API。','']
    path.write_text('\n'.join(lines))
    records=[]
    for row in rows:
        record={}
        for key,value in row.items():
            if key=='per_class':continue
            record[key+'_pct' if key in RATES else key]=round(value*100,6) if key in RATES and value is not None else value
        for label in ['Chache1','Chache2','Chache3','Chache4','Qianyinche','hanging object']:
            for key in ['ground_truth_count','AP50','AP75','AP50_95','precision','recall','f1','f2','tp','fp','fn']:
                value=row['per_class'].get(label,{}).get(key)
                rate=key not in ['ground_truth_count','tp','fp','fn']
                record[f'{label}_{key}'+('_pct' if rate else '')]=round(value*100,6) if rate and value is not None else value
        records.append(record)
    with (ROOT/'outputs/ALL_RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        keys=list(dict.fromkeys(k for row in records for k in row))
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)
    (ROOT/'outputs/ALL_RESULTS.json').write_text(json.dumps(dict(schema='industrial_fewshot.all_algorithms.v1',
        datasets=DATASETS,row_count=len(rows),rows=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Master table:',len(rows),'rows;',dict(Counter(r['algorithm'] for r in rows)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-refresh',action='store_true',help='Use the already refreshed per-dataset indexes.')
    args=parser.parse_args()
    if not args.no_refresh:refresh(datasets=DATASETS)
    rows=collect()
    export(rows)
    navigation=ROOT/'outputs/README.md'
    text=navigation.read_text()
    if '](ALL_RESULTS.md)' not in text:
        entry='[两个300张测试集：全部算法、提示与微调设置总表](ALL_RESULTS.md) · [完整CSV指标](ALL_RESULTS.csv)'
        navigation.write_text(text.replace('\n\n','\n\n'+entry+'\n\n',1))
    from build_final_comparison import build
    build(rows)


if __name__=='__main__':main()

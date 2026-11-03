"""Choose confidence on dev predictions, then apply it to saved test predictions (CPU only)."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from industrial_fewshot.evaluation.detection_metrics import evaluate_score_thresholds
from industrial_fewshot.labels import all_label_names
from summarize_yoloe_optimization import read,write,csv_write,optimization_report_dir

GRID=[.001,.002,.005,.01,.02,.03,.05,.07,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]


def samples(directory):
    evidence=read(directory/'run.json')
    manifest_path=Path(evidence['dataset_manifest'])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=evidence['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed since inference.')
    manifest=read(manifest_path)
    truth={row['relative_path']:row for row in manifest['samples']}
    records=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines() if line.strip()]
    expected={(target['id'],path) for target in evidence['targets'] for path in evidence['image_paths']}
    actual={(r['target_id'],r['relative_path']) for r in records}
    if len(records)!=len(expected) or actual!=expected or any(r['status']!='success' for r in records):
        raise ValueError('Incomplete or duplicate saved predictions.')
    predictions=defaultdict(list)
    for r in records:
        if r['image_sha256']!=truth[r['relative_path']]['image_sha256']:
            raise ValueError('Record image hash differs from frozen GT.')
        predictions[r['relative_path']].extend(r['objects'])
    items=[dict(relative_path=path,ground_truth=truth[path]['ground_truth'],predictions=predictions[path])
           for path in evidence['image_paths']]
    labels=all_label_names(evidence['targets'])
    return items,list(dict.fromkeys(labels)),evidence


def operating(directory,thresholds):
    items,labels,evidence=samples(directory)
    result=evaluate_score_thresholds(items,labels,dict(iou_threshold=.5,score_threshold=.5,
        score_thresholds=thresholds,max_detections_per_image=100))
    return result,evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp',default='yoloe_search_20261029_r2')
    parser.add_argument('--select-only',action='store_true')
    args=parser.parse_args()
    report=optimization_report_dir(args.stamp, project_root=ROOT)
    from summarize_finetuning_best import is_compact
    if is_compact(report):
        print('This completed report uses the compact layout; historical threshold analyses are in the archive recorded in best_evidence.json.')
        return
    runtime=ROOT/'outputs/_shared/runtime'/args.stamp
    state=read(runtime/'state.json')
    scene=read(report/'plan.json').get('scene','chache')
    selection_path=report/'thresholds/dev_selection.json'
    if not selection_path.exists():
        selections=[]
        for family in ['yoloe-head','yoloe-vp']:
            candidates=[]
            for entry in state['jobs'].values():
                job=entry['job']
                if job['method']!=family or '_seed' in job['id'] or entry['status']!='completed' or job.get('role')=='baseline':continue
                directory=Path(entry.get('dev_output',str(Path(job['run_dir'])/'dev_val100')))
                path=next(directory.glob('*/metrics.json'))
                candidates.append((read(path)['mAP50_95'],job['id'],path.parent))
            _,job_id,directory=max(candidates)
            selections.append((job_id,directory))
        if scene=='chache':selections.append(('yolov8_val100_v1',runtime/'yolov8_dev100/fixed_head'))
        else:selections.append(('baseline_yolov8',ROOT/f'outputs/YOLOE/finetune/{scene}_val300/runs'/args.stamp/'baseline_yolov8/dev_val100/fixed_head'))
        frozen=[]
        for job_id,directory in selections:
            values,evidence=operating(directory,GRID)
            if evidence['split']!='dev':raise ValueError('Threshold selection must use dev only.')
            best=max(GRID,key=lambda t:(values[f'{t:g}']['f2'],values[f'{t:g}']['precision'],t))
            item=dict(id=job_id,threshold=best,criterion='dev micro F2; ties: precision, higher threshold',
                dev_directory=str(directory),dev_record_sha256=hashlib.sha256((directory/'records.jsonl').read_bytes()).hexdigest(),
                dev_dataset_sha256=evidence['dataset_sha256'],dev_at_selected=values[f'{best:g}'])
            frozen.append(item);write(report/'thresholds'/f'{job_id}_dev_curve.json',values)
        write(selection_path,dict(grid=GRID,selections=frozen,created=time.time(),
            policy='One global threshold per seed-0 dev winner, then fixed for its seed repeats and VP 5/10shot. No test selection. AP unchanged.'))
    if args.select_only:
        print(selection_path);return
    frozen=read(selection_path)['selections'];rows=[]
    base=ROOT/f'outputs/YOLOE/finetune/{scene}_val300/runs'
    for selected in frozen:
        job_id=selected['id'];threshold=selected['threshold']
        directories=[(job_id,base/job_id/'test_val300')] if job_id=='yolov8_val100_v1' else [
            (name,Path(entry['job']['run_dir'])/'test_val300') for name,entry in state['jobs'].items()
            if name.split('_seed')[0]==job_id]
        for name,directory in directories:
            for path in sorted(directory.glob('*/records.jsonl')):
                values,evidence=operating(path.parent,[threshold]);op=values[f'{threshold:g}'];default=values['0.5']
                split=evidence.get('split','test' if evidence.get('scene_name')=='chache_val300' and evidence['image_count']==300 else None)
                if split!='test':raise ValueError('Expected test for application.')
                row=dict(id=name,prompt_shots=evidence['prompt_shot_per_class'],threshold=threshold,
                    **{key:op[key] for key in ['precision','recall','f2','tp','fp','fn']},
                    **{'default_'+key:default[key] for key in ['precision','recall','f2','tp','fp','fn']})
                rows.append(row);write(report/'thresholds'/name/path.parent.name/'operating.json',
                    dict(selected_on_dev=selected,test_at_frozen_threshold=op,test_at_default=default))
    csv_write(report/'thresholds/test_operating_points.csv',rows)
    lines=['# 只在dev选择置信度阈值','',
        '在seed=0的统一dev预测上，以micro F2选择全局阈值；并列时优先P、再选更高阈值。',
        '冻结后应用于同一配置的seed=0/1/2测试。视觉提示以10shot dev选阈值，5shot沿用；没有为5shot查看测试结果挑阈值。',
        '这只改变工作点P/R/F2，AP保持原总表数值；F2强调召回，不等于业务最优阈值。','',
        '| 实验 | 提示shot | dev所选阈值 | test P | test R | test F2 | 默认0.5 F2 | TP/FP/FN |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f'| {row["id"]} | {row["prompt_shots"] or "不适用"} | {row["threshold"]} | '
            +' | '.join(f'{100*row[k]:.2f}' for k in ['precision','recall','f2','default_f2'])
            +f' | {row["tp"]}/{row["fp"]}/{row["fn"]} |')
    lines+=['','[冻结阈值和dev输入指纹](dev_selection.json)；[CSV](test_operating_points.csv)。',
        '这里只重评已有预测，未运行模型、未修改原metrics.json，也没有调整NMS。','']
    (report/'thresholds/README.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Threshold test settings:',len(rows))


if __name__=='__main__':main()

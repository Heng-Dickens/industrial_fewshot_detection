"""Fill step two of the Qwen report with nested 20shot and missing 1shot runs.

Uses the resumable 27B worker/controller from run_qwen27b_input_supplement,
with separate configuration, reference preparation, runtime state and reports.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import run_qwen27b_input_supplement as batch
from industrial_fewshot.cache import digest
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json,write_json
from industrial_fewshot.data.prepare import prepare_dataset
from industrial_fewshot.experiments import run_ablation
from industrial_fewshot.paths import resolve_location

ROOT=batch.ROOT
RUNTIME=ROOT/'outputs/_shared/runtime/qwen27b_shot_supplement_20261103'
REPORT=ROOT/'outputs/Qwen/comparisons/shot_supplement_20261103'
EXT=ROOT/'data/references/qwen_shot_extension_20261103'
CONFIGS={scene:ROOT/f'configs/specific/{scene}_qwen27b_shot_supplement.yaml'
         for scene in ('chache','diaowu')}
CLASSES={'chache':['Chache1','Chache2','Chache3','Chache4','Qianyinche'],
         'diaowu':['hanging object']}


def checked_copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if batch.sha(source)!=batch.sha(destination):raise ValueError(f'Existing asset differs: {destination}')
    else:shutil.copy2(source,destination)


def verify_frozen(path):
    frozen=read_json(path)
    for row in frozen['positive_samples']:
        for key,hash_key in [('image_path','image_sha256'),('json_path','annotation_sha256')]:
            if batch.sha(resolve_location(Path(row[key])))!=row[hash_key]:
                raise ValueError(f'Frozen reference content changed: {row[key]}')
    return frozen


def extend_reference(scene,label):
    canonical=ROOT/'data/references'/scene
    target=EXT/scene
    if scene=='chache':canonical/=label;target/=label
    old_path=canonical/'10shot/seed_0.json'
    old=verify_frozen(old_path)
    final=target/'20shot/seed_0.json'
    if final.exists():
        saved=verify_frozen(final)
        if ([r['image_sha256'] for r in saved['positive_samples'][:10]]
                !=[r['image_sha256'] for r in old['positive_samples']]):
            raise ValueError('20shot lost the existing 10shot prefix.')
        return saved['selection_provenance']
    original=read_json(canonical/'manifest.json')
    by_id={r['image']:r for r in original['images'] if r['split']=='positive'}
    selected=[copy.deepcopy(by_id[r['sample_id']]) for r in old['positive_samples']]
    used={r['sample_id'] for r in old['positive_samples']}
    existing_hashes={r['image_sha256'] for r in old['positive_samples']}
    test=read_json(ROOT/f'data/raw/{scene}_val300/manifest.json')
    test_hashes={s['image_sha256'] for s in test['samples']}
    added=[]
    if len(by_id)>=20:
        remaining=sorted((r for sid,r in by_id.items() if sid not in used),
            key=lambda r:(digest({'seed':0,'sample_id':r['image']}),r['image']))
        selected.extend(copy.deepcopy(remaining[:10]))
        origin='original_reference_pool'
        added=[dict(sample_id=r['image'],image_sha256=batch.sha(canonical/r['image'])) for r in remaining[:10]]
    else:
        if len(by_id)!=10:raise ValueError('The authorized fallback expects an existing 10-image reference pool.')
        val_root=ROOT/f'data/raw/{scene}_val100/images'
        val_manifest=ROOT/f'data/raw/{scene}_val100/manifest.json'
        val=read_json(val_manifest)
        candidates=[]
        for sample in sorted(val['samples'],key=lambda r:r['relative_path']):
            truth=sample['ground_truth']
            if len(truth)!=1 or truth[0]['label']!=label:continue
            image=val_root/sample['relative_path']
            if batch.sha(image)!=sample['image_sha256']:raise ValueError('Validation image differs from its frozen manifest.')
            if sample['image_sha256'] not in test_hashes|existing_hashes:candidates.append(sample)
        if len(candidates)<10:raise ValueError(f'Fewer than 10 disjoint single-instance val100 samples: {label}')
        chosen=random.Random(0).sample(candidates,10)
        extra_root=target/'extra_val100'
        prepared=(read_json(extra_root/'manifest.json') if (extra_root/'manifest.json').exists() else
            prepare_dataset(val_root,extra_root,selected_paths=[val_root/s['relative_path'] for s in chosen]))
        by_source={r['source_image']:r for r in prepared['images']}
        for sample in chosen:
            row=copy.deepcopy(by_source[sample['relative_path']])
            for key in ('image','annotation','boxed_image'):row[key]='extra_val100/'+row[key]
            for crop in row['crops']:crop['image']='extra_val100/'+crop['image']
            selected.append(row)
            added.append(dict(sample_id=row['image'],source_image=str(val_root/sample['relative_path']),
                source_annotation=str((val_root/sample['relative_path']).with_suffix('.json')),
                image_sha256=sample['image_sha256'],annotation_sha256=sample['annotation_sha256']))
        origin='independent_val100_seed0'
    # Original support assets are copied byte-for-byte, preserving their renderings and annotations.
    for row in selected:
        if row['image'].startswith('extra_val100/'):continue
        for relative in [row['image'],row['annotation'],row['boxed_image'],*[c['image'] for c in row['crops']]]:
            checked_copy(canonical/relative,target/relative)
    selected_hashes=[batch.sha(target/r['image']) for r in selected]
    if len(selected)!=20 or len(set(selected_hashes))!=20 or set(selected_hashes)&test_hashes:
        raise ValueError('20shot must contain 20 unique reference images disjoint from the test cohort.')
    manifest=copy.deepcopy(original)
    manifest.update(output_dir=str(target),images=selected,image_count=20,positive_image_count=20,
        negative_image_count=0,crop_count=sum(len(r['crops']) for r in selected))
    write_json(target/'manifest.json',manifest)
    checked_copy(canonical/'classes.json',target/'classes.json')
    provenance=dict(scene=scene,label=label,seed=0,original_10shot=str(old_path),
        original_10shot_sha256=batch.sha(old_path),prefix_image_sha256=selected_hashes[:10],
        added_reference_source=origin,additional_count=10,added_samples=added,
        excluded_test_manifest_sha256=batch.sha(ROOT/f'data/raw/{scene}_val300/manifest.json'),
        test_overlap_count=0)
    if origin.startswith('independent_val100'):
        provenance['val_manifest_sha256']=batch.sha(ROOT/f'data/raw/{scene}_val100/manifest.json')
        provenance['val_usage']='Adaptation references; val100 is not an independent evaluation cohort for this 20shot condition.'
    def entry(row):
        return dict(sample_id=row['image'],image_path=str(target/row['image']),
            image_sha256=batch.sha(target/row['image']),json_path=str(target/row['annotation']),
            annotation_sha256=batch.sha(target/row['annotation']),boxed_image=str(target/row['boxed_image']),
            crops=[str(target/c['image']) for c in row['crops']])
    write_json(final,dict(schema='industrial_fewshot.support.v1',positive_count=20,selection_seed=0,
        reference_root=str(target),positive_samples=[entry(r) for r in selected],negative_samples=[],
        selection_policy='explicit',selection_provenance=provenance,
        query_dataset_sha256=test['dataset_sha256']))
    return provenance


def configuration(job):
    cfg=load_config(CONFIGS[job['scene']])
    cfg['experiment_suite'].update(positive_counts=[job['shot']],positive_variants=[job['variant']],
        negative_variants=[False],include_prompt_only=False)
    cfg['resolved_config']['experiment_suite']=copy.deepcopy(cfg['experiment_suite'])
    if job['shot']==20:
        root=EXT/job['scene']
        cfg['support_manifest_dir']=root
        cfg['reference_image_dir']=root
        cfg['resolved_config'].update(reference_image_dir=str(root),support_manifest_dir=str(root))
        for target in cfg['targets']:
            target['config'].update(reference_image_dir=str(root),support_manifest_dir=str(root))
    return cfg


async def prepare():
    if (RUNTIME/'plan.json').exists():raise FileExistsError('Plan exists; use --probe/--background/--status.')
    document=read_json(ROOT/'outputs/FINAL_COMPARISON.json')
    rows={r['run_dir']:r for r in document['primary_rows']}
    required=[]
    for stage in document['qwen_analysis']['stages']:
        if stage['stage']!='shots':continue
        for condition in stage['conditions']:
            if condition['run_dir'] and rows[condition['run_dir']]['eligible_for_comparison']:continue
            if condition['inference_shot_per_class'] not in (1,20):raise ValueError('Unexpected missing shot condition.')
            scene=condition['dataset'].removesuffix('_val300')
            if scene=='chache' and condition['inference_shot_per_class']==20:continue
            required.append(dict(scene=scene,shot=condition['inference_shot_per_class'],variant=condition['input_variant'],
                id=f'{scene}_{condition["inference_shot_per_class"]}shot_{condition["input_variant"].replace("+", "")}'))
    provenance=[]
    for scene in {j['scene'] for j in required if j['shot']==20}:
        for label in CLASSES[scene]:provenance.append(extend_reference(scene,label))
    if not required:raise ValueError('No missing step-two conditions.')
    write_json(EXT/'selection_provenance.json',dict(seed=0,selections=provenance,
        cancelled_conditions=[dict(scene='chache',shot=20,reason='Cancelled by user; only chache 1shot is requested.')]))
    for job in required:
        result=await run_ablation(configuration(job),dry_run=True)
        group=result['groups'][0];target=group['targets'][0]
        reference=next(r for r in rows.values() if r['dataset']==job['scene']+'_val300'
            and r['model']==batch.MODEL and r['input_variant']==job['variant']
            and r['inference_shot_per_class']==10 and r['negative_references']==0 and r['eligible_for_comparison'])
        if (result['group_count']!=1 or result['image_count']!=300 or result['excluded_reference_overlap']
                or group['dataset_sha256']!=reference['dataset_sha256']
                or target['model']!=batch.MODEL or target.get('labels',[target['label']])!=CLASSES[job['scene']]
                or target['prompt_snapshot']['text_sha256']!=reference['prompt_sha256']):
            raise ValueError('Shot supplementation changed the cohort, prompt, class scope or model.')
        refs=target['reference_count']
        expected=job['shot']*len(CLASSES[job['scene']])
        if refs['reference_originals']!=expected or refs['reference_negatives']:
            raise ValueError('Wrong support count.')
        write_json(batch.state_path(job),dict(job,status='prepared',config_sha256=batch.sha(CONFIGS[job['scene']]),
            dataset_sha256=group['dataset_sha256'],prompt_sha256=target['prompt_sha256'],
            request_signature=target['request_signature'],reference_count=refs,
            experiment_suite=result['experiment_suite'],comparison_dir=result['comparison_dir'],
            reference_encoding=target.get('reference_encoding',{})))
        print('Prepared:',job['id'],refs,flush=True)
    write_json(RUNTIME/'plan.json',dict(model=batch.MODEL,image_tasks=300*len(required),jobs=required,
        reference_provenance=str(EXT/'selection_provenance.json')))
    render()


async def probe():
    batch.isolate_loopback()
    for scene in CONFIGS:
        scene_jobs=[j for j in batch.jobs() if j['scene']==scene]
        if not scene_jobs:continue
        probe_file=RUNTIME/f'probe_{scene}.json'
        if probe_file.exists() and read_json(probe_file).get('status')=='passed' and scene=='diaowu':
            print('Retain passed image probes:',scene,flush=True)
            continue
        for job in scene_jobs:
            cfg=configuration(job)
            cfg.update(results_layout=False,output_dir=RUNTIME/'probes'/job['id'])
            cfg['resolved_config'].update(results_layout=False,output_dir=str(cfg['output_dir']))
            result=await run_ablation(cfg,limit=1)
            if result['failed_request_count'] or result['status']!='complete':
                raise RuntimeError(f'{job["id"]}: image probe failed; inspect saved feedback.')
            print('Image probe passed:',job['id'],flush=True)
        write_json(RUNTIME/f'probe_{scene}.json',dict(status='passed',jobs=[j['id'] for j in scene_jobs]))


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    lines=['# Qwen3.8-27B第二步shot补齐','',
        '本批仅新跑吊物O+B+J的1shot与20shot，各评估原300张测试集，共600张次。',
        '叉车20shot按用户要求取消；叉车O+C+B的1shot已有完整300张结果，直接引用。',
        '吊物20shot保留原10shot为前缀，另从原25张正参考中补选10张，seed0；无需从val补图。',
        '参考与测试图像SHA无重叠。全部使用原有JPEG质量90、1280最长边和独立图片输入方式。',
        '', '[参考来源与抽样清单](../../../../data/references/qwen_shot_extension_20261103/selection_provenance.json)',
        '', '| 场景 | shot/类 | 输入 | 状态 | 成功/总图 | AP50 | AP50:95 | 结果 |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | --- |']
    total=0
    for job in batch.jobs():
        state=read_json(batch.state_path(job));count=batch.successful(state);total+=count
        status=state['status']
        if status=='running' and not batch.live(state.get('pid')):status='interrupted_process_exited'
        metric=state.get('metrics',{})
        result=Path(state['run_dir'])/'evaluation.txt' if state.get('run_dir') else None
        cells=[job['scene'],str(job['shot']),job['variant'],status,f'{count}/300',
            *[f'{100*metric[k]:.2f}' if k in metric else '—' for k in ('mAP50','mAP50_95')],
            f'[查看]({os.path.relpath(result,REPORT)})' if result and result.exists() else '—']
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['',f'累计成功：{total}/{300*len(batch.jobs())}张次。',
        '', '[最终Qwen表：第二步](../../../_shared/comparisons/final_results/01_qwen_inputs.md#step-2)',
        '每组完成后自动刷新最终表、CSV与全局汇总；不完整结果不参与最优值。',
        '', '命令：`python scripts/run_qwen27b_shot_supplement.py --prepare` → `--probe` → `--background`。',
        '刷新进度：`python scripts/run_qwen27b_shot_supplement.py --status`。','']
    plan=read_json(RUNTIME/'plan.json')
    for reused in plan.get('reused_conditions',[]):
        path=ROOT/reused['run_dir']/'evaluation.txt'
        lines+=['叉车1shot已入表：AP50='+f'{100*reused["AP50"]:.2f}%，AP50:95={100*reused["AP50_95"]:.2f}%。'+
                f'[已有原始结果]({os.path.relpath(path,REPORT)})。','']
    (REPORT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


batch.SCRIPT=Path(__file__).resolve()
batch.RUNTIME=RUNTIME
batch.REPORT=REPORT
batch.CONFIGS=CONFIGS
batch.configuration=configuration
batch.prepare=prepare
batch.probe=probe
batch.render=render

if __name__=='__main__':batch.main()

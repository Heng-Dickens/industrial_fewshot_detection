"""Queue matched Flash/Max 10shot evaluations after the current 27B shot batch.

Preparation is offline. Model/image probes and full evaluation start only after
the dependency controller and all of its jobs report successful completion.
Uses the existing resumable engine, per-model response checks and frozen data.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import run_qwen27b_input_supplement as batch
from industrial_fewshot.config import load_config
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.experiments import run_ablation

ROOT=batch.ROOT
RUNTIME=ROOT/'outputs/_shared/runtime/qwen_model10_supplement_20261103'
REPORT=ROOT/'outputs/Qwen/comparisons/model10_supplement_20261103'
DEPENDENCY=ROOT/'outputs/_shared/runtime/qwen27b_shot_supplement_20261103'
CONFIGS={scene:ROOT/f'configs/specific/{scene}_qwen_model10_supplement.yaml'
         for scene in ('chache','diaowu')}
VARIANTS={'chache':'O+C+B','diaowu':'O+B+J'}
MODELS=('qwen3.8-flash','qwen3.8-max')
original_matching=batch.matching_comparison
original_verify=batch.verify
original_launch=batch.launch
original_background=batch.background


def configuration(job):
    cfg=load_config(CONFIGS[job['scene']])
    model=job['model']
    cfg['resolved_config']['model']=model
    # Aliases select existing proxy groups; the required response model remains
    # the real Qwen model. No service routing or provider settings are changed.
    alias='claude-sonnet-4-6' if model=='qwen3.8-flash' else model
    cfg['resolved_config']['proxy_request_model']=alias
    for target in cfg['targets']:
        target.update(model=model,proxy_request_model=alias)
        target['config'].update(model=model,proxy_request_model=alias)
        target['model_config'].update(name=model,proxy_request_model=alias)
    return cfg


def matching_comparison(state):
    batch.MODEL=state['model']
    return original_matching(state)


def verify(group,state):
    batch.MODEL=state['model']
    return original_verify(group,state)


async def prepare():
    if (RUNTIME/'plan.json').exists():raise FileExistsError('Plan exists; use --background/--status.')
    document=read_json(ROOT/'outputs/FINAL_COMPARISON.json')
    rows=document['primary_rows']
    planned=[]
    reused=[]
    for scene,variant in VARIANTS.items():
        reference=next(r for r in rows if r['dataset']==scene+'_val300'
            and r['model']=='qwen3.8-27b' and r['input_variant']==variant
            and r['inference_shot_per_class']==10 and r['negative_references']==0
            and r['support_seed']==0 and r['eligible_for_comparison']
            and r['prompt_version']==('MLLM提示_v2' if scene=='chache' else 'MLLM提示_v1'))
        reused.append(dict(scene=scene,model='qwen3.8-27b',run_dir=reference['run_dir']))
        for model in MODELS:
            existing=[r for r in rows if r['dataset']==reference['dataset'] and r['model']==model
                and r['input_variant']==variant and r['inference_shot_per_class']==10
                and r['negative_references']==0 and r['support_seed']==0
                and r['prompt_version']==reference['prompt_version'] and r['eligible_for_comparison']]
            if existing:
                if len(existing)!=1:raise ValueError('Ambiguous existing comparison.')
                reused.append(dict(scene=scene,model=model,run_dir=existing[0]['run_dir']))
                continue
            job=dict(scene=scene,model=model,shot=10,variant=variant,
                id=f'{scene}_{model}_10shot_{variant.replace("+", "")}')
            cfg=configuration(job)
            result=await run_ablation(cfg,dry_run=True)
            group=result['groups'][0];target=group['targets'][0]
            if (result['group_count']!=1 or result['image_count']!=300 or result['excluded_reference_overlap']
                    or target['model']!=model or group['dataset_sha256']!=reference['dataset_sha256']
                    or target['prompt_snapshot']['text_sha256']!=reference['prompt_sha256']
                    or target.get('labels',[target['label']])!=list(batch.GT[scene])):
                raise ValueError('Model, prompt, cohort or class scope changed.')
            supports=read_json(ROOT/reference['run_dir']/'run.json')['support_manifests']
            if {s['sha256'] for s in group['support_manifests']}!={s['sha256'] for s in supports}:
                raise ValueError('Support references differ from the 27B baseline.')
            refs=target['reference_count']
            if refs['reference_originals']!=10*len(batch.GT[scene]) or refs['reference_negatives']:
                raise ValueError('Wrong support budget.')
            write_json(batch.state_path(job),dict(job,status='queued',config_sha256=batch.sha(CONFIGS[scene]),
                dataset_sha256=group['dataset_sha256'],prompt_sha256=target['prompt_sha256'],
                request_signature=target['request_signature'],reference_count=refs,
                experiment_suite=result['experiment_suite'],comparison_dir=result['comparison_dir'],
                baseline_run=reference['run_dir'],support_manifests=group['support_manifests']))
            planned.append(job)
            print('Prepared offline:',job['id'],flush=True)
    if not planned:raise ValueError('All matched 10shot model conditions already exist.')
    write_json(RUNTIME/'plan.json',dict(jobs=planned,image_tasks=300*len(planned),
        dependency=str(DEPENDENCY),reused_conditions=reused,
        start_policy='After dependency controller and every planned job successfully complete; otherwise stop.'))
    render()


def dependency_status():
    state=read_json(DEPENDENCY/'controller.json')
    if state['status']=='completed':
        for job in read_json(DEPENDENCY/'plan.json')['jobs']:
            if read_json(DEPENDENCY/'jobs'/(job['id']+'.json'))['status']!='completed':
                raise RuntimeError('Dependency controller/job status mismatch; no new requests started.')
        return True
    if state['status']!='running':
        raise RuntimeError('Preceding batch stopped or failed; no new requests started.')
    command=Path(f'/proc/{state.get("pid",0)}/cmdline')
    if not command.exists() or 'run_qwen27b_shot_supplement.py' not in command.read_text():
        raise RuntimeError('Preceding controller is not alive; no new requests started.')
    return False


async def probe():
    if not dependency_status():raise RuntimeError('Preceding batch is still running; probes are deferred.')
    batch.isolate_loopback()
    for scene in CONFIGS:
        scene_jobs=[j for j in batch.jobs() if j['scene']==scene]
        if not scene_jobs:continue
        for job in scene_jobs:
            cfg=configuration(job)
            cfg.update(results_layout=False,output_dir=RUNTIME/'probes'/job['id'])
            cfg['resolved_config'].update(results_layout=False,output_dir=str(cfg['output_dir']))
            result=await run_ablation(cfg,limit=1)
            if result['status']!='complete' or result['failed_request_count']:
                raise RuntimeError(f'{job["id"]}: model/image probe failed; inspect saved feedback before retrying.')
            write_json(RUNTIME/('probe_'+job['id']+'.json'),dict(status='passed',model=job['model'],
                run_dir=result['groups'][0]['run_dir']))
            print('Model/image probe passed:',job['id'],flush=True)
        write_json(RUNTIME/f'probe_{scene}.json',dict(status='passed',jobs=[j['id'] for j in scene_jobs]))


def launch():
    previous=read_json(RUNTIME/'controller.json') if (RUNTIME/'controller.json').exists() else {}
    if previous.get('status') in {'running','waiting_for_previous','probing'} and batch.live(previous.get('pid')):
        raise RuntimeError('Controller already running.')
    try:
        write_json(RUNTIME/'controller.json',dict(status='waiting_for_previous',pid=os.getpid(),started=time.time()))
        while not dependency_status():
            render()
            time.sleep(5)
        write_json(RUNTIME/'controller.json',dict(status='probing',pid=os.getpid(),started=time.time()))
        render()
        batch.run_async(probe())
    except BaseException as exc:
        write_json(RUNTIME/'controller.json',dict(status='blocked_or_failed',pid=os.getpid(),error=str(exc)))
        render()
        raise
    original_launch()


def background():
    previous=read_json(RUNTIME/'controller.json') if (RUNTIME/'controller.json').exists() else {}
    if previous.get('status') in {'running','waiting_for_previous','probing'} and batch.live(previous.get('pid')):
        raise RuntimeError('Controller already running or queued.')
    original_background()


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    controller=read_json(RUNTIME/'controller.json') if (RUNTIME/'controller.json').exists() else {}
    lines=['# 第三步：10shot模型对比补齐','',
        '仅比较10shot/类：叉车O+C+B（v2），吊物O+B+J（v1）；27B已有结果复用，补Flash与Max各两个场景。',
        '共4组、1200张次；无负例、seed0，沿用27B相同参考图SHA、提示、生成与评估参数。',
        '等待当前27B吊物1shot和20shot全部完成并汇总后，再做真实模型/图片探测及正式推理。',
        'Flash沿用Sonnet路由，Max沿用原qwen3.8-max请求名；响应必须核验为相应Qwen模型，路由不符即停止，不改代理全局设置。',
        '叉车两模型依次运行，并发2；吊物两模型并行，各并发2。每组完成自动更新子表、CSV和全局汇总。',
        '',f'控制器状态：`{controller.get("status","prepared_offline")}`。',
        '', '[前置实验进度](../shot_supplement_20261103/README.md)',
        '', '| 场景 | 模型 | shot/类 | 输入 | 状态 | 成功/总图 | AP50 | AP50:95 | 结果 |',
        '| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |']
    total=0
    for job in batch.jobs():
        state=read_json(batch.state_path(job));count=batch.successful(state);total+=count
        status=state['status']
        if status=='running' and not batch.live(state.get('pid')):status='interrupted_process_exited'
        metrics=state.get('metrics',{})
        path=Path(state['run_dir'])/'evaluation.txt' if state.get('run_dir') else None
        cells=[job['scene'],job['model'],'10',job['variant'],status,f'{count}/300',
            *[f'{100*metrics[k]:.2f}' if k in metrics else '—' for k in ('mAP50','mAP50_95')],
            f'[查看]({os.path.relpath(path,REPORT)})' if path and path.exists() else '—']
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['',f'累计成功：{total}/{300*len(batch.jobs())}张次。',
        '', '[最终Qwen表：第三步](../../../_shared/comparisons/final_results/01_qwen_inputs.md#step-3)',
        '', '命令：`python scripts/run_qwen_model10_supplement.py --prepare`（仅离线准备）→ `--background`（等待后自动启动）。',
        '进度：`python scripts/run_qwen_model10_supplement.py --status`。','']
    if controller.get('error'):lines+=['停止原因：'+controller['error'],'']
    for row in read_json(RUNTIME/'plan.json').get('reused_conditions',[]):
        path=ROOT/row['run_dir']/'evaluation.txt'
        lines.append(f'- 复用{row["scene"]} {row["model"]}：[原始评估]({os.path.relpath(path,REPORT)})')
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


batch.SCRIPT=Path(__file__).resolve()
batch.RUNTIME=RUNTIME
batch.REPORT=REPORT
batch.CONFIGS=CONFIGS
batch.configuration=configuration
batch.matching_comparison=matching_comparison
batch.verify=verify
batch.prepare=prepare
batch.probe=probe
batch.render=render
batch.launch=launch
batch.background=background

if __name__=='__main__':batch.main()

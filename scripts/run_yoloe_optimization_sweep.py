"""Auditable independent-GPU YOLOE optimization; all selection uses the frozen dev set."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from industrial_fewshot.comparison_layout import optimization_report_dir
from industrial_fewshot.finetuning_methods import method_policy
PYTHON = Path(sys.executable)
SCENE = 'chache'
STAMP = 'yoloe_search_20261029_r2'
REPORT = optimization_report_dir(STAMP, project_root=ROOT, scene=SCENE)
RUNTIME = ROOT / 'outputs/_shared/runtime' / STAMP
PROJECT = ROOT / 'outputs/YOLOE/finetune/chache_val300/runs' / STAMP
LOCK = threading.Lock()
STATE = {}


def configure(scene, stamp=None):
    global SCENE, STAMP, REPORT, RUNTIME, PROJECT
    SCENE = scene
    STAMP = stamp or ('yoloe_search_20261029_r2' if scene=='chache' else 'yoloe_diaowu_search_20261029')
    if Path(STAMP).name!=STAMP or STAMP in {'.','..',''}:
        raise ValueError('Stamp must be a simple directory name.')
    REPORT=optimization_report_dir(STAMP, project_root=ROOT, scene=SCENE)
    RUNTIME=ROOT/'outputs/_shared/runtime'/STAMP
    PROJECT=ROOT/'outputs/YOLOE/finetune'/f'{SCENE}_val300'/'runs'/STAMP


def task_entry(name):
    return ROOT/f'outputs/YOLOE/finetune/scripts/{SCENE}'/name


def baseline_specs():
    return [dict(id=f'baseline_{method.replace("yoloe-", "").replace("yolov8", "yolov8")}',
        method=method,strategy='head',batch=8,settings=dict(lr0=1e-4,epochs=100,patience=10),
        adaptation={},role='baseline') for method in ['yoloe-head','yoloe-vp','yolov8']]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(p)


def specs():
    jobs = []
    def add(name, method, lr, strategy='head', batch=8, epochs=80, adaptation=None, **settings):
        jobs.append(dict(id=name, method=method, strategy=strategy, batch=batch,
            settings=dict(lr0=lr, epochs=epochs, **settings), adaptation=adaptation or {}))
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        add(f'head_linear_lr{lr:g}', 'yoloe-head', lr, epochs=120)
    for scope, rates in [('neck', [3e-5, 1e-4, 3e-4]), ('full', [1e-5, 3e-5, 1e-4, 3e-4])]:
        for lr in rates:
            add(f'head_{scope}_lr{lr:g}', 'yoloe-head', lr, strategy=scope)
    for wd in [0., 1e-4, 5e-3]:
        add(f'head_full_wd{wd:g}', 'yoloe-head', 1e-4, strategy='full', weight_decay=wd)
    for lr in [1e-3, 5e-3]:
        add(f'head_full_sgd_lr{lr:g}', 'yoloe-head', lr, strategy='full', optimizer='SGD', momentum=.9)
    add('head_full_cosine', 'yoloe-head', 1e-4, strategy='full', cos_lr=True)
    add('head_full_aug', 'yoloe-head', 1e-4, strategy='full', mosaic=.2, close_mosaic=10, fliplr=.5, translate=.1, scale=.25)
    add('head_full_960', 'yoloe-head', 1e-4, strategy='full', batch=4, imgsz=960)
    add('head_full_1024', 'yoloe-head', 1e-4, strategy='full', batch=2, imgsz=1024)
    add('head_full_batch4', 'yoloe-head', 1e-4, strategy='full', batch=4)
    add('head_full_short_text', 'yoloe-head', 1e-4, strategy='full')
    for bn in [False, True]:
        for lr in [1e-6, 1e-5, 5e-5, 1e-4, 3e-4, 1e-3]:
            add(f'vp_savpe_lr{lr:g}_bn{int(bn)}', 'yoloe-vp', lr, epochs=40,
                adaptation=dict(freeze_savpe_bn_stats=bn))
    for scope in ['head', 'neck', 'full']:
        for lr in [1e-5, 1e-4]:
            add(f'vp_{scope}_lr{lr:g}', 'yoloe-vp', lr, batch=4, epochs=60,
                adaptation=dict(scope=scope, loss='detection', freeze_savpe_bn_stats=True))
    for bn in [False, True]:
        for lr in [1e-5, 1e-4]:
            add(f'vp_cross_lr{lr:g}_bn{int(bn)}', 'yoloe-vp', lr, batch=4, epochs=40,
                adaptation=dict(protocol='cross_image', freeze_savpe_bn_stats=bn))
    add('vp_cross_neck', 'yoloe-vp', 1e-5, batch=2, epochs=60,
        adaptation=dict(protocol='cross_image', scope='neck', loss='detection', freeze_savpe_bn_stats=True))
    add('vp_savpe_960', 'yoloe-vp', 1e-5, batch=4, epochs=40, imgsz=960,
        adaptation=dict(freeze_savpe_bn_stats=True))
    add('vp_no_color', 'yoloe-vp', 1e-5, epochs=40, hsv_h=0., hsv_s=0., hsv_v=0.,
        adaptation=dict(freeze_savpe_bn_stats=True))
    add('vp_no_aug', 'yoloe-vp', 1e-5, epochs=40, hsv_h=0., hsv_s=0., hsv_v=0., translate=0., scale=0.,
        adaptation=dict(freeze_savpe_bn_stats=True))
    return jobs


def prepare(jobs):
    base = read(REPORT/'base_config.json') if (REPORT/'base_config.json').exists() else read(task_entry('config.json'))
    for job in jobs:
        cfg = json.loads(json.dumps(base))
        cfg['project'] = str(PROJECT)
        cfg['adaptation'] = job['adaptation']
        cfg['training'].update(dict(workers=0, val_interval=1, patience=12, plots=False, save_period=-1))
        cfg['training'].update(job['settings'])
        if job['id'].split('_seed')[0] == 'head_full_short_text':
            cfg['text_prompts'] = (['forklift with forks', 'low pallet truck', 'stand-on stacker',
                                   'forklift with a wide yellow attachment', 'industrial tow tractor']
                                   if SCENE=='chache' else ['a hanging cylindrical load with its hook and taut slings'])
        job['config'] = str(RUNTIME/'configs'/f"{job['id']}.json")
        job['run_dir'] = str(PROJECT/job['id'])
        write(job['config'], cfg)
        write(REPORT/'configs'/f"{job['id']}.json",cfg)
    return jobs


def run_command(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        log = log.with_name(f'{log.stem}_continued_{time.time_ns()}{log.suffix}')
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
        YOLO_AUTOINSTALL='false', PYTHONUNBUFFERED='1', WANDB_MODE='disabled')
    command = [str(v) for v in command]
    if '--device' in command:
        index = command.index('--device') + 1
        env['CUDA_VISIBLE_DEVICES'] = command[index]
        command[index] = '0'
    env['YOLO_CONFIG_DIR'] = str(RUNTIME/'settings'/log.stem)
    with log.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(argv=command, CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES'))) + '\n')
        stream.flush()
        process = subprocess.Popen([str(v) for v in command], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        with LOCK:
            STATE.setdefault('processes', {})[str(process.pid)] = dict(argv=command, log=str(log),
                CUDA_VISIBLE_DEVICES=env.get('CUDA_VISIBLE_DEVICES'))
            write(RUNTIME/'state.json', STATE)
        code = process.wait()
        with LOCK:
            STATE['processes'].pop(str(process.pid), None)
            write(RUNTIME/'state.json', STATE)
    return code


def train_command(job, gpu, smoke=False):
    return [PYTHON, task_entry('train.py'), '--config', job['config'],
        '--method', job['method'], '--strategy', job['strategy'], '--name',
        job['id']+('_smoke' if smoke else ''), '--execute', '--device', str(gpu), '--batch', str(job['batch']),
        *(['--epochs','2'] if smoke else [])]


def eval_command(job, gpu, split, output=None):
    command = [PYTHON, task_entry('evaluate.py'), '--config', job['config'],
        '--run-dir', job['run_dir'], '--split', split, '--execute', '--device', str(gpu)]
    if job['method'] == 'yoloe-vp':
        command += ['--prompt-shots', '10'] if split == 'dev' else ['--prompt-shots', '5', '10']
    if output:
        command += ['--output', str(output)]
    return command


def dev_metric_path(job):
    entry = STATE.get('jobs', {}).get(job['id'], {})
    directory = Path(entry.get('dev_output', str(Path(job['run_dir'])/'dev_val100')))
    paths = list(directory.glob('*/metrics.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected one complete dev result: {directory}')
    return paths[0]


def publish():
    """Call under LOCK; every finished trial remains visible even if later jobs fail."""
    rows = []
    for job_id, entry in STATE.get('jobs', {}).items():
        job = entry['job']
        row = dict(id=job_id, method=job['method'], strategy=job['strategy'], role=job.get('role','search'), status=entry['status'],
            gpu=entry.get('gpu'), batch=job['batch'], **job['settings'], **job['adaptation'])
        try:
            metrics=read(dev_metric_path(job))
            row.update({f'dev_{k}':metrics[k] for k in ['mAP50','mAP50_95','precision','recall','tp','fp','fn']})
        except (ValueError, FileNotFoundError):
            pass
        meta = Path(job['run_dir'])/'finetune_config.json'
        if meta.exists():
            try:
                metadata=read(meta)
                row['trainable_parameters']=metadata.get('trainable_parameter_count')
                row['strategy']=metadata['strategy']
                row.update(method_policy(metadata['method'],metadata['strategy'],freeze_savpe_bn_stats=metadata.get('adaptation',{}).get('freeze_savpe_bn_stats',False),freeze_savpe=metadata.get('adaptation',{}).get('freeze_savpe',False)))
            except json.JSONDecodeError:
                pass  # A training subprocess can be updating its metadata right now.
        rows.append(row)
    rows.sort(key=lambda r: r.get('dev_mAP50_95', -1), reverse=True)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        temp=REPORT/'dev_results.csv.tmp'
        with temp.open('w', encoding='utf-8-sig', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
        temp.replace(REPORT/'dev_results.csv')
    write(REPORT/'status.json', dict(stage=STATE.get('stage'), jobs=rows, updated=time.time()))
    lines=['# YOLOE系统优化实验','',f"阶段：{STATE.get('stage')}；更新：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        '', f'场景：{SCENE}。训练池固定{50 if SCENE=="chache" else 10}张（每类10张），dev固定100张，test固定300张。每张GPU独立训练一个模型，不使用DDP。',
        'FP32；每轮验证；全部重新从原始预训练权重初始化。排序只使用dev，测试候选在最终测试前冻结。',
        '固定头与视觉提示各自排序；训练范围、最大epoch、分辨率、batch不同，表中逐项记录。',
        '', '| 实验 | 方法 | 状态 | GPU | dev AP50 | dev AP50:95 |', '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['method']}/{r['strategy']} | {r['status']} | {r.get('gpu','')} | "
            f"{100*r['dev_mAP50']:.2f}"+' | '+f"{100*r['dev_mAP50_95']:.2f} |" if 'dev_mAP50' in r else
            f"| {r['id']} | {r['method']}/{r['strategy']} | {r['status']} | {r.get('gpu','')} | — | — |")
    lines += ['', '[完整参数与dev成绩CSV](dev_results.csv)。',
              f'运行配置、命令、退出码、日志、失败原因在项目outputs/_shared/runtime/{STAMP}/；',
              f'权重与逐图预测在outputs/YOLOE/finetune/{SCENE}_val300/runs/{STAMP}/。',
              '当前已查看过原测试集，后续优化成绩属于探索实验；并非一个全新的盲测。', '']
    if (REPORT/'test_results.md').exists():
        lines += ['[最终测试结果](test_results.md)', '']
    (REPORT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def worker(job, gpu):
    run_dir=Path(job['run_dir'])
    metadata_path=run_dir/'finetune_config.json'
    already_trained=metadata_path.is_file() and read(metadata_path).get('status')=='completed'
    resume=not already_trained and (run_dir/'weights/last.pt').is_file()
    with LOCK:
        previous=STATE.setdefault('jobs', {}).get(job['id'], {})
        STATE['jobs'][job['id']] = dict(**{k:v for k,v in previous.items() if k not in {'job','gpu','status','start'}},
            job=job, gpu=gpu, status='training' if not already_trained else 'dev_evaluating', start=time.time())
        publish(); write(RUNTIME/'state.json', STATE)
    code=0
    if not already_trained:
        command=train_command(job,gpu)
        if resume:
            backup=run_dir/f'interrupted_before_resume_{time.time_ns()}'
            backup.mkdir()
            for relative in ['weights/last.pt','weights/best.pt','finetune_config.json','results.csv','args.yaml']:
                source=run_dir/relative
                if source.exists():
                    target=backup/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            command+=['--resume']
        code=run_command(command,RUNTIME/'logs'/f"{job['id']}_train.log")
    if code == 0:
        with LOCK:
            STATE['jobs'][job['id']]['status']='dev_evaluating'; publish()
        output=run_dir/'dev_val100'
        if output.exists():
            output=run_dir/f'dev_recovered_{time.time_ns()}'
        with LOCK:
            STATE['jobs'][job['id']]['dev_output']=str(output)
        code=run_command(eval_command(job,gpu,'dev',output),RUNTIME/'logs'/f"{job['id']}_dev.log")
    with LOCK:
        STATE['jobs'][job['id']].update(status='completed' if code==0 else 'failed',exit_code=code,end=time.time())
        publish(); write(RUNTIME/'state.json', STATE)
    print(f"DONE {job['id']} gpu={gpu} code={code}",flush=True)
    return code


def dispatch(jobs,gpus):
    slots=queue.Queue()
    for gpu in gpus:slots.put(gpu)
    def execute(job):
        gpu=slots.get()
        try:return worker(job,gpu)
        finally:slots.put(gpu)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures=[executor.submit(execute,j) for j in jobs]
        for future in as_completed(futures):future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Continue existing state; preserve successful runs and logs.')
    parser.add_argument('--scene',choices=['chache','diaowu'],default='chache')
    parser.add_argument('--stamp')
    args=parser.parse_args()
    configure(args.scene,args.stamp)
    from summarize_finetuning_best import is_compact,process_scene
    if is_compact(REPORT):
        if args.execute:raise ValueError('This completed search is archived; use a new --stamp for a new search.')
        process_scene(args.scene)
        print(f'Updated compact report: {REPORT / "ALL_EXPERIMENTS.md"}')
        return
    REPORT.mkdir(parents=True,exist_ok=True); RUNTIME.mkdir(parents=True,exist_ok=True)
    if args.resume:
        STATE.update(read(RUNTIME/'state.json'))
        for pid in STATE.get('processes',{}):
            try:os.kill(int(pid),0)
            except ProcessLookupError:continue
            raise RuntimeError(f'Prior process may still be running: {pid}; inspect before resuming.')
        STATE['processes']={}
        jobs=read(REPORT/'plan.json')['jobs']
        for entry in STATE.get('jobs',{}).values():
            repaired=Path(entry['job']['run_dir'])/'dev_repaired'
            if list(repaired.glob('*/metrics.json')):
                entry.setdefault('history',[]).append(dict(status=entry['status'],exit_code=entry.get('exit_code'),note='Original evaluation failure; repaired dev output retained.'))
                entry.update(status='completed',dev_output=str(repaired),exit_code=0)
        publish();write(RUNTIME/'state.json',STATE)
    else:
        if PROJECT.exists():raise FileExistsError(PROJECT)
        write(REPORT/'base_config.json',read(task_entry('config.json')))
        jobs=prepare((baseline_specs() if SCENE=='diaowu' else [])+specs())
        write(REPORT/'plan.json',dict(scene=SCENE,jobs=jobs,gpus=args.gpus,selection='dev AP50:95, separate head and VP; baseline jobs excluded',
            test_policy='freeze top3 each family and winner seed repeats before reading test results'))
        STATE.update(stage='prepared',scene=SCENE,jobs={},processes={});publish()
    if not args.execute:
        print(f'{len(jobs)} planned jobs; {REPORT}');return
    # Fresh invocation only: preserve all existing output/checkpoints on rerun.
    if not args.resume:
        PROJECT.mkdir(parents=True)
        STATE['stage']='smoke';publish()
    smoke_ids=['head_full_lr0.0001','head_neck_lr0.0001','vp_savpe_lr1e-05_bn1',
        'vp_full_lr1e-05','vp_cross_lr1e-05_bn1','vp_cross_neck','head_full_960','head_full_1024']
    if SCENE=='diaowu':smoke_ids.append('baseline_yolov8')
    smoke_jobs=[next(j for j in jobs if j['id']==name) for name in smoke_ids]
    smoke_slots=queue.Queue()
    for gpu in args.gpus:smoke_slots.put(gpu)
    def smoke(index_job):
        index,job=index_job;gpu=smoke_slots.get()
        try:
            code=run_command(train_command(job,gpu,True),RUNTIME/'logs'/f"{job['id']}_smoke.log")
            print(f"SMOKE {job['id']} code={code}",flush=True)
            return job['id'],code
        finally:smoke_slots.put(gpu)
    if args.resume:
        outcomes=read(REPORT/'smoke_results.json')['outcomes']
    else:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            outcomes=list(executor.map(smoke,enumerate(smoke_jobs)))
        write(REPORT/'smoke_results.json',dict(outcomes=outcomes))
    if any(code for _,code in outcomes):
        STATE['stage']='smoke_failed';publish();raise RuntimeError(outcomes)
    if args.smoke_only:return
    STATE['stage']='stage1';publish();dispatch([j for j in jobs if STATE['jobs'].get(j['id'],{}).get('status')!='completed'],args.gpus)
    # Evaluate every completed trial on dev using the exact final inference protocol.
    winners=[]
    for family in ['yoloe-head','yoloe-vp']:
        candidates=[e['job'] for e in STATE['jobs'].values() if e['status']=='completed' and e['job']['method']==family and '_seed' not in e['job']['id'] and e['job'].get('role')!='baseline']
        def score(job):
            return read(dev_metric_path(job))['mAP50_95']
        candidates.sort(key=score,reverse=True)
        if candidates:winners.append(candidates[0])
    repeats=[]
    for winner in winners:
        for seed in [1,2]:
            job=json.loads(json.dumps(winner));job['id']=winner['id']+f'_seed{seed}';job['settings']['seed']=seed
            repeats.append(job)
    STATE['stage']='seed_repeats';publish();dispatch([j for j in prepare(repeats) if STATE['jobs'].get(j['id'],{}).get('status')!='completed'],args.gpus)
    selected=[]
    for family in ['yoloe-head','yoloe-vp']:
        candidates=[e['job'] for e in STATE['jobs'].values() if e['status']=='completed' and e['job']['method']==family and '_seed' not in e['job']['id'] and e['job'].get('role')!='baseline']
        candidates.sort(key=score,reverse=True);selected.extend(candidates[:3])
    selected.extend(j for j in repeats if STATE['jobs'][j['id']]['status']=='completed')
    selected.extend(e['job'] for e in STATE['jobs'].values() if e['status']=='completed' and e['job'].get('role')=='baseline')
    selection_path=REPORT/'test_selection.json'
    if selection_path.exists():
        if [j['id'] for j in read(selection_path)['selected']] != [j['id'] for j in selected]:
            raise RuntimeError('Frozen test selection differs; inspect instead of overwriting it.')
    else:
        write(selection_path,dict(criterion='dev AP50:95 only',selected=selected,created=time.time()))
    STATE['stage']='final_tests';publish()
    def test(index_job):
        i,job=index_job;gpu=args.gpus[i%len(args.gpus)]
        code=run_command(eval_command(job,gpu,'test'),RUNTIME/'logs'/f"{job['id']}_test.log")
        return job,code
    # Allocate at most one test process per GPU, including the tail of the queue.
    slots=queue.Queue()
    for gpu in args.gpus:slots.put(gpu)
    def test_slot(job):
        gpu=slots.get()
        try:
            expected=['visual_5shot','visual_10shot'] if job['method']=='yoloe-vp' else ['fixed_head']
            output=Path(job['run_dir'])/'test_val300'
            if all((output/name/'metrics.json').is_file() for name in expected):
                return job,0
            if output.exists():
                raise RuntimeError(f'Partial test output retained; inspect before recovery: {output}')
            code=run_command(eval_command(job,gpu,'test'),RUNTIME/'logs'/f"{job['id']}_test.log")
            return job,code
        finally:slots.put(gpu)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        tests=list(executor.map(test_slot,selected))
    rows=[]
    for job,code in tests:
        for path in (Path(job['run_dir'])/'test_val300').glob('*/metrics.json'):
            metric=read(path);row=dict(id=job['id'],method=job['method'],prompt_shots=5 if path.parent.name=='visual_5shot' else 10 if path.parent.name=='visual_10shot' else None,
                exit_code=code,**{k:metric[k] for k in ['mAP50','mAP50_95','precision','recall','tp','fp','fn']})
            rows.append(row)
            write(REPORT/'metrics'/job['id']/path.parent.name/'metrics.json',metric)
    write(REPORT/'test_results.json',dict(rows=rows,selection=str(REPORT/'test_selection.json')))
    if rows:
        with (REPORT/'test_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 按dev选择后的300张测试结果','', '| 实验 | 提示shot | AP50 | AP50:95 | P≥0.5 | R≥0.5 | TP/FP/FN |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['prompt_shots'] or '不适用'} | "+' | '.join(f"{100*r[k]:.2f}" for k in ['mAP50','mAP50_95','precision','recall'])+f" | {r['tp']}/{r['fp']}/{r['fn']} |")
    lines+=['','各参数组合按dev选择，未按test选择。不同训练范围/分辨率/预算见dev_results.csv。',
             f'场景：{SCENE}；本场景基线与原始YOLOE结果见独立总表。', '']
    (REPORT/'test_results.md').write_text('\n'.join(lines),encoding='utf-8')
    STATE['stage']='complete' if all(code==0 for _,code in tests) else 'complete_with_test_failures'
    publish();write(RUNTIME/'state.json',STATE)
    print(f"COMPLETE {REPORT}",flush=True)


if __name__=='__main__':main()

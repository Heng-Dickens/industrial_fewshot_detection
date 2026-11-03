"""Freeze dev candidates and test finished seed-0 models on otherwise idle GPUs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from summarize_yoloe_optimization import ROOT,read,write,optimization_report_dir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp',required=True)
    parser.add_argument('--gpus',nargs='+',type=int,required=True)
    args=parser.parse_args()
    runtime=ROOT/'outputs/_shared/runtime'/args.stamp
    report=optimization_report_dir(args.stamp, project_root=ROOT)
    state=read(runtime/'state.json')
    if state['stage']!='seed_repeats':raise RuntimeError('Only run while fixed-config seed repeats are training.')
    occupied={p.get('CUDA_VISIBLE_DEVICES') for p in state['processes'].values()}
    if any(str(g) in occupied for g in args.gpus):raise RuntimeError('A requested GPU is allocated to an active training process.')
    scene=read(report/'plan.json').get('scene','chache')
    def score(job):
        entry=state['jobs'][job['id']]
        directory=Path(entry.get('dev_output',str(Path(job['run_dir'])/'dev_val100')))
        return read(next(directory.glob('*/metrics.json')))['mAP50_95']
    selected=[];winners=[]
    for family in ['yoloe-head','yoloe-vp']:
        jobs=[e['job'] for e in state['jobs'].values() if e['status']=='completed' and e['job']['method']==family
              and '_seed' not in e['job']['id'] and e['job'].get('role')!='baseline']
        jobs.sort(key=score,reverse=True);selected.extend(jobs[:3]);winners.append(jobs[0])
    for winner in winners:
        for seed in [1,2]:selected.append(state['jobs'][winner['id']+f'_seed{seed}']['job'])
    selected.extend(e['job'] for e in state['jobs'].values() if e['status']=='completed' and e['job'].get('role')=='baseline')
    frozen=report/'test_selection.json'
    if frozen.exists():
        if [j['id'] for j in read(frozen)['selected']]!=[j['id'] for j in selected]:raise RuntimeError('Frozen selection differs.')
    else:
        write(frozen,dict(criterion='dev AP50:95 only',selected=selected,created=time.time(),
            note='Seed-0 candidates and predeclared baselines tested during fixed-config seed repeats.'))
    print('FROZEN',[j['id'] for j in selected],flush=True)
    slots=queue.Queue()
    for gpu in args.gpus:slots.put(gpu)
    lock=threading.Lock();outcomes=[]
    def run(job):
        gpu=slots.get()
        try:
            output=Path(job['run_dir'])/'test_val300'
            if output.exists():
                expected=['visual_5shot','visual_10shot'] if job['method']=='yoloe-vp' else ['fixed_head']
                if all((output/name/'metrics.json').is_file() and read(output/name/'run.json').get('status')=='completed' for name in expected):
                    print('REUSE',job['id'],flush=True);return 0
                raise RuntimeError(f'Refusing partial result directory: {output}')
            command=[sys.executable,str(ROOT/f'outputs/YOLOE/finetune/scripts/{scene}/evaluate.py'),
                '--config',job['config'],'--run-dir',job['run_dir'],'--split','test','--execute','--device','0']
            if job['method']=='yoloe-vp':command+=['--prompt-shots','5','10']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='2',YOLO_AUTOINSTALL='false',WANDB_MODE='disabled',PYTHONUNBUFFERED='1',
                YOLO_CONFIG_DIR=str(runtime/'settings'/('early_test_'+job['id'])))
            log=runtime/'logs'/f"{job['id']}_test_early.log"
            with log.open('x') as stream:
                stream.write(json.dumps(dict(argv=command,CUDA_VISIBLE_DEVICES=str(gpu)))+'\n');stream.flush()
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                code=process.wait()
            result=dict(id=job['id'],gpu=gpu,exit_code=code,log=str(log))
            with lock:
                outcomes.append(result);write(runtime/'early_tests.json',outcomes)
            print('TEST',result,flush=True)
            return code
        finally:slots.put(gpu)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        codes=list(executor.map(run,[j for j in selected if '_seed' not in j['id']]))
    if any(codes):raise RuntimeError('One or more early test jobs failed; preserve logs and inspect.')


if __name__=='__main__':main()

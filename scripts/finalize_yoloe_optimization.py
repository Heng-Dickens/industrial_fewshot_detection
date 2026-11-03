"""Finish raw dev baseline, archive tables, and plot results for an existing search."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from summarize_yoloe_optimization import ROOT,read,write,optimization_report_dir


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp',required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    runtime=ROOT/'outputs/_shared/runtime'/args.stamp
    report=optimization_report_dir(args.stamp, project_root=ROOT)
    from summarize_finetuning_best import is_compact,process_scene
    if is_compact(report):
        process_scene('diaowu' if report.parent.parent.name=='diaowu_val300' else 'chache')
        print(f'Updated compact report: {report / "ALL_EXPERIMENTS.md"}')
        return
    plan=read(report/'plan.json');scene=plan.get('scene','chache')
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',YOLO_AUTOINSTALL='false')
    baseline=runtime/'baseline_dev100'
    raw_done=(baseline/'visual_10shot/metrics.json').is_file()
    baseline_source=next(j for j in plan['jobs'] if j['id']=='vp_savpe_lr1e-05_bn1')
    latest_partial=0
    while True:
        state=read(runtime/'state.json')
        smoke_run=Path(baseline_source['run_dir']+'_smoke')
        metadata=smoke_run/'finetune_config.json'
        if not raw_done and metadata.is_file() and read(metadata).get('status')=='completed':
            command=[sys.executable,str(ROOT/f'outputs/YOLOE/finetune/scripts/{scene}/evaluate.py'),
                '--config',baseline_source['config'],'--run-dir',str(smoke_run),'--split','dev',
                '--pretrained-baseline','--prompt-shots','10','--output',str(baseline),'--device','cpu','--execute']
            with (runtime/'logs/raw_baseline_dev_cpu.log').open('x') as stream:
                stream.write(json.dumps(dict(argv=command,CUDA_VISIBLE_DEVICES=''))+'\n');stream.flush()
                subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
            raw_done=True
            print('Original-weight unified dev baseline completed.',flush=True)
        if state['stage']=='complete':break
        if state['stage'] in {'smoke_failed','complete_with_test_failures'}:
            raise RuntimeError(f"Search needs attention: {state['stage']}")
        if not args.wait:raise RuntimeError(f"Search still running: {state['stage']}")
        if time.time()-latest_partial>120 and state.get('jobs'):
            try:
                subprocess.run([sys.executable,str(ROOT/'scripts/summarize_yoloe_optimization.py'),'--stamp',args.stamp,'--partial'],cwd=ROOT,env=env,check=True)
                latest_partial=time.time()
            except subprocess.CalledProcessError:
                print('Partial archive encountered an in-progress file; retry later.',flush=True)
        time.sleep(30)
    for script in ['summarize_yoloe_optimization.py','calibrate_yoloe_search_thresholds.py','plot_yoloe_optimization.py','summarize_yoloe_optimization.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--stamp',args.stamp],cwd=ROOT,env=env,check=True)
    write(runtime/'finalized.json',dict(status='completed',scene=scene,stamp=args.stamp,time=time.time(),raw_dev_baseline=raw_done))
    print(f'FINALIZED {report}',flush=True)


if __name__=='__main__':main()

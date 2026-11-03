"""Drain existing experiment workers, recover failures, verify and publish summaries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path.insert(0,str(Path(__file__).parent))
from generate_candidates import read,write,folder

STAMP='yoloe_v8l_10shot_qwen38_v1_20261030'
RUNTIME=ROOT/'outputs/_shared/runtime/ovd_mllm_filtering_20261030'
STATE=RUNTIME/'supervisor.json'


def alive(pid):
    try:
        status=Path(f'/proc/{pid}/stat').read_text()
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
        return status.split()[2]!='Z' and b'outputs/OVD-MLLM/scripts/experiment.py' in command
    except (OSError,IndexError):return False


def passed(scene):
    for split in ['val100','val300']:
        root=folder(scene,split,STAMP)
        if not (root/'verification.json').exists() or read(root/'verification.json').get('status')!='passed':return False
        if not (root/'experiment_results.json').exists():return False
        stages={r['stage'] for r in read(root/'experiment_results.json').get('rows',[])}
        if stages!={'source','baseline','filtered','deduplicated','refined'}:return False
    return True


def main():
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
    previous=read(STATE) if STATE.exists() else {}
    state=dict(status='running',pid=os.getpid(),started_at=previous.get('started_at',time.time()),scenes={})
    for scene in ['chache','diaowu']:
        execution=read(RUNTIME/f'{scene}_execution.json')
        prior=previous.get('scenes',{}).get(scene,{})
        state['scenes'][scene]=prior if prior.get('pid') and alive(prior['pid']) else dict(pid=execution['pid'],phase='initial_worker',repairs=0)
    write(STATE,state)
    while True:
        for scene,item in state['scenes'].items():
            if item.get('phase') in {'complete','incomplete'}:continue
            if alive(item['pid']):continue
            if passed(scene):
                item['phase']='complete';continue
            if item['repairs']>=3:
                item['phase']='incomplete';continue
            item['repairs']+=1
            argv=[sys.executable,'-u','outputs/OVD-MLLM/scripts/experiment.py','--scene',scene,
                  '--split','both','--threshold','0.01','--refine','--concurrency','4']
            log=(RUNTIME/f'{scene}_recovery.log').open('a')
            process=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();item.update(pid=process.pid,phase='recovering')
            write(RUNTIME/f'{scene}_execution.json',dict(pid=process.pid,argv=argv,status='recovering'))
            print('RECOVERY',scene,'pass',item['repairs'],'pid',process.pid,flush=True)
        write(STATE,state)
        if all(v['phase'] in {'complete','incomplete'} for v in state['scenes'].values()):break
        time.sleep(15)
    command=[sys.executable,'outputs/OVD-MLLM/scripts/summarize_experiment.py','--audit','--previews']
    subprocess.run(command,cwd=ROOT,env=env,check=True)
    if all(v['phase']=='complete' for v in state['scenes'].values()):
        subprocess.run([sys.executable,'scripts/summarize_results.py','--dataset','chache_val300',
                        '--dataset','diaowu_val300'],cwd=ROOT,env=env,check=True)
        path=ROOT/'outputs/OVD-MLLM/scripts/README.md';text=path.read_text()
        lines=text.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('当前状态：'):
                lines[i]='当前状态：两场景共800张图的YOLOE候选、Qwen筛选/去重及坐标微调已完成，四个开发/测试队列均通过完整性校验。实际成绩、误删与成本见下方统一结果入口。'
                break
        path.write_text('\n'.join(lines)+'\n')
        state['status']='complete'
        for scene in state['scenes']:
            execution=read(RUNTIME/f'{scene}_execution.json');execution['status']='complete'
            write(RUNTIME/f'{scene}_execution.json',execution)
    else:state['status']='incomplete'
    state['finished_at']=time.time();write(STATE,state)
    print('SUPERVISOR',state['status'],flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        state=read(STATE) if STATE.exists() else {}
        state.update(status='failed',error=type(exc).__name__+': '+str(exc),finished_at=time.time())
        write(STATE,state)
        raise

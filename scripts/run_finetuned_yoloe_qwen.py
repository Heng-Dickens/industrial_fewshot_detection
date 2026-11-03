"""Evaluate three best 10shot YOLOE adaptations through the same Qwen27B cascade.

Reuses frozen predictions, never trains/reruns YOLOE. Each scene uses its original
300-image cohort. Qwen receives one boxed reference per class, candidate batches
of at most 20, then only retained boxes for coordinate refinement. Successful
responses are cached by full input identity; every paid attempt records usage.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from PIL import Image, ImageDraw, ImageFont
from industrial_fewshot.config import load_config
from industrial_fewshot.credentials import resolve_api_key
from industrial_fewshot.data.labelme import load_evaluation_truth
from industrial_fewshot.detectors.claude_messages import complete_message
from industrial_fewshot.evaluation.evaluator import evaluate_records
from industrial_fewshot.evaluation.reporting import save_text_report
from industrial_fewshot.postprocess.boxes import box_iou
from industrial_fewshot.postprocess.schema import clean_json_response
from industrial_fewshot.service_errors import terminal_service_error
from industrial_fewshot.prompts.references import encode_data_url

SCRIPT=Path(__file__).resolve()
RUNTIME=ROOT/'outputs/_shared/runtime/finetuned_yoloe_qwen_20261103'
REPORT=ROOT/'outputs/OVD-MLLM/comparisons/finetuned_yoloe_qwen_20261103'
DEPENDENCY=ROOT/'outputs/_shared/runtime/qwen_model10_supplement_20261103'
METHODS={'head':'移除','savpe_train':'训练','savpe_frozen':'冻结'}
NAMES={'head':'固定头／移除SAVPE','savpe_train':'视觉提示／更新SAVPE','savpe_frozen':'视觉提示／冻结SAVPE'}
STAGES=('source','baseline','filtered','deduplicated','refined')
PROTOCOL='finetuned.cascade.original1280+overlay24px+context1.5.crop640.v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)
def jobs():return read(RUNTIME/'plan.json')['jobs']
def state_path(job):return RUNTIME/'jobs'/(job['id']+'.json')
def output(job):return ROOT/job['output_dir']
def live(pid):
    try:return SCRIPT.name in Path(f'/proc/{pid}/cmdline').read_text()
    except OSError:return False
def relative(path,page=REPORT/'README.md'):return os.path.relpath(path,page.parent)


def source_data(job):
    source=ROOT/job['source_run']
    if sha(source/'records.jsonl')!=job['source_records_sha256'] or sha(source/'metrics.json')!=job['source_metrics_sha256']:
        raise ValueError('Frozen source predictions or metrics changed.')
    manifest=read(ROOT/job['manifest'])
    if sha(ROOT/job['manifest'])!=job['manifest_sha256']:raise ValueError('Frozen cohort changed.')
    samples={s['relative_path']:s for s in manifest['samples']}
    ids={t['id'] for t in read(source/'run.json')['targets']}
    seen=set();merged={p:[] for p in samples}
    for line in (source/'records.jsonl').read_text().splitlines():
        r=json.loads(line);key=(r['relative_path'],r['target_id'])
        if key in seen or r['status']!='success' or r['relative_path'] not in samples or r['target_id'] not in ids:
            raise ValueError('Invalid, incomplete or duplicated source predictions.')
        if r['image_sha256']!=samples[r['relative_path']]['image_sha256']:raise ValueError('Source image SHA differs.')
        seen.add(key);merged[r['relative_path']].extend(copy.deepcopy(r['objects']))
    if len(samples)!=300 or len(seen)!=300*len(ids):raise ValueError('Source is not the complete 300-image cohort.')
    records=[]
    for path,sample in samples.items():
        records.append(dict(relative_path=path,target_id='cascade',status='success',image_sha256=sample['image_sha256'],
            objects=merged[path],target_count=len(merged[path]),annotation_status='annotated' if merged[path] else 'no_target',
            reason_code='target_detected' if merged[path] else 'no_target',reason='Frozen source predictions.'))
    return manifest,records


def candidates(record,labels):
    kept=[]
    for label in labels:
        kept.extend(sorted((copy.deepcopy(o) for o in record['objects']
            if o['label']==label and o['confidence']>=.01),key=lambda o:-o['confidence'])[:100])
    kept.sort(key=lambda o:-o['confidence'])
    for i,o in enumerate(kept):o['candidate_id']=f'c{i:04d}'
    return kept


def reference_samples(scene,labels,test_hashes):
    references=[]
    for label in labels:
        root=ROOT/'data/references'/scene
        if scene=='chache':root/=label
        frozen=read(root/'1shot/seed_0.json')
        rows=frozen['positive_samples']
        if len(rows)!=1:raise ValueError('Expected one frozen positive reference per class.')
        r=rows[0]
        from industrial_fewshot.paths import resolve_location
        image=resolve_location(Path(r['image_path']));annotation=resolve_location(Path(r['json_path']))
        if sha(image)!=r['image_sha256'] or sha(annotation)!=r['annotation_sha256']:
            raise ValueError('Frozen reference changed.')
        if r['image_sha256'] in test_hashes:raise ValueError('Reference overlaps test cohort.')
        with Image.open(image) as im:w,h=im.size
        boxes=load_evaluation_truth(annotation,w,h,{label},'error')[0]
        references.append(dict(label=label,image=str(image),annotation=str(annotation),
            image_sha256=sha(image),annotation_sha256=sha(annotation),
            boxes=[b['bbox_pixels_xyxy'] for b in boxes],manifest=str(root/'1shot/seed_0.json'),
            manifest_sha256=sha(root/'1shot/seed_0.json')))
    return references


def prepare():
    if (RUNTIME/'plan.json').exists():raise FileExistsError('Plan exists; use --background/--status.')
    rows=read(ROOT/'outputs/FINAL_COMPARISON.json')['primary_rows'];planned=[]
    for scene in ('chache','diaowu'):
        dataset=scene+'_val300'
        manifest_path=ROOT/f'data/raw/{dataset}/manifest.json'
        manifest=read(manifest_path)
        protocol=read(ROOT/f'outputs/OVD-MLLM/experiments/{dataset}/yoloe_v8l_10shot_qwen38_v1_20261030/experiment_plan.json')
        for method,savpe in METHODS.items():
            pool=[r for r in rows if r['dataset']==dataset and r['algorithm']=='YOLOE'
                and r['training_shot_per_class']==10 and r['savpe']==savpe and r['eligible_for_comparison']
                and (method=='head' or r['inference_shot_per_class']==10)]
            source=sorted(pool,key=lambda r:(-r['AP50'],r['run_dir']))[0]
            job=dict(id=scene+'_'+method,scene=scene,dataset=dataset,method=method,savpe=savpe,
                source_run=source['run_dir'],source_AP50=source['AP50'],source_AP50_95=source['AP50_95'],
                source_records_sha256=sha(ROOT/source['run_dir']/'records.jsonl'),
                source_metrics_sha256=sha(ROOT/source['run_dir']/'metrics.json'),
                manifest=str(manifest_path.relative_to(ROOT)),manifest_sha256=sha(manifest_path),
                output_dir=f'outputs/OVD-MLLM/experiments/{dataset}/finetuned_{method}_10shot_qwen27b_1shot_20261103',
                labels=[r['name'] for r in protocol['definitions']],definitions=protocol['definitions'],
                model=protocol['model'],filter_prompt=protocol['filter_prompt'],refine_prompt=protocol['refine_prompt'],
                input_protocol=PROTOCOL,training_shot_per_class=10,prompt_shot_per_class=None if method=='head' else 10,
                mllm_reference_shot_per_class=1,source_input_size=source['input_size'],
                evaluation=read(ROOT/source['run_dir']/'metrics.json')['settings'])
            job['references']=reference_samples(scene,job['labels'],{s['image_sha256'] for s in manifest['samples']})
            _,records=source_data(job)
            selected=[candidates(r,job['labels']) for r in records]
            job.update(candidate_count=sum(map(len,selected)),nonempty_images=sum(bool(x) for x in selected),
                filter_batches=sum(math.ceil(len(x)/20) for x in selected))
            job['identity']=digest(job)
            write(state_path(job),dict(status='queued',completed_images=0,stage='queued',identity=job['identity']))
            write(output(job)/'experiment_plan.json',job)
            planned.append(job)
            print(job['id'],'candidate boxes',job['candidate_count'],'filter requests',job['filter_batches'],flush=True)
    write(RUNTIME/'plan.json',dict(schema='finetuned_yoloe_qwen.v1',jobs=planned,image_tasks=1800,
        dependency=str(DEPENDENCY),selection='One actual highest AP50 run per scene/method, training10 and visual prompt10; exploratory test selection.',
        candidate_floor=.01,batch_size=20,dedup_min_iou=.5,keep_uncertain=True,
        source_policy='Saved predictions only; no new training or GPU inference.',reference_shot=1,
        cost_policy='Skip images with no candidates; cache validated full-identity responses; max3 attempts per request; no nested retries.'))
    render()


def image_url(im,side=1280):
    im=im.convert('RGB');im.thumbnail((side,side),Image.Resampling.LANCZOS)
    data=io.BytesIO();im.save(data,format='JPEG',quality=90)
    return encode_data_url(data.getvalue(),'image/jpeg')


def overlay(im,boxes):
    im=im.convert('RGB');w,h=im.size;im.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    sx,sy=im.width/w,im.height/h;draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    for o in boxes:
        x1,y1,x2,y2=o['bbox_pixels_xyxy'];b=(x1*sx,y1*sy,x2*sx,y2*sy)
        draw.rectangle(b,outline='yellow',width=3)
        name=o.get('candidate_id',o['label']);x=max(0,min(b[0],im.width-110));y=max(0,b[1]-26)
        draw.rectangle((x,y,x+110,y+27),fill='black');draw.text((x,y),name,fill='yellow',font=font)
    return im


def visual_references(job):
    content=[]
    for r in job['references']:
        if sha(r['image'])!=r['image_sha256'] or sha(r['annotation'])!=r['annotation_sha256']:
            raise ValueError('Reference asset changed.')
        with Image.open(r['image']) as im:
            content.extend([{'type':'text','text':f'类别{r["label"]}的唯一参考：原图，随后为人工标注框图。'},
                {'type':'image_url','image_url':{'url':image_url(im.copy())}},
                {'type':'image_url','image_url':{'url':image_url(overlay(im.copy(),[
                    dict(label=r['label'],bbox_pixels_xyxy=b) for b in r['boxes']]))}}])
    return content


def query_content(job,sample,boxes,stage,refs):
    image=ROOT/f'data/raw/{job["dataset"]}/images'/sample['relative_path']
    if sha(image)!=sample['image_sha256']:raise ValueError('Test image SHA differs.')
    # Deliberately accept only geometry and image identity here. Ground truth is
    # used by evaluation only and must never enter model messages/cache inputs.
    w,h=sample['width'],sample['height']
    descriptors=[dict(candidate_id=o['candidate_id'],label=o['label'],confidence=o['confidence'],
        bbox_1000=[round(x/s*1000,3) for x,s in zip(o['bbox_pixels_xyxy'],[w,h,w,h])]) for o in boxes]
    prompt=job['filter_prompt'] if stage=='filter' else job['refine_prompt']
    text=prompt+'\n类别定义：'+json.dumps(job['definitions'],ensure_ascii=False)+\
        '\nrequested_ids='+json.dumps([o['candidate_id'] for o in boxes])+\
        '\n候选（坐标属于整张原图）：'+json.dumps(descriptors,ensure_ascii=False)
    content=copy.deepcopy(refs)
    content.append({'type':'text','text':text})
    with Image.open(image) as im:
        content.extend([{'type':'text','text':'本次待审原图：'},
            {'type':'image_url','image_url':{'url':image_url(im.copy())}},
            {'type':'text','text':'同一待审原图的编号候选框：'},
            {'type':'image_url','image_url':{'url':image_url(overlay(im.copy(),boxes))}}])
        for o in boxes:
            x1,y1,x2,y2=o['bbox_pixels_xyxy'];cx,cy=(x1+x2)/2,(y1+y2)/2
            dx,dy=max(x2-x1,2)*.75,max(y2-y1,2)*.75
            crop=im.crop((max(0,math.floor(cx-dx)),max(0,math.floor(cy-dy)),
                min(w,math.ceil(cx+dx)),min(h,math.ceil(cy+dy))))
            content.extend([{'type':'text','text':f'{o["candidate_id"]}上下文裁图；输出坐标仍属于整张原图：'},
                {'type':'image_url','image_url':{'url':image_url(crop,640)}}])
    return content


def validate_reviews(document,boxes,stage):
    rows=document.get('reviews');ids={o['candidate_id'] for o in boxes}
    if not isinstance(rows,list) or len(rows)!=len(ids) or any(not isinstance(r,dict) for r in rows):
        raise ValueError('Review result must contain every requested candidate exactly once.')
    if {r.get('candidate_id') for r in rows}!=ids:raise ValueError('Missing or foreign candidate IDs.')
    for r in rows:
        if not isinstance(r.get('reason'),str) or not r['reason'].strip():raise ValueError('Missing candidate reason.')
        if stage=='filter':
            if r.get('decision') not in {'keep','remove','uncertain'}:raise ValueError('Invalid review decision.')
            other=r.get('duplicate_of')
            if other is not None and (other not in ids or other==r['candidate_id']):raise ValueError('Invalid duplicate link.')
        else:
            if r.get('action') not in {'keep','refine'}:raise ValueError('Invalid coordinate action.')
            if r['action']=='refine':
                b=r.get('bbox_1000')
                if (not isinstance(b,list) or len(b)!=4 or any(isinstance(v,bool) or not isinstance(v,(float,int))
                    or not math.isfinite(v) or not 0<=v<=1000 for v in b) or not (b[0]<b[2] and b[1]<b[3])):
                    raise ValueError('Invalid normalized full-image coordinates.')
    return rows


async def request(client,job,sample,boxes,stage,refs):
    content=query_content(job,sample,boxes,stage,refs)
    key=digest(dict(identity=job['identity'],stage=stage,messages=content))
    cache=RUNTIME/'responses'/job['id']/(key+'.json')
    if cache.exists():
        saved=read(cache)
        if saved.get('key')!=key:raise ValueError('Cache identity mismatch.')
        validate_reviews({'reviews':saved['reviews']},boxes,stage)
        return saved['reviews']
    for attempt in range(3):
        event=dict(key=key,job=job['id'],stage=stage,image=sample['relative_path'],attempt=attempt+1,started=time.time())
        try:
            raw,finish,usage,_=await complete_message(client,job['model'],[{'role':'user','content':content}],{'type':'json_object'})
            event.update(usage=usage,finish_reason=finish)
            if finish!='stop':raise ValueError('Truncated or abnormal completion.')
            doc=clean_json_response(raw);rows=validate_reviews(doc,boxes,stage)
            write(cache,dict(key=key,reviews=rows,usage=usage,finish_reason=finish))
            event['status']='success'
            return rows
        except Exception as exc:
            event.update(status='failed',error=str(exc))
            if terminal_service_error(exc) or attempt==2:raise
            await asyncio.sleep(2)
        finally:
            event['finished']=time.time()
            with (RUNTIME/'requests.jsonl').open('a') as stream:stream.write(json.dumps(event,ensure_ascii=False)+'\n')


def apply_filter(boxes,reviews):
    by_id={r['candidate_id']:r for r in reviews}
    return [copy.deepcopy(o) for o in boxes if by_id[o['candidate_id']]['decision']!='remove']


def deduplicate(boxes,reviews):
    by_id={o['candidate_id']:o for o in boxes};parent={k:k for k in by_id}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    for r in reviews:
        a,b=r['candidate_id'],r.get('duplicate_of')
        if a in by_id and b in by_id and by_id[a]['label']==by_id[b]['label'] and \
                box_iou(by_id[a]['bbox_pixels_xyxy'],by_id[b]['bbox_pixels_xyxy'])>=.5:
            parent[root(a)]=root(b)
    winners={}
    for o in boxes:
        key=root(o['candidate_id'])
        if key not in winners or o['confidence']>winners[key]['confidence']:winners[key]=o
    return [copy.deepcopy(o) for o in boxes if o is winners[root(o['candidate_id'])]]


def apply_refinement(boxes,reviews,w,h):
    by_id={r['candidate_id']:r for r in reviews};out=copy.deepcopy(boxes)
    for o in out:
        r=by_id[o['candidate_id']]
        if r['action']=='refine':
            o['bbox_2d']=r['bbox_1000']
            o['bbox_pixels_xyxy']=[v*s/1000 for v,s in zip(r['bbox_1000'],[w,h,w,h])]
    return out


def prediction(original,objects):
    r=copy.deepcopy(original);r.update(objects=objects,target_count=len(objects),
        annotation_status='annotated' if objects else 'no_target',reason_code='target_detected' if objects else 'no_target')
    return r


def save_stage(job,stage,manifest,records):
    directory=output(job)/stage;directory.mkdir(parents=True,exist_ok=True)
    cascade={k:job[k] for k in ('method','savpe','source_run','training_shot_per_class',
        'prompt_shot_per_class','mllm_reference_shot_per_class','source_input_size','identity')}
    mllm_shot=1 if stage not in {'source','baseline'} else 0
    run=dict(scene_name=job['dataset'],backend='ovd_mllm',stage=stage,status='completed',
        image_dir=str(ROOT/f'data/raw/{job["dataset"]}/images'),dataset_manifest=str(ROOT/job['manifest']),
        dataset_sha256=manifest['dataset_sha256'],image_paths=[s['relative_path'] for s in manifest['samples']],image_count=300,
        labels=job['definitions'],targets=[dict(id='cascade',backend='ovd_mllm',model='yoloe→qwen3.8-27b',labels=job['labels'])],
        evaluation=job['evaluation'],cascade=cascade,model='yoloe→qwen3.8-27b',run_dir=str(directory),
        experiment=dict(variant=stage,positive_count=job['prompt_shot_per_class'],negative_count=0),
        candidate_policy=dict(threshold=.01,limit_per_class=100),
        result_layout=dict(prompt_version='finetuned_cascade_v1'))
    with (directory/'records.jsonl').open('w') as stream:
        for r in records:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(directory/'run.json',run)
    prompt='' if not mllm_shot else job['filter_prompt']+('\n'+job['refine_prompt'] if stage=='refined' else '')
    write(directory/'prompt.json',dict(text=prompt,text_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        reference_shot_per_class=mllm_shot,input_protocol=PROTOCOL))
    write(directory/'experiment.json',dict(method='OVD-MLLM',dataset=job['dataset'],stage=stage,source_run=job['source_run']))
    metrics=evaluate_records(manifest['samples'],records,job['labels'],job['evaluation'])
    if metrics['evaluation_status']!='complete' or metrics['evaluated_image_count']!=300:raise ValueError('Incomplete evaluation.')
    if stage=='source' and (abs(metrics['mAP50']-job['source_AP50'])>1e-9 or
            abs(metrics['mAP50_95']-job['source_AP50_95'])>1e-9):
        raise ValueError('Re-evaluated source metrics differ from the selected original run.')
    write(directory/'metrics.json',metrics)
    save_text_report(directory,dict(run,metrics=metrics))
    return metrics


async def run_job(job):
    manifest,records=source_data(job)
    state=read(state_path(job))
    if state.get('identity')!=job['identity']:raise ValueError('Plan identity mismatch.')
    if state['status']=='completed':return
    state.update(status='running',pid=os.getpid(),started=time.time(),stage='source')
    write(state_path(job),state);render()
    save_stage(job,'source',manifest,records)
    baselines=[prediction(r,candidates(r,job['labels'])) for r in records]
    save_stage(job,'baseline',manifest,baselines)
    refs=visual_references(job)
    cfg=load_config(ROOT/f'configs/specific/{job["scene"]}_qwen27b_input_supplement.yaml')
    from openai import AsyncOpenAI
    client=AsyncOpenAI(api_key=resolve_api_key(cfg['targets'][0]['model_config']),
        base_url=job['model']['base_url'],timeout=180,max_retries=0)
    checkpoint=RUNTIME/'images'/job['id'];checkpoint.mkdir(parents=True,exist_ok=True)
    semaphore=asyncio.Semaphore(2)
    async def one(sample,original,baseline):
        path=checkpoint/(digest(sample['relative_path'])+'.json')
        if path.exists():
            cached=read(path)
            if cached.get('identity')!=job['identity']:raise ValueError('Image checkpoint identity mismatch.')
            return cached
        async with semaphore:
            boxes=baseline['objects'];reviews=[]
            for offset in range(0,len(boxes),20):
                reviews.extend(await request(client,job,sample,boxes[offset:offset+20],'filter',refs))
            filtered=apply_filter(boxes,reviews)
            dedup=deduplicate(filtered,reviews);refined=[]
            for offset in range(0,len(dedup),20):
                batch=dedup[offset:offset+20]
                reply=await request(client,job,sample,batch,'refine',refs)
                refined.extend(apply_refinement(batch,reply,sample['width'],sample['height']))
            result=dict(identity=job['identity'],relative_path=sample['relative_path'],
                filtered=prediction(original,filtered),deduplicated=prediction(original,dedup),refined=prediction(original,refined))
            # Refinement must preserve the accepted candidate set, class and score.
            if [(o['candidate_id'],o['label'],o['confidence']) for o in dedup]!=\
                    [(o['candidate_id'],o['label'],o['confidence']) for o in refined]:
                raise ValueError('Refinement changed candidate IDs/classes/scores.')
            write(path,result)
            state.update(completed_images=len(list(checkpoint.glob('*.json'))),stage='filter/refine')
            write(state_path(job),state)
            return result
    tasks=[asyncio.create_task(one(s,r,b)) for s,r,b in zip(manifest['samples'],records,baselines)]
    try:
        results=await asyncio.gather(*tasks)
        metrics={}
        for stage in ('filtered','deduplicated','refined'):
            metrics[stage]=save_stage(job,stage,manifest,[r[stage] for r in results])
        state.update(status='completed',completed_images=300,stage='completed',finished=time.time(),
            metrics={k:{name:v[name] for name in ('mAP50','mAP50_95','precision','recall')} for k,v in metrics.items()})
        write(output(job)/'verification.json',dict(status='passed',all_images_evaluated=300,stages=5,
            retained_fields_verified=True,refinement_candidate_set_fixed=True,failed_requests=0,identity=job['identity']))
    except BaseException as exc:
        for t in tasks:
            if not t.done():t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        state.update(status='interrupted' if isinstance(exc,asyncio.CancelledError) else 'failed',error=str(exc))
        raise
    finally:
        write(state_path(job),state);await client.close();render()


def refresh():subprocess.run([sys.executable,str(ROOT/'scripts/summarize_all_algorithms.py')],cwd=ROOT,check=True)


async def worker(index):
    isolate()
    for job in [j for j in jobs() if j['scene']==('chache' if index==0 else 'diaowu')]:
        await run_job(job)


def isolate():
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
    if any(j['model']['base_url']!='http://127.0.0.1:15721/v1' for j in jobs()):raise ValueError('Unexpected service endpoint.')


def dependency_ready():
    state=read(DEPENDENCY/'controller.json')
    if state['status']=='completed':return True
    if state['status']!='running' or not Path(f'/proc/{state.get("pid",0)}/cmdline').exists():
        raise RuntimeError('Preceding model batch stopped/failed; no new requests started.')
    return False


def run():
    previous=read(RUNTIME/'controller.json') if (RUNTIME/'controller.json').exists() else {}
    if live(previous.get('pid')):raise RuntimeError('Controller already running.')
    processes=[];completed=set()
    write(RUNTIME/'controller.json',dict(status='waiting_for_previous',pid=os.getpid(),started=time.time()))
    try:
        while not dependency_ready():render();time.sleep(5)
        write(RUNTIME/'controller.json',dict(status='running',pid=os.getpid(),started=time.time()))
        for index in range(2):
            with (RUNTIME/f'worker_{index}.log').open('a') as log:
                processes.append(subprocess.Popen([sys.executable,str(SCRIPT),'--worker',str(index)],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1')))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('A worker failed; paid responses retained.')
            render()
            current={j['id'] for j in jobs() if read(state_path(j))['status']=='completed'}
            if current!=completed:refresh();completed=current
            time.sleep(5)
        if any(p.returncode for p in processes):raise RuntimeError('A worker failed; paid responses retained.')
        refresh();write(RUNTIME/'controller.json',dict(status='completed',pid=os.getpid(),finished=time.time()));render()
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:p.terminate()
        write(RUNTIME/'controller.json',dict(status='interrupted_or_failed',pid=os.getpid(),error=str(exc)));render()
        raise


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    controller=read(RUNTIME/'controller.json') if (RUNTIME/'controller.json').exists() else {}
    lines=['# 三种YOLOE微调→Qwen27B（1shot）','',
        '两个场景各选训练10shot、视觉提示10shot（固定头不需要提示）的AP50最佳实际配置，共6组，测试各300张。',
        'Qwen每类仅1张原图+人工带框参考，无负例，seed0。直接复用原YOLOE预测，不重新训练或运行GPU推理。',
        'S0原始预测→B0候选score≥0.01/每类最多100框→B1筛选→B1D同类同实例去重→B2坐标修正。',
        '每请求最多20个候选；无候选不调用Qwen，uncertain保留。筛选禁止补框/改类/改分数；框修正保留候选集合、类别和分数。',
        '输入含原图、编号整图与1.5倍上下文裁图；整图最长边1280、裁图640、JPEG90。旧未微调级联保留作为历史对照。',
        '每场景并发2个图像任务，共最多4；逐请求记录usage及重试，验证成功的请求按完整输入哈希缓存。',
        'AP50最佳来自现有测试结果的探索性选择，输入尺寸/seed/训练范围随选中配置保留，不当作等预算独立验证。',
        '',f'控制器状态：`{controller.get("status","prepared_offline")}`。',
        '', '| 场景 | 微调方式 | 原始AP50 | 候选框 | 筛选批数 | 状态 | 完成/300 | B1 AP50 | B2 AP50 |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for j in jobs():
        s=read(state_path(j));m=s.get('metrics',{})
        values=[j['scene'],NAMES[j['method']],f'{100*j["source_AP50"]:.2f}',j['candidate_count'],j['filter_batches'],
            s['status'],s.get('completed_images',0),
            *[f'{100*m[k]["mAP50"]:.2f}' if k in m else '—' for k in ('filtered','refined')]]
        lines.append('| '+' | '.join(map(str,values))+' |')
    accounting=Counter()
    if (RUNTIME/'requests.jsonl').exists():
        for line in (RUNTIME/'requests.jsonl').read_text().splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            accounting['attempts']+=1
            accounting['failures']+=event['status']!='success'
            for key in ('input_tokens','cache_read_input_tokens','output_tokens'):
                accounting[key]+=(event.get('usage') or {}).get(key,0) or 0
    lines+=['',f'已落盘请求尝试{accounting["attempts"]}次（失败{accounting["failures"]}次）；'
        f'非缓存输入token {accounting["input_tokens"]:,}，缓存输入token {accounting["cache_read_input_tokens"]:,}，输出token {accounting["output_tokens"]:,}。'
        '仅统计收到usage的请求，不将缓存读入与非缓存输入按同价合计；实际费用以平台账单为准。']
    lines+=['','[第四部分最终比较表](../../../_shared/comparisons/final_results/04_yoloe_qwen.md)',
        '[前置10shot模型对比](../../../Qwen/comparisons/model10_supplement_20261103/README.md)',
        '', '命令：`python scripts/run_finetuned_yoloe_qwen.py --prepare`（离线）→ `--background`；`--status`查看。','']
    for j in jobs():lines.append(f'- {j["id"]}：[选中原始预测]({relative(ROOT/j["source_run"]/"evaluation.txt")}) · [本组结果]({relative(output(j)/"experiment_plan.json")})')
    if controller.get('error'):lines+=['','停止原因：'+controller['error']]
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def background():
    if not (RUNTIME/'plan.json').exists():raise ValueError('Run --prepare first.')
    previous=read(RUNTIME/'controller.json') if (RUNTIME/'controller.json').exists() else {}
    if live(previous.get('pid')):raise RuntimeError('Controller already queued or running.')
    with (RUNTIME/'controller.log').open('a') as log:
        p=subprocess.Popen([sys.executable,str(SCRIPT),'--run'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
    print('Detached controller PID:',p.pid)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for option in ('prepare','background','run','status'):g.add_argument('--'+option,action='store_true')
    g.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.prepare:prepare()
    elif args.background:background()
    elif args.run:run()
    elif args.worker is not None:asyncio.run(worker(args.worker))
    else:render();print((REPORT/'README.md').read_text())

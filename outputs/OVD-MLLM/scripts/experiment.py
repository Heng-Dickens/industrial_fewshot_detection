"""Paired YOLOE -> Qwen filtering, deduplication and coordinate refinement."""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(Path(__file__).parent)]
from generate_candidates import folder, read, sha, write
from industrial_fewshot.evaluation.detection_metrics import box_iou
from industrial_fewshot.evaluation.evaluator import evaluate_records
from industrial_fewshot.evaluation.reporting import format_metrics
from industrial_fewshot.postprocess.schema import clean_json_response
from scripts.analyze_box_precision import match_counts, tolerance_ap

STAMP = 'yoloe_v8l_10shot_qwen38_v1_20261030'
MODEL = dict(model='qwen3.8-27b', base_url='http://127.0.0.1:15721/v1',
             proxy_request_model='qwen3.8-27b',
             api_protocol='anthropic_messages', temperature=.1, top_p=.9,
             max_tokens=8192, stream=False, enable_thinking=False)
FILTER_PROMPT = '''你是工业目标候选审核员。参考图仅定义类别，不是待测目标。
输入包含无标记原图、编号框整图和候选上下文裁图。已有框和类别只是待审假设。
逐个审核 requested_ids，必须完整且恰好返回一次：
keep：可见结构支持当前类别；remove：明确是背景、非目标或不属于当前类别；
uncertain：模糊、遮挡、过小、边界或类别依据不足。不要猜测，不因框不准而删真实目标。
同一真实目标可能有多个框。仅当能明确确认同类、同一实例时，在保留候选中填写
duplicate_of 指向另一编号；两个相邻目标绝不能仅因重叠而认作重复。
去重代表框由程序按原分数选择，你无需选择更好的定位框。
不能改类别、坐标和分数，不能补框。不要把人、货物、地面设备当作车辆。
对于吊物，须按场景定义判断被吊起物及相连吊装组件，不能仅凭吊索或空钩判断成立。
只输出JSON：{"reviews":[{"candidate_id":"编号","decision":"keep/remove/uncertain",
"duplicate_of":null,"reason":"不超过30字的可见依据"}]}。duplicate_of只能使用本批输入编号。
'''
REFINE_PROMPT = '''你是工业目标边界复核员。以下候选已经完成目标确认与去重。
本轮固定候选集合、类别和分数，只允许修正同一目标的边界；不得删框、补框、改类。
结合无标记原图、编号框整图、上下文裁图和类别参考，紧贴目标的可见整体边界。
车辆按参考标注包含对应门架、货叉/专用属具等部件，避免邻近人员和背景；
吊物按参考标注包含被吊起物及相连吊装组件，不能只框吊索，不能擅自猜测遮挡边界。
坐标统一是【整张原图】归一化0~1000的[x1,y1,x2,y2]，绝不是裁图坐标。
无法可靠改进时action=keep且bbox_1000=null；否则action=refine并给出坐标。
逐个requested_ids完整返回，不输出额外编号。只输出JSON：
{"reviews":[{"candidate_id":"编号","action":"keep/refine","bbox_1000":null,
"reason":"不超过30字的边界依据"}]}。
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def select_objects(objects, threshold, limit):
    selected = sorted((p for p in objects if p['confidence'] >= threshold), key=lambda p:-p['confidence'])[:limit]
    selected_ids = {p['candidate_id'] for p in selected}
    return [copy.deepcopy(p) for p in objects if p['candidate_id'] in selected_ids]


def samples_for(items, records):
    by_path = {r['relative_path']:r for r in records}
    if set(by_path) != {s['relative_path'] for s in items} or len(by_path) != len(records):
        raise ValueError('Prediction/image matrix is incomplete or duplicated')
    return [dict(s, predictions=by_path[s['relative_path']]['objects']) for s in items]


def freeze_selection(scene, stamp, threshold=.01):
    output = folder(scene, 'val100', stamp)
    path = output / 'selection.json'
    if not 0 < threshold <= 1:
        raise ValueError('Candidate confidence must be in (0,1]')
    if path.exists():
        chosen = read(path)
        if chosen['source_sha256'] != sha(output / 'source_records.jsonl'):
            raise ValueError('Source predictions changed after selection')
        if chosen.get('selection_policy') == 'user_fixed_floor':
            if chosen['threshold'] != threshold and ((output/'experiment_plan.json').exists() or (output/'pilot_verification.json').exists()):
                raise ValueError('User threshold changed: use a new stamp')
            if chosen['threshold'] == threshold:
                if not (output/'experiment_plan.json').exists():
                    chosen.pop('grid',None)
                    chosen.pop('dev_source_counts',None)
                    write(path,chosen)
                return chosen
        if (output/'experiment_plan.json').exists():
            raise ValueError('A different threshold was already used: use a new stamp')
    cohort = read(output / 'cohort.json')
    source = load_records(output / 'source_records.jsonl')
    labels = read(output / 'source_complete.json')['labels']
    initial = match_counts(samples_for(cohort['samples'], source), labels, 0.)
    truth_count = initial['ground_truth_count']
    records = [dict(r, objects=select_objects(r['objects'], threshold, 500)) for r in source]
    counts = match_counts(samples_for(cohort['samples'], records), labels, 0.)
    chosen = dict(threshold=threshold, limit=500, candidates=sum(len(r['objects']) for r in records),
                  recall50=counts['tp50']/truth_count if truth_count else 0.,
                  recall85=counts['tp85']/truth_count if truth_count else 0.,
                  box20_coverage=counts['joint20_gt_coverage'],
                  lost_recall50=(initial['tp50']-counts['tp50'])/truth_count if truth_count else 0.,
                  selection_policy='user_fixed_floor',
                  criterion=f'User fixed score>={threshold}; no additional global truncation beyond source 100/class (500 total). Dev recall loss is reported, not optimized away.',
                  source_sha256=sha(output / 'source_records.jsonl'))
    write(path, chosen)
    return chosen


def make_batches(objects, size=20):
    """Keep overlapping candidates together; large groups retain a context anchor."""
    indices = set(range(len(objects)))
    groups = []
    while indices:
        seed = min(indices); indices.remove(seed)
        group = [seed]; queue = [seed]
        while queue:
            i = queue.pop()
            neighbors = [j for j in indices if box_iou(objects[i]['bbox_pixels_xyxy'], objects[j]['bbox_pixels_xyxy']) >= .5]
            for j in neighbors:
                indices.remove(j); group.append(j); queue.append(j)
        groups.append(sorted(group, key=lambda i:-objects[i]['confidence']))
    result, pending = [], []
    for group in groups:
        if len(group) > size:
            if pending:
                result.append((pending, [])); pending = []
            for offset in range(0, len(group), size):
                chunk = group[offset:offset+size]
                result.append(([objects[i] for i in chunk], [objects[group[0]]] if offset else []))
        else:
            if len(pending) + len(group) > size:
                result.append((pending, [])); pending = []
            pending.extend(objects[i] for i in group)
    if pending:
        result.append((pending, []))
    return result


def validate_reviews(value, requested, allowed, phase):
    reviews = value.get('reviews')
    if not isinstance(reviews, list) or any(not isinstance(r, dict) for r in reviews):
        raise ValueError('reviews must be an object array')
    ids = [r.get('candidate_id') for r in reviews]
    if len(ids) != len(set(ids)) or set(ids) != set(requested):
        raise ValueError('Candidate IDs missing, duplicated or fabricated')
    for r in reviews:
        if not isinstance(r.get('reason'), str) or not r['reason'].strip():
            raise ValueError('Missing reason')
        if phase == 'filter':
            if r.get('decision') not in {'keep','remove','uncertain'}:
                raise ValueError('Invalid filtering decision')
            duplicate = r.get('duplicate_of')
            if duplicate is not None and (duplicate not in allowed or duplicate == r['candidate_id']):
                raise ValueError('Invalid duplicate reference')
        else:
            if r.get('action') not in {'keep','refine'}:
                raise ValueError('Invalid refinement action')
            if r['action'] == 'refine':
                b = r.get('bbox_1000')
                if not isinstance(b, list) or len(b)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in b):
                    raise ValueError('Invalid box coordinates')
                if not (0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000):
                    raise ValueError('Refined box out of image')
    return {r['candidate_id']:r for r in reviews}


def parse_review_response(raw):
    """Accept literal newlines in model reason strings, never invent structure."""
    try:
        return clean_json_response(raw),False
    except json.JSONDecodeError:
        start,end=raw.find('{'),raw.rfind('}')
        if start<0 or end<=start:raise
        value=json.loads(raw[start:end+1],strict=False)
        if not isinstance(value,dict):raise ValueError('Response is not an object')
        return value,True


def apply_filter(objects, reviews):
    return [copy.deepcopy(p) for p in objects if reviews[p['candidate_id']]['decision'] != 'remove']


def deduplicate(objects, reviews):
    """Only VLM-confirmed, same-class overlap; retain highest original score."""
    by_id = {p['candidate_id']:p for p in objects}
    parent = {k:k for k in by_id}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    edges = []
    for key,p in by_id.items():
        other = reviews[key].get('duplicate_of')
        if (other not in by_id or reviews[key]['decision'] != 'keep' or reviews[other]['decision'] != 'keep'
                or p['label'] != by_id[other]['label']):
            continue
        if box_iou(p['bbox_pixels_xyxy'], by_id[other]['bbox_pixels_xyxy']) < .5:
            continue
        parent[find(key)] = find(other)
        edges.append([key, other])
    groups = {}
    for key in by_id:
        groups.setdefault(find(key), []).append(key)
    winners = {max(ids, key=lambda k:(by_id[k]['confidence'], -list(by_id).index(k))) for ids in groups.values()}
    return [copy.deepcopy(p) for p in objects if p['candidate_id'] in winners], edges


def apply_refinement(objects, reviews, width, height):
    result = []
    for obj in objects:
        p = copy.deepcopy(obj); r = reviews[p['candidate_id']]
        if r['action'] == 'refine':
            b = r['bbox_1000']
            p['bbox_pixels_xyxy'] = [b[0]*width/1000, b[1]*height/1000, b[2]*width/1000, b[3]*height/1000]
            p['bbox_2d'] = list(b)
        result.append(p)
    return result


def jpeg(image, side=1280):
    image = image.copy().convert('RGB'); image.thumbnail((side,side))
    out = io.BytesIO(); image.save(out, format='JPEG', quality=88)
    return out.getvalue()


def image_block(data):
    import base64
    return {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()}}


def reference_panel(references):
    from PIL import Image, ImageDraw, ImageFont
    first = {}
    for ref in references:
        first.setdefault(ref['label'], ref)
    panel = Image.new('RGB', (320*len(first), 300), 'white')
    for i,(label,ref) in enumerate(first.items()):
        if sha(ref['image_path']) != ref['image_sha256']:
            raise ValueError('Reference image changed')
        with Image.open(ref['image_path']) as source:
            box = ref['bboxes'][0]
            crop = source.convert('RGB').crop(tuple(map(int,box)))
            crop.thumbnail((310,260))
            panel.paste(crop,(i*320+(320-crop.width)//2,30+(260-crop.height)//2))
        ImageDraw.Draw(panel).text((i*320+8,8), label, fill='black')
    return jpeg(panel,1600)


def request_content(image_path, sample, objects, context, ref_image, definitions, phase):
    from PIL import Image, ImageDraw
    data = image_path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=sample['image_sha256']:
        raise ValueError('Query image changed')
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert('RGB')
    overlay = image.copy(); draw = ImageDraw.Draw(overlay)
    try:
        from PIL import ImageFont
        font=ImageFont.truetype('DejaVuSans.ttf',24)
    except OSError:
        font=None
    shown = objects+context
    for p in shown:
        b = p['bbox_pixels_xyxy']; draw.rectangle(b, outline='red', width=4)
        draw.text((b[0]+3,b[1]+3), p['candidate_id']+' '+p['label'], fill='yellow', stroke_width=1, stroke_fill='black',font=font)
    listing = [dict(candidate_id=p['candidate_id'], label=p['label'],
                    bbox_1000=[round(v/(image.width if j%2==0 else image.height)*1000,3) for j,v in enumerate(p['bbox_pixels_xyxy'])]) for p in shown]
    prompt = (FILTER_PROMPT if phase=='filter' else REFINE_PROMPT)
    prompt += '\n类别定义：'+json.dumps(definitions,ensure_ascii=False)
    prompt += '\nrequested_ids='+json.dumps([p['candidate_id'] for p in objects])
    prompt += '\ncontext_ids='+json.dumps([p['candidate_id'] for p in context])
    prompt += '\n候选信息='+json.dumps(listing,ensure_ascii=False)
    content = [dict(type='text',text=prompt),dict(type='text',text='类别参考图，不是待测图'),image_block(ref_image),
               dict(type='text',text='待测无标记原图'),image_block(jpeg(image)),
               dict(type='text',text='待测编号框整图'),image_block(jpeg(overlay))]
    for p in shown:
        x1,y1,x2,y2 = p['bbox_pixels_xyxy']; dx=(x2-x1)*.25; dy=(y2-y1)*.25
        window=(max(0,int(x1-dx)),max(0,int(y1-dy)),min(image.width,math.ceil(x2+dx)),min(image.height,math.ceil(y2+dy)))
        content.extend([dict(type='text',text=f"候选{p['candidate_id']}上下文裁图；窗口原图像素{window}，坐标输出仍相对整图"),
                        image_block(jpeg(image.crop(window),640))])
    return content, prompt, listing


async def review_stage(output, records, phase, definitions, concurrency=4):
    import httpx
    from openai import AsyncOpenAI
    from industrial_fewshot.detectors.claude_messages import complete_message
    from industrial_fewshot.service_errors import ModelServiceError
    cohort = read(output / 'cohort.json')
    samples = {s['relative_path']:s for s in cohort['samples']}
    refs = read(output / 'references.json')['examples']
    reference = reference_panel(refs)
    semaphore = asyncio.Semaphore(concurrency)
    runtime = ROOT / 'outputs/_shared/runtime/ovd_mllm_filtering_20261030' / output.parent.name / output.name / phase
    runtime.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(base_url=MODEL['base_url'],api_key='EMPTY',timeout=180.,max_retries=0,
                         http_client=httpx.AsyncClient(timeout=180.,trust_env=False))
    jobs = []
    reviews_by_path = {r['relative_path']:{} for r in records}
    accounting = dict(requests=0, cached=0, recovered_cached=0, failed=0, retries=0, input_tokens=0, output_tokens=0, seconds=0.)
    completed = 0; started = time.perf_counter()
    async def one(record, objects, context):
        nonlocal completed
        async with semaphore:
            sample = samples[record['relative_path']]
            signature = digest(dict(model=MODEL,phase=phase,prompt=FILTER_PROMPT if phase=='filter' else REFINE_PROMPT,
                definitions=definitions,reference_sha=hashlib.sha256(reference).hexdigest(),
                image_sha256=sample['image_sha256'],objects=objects,context=context,
                requested_ids=[p['candidate_id'] for p in objects],input_protocol='original+overlay24px+context1.5.crop640.v2'))
            cache = runtime / f'{signature}.json'
            if cache.exists():
                previous=read(cache)
                if previous.get('status')=='failed' and previous.get('signature')==signature:
                    for attempt in reversed(previous.get('attempts',[])):
                        if not isinstance(attempt.get('raw_response'),str):continue
                        try:
                            recovered,repaired=parse_review_response(attempt['raw_response'])
                            validate_reviews(recovered,[p['candidate_id'] for p in objects],
                                {p['candidate_id'] for p in objects+context},phase)
                        except (ValueError,TypeError,KeyError):continue
                        previous.update(status='success',response=recovered,response_format_recovered=repaired,
                            recovered_from_saved_response=True,requested_ids=[p['candidate_id'] for p in objects])
                        write(cache,previous);accounting['recovered_cached']+=1;break
            if cache.exists() and read(cache).get('status')=='success':
                entry=read(cache); accounting['cached']+=1
                checked=validate_reviews(entry['response'],[p['candidate_id'] for p in objects],
                    {p['candidate_id'] for p in objects+context},phase)
            else:
                content,prompt,listing=request_content(Path(cohort['image_root'])/sample['relative_path'],sample,
                    objects,context,reference,definitions,phase)
                attempts=read(cache).get('attempts',[]) if cache.exists() else []
                attempt_start=len(attempts);checked=None
                for attempt in range(2):
                    t=time.perf_counter(); accounting['requests']+=1
                    raw=None
                    feedback=None
                    request_blocks=content
                    if attempt>0 or attempt_start:
                        feedback=('上次响应未通过JSON/编号校验。只修复输出格式：必须对requested_ids逐条返回，'
                            '不能输出{}，不能遗漏或增加编号。没有目标也应逐条给remove；'
                            'reason中的换行须转义。requested_ids='+json.dumps([p['candidate_id'] for p in objects]))
                        request_blocks=content+[dict(type='text',text=feedback)]
                    if accounting['requests'] == 1 or accounting['requests']%10==0:
                        write(runtime/'status.json',dict(phase=phase,completed=completed,total=len(jobs),**accounting))
                        print(output.parent.name,phase,'request',accounting['requests'],'image',record['relative_path'],flush=True)
                    try:
                        raw,finish,usage,_=await complete_message(client,MODEL,[{'role':'user','content':request_blocks}],{'type':'json_object'})
                        response,repaired=parse_review_response(raw)
                        checked=validate_reviews(response,[p['candidate_id'] for p in objects],
                            {p['candidate_id'] for p in objects+context},phase)
                        attempts.append(dict(attempt=attempt_start+attempt+1,status='success',raw_response=raw,finish_reason=finish,
                                             response_format_recovered=repaired,
                                             format_retry_feedback=feedback,
                                             usage=usage,seconds=time.perf_counter()-t))
                        usage=usage or {}; accounting['input_tokens']+=usage.get('input_tokens',0)
                        accounting['output_tokens']+=usage.get('output_tokens',0)
                        write(cache,dict(status='success',signature=signature,model=MODEL,phase=phase,
                            relative_path=record['relative_path'],image_sha256=sample['image_sha256'],
                            prompt=prompt,candidates=listing,reference_sources=[r['image_path'] for r in refs[::10]],
                            requested_ids=[p['candidate_id'] for p in objects],response=response,attempts=attempts))
                        break
                    except ModelServiceError:
                        raise
                    except Exception as exc:
                        if getattr(exc,'status_code',None) in {401,403,404}:
                            raise ModelServiceError(f'Model endpoint returned HTTP {exc.status_code}; stopped before bulk retries') from None
                        attempts.append(dict(attempt=attempt_start+attempt+1,status='failed',error=type(exc).__name__+': '+str(exc)[:350],
                                             format_retry_feedback=feedback,
                                             seconds=time.perf_counter()-t,raw_response=locals().get('raw')))
                        if attempt==0:
                            accounting['retries']+=1; await asyncio.sleep(1)
                    finally:
                        accounting['seconds']+=time.perf_counter()-t
                if checked is None:
                    accounting['failed']+=1
                    checked={p['candidate_id']:dict(candidate_id=p['candidate_id'],
                        **({'decision':'uncertain','duplicate_of':None} if phase=='filter' else {'action':'keep','bbox_1000':None}),
                        reason='request_failed: retain original',review_failed=True) for p in objects}
                    write(cache,dict(status='failed',signature=signature,relative_path=record['relative_path'],attempts=attempts))
            reviews_by_path[record['relative_path']].update(checked)
            completed+=1
            if completed%10==0 or completed==len(jobs):
                write(runtime/'status.json',dict(phase=phase,completed=completed,total=len(jobs),**accounting))
                print(output.parent.name,phase,completed,'/',len(jobs),'failed',accounting['failed'],flush=True)
    for record in records:
        for objects,context in make_batches(record['objects']):
            jobs.append((record,objects,context))
    tasks=[asyncio.create_task(one(*job)) for job in jobs]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        raise
    finally:
        await client.close()
    accounting['wall_seconds']=time.perf_counter()-started
    entries=[dict(relative_path=path,reviews=list(reviews.values())) for path,reviews in reviews_by_path.items()]
    (output/f'{phase}_reviews.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in entries))
    write(output/f'{phase}_accounting.json',accounting)
    return reviews_by_path,accounting


def stage_metrics(output, name, records, selected, model_metadata):
    cohort=read(output/'cohort.json'); labels=read(output/'source_complete.json')['labels']
    settings=dict(iou_threshold=.5,score_threshold=.5,score_thresholds=[.001,.01,.03,.05,.1,.5,.7],max_detections_per_image=100)
    metrics=evaluate_records(cohort['samples'],records,labels,settings)
    tolerance=tolerance_ap(samples_for(cohort['samples'],records),labels)
    metrics['AP_box20']=tolerance['AP_box20']
    for label,values in tolerance['per_class'].items(): metrics['per_class'][label]['AP_box20']=values['AP_box20']
    for level in (85,90,95):
        values=[r['ap_by_iou'].get(str(level/100)) for r in metrics['per_class'].values() if r['ground_truth_count']]
        metrics[f'AP{level}']=sum(values)/len(values) if values else None
    target=output/name; target.mkdir(exist_ok=True)
    (target/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write(target/'metrics.json',metrics)
    (target/'evaluation.txt').write_text(format_metrics(metrics)+'\n')
    run=dict(scene_name=output.parent.name,backend='ovd_mllm',stage=name,
        image_dir=cohort['image_root'],dataset_manifest=str(output/'cohort.json'),
        dataset_sha256=cohort['dataset_sha256'],image_paths=[s['relative_path'] for s in cohort['samples']],
        image_count=len(records),labels=labels,targets=[dict(id='all',labels=labels,backend='OVD-MLLM',
            model='yoloe-v8l-seg' if name in {'source','baseline'} else 'yoloe-v8l-seg+qwen3.8-27b')],evaluation=settings,
        candidate_policy=selected,model=model_metadata,run_dir=str(target),
        result_layout=dict(prompt_version='qwen_filter_v1_conf01'),
        experiment=dict(method='OVD-MLLM',model='yoloe-v8l-seg -> qwen3.8-27b',stage=name,
                        positive_count=10,negative_count=0,variant=name,shot_per_class=10))
    write(target/'run.json',run)
    write(target/'experiment.json',dict(method='OVD-MLLM',dataset=output.parent.name,
        stage=name,source_run=str(output),protocol='Frozen original scores/classes; paired candidate filtering/dedup/refinement.'))
    prompt=REFINE_PROMPT if name=='refined' else FILTER_PROMPT if name in {'filtered','deduplicated'} else ''
    write(target/'prompt.json',dict(text=prompt,text_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),model=MODEL,
        reference_shot_per_class=1 if prompt else 0,input_protocol='original+overlay24px+context1.5.crop640.v2'))
    return metrics


def record_with(record, objects):
    return dict(record,objects=objects,target_count=len(objects))


def assert_subset(before,after):
    by_id={p['candidate_id']:p for p in before}
    ids=[p['candidate_id'] for p in after]
    if len(ids)!=len(set(ids)) or any(p!=by_id.get(p['candidate_id']) for p in after):
        raise ValueError('Filtering changed candidate coordinates, class, confidence, or identity')
    if ids != [p['candidate_id'] for p in before if p['candidate_id'] in set(ids)]:
        raise ValueError('Filtering reordered candidates')


def deletion_diagnostics(items,baseline,after,labels,score=.01,iou=.5,max_dets=100):
    """Distinguish deletion of a baseline TP from net recall loss after rematching."""
    before_by_path={r['relative_path']:r['objects'] for r in baseline}
    after_by_path={r['relative_path']:r['objects'] for r in after}
    result=dict(baseline_tp=0,baseline_fp=0,deleted_baseline_tp=0,deleted_baseline_fp=0,deleted_correct_candidates=[])
    for sample in items:
        path=sample['relative_path'];kept={p['candidate_id'] for p in after_by_path[path]}
        assert_subset(before_by_path[path],after_by_path[path])
        for label in labels:
            truth=[g for g in sample['ground_truth'] if g['label']==label];claimed=set()
            ranked=sorted((p for p in before_by_path[path] if p['label']==label),key=lambda p:-p['confidence'])[:max_dets]
            for p in ranked:
                if p['confidence']<score:continue
                eligible=[(box_iou(p['bbox_pixels_xyxy'],g['bbox_pixels_xyxy']),j) for j,g in enumerate(truth)
                          if j not in claimed and box_iou(p['bbox_pixels_xyxy'],g['bbox_pixels_xyxy'])>=iou]
                is_tp=bool(eligible)
                if is_tp:claimed.add(max(eligible)[1])
                result['baseline_tp' if is_tp else 'baseline_fp']+=1
                if p['candidate_id'] not in kept:
                    result['deleted_baseline_tp' if is_tp else 'deleted_baseline_fp']+=1
                    if is_tp:result['deleted_correct_candidates'].append(dict(relative_path=path,candidate_id=p['candidate_id'],label=label))
    result['baseline_tp_deleted_rate']=result['deleted_baseline_tp']/result['baseline_tp'] if result['baseline_tp'] else None
    result['baseline_fp_deleted_rate']=result['deleted_baseline_fp']/result['baseline_fp'] if result['baseline_fp'] else None
    return result


def report(output, results, audit):
    write(output/'experiment_results.json',dict(dataset=output.parent.name,rows=results,audit=audit))
    columns=['stage','candidates','mAP50','mAP75','mAP50_95','AP85','AP90','AP95','AP_box20','precision','recall','f2','tp','fp','fn']
    with (output/'experiment_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=columns);w.writeheader();w.writerows({k:r.get(k) for k in columns} for r in results)
    lines=['# YOLOE → Qwen3.8-27B 配对实验','',f'数据集：{output.parent.name}。YOLOE原始10shot视觉提示，候选下限0.001，seed=0；非新微调权重。',
        '',f"用户指定候选阈值：{audit['selection']['threshold']}；全图上限：{audit['selection']['limit']}。开发集仅核查召回变化，不再搜索阈值。",
        '大模型参考：每类1张固定支持图裁图；temperature=0.1，整图+编号框+上下文裁图，原分数不展示。',
        'B1只筛选；B1D在B1基础上按Qwen确认的同类同实例关系去重，保留原分数最高的框；B2只改B1D框坐标。',
        'AP用全部保存分数，P/R/F2及TP/FP/FN为score≥0.5、IoU≥0.5。AP/P/R/F2以百分数显示。','',
        '| 阶段 | 框数 | AP50 | AP75 | AP50:95 | AP85 | AP90 | AP95 | 四项≤20% AP | P | R | F2 | TP/FP/FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in results:
        values=[f"{row[k]*100:.2f}" if row.get(k) is not None else '—' for k in columns[2:12]]
        lines.append('| '+' | '.join([row['stage'],str(row['candidates']),*values,f"{row['tp']}/{row['fp']}/{row['fn']}"])+' |')
    baseline=next((r for r in results if r['stage']=='baseline'),None)
    if baseline:
        lines+=['','与B0比较（百分点）：','']
        for r in results:
            if r['stage'] in {'filtered','deduplicated'}:
                lines.append(f"- {r['stage']}：AP50 {100*(r['mAP50']-baseline['mAP50']):+.2f}，AP50:95 {100*(r['mAP50_95']-baseline['mAP50_95']):+.2f}，召回 {100*(r['recall']-baseline['recall']):+.2f}；FP净减少 {baseline['fp']-r['fp']}。")
    prior=next((r for r in results if r['stage']=='deduplicated'),None)
    after=next((r for r in results if r['stage']=='refined'),None)
    if prior and after:
        lines+=['',f"B2−B1D：AP50:95 {100*(after['mAP50_95']-prior['mAP50_95']):+.2f}，AP85 {100*(after['AP85']-prior['AP85']):+.2f} 个百分点。"]
    lines+=['','## 审核审计','', '```json',json.dumps(audit,ensure_ascii=False,indent=2),'```','',
        '限制：固定测试集已有历史结果；开发/测试可能共享视频序列，详见split_audit.json，不宣称序列独立泛化。',
        '已保留不确定与请求失败的原框；失败率见审计，存在失败时不能称为完整成功的Qwen审核。',
        '各阶段metrics.json含完整逐类和各置信度指标；原始请求与回答位于outputs/_shared/runtime/ovd_mllm_filtering_20261030。']
    (output/'experiment_results.md').write_text('\n'.join(lines)+'\n')
    parent=output.parent
    (parent/'experiment_results.md').write_text('# OVD-MLLM实验索引\n\n'+''.join(
        f'- [{p.parent.name}]({p.parent.name}/experiment_results.md)\n' for p in sorted(parent.glob('*/experiment_results.md'))))
    from summarize_experiment import summary
    summary()


async def run_split(scene,split,stamp,selection,refine,concurrency):
    import yaml
    output=folder(scene,split,stamp); cohort=read(output/'cohort.json')
    config=yaml.safe_load((ROOT/f'configs/common/{"chache_val300" if scene=="chache" else "diaowu"}.yaml').read_text())
    definitions=config['labels']
    source=load_records(output/'source_records.jsonl')
    baseline=[record_with(r,select_objects(r['objects'],selection['threshold'],selection['limit'])) for r in source]
    plan=dict(schema='ovd_mllm.paired.v1',model=MODEL,selection=selection,
        source_sha256=sha(output/'source_records.jsonl'),manifest_sha256=sha(output/'cohort.json'),
        definitions=definitions,filter_prompt=FILTER_PROMPT,refine_prompt=REFINE_PROMPT,
        stages=['source','baseline','filtered','deduplicated','refined'],dedup_min_iou=.5,dedup_same_class=True,
        review_batch_size=20,input_protocol='original+overlay24px+context1.5.crop640.v2')
    if (output/'experiment_plan.json').exists() and read(output/'experiment_plan.json')!=plan:
        raise ValueError('Experiment inputs changed: choose new stamp')
    write(output/'experiment_plan.json',plan)
    results=[]; audit=dict(selection={k:v for k,v in selection.items() if k not in {'grid','dev_source_counts'}},
        split_audit=read(output/'split_audit.json'))
    def evaluate(name,records):
        m=stage_metrics(output,name,records,selection,MODEL)
        row={k:m.get(k) for k in ['mAP50','mAP75','mAP50_95','AP85','AP90','AP95','AP_box20','precision','recall','f2','tp','fp','fn']}
        row.update(stage=name,candidates=sum(len(r['objects']) for r in records));results.append(row)
        report(output,results,audit)
    evaluate('source',source);evaluate('baseline',baseline)
    reviews,accounting=await review_stage(output,baseline,'filter',definitions,concurrency)
    filtered=[];dedup=[];decision_counts={};edges=[]
    for record in baseline:
        rr=reviews[record['relative_path']]
        for review in rr.values():decision_counts[review['decision']]=decision_counts.get(review['decision'],0)+1
        objects=apply_filter(record['objects'],rr); unique,pairs=deduplicate(objects,rr)
        assert_subset(record['objects'],objects);assert_subset(objects,unique)
        filtered.append(record_with(record,objects));dedup.append(record_with(record,unique))
        edges.extend(dict(relative_path=record['relative_path'],edge=p) for p in pairs)
    audit.update(filter_accounting=accounting,decisions=decision_counts,duplicate_edges=len(edges),
                 invariant_check='passed: retained coordinates/classes/scores/order unchanged')
    write(output/'dedup_edges.json',{'edges':edges})
    evaluate('filtered',filtered);evaluate('deduplicated',dedup)
    if refine:
        rr,accounting=await review_stage(output,dedup,'refine',definitions,concurrency)
        refined=[];changed=0
        by_path={s['relative_path']:s for s in cohort['samples']}
        for record in dedup:
            s=by_path[record['relative_path']]
            objects=apply_refinement(record['objects'],rr[record['relative_path']],s['width'],s['height'])
            if [(p['candidate_id'],p['label'],p['confidence']) for p in objects]!=[(p['candidate_id'],p['label'],p['confidence']) for p in record['objects']]:
                raise ValueError('Refinement changed candidate set, class or confidence')
            changed+=sum(a['bbox_pixels_xyxy']!=b['bbox_pixels_xyxy'] for a,b in zip(objects,record['objects']))
            refined.append(record_with(record,objects))
        audit.update(refine_accounting=accounting,refined_boxes=changed)
        evaluate('refined',refined)
    if sha(output/'source_records.jsonl')!=plan['source_sha256'] or sha(output/'cohort.json')!=plan['manifest_sha256']:
        raise ValueError('Source data changed during experiment')
    failed=accounting['failed']+audit['filter_accounting']['failed'] if refine else accounting['failed']
    write(output/'verification.json',dict(status='passed' if not failed else 'incomplete_with_fallback',
        all_images_evaluated=len(source),stages=len(results),retained_fields_verified=True,
        refinement_candidate_set_fixed=True,failed_requests=failed))
    from summarize_experiment import summary
    summary()
    print('COMPLETE',scene,split,flush=True)


async def main(args):
    selection=freeze_selection(args.scene,args.stamp,args.threshold)
    if args.pilot:
        import yaml
        output=folder(args.scene,'val100',args.stamp)
        cohort=read(output/'cohort.json')
        config=yaml.safe_load((ROOT/f'configs/common/{"chache_val300" if args.scene=="chache" else "diaowu"}.yaml').read_text())
        source=load_records(output/'source_records.jsonl')
        source=sorted(source,key=lambda r:digest(r['image_sha256']))[:args.pilot]
        baseline=[record_with(r,select_objects(r['objects'],selection['threshold'],selection['limit'])) for r in source]
        reviews,accounting=await review_stage(output,baseline,'filter',config['labels'],args.concurrency)
        filtered=[];dedup=[]
        for r in baseline:
            objects=apply_filter(r['objects'],reviews[r['relative_path']]);unique,_=deduplicate(objects,reviews[r['relative_path']])
            assert_subset(r['objects'],objects);assert_subset(objects,unique)
            filtered.append(record_with(r,objects));dedup.append(record_with(r,unique))
        paths={r['relative_path'] for r in baseline}
        items=[s for s in cohort['samples'] if s['relative_path'] in paths]
        labels=read(output/'source_complete.json')['labels']
        metrics={name:evaluate_records(items,records,labels,dict(iou_threshold=.5,score_threshold=.5,max_detections_per_image=100))
                 for name,records in [('baseline',baseline),('filtered',filtered),('deduplicated',dedup)]}
        result=dict(status='passed' if not accounting['failed'] else 'failed_requests',images=len(items),
                    paths=sorted(paths),accounting=accounting,metrics=metrics,retained_fields_verified=True,
                    note='Development-only smoke; no claim of full cohort performance.')
        write(output/'pilot_verification.json',result)
        print('PILOT',result['status'],'images',len(items),'accounting',accounting,flush=True)
        if accounting['failed']:
            raise RuntimeError('Pilot has failed requests; resolve before full run')
        return
    splits=['val100','val300'] if args.split=='both' else [args.split]
    for split in splits:
        await run_split(args.scene,split,args.stamp,selection,args.refine,args.concurrency)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['chache','diaowu'],required=True)
    p.add_argument('--split',choices=['val100','val300','both'],default='both')
    p.add_argument('--stamp',default=STAMP)
    p.add_argument('--refine',action='store_true')
    p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--pilot',type=int,default=0,help='Run only a deterministic development smoke subset, cache reusable.')
    p.add_argument('--threshold',type=float,default=.01,help='User-authorized fixed candidate floor; default 0.01.')
    asyncio.run(main(p.parse_args()))

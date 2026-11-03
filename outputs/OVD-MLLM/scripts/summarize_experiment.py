"""One compact, honest cross-scene table from the paired experiment artifacts."""
from pathlib import Path
import csv
import json
import os
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from generate_candidates import folder,read,write

STAMP='yoloe_v8l_10shot_qwen38_v1_20261030'
REPORT=ROOT/'outputs/OVD-MLLM/comparisons/yoloe_qwen_filtering_20261030'
TITLES={'source':'S0 完整YOLOE','baseline':'B0 score≥0.01','filtered':'B1 仅筛选',
        'deduplicated':'B1D 筛选+去重','refined':'B2 再改坐标'}


def _summary_unlocked():
    REPORT.mkdir(parents=True,exist_ok=True)
    rows=[];states=[];costs=[]
    for scene in ('chache','diaowu'):
        for split in ('val100','val300'):
            root=folder(scene,split,STAMP)
            if not (root/'source_complete.json').exists():continue
            complete=read(root/'source_complete.json')
            verification=read(root/'verification.json') if (root/'verification.json').exists() else {}
            pilot=read(root/'pilot_verification.json') if (root/'pilot_verification.json').exists() else {}
            state=dict(dataset=f'{scene}_{split}',images=complete['images'],
                source_candidates=complete['predictions'],pilot_status=pilot.get('status','待完成'),
                pipeline_status=verification.get('status','未完成'),run=str(root.relative_to(ROOT)))
            if split=='val300':
                dev=folder(scene,'val100',STAMP)/'pilot_verification.json'
                state['pilot_status']='复用开发集验证' if dev.exists() and read(dev).get('status')=='passed' else '待开发验证'
            states.append(state)
            result=read(root/'experiment_results.json') if (root/'experiment_results.json').exists() else {}
            for phase in ('filter','refine'):
                accounting=result.get('audit',{}).get(phase+'_accounting')
                if accounting:
                    costs.append(dict(dataset=state['dataset'],phase=phase,**accounting))
            for r in result.get('rows',[]):
                row=dict(dataset=state['dataset'],split='开发集' if split=='val100' else '历史固定测试集',
                    status=state['pipeline_status'],threshold=.001 if r['stage']=='source' else .01,
                    **r,report=str((root/'experiment_results.md').relative_to(ROOT)))
                metrics_path=root/r['stage']/'metrics.json'
                if metrics_path.exists():
                    point=read(metrics_path).get('score_threshold_results',{}).get('0.01',{})
                    row.update({f'{key}_score01':point.get(key) for key in ['tp','fp','fn','precision','recall','f2']})
                baseline=next((b for b in result['rows'] if b['stage']=='baseline'),None)
                if baseline and r['stage'] in {'filtered','deduplicated'}:
                    row.update(delta_AP50_pp=100*(r['mAP50']-baseline['mAP50']),
                        delta_AP50_95_pp=100*(r['mAP50_95']-baseline['mAP50_95']),
                        delta_recall_pp=100*(r['recall']-baseline['recall']),fp_removed=baseline['fp']-r['fp'])
                rows.append(row)
    cumulative=[]
    runtime=ROOT/'outputs/_shared/runtime/ovd_mllm_filtering_20261030'
    grouped={}
    for path in runtime.glob(f'*/{STAMP}/*/*.json'):
        entry=read(path)
        if not entry.get('signature') or not isinstance(entry.get('attempts'),list):continue
        dataset,_,phase,_=path.relative_to(runtime).parts
        totals=grouped.setdefault((dataset,phase),dict(request_attempts=0,known_input_tokens=0,known_output_tokens=0,attempts_without_usage=0))
        for attempt in entry['attempts']:
            totals['request_attempts']+=1
            usage=attempt.get('usage')
            if not isinstance(usage,dict):totals['attempts_without_usage']+=1;continue
            totals['known_input_tokens']+=usage.get('input_tokens',0)
            totals['known_output_tokens']+=usage.get('output_tokens',0)
    cumulative=[dict(dataset=d,phase=p,**v) for (d,p),v in sorted(grouped.items())]
    write(REPORT/'results.json',dict(stamp=STAMP,candidate_floor=.01,model='qwen3.8-27b',states=states,rows=rows,costs=costs,cumulative_request_usage=cumulative))
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['dataset','stage','status']
    with (REPORT/'results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    lines=['# YOLOE候选 → Qwen3.8-27B：统一实验结果','',
        '**Qwen输入门槛固定为 score≥0.01（用户指定）。** YOLOE原始10shot视觉提示，未使用新微调权重。',
        'S0保留生成时的0.001完整输出作审计参照；仅B0及其后续阶段进入Qwen流程。',
        'B1只筛选，B1D独立增加同类同实例去重；B2固定B1D候选集合、类别和原分数，只修改坐标。',
        '去重由Qwen确认关系，程序保留原分数最高的代表框；跨类别冲突不擅自改类。',
        '不确定或请求失败时保留原框。所有图片和GT保留在分母；没有结果的阶段不填数值。','',
        '[筛选/去重/改框流程](../../scripts/README.md) · [CSV](results.csv) · [JSON](results.json)','',
        '## 执行状态','', '| 数据集 | 图片 | YOLOE原始框 | 小批量验证 | 完整流程 |', '|---|---:|---:|---|---|']
    for state in states:
        lines.append(f"| {state['dataset']} | {state['images']} | {state['source_candidates']} | {state['pilot_status']} | {state['pipeline_status']} |")
    lines+=['','## 配对结果','',
        'AP、P/R/F2均为百分数；AP使用全部保存候选，P/R/F2与TP/FP/FN使用score≥0.5、IoU≥0.5。',
        '低阈值工作点和逐类别指标见每组原始metrics.json；开发集与300张测试结果分开，不混合排名。','']
    for state in states:
        subset=[r for r in rows if r['dataset']==state['dataset']]
        lines += [f"### {state['dataset']}",'']
        if not subset:
            lines+=['YOLOE候选已完成，审核及配对评估尚未完成。',''];continue
        lines+=['| 阶段 | 框数 | AP50 | AP75 | AP50:95 | AP85 | AP90 | AP95 | 四项≤20% AP | P | R | F2 | TP/FP/FN |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        keys=['mAP50','mAP75','mAP50_95','AP85','AP90','AP95','AP_box20','precision','recall','f2']
        for r in subset:
            vals=[f"{100*r[k]:.2f}" if r.get(k) is not None else '—' for k in keys]
            lines.append('| '+' | '.join([TITLES[r['stage']],str(r['candidates']),*vals,f"{r['tp']}/{r['fp']}/{r['fn']}"])+' |')
        for r in subset:
            if 'delta_AP50_pp' in r:
                lines.append(f"\n{TITLES[r['stage']]}−B0：AP50 {r['delta_AP50_pp']:+.2f}，AP50:95 {r['delta_AP50_95_pp']:+.2f}，召回 {r['delta_recall_pp']:+.2f} 个百分点，FP净减少 {r['fp_removed']}。")
        filtered=next((r for r in subset if r['stage']=='filtered'),None)
        dedup=next((r for r in subset if r['stage']=='deduplicated'),None)
        refined=next((r for r in subset if r['stage']=='refined'),None)
        for before,after,label in [(filtered,dedup,'去重 B1D−B1'),(dedup,refined,'改框 B2−B1D')]:
            if before and after:
                lines.append(f"\n{label}：AP50 {100*(after['mAP50']-before['mAP50']):+.2f}、AP50:95 {100*(after['mAP50_95']-before['mAP50_95']):+.2f}、AP85 {100*(after['AP85']-before['AP85']):+.2f} 个百分点。")
        low=[r for r in subset if r.get('tp_score01') is not None]
        if low:
            lines+=['','**实际候选门槛score≥0.01下的工作点（IoU≥0.5）**','',
                '| 阶段 | Precision% | Recall% | F2% | TP/FP/FN |', '|---|---:|---:|---:|---|']
            for r in low:
                vals=[f"{100*r[k+'_score01']:.2f}" for k in ['precision','recall','f2']]
                lines.append('| '+' | '.join([TITLES[r['stage']],*vals,f"{r['tp_score01']}/{r['fp_score01']}/{r['fn_score01']}"])+' |')
        relative=os.path.relpath(ROOT/state['run']/'experiment_results.md', REPORT)
        lines+=['',f'[本组详细审计与逐阶段结果]({relative})','']
    if costs:
        lines+=['## 请求与耗时','',
            '以下为本次执行的新请求用量；缓存命中数单列，命中的历史请求token未再次计费，也未重复加到本次用量。耗时为该阶段墙钟时间。','',
            '| 数据集 | 阶段 | 新请求（含重试） | 缓存命中 | 失败批次 | 输入/输出token | 耗时秒 |',
            '|---|---|---:|---:|---:|---|---:|']
        for c in costs:
            lines.append(f"| {c['dataset']} | {c['phase']} | {c['requests']} | {c['cached']} | {c['failed']} | {c['input_tokens']}/{c['output_tokens']} | {c.get('wall_seconds',0):.1f} |")
    if cumulative:
        lines+=['','**累计请求记录（含小批量验证与恢复执行）**','',
            '按成功缓存及失败响应中的原始attempt去重累计，不把缓存读取再算作请求。格式/网络失败可能未保存usage；只报告已知token，不推算计费金额，也不包含无图连通性预检。','',
            '| 数据集 | 阶段 | 记录的请求尝试 | 已知输入/输出token | 缺usage的尝试 |', '|---|---|---:|---|---:|']
        for c in cumulative:
            lines.append(f"| {c['dataset']} | {c['phase']} | {c['request_attempts']} | {c['known_input_tokens']}/{c['known_output_tokens']} | {c['attempts_without_usage']} |")
    lines+=['## 限制','',
        '- 现有开发与测试队列可能共享视频序列，不宣称序列独立泛化；精确图像哈希不重叠，详见split_audit.json。',
        '- 固定300张测试集已有历史实验；本轮不根据测试成绩调整阈值或提示词。',
        '- 四项≤20%是相对GT的匹配指标，不等于修框幅度限制。',
        '- 去重需Qwen明确确认、同类且原框IoU≥0.5，覆盖差异较大或未确认关系的同实例框可能残留；改框后不再去重，以保持B2候选集合固定。',
        '- 格式恢复仅解析理由文本中未转义的控制字符，不补造编号或决定；结构缺失必须重新请求，原始attempt保留。',
        '- `未完成`表示对应后续阶段尚未完成，已写出的阶段指标仅代表该阶段，不伪造后续成绩。','',
        '更新：`python outputs/OVD-MLLM/scripts/summarize_experiment.py`。']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    print('WROTE',REPORT,'rows',len(rows))


def previews(per_cohort=4):
    """Fixed hash-selected post-evaluation previews; GT never enters requests."""
    import hashlib
    from PIL import Image,ImageDraw,ImageOps,ImageFont
    entries=[]
    for scene in ('chache','diaowu'):
        for split in ('val100','val300'):
            root=folder(scene,split,STAMP)
            if not (root/'refined/records.jsonl').exists():continue
            cohort=read(root/'cohort.json');stages={}
            for stage in ('baseline','filtered','deduplicated','refined'):
                p=root/stage/'records.jsonl'
                stages[stage]={r['relative_path']:r['objects'] for r in (json.loads(s) for s in p.read_text().splitlines())}
            selected=sorted(cohort['samples'],key=lambda s:hashlib.sha256(s['image_sha256'].encode()).hexdigest())[:per_cohort]
            for sample in selected:
                with Image.open(Path(cohort['image_root'])/sample['relative_path']) as image:
                    original=image.convert('RGB')
                grid=Image.new('RGB',(1920,1160),'white')
                try:font=ImageFont.truetype('DejaVuSans.ttf',15)
                except OSError:font=None
                for index,(stage,records) in enumerate(stages.items()):
                    tile=Image.new('RGB',(960,580),'white');display=ImageOps.contain(original,(960,540))
                    xoff=(960-display.width)//2;yoff=40+(540-display.height)//2
                    tile.paste(display,(xoff,yoff));draw=ImageDraw.Draw(tile)
                    sx=display.width/sample['width'];sy=display.height/sample['height']
                    def box(obj,color,label):
                        b=obj['bbox_pixels_xyxy'];b=[b[0]*sx+xoff,b[1]*sy+yoff,b[2]*sx+xoff,b[3]*sy+yoff]
                        draw.rectangle(b,outline=color,width=2)
                        draw.text((max(0,b[0]),max(40,b[1]-16)),label,fill=color,font=font,stroke_width=1,stroke_fill='black')
                    for obj in sample['ground_truth']:box(obj,'lime','GT '+obj['label'])
                    objects=sorted(records[sample['relative_path']],key=lambda p:-p['confidence'])
                    shown=[p for p in objects if p['confidence']>=.01][:40]
                    for obj in shown:box(obj,'red',f"{obj['label']} {obj['confidence']:.3f}")
                    draw.text((8,8),f'{stage}: {len(objects)} boxes; top {len(shown)} shown; RED prediction / GREEN GT',font=font,fill='black')
                    grid.paste(tile,((index%2)*960,(index//2)*580))
                destination=REPORT/'previews'/f'{scene}_{split}_{Path(sample["relative_path"]).stem}.jpg'
                destination.parent.mkdir(parents=True,exist_ok=True);grid.save(destination,quality=88)
                entries.append(dict(dataset=f'{scene}_{split}',relative_path=sample['relative_path'],
                    preview=str(destination.relative_to(REPORT)),selection='fixed image-hash order; no outcome cherry-picking'))
    if entries:
        write(REPORT/'previews/manifest.json',{'entries':entries,'note':'GT only shown after evaluation; never sent to Qwen.'})
        path=REPORT/'README.md';contents=path.read_text()
        if '## 可视化' not in contents:
            path.write_text(contents+'\n## 可视化\n\n每个已完成队列按固定图像哈希选4张，红色为预测、绿色为GT；每阶段最多展示40个最高分框，完整指标不受此显示上限影响。\n\n'+''.join(
                f"- [{e['dataset']} / {e['relative_path']}]({e['preview']})\n" for e in entries))
    print('PREVIEWS',len(entries))


def audit_deletions():
    from experiment import load_records,deletion_diagnostics
    data=read(REPORT/'results.json');audits=[];lines=['','## 基线正确框误删核查','',
        'score≥0.01、IoU≥0.5，固定B0的TP/FP标记统计删除；净TP损失按过滤后的全组重新匹配计算，两者含义不同。','',
        '| 数据集 | 阶段 | 基线TP框被删 | 基线FP框被删 | 重新匹配后净TP损失 |', '|---|---|---:|---:|---:|']
    for state in data['states']:
        root=ROOT/state['run']
        if not (root/'verification.json').exists():continue
        cohort=read(root/'cohort.json');baseline=load_records(root/'baseline/records.jsonl')
        labels=read(root/'source_complete.json')['labels']
        basepoint=read(root/'baseline/metrics.json')['score_threshold_results']['0.01']
        for stage in ['filtered','deduplicated']:
            after=load_records(root/stage/'records.jsonl')
            result=deletion_diagnostics(cohort['samples'],baseline,after,labels)
            if result['baseline_tp']!=basepoint['tp'] or result['baseline_fp']!=basepoint['fp']:
                raise ValueError('Baseline TP/FP audit disagrees with evaluator')
            point=read(root/stage/'metrics.json')['score_threshold_results']['0.01']
            result.update(dataset=state['dataset'],stage=stage,net_tp_loss=basepoint['tp']-point['tp'])
            audits.append(result)
            lines.append(f"| {state['dataset']} | {TITLES[stage]} | {result['deleted_baseline_tp']}/{result['baseline_tp']} | {result['deleted_baseline_fp']}/{result['baseline_fp']} | {result['net_tp_loss']} |")
    data['deletion_audit']=audits;write(REPORT/'results.json',data)
    path=REPORT/'README.md';text=path.read_text()
    if '## 基线正确框误删核查' not in text:path.write_text(text+'\n'.join(lines)+'\n')
    print('DELETION AUDITS',len(audits))


def summary():
    import fcntl
    lock=ROOT/'outputs/_shared/runtime/ovd_mllm_filtering_20261030/summary.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try:_summary_unlocked()
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--previews',action='store_true')
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args();summary()
    if args.audit:audit_deletions()
    if args.previews:previews()

"""Build one traceable table and a concise analysis from every saved Chache run; no model calls."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import io
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from industrial_fewshot.data.labelme import read_json,write_json
from industrial_fewshot.run_discovery import saved_runs

LABELS=['Chache1','Chache2','Chache3','Chache4','Qianyinche']
COUNTS=dict(zip(LABELS,[40,30,40,30,48]))
VARIANTS=['text_only','O+J','O+C','O+B','O+C+J','O+B+J','O+C+B','O+C+B+J']


def pct(value):return '—' if value is None else f'{value*100:.2f}'


def records(directory):
    file=directory/'records.jsonl'
    if not file.is_file():return []
    result=[]
    import json
    for line in file.read_text(encoding='utf-8-sig').splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:continue
    return result


def collect(root):
    marker=read_json(root/'dataset.json')
    query=read_json(ROOT/'data/raw/chache_val300/manifest.json')
    query_hashes={s['relative_path']:s['image_sha256'] for s in query['samples']}
    current={}
    statefile=ROOT/'outputs/runtime/chache_qwen27b_full5shot_top3/status.json'
    if statefile.is_file():
        state=read_json(statefile)
        current={str(Path(j['run_dir']).resolve()):j['status'] for j in state.get('jobs',[])}
    rows=[]
    for file in saved_runs(root):
        directory=file.parent
        initial=read_json(file)
        summary=directory/'summary.json'
        saved=read_json(summary) if summary.is_file() else initial
        target=saved['targets'][0]
        data=records(directory)
        successes=sum(r.get('status')=='success' for r in data)
        failures=sum(r.get('status')=='failed' for r in data)
        unique=len({r['relative_path'] for r in data})
        m=saved.get('metrics') or (read_json(directory/'metrics.json') if (directory/'metrics.json').is_file() else {})
        exp=saved.get('experiment',{})
        shot=exp.get('positive_count',0)
        variant=exp.get('variant','text_only' if shot==0 else 'unknown')
        refs=target.get('reference_count',{})
        version=next((p for p in directory.relative_to(root).parts if re.search(r'_v\d+$',p)),'unknown')
        number=int(version.rsplit('_v',1)[1]) if '_v' in version else 0
        prompt='v1短版' if number==1 else 'v2扩展版' if number==2 else version
        same_dataset=saved.get('dataset_sha256')==marker['dataset_sha256']
        complete=(same_dataset and saved.get('evaluation_status')=='complete' and successes==unique==len(data)==300
                  and failures==0 and {k:c['ground_truth_count'] for k,c in m.get('per_class',{}).items()}==COUNTS
                  and {r['relative_path']:r.get('image_sha256') for r in data}==query_hashes)
        notes=[]
        if not same_dataset:status='其他数据版本';notes.append('不参与本数据集结论')
        elif complete:status='完成'
        elif current.get(str(directory.resolve()))=='running':status='运行中'
        elif failures:status='不完整' if m else '失败'
        else:status='未完成'
        errors=[r.get('error','') for r in data if r.get('status')=='failed']
        if any('与请求模型' in e for e in errors):
            status='失败：模型路由不符';notes.append('上游返回模型与请求不同，无有效该模型评估')
        elif any('reason' in e for e in errors):notes.append('失败响应缺少规定的reason字段；保留原记录')
        if not complete and m:notes.append('指标仅供排查，不参与最优结果和配对结论')
        points=m.get('score_threshold_results',{})
        point5=points.get('0.5') or (m if m.get('settings',{}).get('score_threshold')==.5 else {})
        point7=points.get('0.7',{})
        values=dict(AP50=m.get('mAP50'),AP50_95=m.get('mAP50_95'),
                    **{f'{key}_0_5':point5.get(key) for key in ('precision','recall','f2','tp','fp','fn')},
                    **{f'{key}_0_7':point7.get(key) for key in ('precision','recall','f2','tp','fp','fn')})
        repeat=re.search(r'_第(\d+)次',directory.parent.name)
        rows.append(dict(model=target['model'],prompt=prompt,prompt_version=number,
            prompt_sha256=target.get('prompt_sha256'),shot_per_class=shot,input=variant,
            negative_references=refs.get('reference_negatives',0),**values,
            success_images=successes,failed_images=failures,total_images=saved['image_count'],
            status=status,complete=complete,dataset_sha256=saved.get('dataset_sha256'),
            reference_images=sum(refs.get(k,0) for k in ('reference_originals','reference_crops','reference_boxed_images','reference_negatives')),
            class_AP50={name:m.get('per_class',{}).get(name,{}).get('AP50') for name in LABELS},
            run_dir=directory.relative_to(root).as_posix(),
            report=(directory/('evaluation.txt' if (directory/'evaluation.txt').is_file() else 'run.json')).relative_to(root).as_posix(),
            prompt_file=(directory/'prompt.txt').relative_to(root).as_posix(),repeat=int(repeat[1]) if repeat else 1,
            notes='；'.join(notes)))
    rows.sort(key=lambda r:(0 if r['model']=='qwen3.8-27b' else 1,r['model'],r['prompt_version'],
                            r['shot_per_class'],VARIANTS.index(r['input']) if r['input'] in VARIANTS else 99,r['repeat'],r['run_dir']))
    return rows


def conclusion(rows):
    good=[r for r in rows if r['complete']]
    if not good:return '目前没有满足同一数据指纹、300张全部成功及完整评估条件的运行，暂不比较优劣。'
    best=max(good,key=lambda r:r['AP50'])
    strict=max(good,key=lambda r:r['AP50_95'])
    def get(model,version,shot,variant):
        matches=[r for r in good if (r['model'],r['prompt_version'],r['shot_per_class'],r['input'])==(model,version,shot,variant)]
        return min(matches,key=lambda r:r['repeat']) if matches else None
    def q(version,shot,variant):return get('qwen3.8-27b',version,shot,variant)
    lines=[f"在同一300张、188个GT的特种车辆验证集上，当前完整运行中AP50最高的是{best['model']}（{best['prompt']}、每类{best['shot_per_class']}shot、{best['input']}），"
           f"AP50为{pct(best['AP50'])}%，Precision/Recall/F2为{pct(best['precision_0_5'])}%/{pct(best['recall_0_5'])}%/{pct(best['f2_0_5'])}%；"
           f"更严格的AP50:95最高值为{pct(strict['AP50_95'])}%，来自{strict['model']}的{strict['shot_per_class']}shot {strict['input']}。"]
    pure=q(2,0,'text_only');five=[r for r in good if r['model']=='qwen3.8-27b' and r['prompt_version']==2 and r['shot_per_class']==5]
    if pure and five:
        winner=max(five,key=lambda r:r['AP50'])
        lines.append(f"视觉参考的收益明显：扩展版纯文字AP50为{pct(pure['AP50'])}%，5shot最佳组合达到{pct(winner['AP50'])}%，提高{(winner['AP50']-pure['AP50'])*100:.2f}个百分点。")
    old=q(1,5,'O+B');new=q(2,5,'O+B');oldpure=q(1,0,'text_only')
    if old and new and oldpure and pure:
        lines.append(f"扩展提示词在相同5shot O+B条件下将AP50从{pct(old['AP50'])}%提高至{pct(new['AP50'])}%，纯文字则由{pct(oldpure['AP50'])}%降至{pct(pure['AP50'])}%，提示词扩展对两种输入配置的影响不同。")
    a=q(2,1,'O+C+B');b=q(2,5,'O+C+B');c=q(2,10,'O+C+B')
    if a and b and c:
        lines.append(f"同一O+C+B组合从1shot的{pct(a['AP50'])}%提升到5shot的{pct(b['AP50'])}%，10shot为{pct(c['AP50'])}%，5到10shot仅增加{(c['AP50']-b['AP50'])*100:.2f}个百分点，增益已明显收窄。")
    pairs=[(q(2,5,v),q(2,5,v+'+J')) for v in ['O+C','O+B','O+C+B']]
    if all(a and b and b['AP50']<a['AP50'] for a,b in pairs):
        lines.append('本轮5shot三组配对中，加入参考标注JSON均未提高AP50，堆叠更多参考材料没有表现出一致收益。')
    flash=get('qwen3.8-flash',2,5,'O+B');flash10=get('qwen3.8-flash',2,10,'O+B');q10=q(2,10,'O+B')
    if new and flash and flash10 and q10:
        lines.append(f"Flash与27B在5shot O+B下基本持平（{pct(flash['AP50'])}%与{pct(new['AP50'])}%），但10shot时27B为{pct(q10['AP50'])}%，高于Flash的{pct(flash10['AP50'])}%。")
    weakest=sorted(((name,value) for name,value in best['class_AP50'].items() if value is not None),key=lambda p:p[1])
    if weakest:
        lines.append(f"最佳AP50方案中{weakest[0][0]}仍最弱（该类AP50 {pct(weakest[0][1])}%），后续应重点检查该类的结构区分、类别混淆及框选误差。")
    lines.append('这些结论基于固定验证集上的单次运行；小幅AP差异受模型采样和同分排序影响，尚不能据此认定统计显著优势。')
    return ''.join(lines)


def main(root):
    rows=collect(root)
    counts=Counter(r['status'] for r in rows)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    text=['# Chache特种车辆：全部运行结果总表','',f'更新时间：{stamp}（北京时间）。扫描{len(rows)}个实际运行目录；'+
          '、'.join(f'{name}{count}组' for name,count in counts.items())+'。每个当前运行目录只占一行；comparison、prompt索引、缓存及归档中的被替代重试副本不重复计数。','',
          '数据统一为最终版chache_val300：300张（174张有目标、126张无目标）、188个GT；五类数量为40/30/40/30/48。参考负例均为0；测试集中的无目标图片仍保留。',
          'O=正例原图，C=目标裁剪图，B=带框原图，J=参考标注JSON；所有视觉组同时使用对应版本的文字提示。shot按每类计，纯文字实际为0shot，即使其历史目录名称包含5shot。',
          '表内指标均为百分数；AP50为五类宏平均，P/R/F2与TP/FP/FN为全部类别汇总，IoU=0.5、maxDets=100。AP使用全部保存分数；0.5/0.7仅筛选输出框及对应P/R/F2。所有实验均未启用二次复核。','',
          '| # | 模型 | prompt | shot/类 | 输入 | AP50 | AP50:95 | P/R/F2≥0.5 | P/R/F2≥0.7 | TP/FP/FN≥0.5 | 成功/总图 | 状态 | 详细结果 |',
          '|---:|---|---|---:|---|---:|---:|---|---|---|---|---|---|']
    best=max((r['AP50'] for r in rows if r['complete']),default=None)
    for i,r in enumerate(rows,1):
        triple=lambda threshold:'/'.join(pct(r[f'{k}_{threshold}']) for k in ('precision','recall','f2'))
        count='/'.join(str(r[f'{k}_0_5']) if r[f'{k}_0_5'] is not None else '—' for k in ('tp','fp','fn'))
        ap=pct(r['AP50'])
        if r['complete'] and r['AP50']==best:ap='**'+ap+'**'
        if not r['complete'] and r['AP50'] is not None:ap+='†'
        prompt=f"[{r['prompt']}]({r['prompt_file']})"
        text.append('| '+' | '.join([str(i),r['model'],prompt,str(r['shot_per_class']),r['input'],ap,pct(r['AP50_95']),
                    triple('0_5'),triple('0_7'),count,f"{r['success_images']}/{r['total_images']}",r['status'],f"[查看]({r['report']})"])+ ' |')
    text.extend(['','† 不完整运行的指标仅用于排查，不参与最佳结果和上述配对比较；“—”表示没有有效保存值，不能解释为0。失败的Flash首次尝试是模型路由不符，其后成功重跑另列一行。','',
                 '**总结归纳**','',conclusion(rows),'',
                 '表格导出：[CSV](results_summary.csv)（UTF-8 BOM，可用Excel打开，`_pct`列为百分数）；[JSON](results_summary.json)保留0～1原始指标、类别AP50、指纹与来源。',
                 '更新命令：`.\\.venv-yoloe\\Scripts\\python.exe scripts/summarize_chache_results.py`。此命令只读取已保存记录，不调用模型、不重新计算或改写原指标。'])
    (root/'results_summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    percentages=['AP50','AP50_95',*[f'{k}_{t}' for t in ('0_5','0_7') for k in ('precision','recall','f2')]]
    csv_rows=[]
    for row in rows:
        exported={k+('_pct' if k in percentages else ''):
                  (round(v*100,2) if v is not None else '') if k in percentages else v
                  for k,v in row.items() if k!='class_AP50'}
        exported.update({f'{label}_AP50_pct':round(value*100,2) if value is not None else ''
                         for label,value in row['class_AP50'].items()})
        csv_rows.append(exported)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]) if csv_rows else [])
    writer.writeheader();writer.writerows(csv_rows)
    (root/'results_summary.csv').write_text(stream.getvalue(),encoding='utf-8-sig')
    write_json(root/'results_summary.json',dict(generated_at=stamp,dataset=read_json(root/'dataset.json'),
               counts=dict(counts),row_count=len(rows),rows=rows,conclusion=conclusion(rows)))
    print(f'Runs: {len(rows)}; statuses: {dict(counts)}; report: {root / "results_summary.md"}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/runs/chache_val300')
    main(parser.parse_args().root.resolve())

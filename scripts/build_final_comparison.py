"""Build thematic experiment tables and a compact, traceable global comparison.

Only reads saved results; no inference, training or API calls. All metric cells
in a representative row come from the same run. Full-scope and novel-only
evaluations remain separate comparison populations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
DETAIL = OUT/'_shared/comparisons/final_results'
DATASETS = ['chache_val300', 'diaowu_val300']
SCENES = {'chache_val300': '叉车（五类）', 'diaowu_val300': '吊物（单类）'}
FILES = {'qwen': '01_qwen_inputs.md', 'models': '02_model_comparison.md',
         'finetune': '03_yoloe_finetuning.md', 'cascade': '04_yoloe_qwen.md',
         'supervised': '06_supervised_reference.md'}
TITLES = {'qwen': 'Qwen输入、shot与模型对比', 'models': 'Qwen与开放词汇模型对比',
          'finetune': 'YOLOE微调与YOLOv8对照', 'cascade': 'YOLOE→Qwen分阶段处理',
          'supervised': '定制监督网络参考'}
METRICS = [('AP50↑','AP50'), ('AP75↑','AP75'), ('AP50:95↑','AP50_95'),
           ('P↑','precision_0_5'), ('R↑','recall_0_5'), ('F1↑','f1_0_5'),
           ('F2↑','f2_0_5'), ('负图错误率↓','negative_error_rate')]
SHORT = [METRICS[i] for i in [0,2,3,4,5,6,7]]
RATE_KEYS = {key for _,key in METRICS} | {
    'AP85','AP90','AP95','AP_box20','mean_matched_iou','image_accuracy',
    'negative_false_positive_rate','COCO_AR100','COCO_AP_small','COCO_AP_medium','COCO_AP_large',
    *[f'{metric}_{suffix}' for metric in ['precision','recall','f1','f2'] for suffix in ['0_5','0_7']]}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link(label, target, page):
    relative = os.path.relpath(target, page.parent)
    return f'[{label}]({quote(relative, safe="/._-#")})'


def pct(value):
    return '—' if value is None else f'{100*value:.2f}'


def eligible(row):
    return row.get('eligible_for_comparison', False)


def winners(rows, keys=('AP50','AP50_95')):
    """Union of actual AP winners; retain both runs when winners differ."""
    usable = [r for r in rows if eligible(r)]
    selected = {}
    for key in keys:
        candidates = [r for r in usable if r.get(key) is not None]
        if not candidates:
            continue
        # Stable tie choice; no preference based on another metric or missing data.
        best = sorted(candidates, key=lambda r: (-r[key], r['run_dir']))[0]
        record = selected.setdefault(best['run_dir'], copy.deepcopy(best))
        record.setdefault('selected_by', []).append(key)
    return list(selected.values())


def primary(row):
    if row['algorithm']=='Qwen':
        return 'qwen'
    if row['algorithm']=='OVD-MLLM':
        return 'cascade'
    if row['algorithm']=='JSTY-YOLO':
        return 'supervised'
    if row['algorithm']=='YOLOv8' or row['training_shot_per_class']:
        return 'finetune'
    return 'models'


def training_metadata(row):
    source=row.get('cascade_source_run') or row['run_dir']
    for directory in (ROOT/source).parents:
        path = directory/'finetune_config.json'
        if path.is_file():
            return read(path)
        if directory==ROOT:
            break
    return {}


def enrich(row):
    row = copy.deepcopy(row)
    metadata = training_metadata(row)
    settings = metadata.get('training', {})
    adaptation = metadata.get('adaptation', {})
    row['learning_rate'] = settings.get('lr0')
    row['batch'] = metadata.get('batch')
    row['max_epochs'] = settings.get('epochs')
    row['optimizer'] = settings.get('optimizer')
    row['patience'] = settings.get('patience')
    row['augmentation'] = {k:settings[k] for k in ['mosaic','mixup','copy_paste','close_mosaic',
        'degrees','translate','scale','fliplr','flipud'] if k in settings}
    row['adaptation_loss'] = adaptation.get('loss')
    row['freeze_savpe_bn_stats'] = adaptation.get('freeze_savpe_bn_stats')
    if metadata and row['algorithm']=='YOLOE' and row['savpe']!='移除':
        row['training_protocol'] = adaptation.get('protocol', 'same_image')
    row['primary_topic'] = primary(row)
    row['training_budget_display'] = '监督/训练量未绑定' if row['training_shot_per_class'] is None else str(row['training_shot_per_class'])
    hint = row['inference_shot_per_class']
    row['prompt_budget_display'] = '不适用' if hint is None else str(hint)
    if row['algorithm']=='OVD-MLLM':
        row['prompt_budget_display'] = f'YOLOE {hint if hint is not None else "固定头"} / Qwen {row["mllm_reference_shot_per_class"]}'
    row['metric_evidence'] = ('未产出指标' if not row['metric_file'] else
                              '元数据†' if '元数据' in row['verification'] else '逐图核验')
    row['counts'] = '/'.join(str(row.get(f'{k}_0_5')) if row.get(f'{k}_0_5') is not None else '—'
                           for k in ['tp','fp','fn'])
    return row


class Report:
    def __init__(self, rows):
        self.rows = rows
        self.tables = []
        self.members = defaultdict(set)

    def table(self, page, title, rows, columns, metrics=METRICS, counts=True, status=True, group=None,
              minimum_comparators=1):
        """Bold extrema within the visible comparison population, at shown precision."""
        def population(r):
            return tuple(r[k] for k in group) if group else ()
        best = {}
        for _, key in metrics:
            populations = defaultdict(list)
            for row in rows:
                value = row.get(key)
                if eligible(row) and value is not None and math.isfinite(value):
                    populations[population(row)].append(round(100*value,2))
            for subset, values in populations.items():
                if len(values)>=minimum_comparators:
                    best[subset,key] = min(values) if key=='negative_error_rate' else max(values)
        headers = [c[0] for c in columns] + [m[0] for m in metrics]
        if counts:
            headers.append('TP/FP/FN')
        if status:
            headers.append('状态/证据')
        headers.append('原始结果')
        lines = [f'### {title}', '', '| '+' | '.join(headers)+' |',
                 '| '+' | '.join(['---']*len(columns)+['---:']*len(metrics)
                                +(['---'] if counts else [])+(['---'] if status else [])+['---'])+' |']
        highlighted = []
        for row in rows:
            values = [str(function(row)) for _,function in columns]
            for _, key in metrics:
                value = row.get(key)
                cell = pct(value)
                if value is not None and eligible(row) and round(100*value,2)==best.get((population(row),key)):
                    cell = f'**{cell}**'
                    highlighted.append(dict(source=row['run_dir'], metric=key, value=value))
                values.append(cell)
            if counts:
                values.append(row['counts'])
            if status:
                values.append(row['status']+('†' if row['metric_evidence']=='元数据†' else ''))
            values.append(link('查看', ROOT/row['report'], page) if row.get('report') else '—')
            lines.append('| '+' | '.join(v.replace('|','\\|').replace('\n',' ') for v in values)+' |')
        self.tables.append(dict(page=str(page.relative_to(ROOT)), title=title, rows=len(rows),
            comparison_group_fields=group or [], minimum_comparators=minimum_comparators,
            sources=[r['run_dir'] for r in rows if r.get('run_dir')], highlighted=highlighted))
        return '\n'.join(lines)+'\n\n'

    def page(self, topic, intro):
        path = DETAIL/FILES[topic]
        return path, f'# {TITLES[topic]}\n\n'+intro+'\n\n'+link('全局比较',OUT/'FINAL_COMPARISON.md',path)+' · '+link('子任务导航',DETAIL/'README.md',path)+'\n\n'+rules()+'\n\n'

    def save(self, path, text, rows):
        path.write_text(text, encoding='utf-8')
        unique = {r['run_dir']:r for r in rows if r.get('run_dir')}
        self.members[path.name].update(unique)
        export_csv(path.with_suffix('.csv'), list(unique.values()))


def rules():
    return ('指标均为百分数。AP使用该运行保存的全部候选；P/R/F1/F2使用score≥0.5、IoU=0.5，TP/FP/FN为框数。'
        '箭头表示优化方向；每张表内逐列加粗最优值，按显示精度并列最优同时加粗。参数值不作最优标记。'
        '未完成、队列不符或类别范围不全的记录保留展示，但不参与最优值计算。'
        '†表示本机仅保存指标和元数据，尚无逐图记录可复核；仍可作为有此证据限制的结果对比。'
        'P或负图错误率单独最优不等于检测最好，需结合R和TP/FP/FN阅读。')


def export_csv(path, rows):
    if not rows:
        return
    converted=[]
    for row in rows:
        record={}
        for key,value in row.items():
            rate = key in RATE_KEYS or bool(re.fullmatch(
                r'(novel|base|all)_(mAP50|mAP75|mAP50_95|precision|recall|f1|f2|negative_error_rate|image_accuracy|mean_matched_iou)',key))
            field = key+'_pct' if rate else key
            record[field] = round(100*value,6) if rate and value is not None else value
        converted.append(record)
    fields = list(dict.fromkeys(k for row in converted for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in converted:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})


def fine_name(row):
    if row['algorithm']=='YOLOv8':
        return 'YOLOv8-L微调' if row['training_shot_per_class'] else 'YOLOv8-L目标输出初始化'
    if not row['training_shot_per_class']:
        return '原始YOLOE视觉提示'
    return {'移除':'YOLOE固定头／移除SAVPE', '训练':'YOLOE视觉提示／更新SAVPE',
            '冻结':'YOLOE视觉提示／冻结SAVPE'}[row['savpe']]


def fine_trial(row):
    # Actual checkpoint parent, not a reconstructed/shortened method alias.
    return Path(row['run_dir']).parents[1].name if row['training_shot_per_class'] else row['model']


def fine_parameters(row):
    scope = row['training_scope'] or '不更新'
    protocol = row['training_protocol'] or '—'
    augmentation = row.get('augmentation', {})
    hint = f'; mosaic={augmentation["mosaic"]},翻转={augmentation.get("fliplr",0)}' if augmentation.get('mosaic',0)>0 or augmentation.get('fliplr',0)>0 else ''
    return f'{scope}/{protocol}; lr={row["learning_rate"] if row["learning_rate"] is not None else "—"}; 输入={row["input_size"] or "—"}; batch={row["batch"] or "—"}'+hint


def qwen_representatives(rows):
    pools = defaultdict(list)
    for row in rows:
        if row['algorithm']=='Qwen':
            pools[row['model'],row['inference_shot_per_class']].append(row)
    return [row for pool in pools.values() for row in winners(pool)]


QWEN_INPUTS = ('O+B','O+C','O+J','O+B+J','O+C+J','O+C+B','O+C+B+J')
QWEN_VERSIONS = {'chache_val300':'MLLM提示_v2','diaowu_val300':'MLLM提示_v1'}
QWEN_SHOT_COUNTS = {'chache_val300':(0,1,5,10), 'diaowu_val300':(0,1,5,10,20)}
QWEN_MODEL_INPUTS = {'chache_val300':'O+C+B', 'diaowu_val300':'O+B+J'}


def qwen_condition(rows, dataset, model, shot, variant):
    matches=[r for r in rows if r['dataset']==dataset and r['model']==model
        and r['prompt_version']==QWEN_VERSIONS[dataset]
        and r['inference_shot_per_class']==shot and r['input_variant']==variant
        and r['negative_references']==0 and r['support_seed']==0]
    complete=[r for r in matches if eligible(r)]
    if len(complete)>1:
        raise ValueError(f'Ambiguous Qwen condition: {(dataset,model,shot,variant)}')
    if complete or matches:
        # Show the latest incomplete attempt while a supplement is running;
        # every other attempt remains in the appendix.
        matches.sort(key=lambda r:((ROOT/r['run_dir']/'run.json').stat().st_mtime_ns,r['run_dir']),reverse=True)
        return copy.deepcopy(complete[0] if complete else matches[0])
    return dict(dataset=dataset,algorithm='Qwen',model=model,
        inference_shot_per_class=shot,input_variant=variant,prompt_version=QWEN_VERSIONS[dataset],
        negative_references=0,support_seed=0,run_dir=None,report=None,
        eligible_for_comparison=False,status='待补跑（当前300张无结果）',
        metric_evidence='未产出指标',counts='—/—/—')


def qwen_input_selection(rows, dataset):
    candidates=[]
    for variant in QWEN_INPUTS:
        pair=[qwen_condition(rows,dataset,'qwen3.8-27b',shot,variant) for shot in (5,10)]
        if all(eligible(r) for r in pair):
            candidates.append(dict(input_variant=variant,
                mean_AP50=sum(r['AP50'] for r in pair)/2,
                mean_AP50_95=sum(r['AP50_95'] for r in pair)/2,
                sources=[r['run_dir'] for r in pair]))
    if not candidates:
        raise ValueError(f'No complete 5/10shot input pair: {dataset}')
    candidates.sort(key=lambda c:(-c['mean_AP50'],-c['mean_AP50_95'],c['input_variant']))
    return dict(prompt_version=QWEN_VERSIONS[dataset],support_seed=0,negative_references=0,
                selected_input=candidates[0]['input_variant'],candidates=candidates)


def make_qwen(report):
    rows = [r for r in report.rows if r['algorithm']=='Qwen']
    path, text = report.page('qwen',
        '按实验顺序组织为三个阶段：输入对比按场景和shot分为4张表，shot对比和模型对比各按场景分为2张表，正文共8张表。'
        '先只用27B选择输入，再只用27B比较shot，最后比较27B、Flash、Max。'
        'O=参考原图，C=目标裁剪图，B=带框参考图，J=标注JSON。0shot没有视觉参考，但仍输入测试查询图。')
    text+=('主线统一负参考=0、support seed=0；叉车使用提示v2，吊物使用v1，测试集均为各自固定300张。'
        '缺少的条件显示待补跑，不用其他输入、其他shot或旧测试集结果代填。已有其他条件与Max旧结果收在折叠附录，CSV仍保留全部真实运行。\n\n'
        '- [第一步：比较27B的输入方式](#step-1)\n'
        '- [第二步：固定输入比较shot](#step-2)\n'
        '- [第三步：固定10shot比较模型](#step-3)\n\n')
    supplement=OUT/'Qwen/comparisons/input_supplement_20261103/README.md'
    if supplement.is_file():
        text+=link('27B输入消融缺项补跑进度与新增结果',supplement,path)+'\n\n'
    shot_supplement=OUT/'Qwen/comparisons/shot_supplement_20261103/README.md'
    if shot_supplement.is_file():
        text+=link('第二步1shot/20shot补跑进度与参考来源',shot_supplement,path)+'\n\n'
    analysis=dict(selection_rule='Within each scene, maximize mean AP50 of complete no-negative 27B 5/10shot pairs; '
        'break ties by mean AP50_95, then input name. Fixed prompt version and support seed 0.',
        datasets={dataset:qwen_input_selection(rows,dataset) for dataset in DATASETS},stages=[])
    displayed=[]

    def table(stage,title,pool,columns,**kwargs):
        nonlocal text
        displayed.extend(pool)
        text+=report.table(path,title,pool,columns,**kwargs)
        analysis['stages'].append(dict(stage=stage,title=title,conditions=[
            {key:r.get(key) for key in ('dataset','model','inference_shot_per_class','input_variant',
                'prompt_version','negative_references','support_seed','run_dir','status')}
            for r in pool]))

    text+=('<a id="step-1"></a>\n\n## 第一步：27B的5shot与10shot输入对比\n\n'
        '本阶段仅比较qwen3.8-27b。每个场景分别列5shot与10shot两张表，共4张；'
        '每张表内加粗最优指标，比较相同shot下输入组成的影响。'
        '后续只固定一种输入：在5shot与10shot均已完成的候选中，选择两档AP50算术平均最高者；'
        '平均AP50相同再比较平均AP50:95。此规则不会把两档各自最好的不同输入拼成shot曲线。'
        '只有5shot完成的输入保留，但须补齐10shot才能参与这项选择。\n\n')
    for dataset in DATASETS:
        for shot in (5,10):
            pool=[qwen_condition(rows,dataset,'qwen3.8-27b',shot,variant) for variant in QWEN_INPUTS]
            table('inputs',f'{SCENES[dataset]} · 27B输入对比 · {shot}shot/类',pool,
                  [('输入',lambda r:r['input_variant'])])
        selection=analysis['datasets'][dataset]
        best=selection['candidates'][0]
        selected=[qwen_condition(rows,dataset,'qwen3.8-27b',shot,selection['selected_input']) for shot in (5,10)]
        candidate_text='；'.join(f'{c["input_variant"]}={pct(c["mean_AP50"])}%'
                                for c in selection['candidates'])
        text+=(f'**{SCENES[dataset]}后续固定{selection["selected_input"]}**。'
            f'已完成两档的候选平均AP50：{candidate_text}。所选输入5shot/10shot的AP50分别为'
            f'{pct(selected[0]["AP50"])}/{pct(selected[1]["AP50"])}%，平均{pct(best["mean_AP50"])}%。\n\n')
    text+=('这是现有测试结果的探索性选择；'+
        ('未补齐的候选仍可能改变最优输入。' if any(len(s['candidates'])<len(QWEN_INPUTS)
            for s in analysis['datasets'].values()) else '两个场景的七种输入均已补齐5shot与10shot。')+
        '正式确认输入方案宜使用独立验证集，不能把按测试集选出的最高值当作独立验证结论。\n\n')

    text+=('<a id="step-2"></a>\n\n## 第二步：固定输入比较27B的shot（叉车至10shot，吊物至20shot）\n\n'
        '本阶段仍只比较qwen3.8-27b，每个场景一张表。采用第一步选定的输入，提示版本、负参考数与seed保持一致。0shot是同版本文字提示基线，'
        '不含O/C/B/J参考；其他shot使用固定的视觉输入。叉车比较0/1/5/10shot，吊物比较0/1/5/10/20shot。'
        'ΔAP50相对本场景0shot计算，单位为百分点；尚缺的条件保留空位，不假定增加shot必然提升。\n\n')
    extension=ROOT/'data/references/qwen_shot_extension_20261103/selection_provenance.json'
    if extension.is_file():
        text+=('吊物20shot保留原10shot为前缀，再从原25张正参考池中补选10张，seed0；参考与300张测试图像SHA无重叠。'
            '参考使用原有JPEG质量90、最长边1280与独立图片格式。叉车20shot按用户要求取消，叉车1shot直接引用已有完整结果。'+
            link('吊物20shot参考来源与抽样清单',extension,path)+'。\n\n')
    for dataset in DATASETS:
        variant=analysis['datasets'][dataset]['selected_input']
        pool=[qwen_condition(rows,dataset,'qwen3.8-27b',shot,'text_only' if shot==0 else variant)
              for shot in QWEN_SHOT_COUNTS[dataset]]
        zero=pool[0].get('AP50') if eligible(pool[0]) else None
        for r in pool:
            r['delta_vs_zero_pp']=(100*(r['AP50']-zero) if zero is not None and eligible(r) else None)
        table('shots',f'{SCENES[dataset]} · 27B · 固定{variant}',pool,
            [('shot/类',lambda r:r['inference_shot_per_class']),('输入',lambda r:r['input_variant']),
             ('ΔAP50/百分点',lambda r:'—' if r['delta_vs_zero_pp'] is None else f'{r["delta_vs_zero_pp"]:+.2f}')])
        valid=[r for r in pool if eligible(r)]
        curve=' → '.join(f'{r["inference_shot_per_class"]}shot {pct(r["AP50"])}%' for r in valid)
        pending=[str(r['inference_shot_per_class'])+'shot' for r in pool if not eligible(r)]
        text+=f'已完成AP50序列：{curve}。'+('待补齐：'+'、'.join(pending)+'。' if pending else f'{len(pool)}档shot均已完成。')+'\n\n'

    text+=('<a id="step-3"></a>\n\n## 第三步：固定10shot，比较27B、Flash、Max\n\n'
        '到本阶段才比较模型。每个场景一张表，只比较10shot/类的三个模型。'
        '固定叉车O+C+B、吊物O+B+J，在同场景、同shot、同版本、无负例和seed0条件下比较三个模型。'
        '不为某个模型单独改用其最优输入；不把缺失Max替换成旧99张结果。'
        '每张表至少两个模型有完整可比结果才加粗最优值，仅有一个模型时不判断模型优胜。\n\n')
    model_supplement=OUT/'Qwen/comparisons/model10_supplement_20261103/README.md'
    if model_supplement.is_file():
        text+=link('Flash与Max的10shot补跑队列（第二步当前批次完成后启动）',model_supplement,path)+'\n\n'
    for dataset in DATASETS:
        variant=QWEN_MODEL_INPUTS[dataset]
        pool=[qwen_condition(rows,dataset,model,10,variant)
              for model in ('qwen3.8-27b','qwen3.8-flash','qwen3.8-max')]
        table('models',f'{SCENES[dataset]} · 固定{variant} · 模型对比（10shot）',pool,
              [('shot/类',lambda r:r['inference_shot_per_class']),('模型',lambda r:r['model'])],
              group=['inference_shot_per_class'],minimum_comparators=2)
    for dataset in DATASETS:
        variant=QWEN_MODEL_INPUTS[dataset]
        missing=[f'{model}/{shot}shot' for shot in (10,)
                 for model in ('qwen3.8-27b','qwen3.8-flash','qwen3.8-max')
                 if not eligible(qwen_condition(rows,dataset,model,shot,variant))]
        text+=f'{SCENES[dataset]}固定{variant}：'+('待补齐'+ '、'.join(missing) if missing else '三个模型10shot均已完成')+'。\n\n'
    text+='Max的旧吊物99张结果单列在附录，不代替当前300张结果；其他输入的Flash结果也保留在附录。\n\n'
    analysis['missing_conditions']=[condition for stage in analysis['stages'] for condition in stage['conditions']
                                     if not condition['run_dir']]
    # Remaining saved runs are available for provenance without obscuring the three main steps.
    used={r['run_dir'] for r in displayed if r.get('run_dir')}
    remaining=[r for r in rows if r['run_dir'] not in used]
    if remaining:
        text+=('<details>\n<summary>附录A：其他输入、负例、提示版本和Flash已完成/历史尝试</summary>\n\n'
            '以下条件保留原始指标，不参与上述固定条件主线；按shot、负参考数和提示版本分别计算粗体。\n\n')
        for dataset in DATASETS:
            pool=sorted([r for r in remaining if r['dataset']==dataset],key=lambda r:(
                r['inference_shot_per_class'],r['negative_references'],r['prompt_version'],
                r['input_variant'],r['model'],not eligible(r),r['run_dir']))
            if pool:
                displayed.extend(pool)
                text+=report.table(path,f'{SCENES[dataset]} · 其他条件',pool,
                    [('模型',lambda r:r['model']),('shot/类',lambda r:r['inference_shot_per_class']),
                     ('输入',lambda r:r['input_variant']),('负参考',lambda r:r['negative_references']),
                     ('版本',lambda r:r['prompt_version'])],
                    group=['inference_shot_per_class','negative_references','prompt_version'])
        text+='</details>\n\n'
    historical=historical_max_rows()
    if historical:
        text+=('<details>\n<summary>附录B：Qwen3.8-Max旧吊物99张结果</summary>\n\n'
            'Max有8组旧99张、55个真值框的结果，7组完成、1组未完成。队列不同，'
            '仅在此单独展示，不参与当前300张模型比较。目录中的0shot文字条件按实际输入标记。\n\n')
        text+=report.table(path,'旧吊物99张 · qwen3.8-max',historical,
            [('输入',lambda r:r['input_variant']),('shot/类',lambda r:r['inference_shot_per_class']),
             ('负参考',lambda r:r['negative_references'])])
        text+=link('旧99张全部模型明细',OUT/'Qwen/comparisons/diaowu_99/results_summary.md',path)+'\n\n</details>\n\n'
    text+=link('Flash补跑原始记录',OUT/'Qwen/comparisons/flash_supplement_20261103/README.md',path)+' · '
    text+=link('完整真实运行CSV（含附录，不含待补跑占位）',path.with_suffix('.csv'),path)+'\n'
    report.save(path,text,displayed+historical)
    return historical,analysis


def historical_max_rows():
    directory=OUT/'Qwen/comparisons/diaowu_99'
    document=read(directory/'results_summary.json')
    rows=[]
    for indexed in document['rows']:
        if indexed['model']!='qwen3.8-max':continue
        source=(directory/indexed['run_dir']).resolve()
        metric_path=source/'metrics.json'
        metric=read(metric_path)
        row=copy.deepcopy(indexed)
        row.update(dataset='diaowu_99',algorithm='Qwen',historical=True,
            run_dir=str(source.relative_to(ROOT)),report=str((source/'evaluation.txt').relative_to(ROOT)),
            metric_file=str(metric_path.relative_to(ROOT)),metrics_sha256=sha(metric_path),
            input_variant=indexed['input'],training_shot_per_class=0,
            inference_shot_per_class=indexed['shot_per_class'],metric_evidence='元数据†',
            negative_error_rate=metric.get('negative_error_rate') if metric['settings']['score_threshold']==.5 else None,
            counts='/'.join(str(indexed.get(f'{k}_0_5','—')) for k in ['tp','fp','fn']))
        p,r=row['precision_0_5'],row['recall_0_5']
        row['f1_0_5']=2*p*r/(p+r) if p is not None and r is not None and p+r else 0.
        if metric['evaluated_image_count']!=99 or sum(v['ground_truth_count'] for v in metric['per_class'].values())!=55:
            raise ValueError('Historical Max cohort identity differs.')
        rows.append(row)
    rows.sort(key=lambda r:(r['inference_shot_per_class'],r['input_variant']))
    return rows


def ovd_prompt_metadata(row):
    """Human-reviewed prompt categories bound to the exact saved prompt text."""
    categories={
        'caefe918edccd435d60f54d7deaa7b299d7e9af7870a1f121f91f08fb5d7b583':
            ('简单提示','基础类别名；例：Chache1=叉车'),
        'c35f2333d2bb48c26780054f116475e91a2ec52e8a0a7d4038e545269602a9c2':
            ('简单提示','细分类名；例：Chache1=平衡重叉车'),
        'a26a99e47a3ced745975a6e80f2bcd9953bce9a1b5ccb5638a13fdb1b2fc46c4':
            ('复杂提示','简短结构描述；例：有护顶架和普通货叉的座驾叉车'),
        'b62c5d5a1171bf89d8b992fe623518e8bf5679d89a7587f3998f517ec5f1abd4':
            ('复杂提示','详细外观定义，包含结构、驾驶姿态、部件与颜色'),
        'fb4fc59a71a21423955cfa0a59d6cc2d0c08d9a135503b22f8b02bc805eac609':
            ('简单提示','类别名：吊物'),
        'dcb1adc5ee3c4f2d266b549dacd644513d844b6aa17fa9e2b25421f82a3dc527':
            ('复杂提示','属性限定短语：被吊起的圆筒吊物；不是长篇指令'),
    }
    text=(ROOT/row['prompt_file']).read_text(encoding='utf-8-sig')
    fingerprint=hashlib.sha256(text.encode('utf-8')).hexdigest()
    if fingerprint not in categories:
        raise ValueError(f'Review the changed OVD prompt before classifying: {row["prompt_file"]}')
    kind,summary=categories[fingerprint]
    return dict(text_prompt_type=kind,text_prompt_summary=summary,text_prompt_text=text.strip(),
                text_prompt_sha256=fingerprint)


def make_models(report):
    path, text = report.page('models',
        '对比Qwen3.8-27B、GroundingDINO、WeDetect、原始YOLOE和PET-DINO，不含任务微调。'
        '按提示shot=0/1/5/10分别成表；0shot包含文字提示，1/5/10shot为视觉参考。'
        '本页Qwen仅用27B：0shot固定文字提示，1/5/10shot固定O+C（参考原图+目标裁剪图），'
        '均无负例、support seed=0；叉车统一提示v2，吊物统一提示v1。'
        '按固定条件引用已有结果，不按最高指标更换输入或提示版本。其余方法保留所有已有设置。')
    text+=('提示类型按实际保存文本标注：简单提示是基础或细分类名；复杂提示是增加属性、结构或外观定义的描述。'
        '吊物v2虽列为复杂提示，实际仅为属性限定短语，不等同于Qwen的长篇场景指令。'
        '下表的提示版本可点击查看完整文本，同名版本的WeDetect/GroundingDINO使用相同文本。'
        '本页所需27B的两个场景×四档shot共8项均已有完成结果，直接补入，不重复请求模型。'
        '同shot仅对齐正参考张数，输入类型、模型规模、预训练和提示复杂度仍不同。\n\n')

    def prompt_type(row):
        if row['algorithm'] in ('WeDetect','GroundingDINO'):
            return row['text_prompt_type']+'（'+row['text_prompt_summary']+'）'
        if row['algorithm']=='Qwen':
            return '复杂文字提示（场景定义与检测规则）' if row['inference_shot_per_class']==0 else '视觉提示（O+C）'
        return '视觉提示'

    def model_input(row):
        version=row['prompt_version']
        if row.get('prompt_file'):
            version=link(version,ROOT/row['prompt_file'],path)
        return f'{version} / {row["input_variant"]} / 负参考{row["negative_references"]}'

    displayed = []
    for dataset in DATASETS:
        source = [r for r in report.rows if r['dataset']==dataset]
        qwen=[qwen_condition(source,dataset,'qwen3.8-27b',shot,'text_only' if shot==0 else 'O+C')
              for shot in (0,1,5,10)]
        others = [copy.deepcopy(r) for r in source if r['primary_topic']=='models']
        for row in others:
            if row['algorithm'] in ('WeDetect','GroundingDINO'):
                row.update(ovd_prompt_metadata(row))
        for shot in [0,1,5,10]:
            pool = [r for r in qwen+others if r['inference_shot_per_class']==shot]
            if not pool:
                continue
            pool.sort(key=lambda r:(r['algorithm'],r['model'],r['prompt_version'],r['input_variant'],r['run_dir']))
            displayed.extend(pool)
            text += report.table(path,f'{SCENES[dataset]} · {shot}shot',pool,
                [('算法/模型',lambda r:r['algorithm']+'/'+r['model']),
                 ('提示类型',prompt_type),('关键输入/提示文本',model_input)])
    text += ('GroundingDINO和WeDetect目前是文字0shot；1/5/10shot表中未填造它们的视觉提示结果。'
        '低置信度检测器在score≥0.5时可能没有输出，但AP仍非零，不能把AP和固定阈值下的P/R当作同一个量。\n')
    report.save(path,text,displayed)


def finetune_representatives(rows):
    """Keep actual AP winners for each method and inference budget separately."""
    pools=defaultdict(list)
    for row in rows:
        pools[fine_name(row),row['inference_shot_per_class']].append(row)
    return [row for pool in pools.values() for row in winners(pool)]


def compact_fine_parameters(row):
    if not row['training_shot_per_class']:
        return f'原始权重；输入{row["input_size"]}'
    scope=row['training_scope'] or '—'
    protocol={'same_image':'同图','cross_image':'跨图'}.get(row['training_protocol'],'')
    value=f'输入{row["input_size"]}；{scope}'+('/'+protocol if protocol else '')
    value+=f'；lr={row["learning_rate"]}；seed={row["seed"]}'
    augmentation=row.get('augmentation',{})
    if augmentation.get('mosaic',0)>0 or augmentation.get('fliplr',0)>0:
        value+=f'；mosaic={augmentation.get("mosaic",0)}/翻转={augmentation.get("fliplr",0)}'
    return value


def make_finetune(report, novel):
    path=DETAIL/FILES['finetune']
    text=('# YOLOE微调、YOLOv8对照与新类别扩展\n\n'
        '已见类别结果按场景、训练5shot/10shot分为四张表，之后展示Chache4新类别提示迁移，最后归纳YOLOE微调的优点与证据范围。'
        '每种微调方式、每档推理shot仅保留AP50或AP50:95最高的实际配置；'
        '两项最优来自同一实验时合并，来自不同实验时分别列出。每行全部指标来自同一次测试推理。\n\n'
        '微调方式只分为固定头（移除SAVPE）、视觉提示（更新SAVPE）、视觉提示（冻结SAVPE）和YOLOv8微调。'
        '学习率、更新范围、同图/跨图、输入尺寸和seed均作为配置选择，不再逐项展开所有尝试。'
        '训练K与推理提示K分开列出，训练10shot后用5shot提示仍属于10shot微调。\n\n')
    text+=link('全局比较',OUT/'FINAL_COMPARISON.md',path)+' · '+link('完整测试明细CSV',path.with_suffix('.csv'),path)+'\n\n'
    text+='[已见类别微调对比](#seen-class) · [Chache4新类别扩展](#novel-chache4) · [如何说明YOLOE微调的优点](#yoloe-advantages)\n\n'
    text+=('指标为百分数，均来自各自300张测试集；P/R/F1/F2与TP/FP/FN采用score≥0.5、IoU=0.5。'
        '每张表逐列加粗最优值。原始YOLOE同shot结果作为未微调基线；未训练YOLOv8初始化与其他搜索记录收在完整CSV中。\n\n')
    text+='<a id="seen-class"></a>\n\n## 一、已见类别：各方式最佳微调结果\n\n'
    full_rows=[]
    selected_rows=[]
    for dataset in DATASETS:
        for budget in (5,10):
            trained=[r for r in report.rows if r['dataset']==dataset and r['primary_topic']=='finetune'
                     and r['training_shot_per_class']==budget]
            # Preserve prompt K=5 and K=10 as separate comparisons under 10shot training.
            selected=finetune_representatives(trained)
            baselines=[copy.deepcopy(r) for r in report.rows if r['dataset']==dataset
                and r['algorithm']=='YOLOE' and r['training_shot_per_class']==0
                and r['inference_shot_per_class'] in ((5,) if budget==5 else (5,10)) and eligible(r)]
            for row in baselines:row['selected_by']=['baseline']
            selected=baselines+selected
            selected.sort(key=lambda r:(bool(r['training_shot_per_class']),fine_name(r),
                r['inference_shot_per_class'] or 0,-(r['AP50'] or 0),r['run_dir']))
            selected_rows.extend(selected)
            text+=report.table(path,f'{SCENES[dataset]} · 训练每类{budget}张',selected,
                [('方法',fine_name),('训练K/提示K',lambda r:r['training_budget_display']+'/'+r['prompt_budget_display']),
                 ('最佳配置',compact_fine_parameters),
                 ('保留依据',lambda r:'未微调基线' if r['selected_by']==['baseline'] else
                    '两项AP最高' if set(r['selected_by'])=={'AP50','AP50_95'} else
                    '、'.join('AP50:95最高' if k=='AP50_95' else 'AP50最高' for k in r['selected_by']))],
                metrics=SHORT,counts=True,status=False)
            full_rows.extend(trained+baselines)
        full_rows.extend(r for r in report.rows if r['dataset']==dataset and r['primary_topic']=='finetune'
                         and r['training_shot_per_class']==0)
    unique_full={r['run_dir']:r for r in full_rows}
    text+=f'已见类别部分共{len(selected_rows)}行最佳结果与基线；同名CSV保留{len(unique_full)}条完整测试记录及详细参数。\n\n'
    for dataset in DATASETS:
        base=OUT/f'YOLOE/finetune/{dataset}/comparisons/yoloe_search_20261029'
        text+=link(f'{SCENES[dataset]}全部实验',base/'ALL_EXPERIMENTS.md',path)+' · '
        text+=link('训练/dev搜索CSV',base/'all_trials.csv',path)+' · '
        text+=link('SAVPE方法说明',base/'SAVPE_METHODS.md',path)+'\n\n'
    text+=('full表示更新backbone、neck与头；neck表示冻结backbone。以上为已测试配置的最高值回顾，'
        '训练预算相同时，输入尺寸和搜索次数仍可能不同。下节的新类单类AP与本节全类别mAP分开比较。\n\n')
    text+=render_novel(report,path,novel)
    text+=finetune_advantages(report.rows,novel)
    report.save(path,text,full_rows)


def audit_cascade_stages(rows):
    """Only merge stages after comparing complete prediction records, not rounded AP."""
    groups=defaultdict(dict)
    for r in rows:
        if r['algorithm']=='OVD-MLLM':
            groups[r['dataset'],r.get('cascade_method') or 'pretrained'][r['input_variant']]=r
    audits={}
    for key,group in groups.items():
        if not {'source','baseline','filtered','deduplicated'}<=set(group):continue
        paths={stage:ROOT/group[stage]['run_dir']/'records.jsonl'
               for stage in ('source','baseline','filtered','deduplicated')}
        if not all(p.is_file() for p in paths.values()):continue
        records={stage:{(r['relative_path'],r['target_id']):r for r in
            (json.loads(line) for line in p.read_text().splitlines() if line.strip())}
            for stage,p in paths.items()}
        if any(len(r)!=300 or any(z['status']!='success' for z in r.values()) for r in records.values()):continue
        if any(set(r)!=set(records['source']) for r in records.values()):continue
        def operating(r):
            return sorted((o['label'],o['confidence'],tuple(o['bbox_pixels_xyxy']))
                          for o in r['objects'] if o['confidence']>=.5)
        audit=dict(dataset=key[0],method=key[1],image_count=300,
            record_files={stage:str(p.relative_to(ROOT)) for stage,p in paths.items()},
            record_sha256={stage:sha(p) for stage,p in paths.items()},
            box_counts={stage:sum(len(r['objects']) for r in rs.values()) for stage,rs in records.items()},
            filtered_deduplicated_identical=records['filtered']==records['deduplicated'],
            filtered_deduplicated_metrics_identical=read(ROOT/group['filtered']['metric_file'])==read(ROOT/group['deduplicated']['metric_file']),
            dedup_changed_images=sum(records['filtered'][p]!=records['deduplicated'][p] for p in records['filtered']),
            source_baseline_operating_predictions_identical=all(operating(records['source'][p])==operating(records['baseline'][p])
                                                               for p in records['source']))
        audit['dedup_removed_boxes']=audit['box_counts']['filtered']-audit['box_counts']['deduplicated']
        audits[key]=audit
    return audits


def make_cascade(report):
    path, text = report.page('cascade',
        '比较三种YOLOE微调获得候选后，再交由Qwen3.8-27B筛选、去重和修正框的效果。'
        '每个场景分别选训练10shot的固定头（移除SAVPE）、视觉提示（更新SAVPE）、视觉提示（冻结SAVPE）各一组；'
        '视觉提示推理使用10shot，Qwen统一仅每类1shot，seed0、无负例。原始未微调级联与直接Qwen结果保留在附录。')
    names = {'source':'S0 原始YOLOE', 'baseline':'B0 候选截断', 'filtered':'B1 Qwen筛选',
             'deduplicated':'B1D 去重', 'refined':'B2 框修正'}
    methods={'head':'固定头／移除SAVPE','savpe_train':'视觉提示／更新SAVPE','savpe_frozen':'视觉提示／冻结SAVPE'}
    order=list(names);displayed=[]
    audits=audit_cascade_stages(report.rows)
    audit_path=DETAIL/'04_yoloe_qwen_stage_audit.json'
    audit_path.write_text(json.dumps(dict(schema='cascade.stage_equivalence.v1',
        merge_rule='Merge filtered/deduplicated only when all 300 complete records are exactly identical; '
            'retain both actual runs in the topic CSV and raw ledger.',
        groups=list(audits.values())),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    progress=OUT/'OVD-MLLM/comparisons/finetuned_yoloe_qwen_20261103/README.md'
    if progress.is_file():text+=link('三种微调级联：6组补跑进度与选中权重',progress,path)+'\n\n'
    text+=('S0保留原始逐图预测，B0仅保留score≥0.01、每类最多100框的候选；'
        'B1只删除明确非目标，uncertain保留；B1D仅按Qwen明确给出的同类同实例关系且IoU≥0.5去重；'
        'B2只改坐标，候选集合、类别和原始分数固定，不补框。没有候选的图片不调用Qwen。'
        '候选score下限与P/R工作阈值0.5不同。S0→B0反映截断影响，B0→后续阶段反映Qwen处理影响。'
        '选择依据为现有测试AP50最高的实际配置，尺寸、seed和训练范围可能不同，不能当作等预算独立验证。\n\n')
    merged=[a for a in audits.values() if a['method']!='pretrained' and a['filtered_deduplicated_identical']
            and a['filtered_deduplicated_metrics_identical']]
    if merged:
        text+=(f'**相同阶段合并展示：**已逐图核验{len(merged)}组B1与B1D的全部300条预测完全相同，'
            '独立去重阶段额外删除0框；表中合为“B1筛选 / B1D去重（无额外变化）”，不算两次独立改进。'
            '原先相邻两行相同是实际输出未变，重复列出没有增加信息。B1D是本地后处理，未再次调用Qwen。'
            '本批去重关联多数涉及筛选已删除的框，其余关联不满足同类或IoU门限，因此没有额外删除。'
            'S0与B0仍分别保留：删去score<0.01的低分框会影响AP，而P/R使用score≥0.5，因此P/R可能完全相同。'+
            link('逐图核验、框数和SHA256',audit_path,path)+' · '+
            link('所有阶段完整CSV',path.with_suffix('.csv'),path)+'。\n\n')
    for dataset in DATASETS:
        stages=[copy.deepcopy(r) for r in report.rows if r['dataset']==dataset and r['algorithm']=='OVD-MLLM' and r.get('cascade_method')]
        stages.sort(key=lambda r:(list(methods).index(r['cascade_method']),order.index(r['input_variant'])))
        visible=[]
        for r in stages:
            r['stage_name']=names[r['input_variant']]
            audit=audits.get((dataset,r['cascade_method']),{})
            if audit.get('filtered_deduplicated_identical') and audit.get('filtered_deduplicated_metrics_identical'):
                if r['input_variant']=='deduplicated':continue
                if r['input_variant']=='filtered':r['stage_name']='B1筛选 / B1D去重（无额外变化）'
            visible.append(r)
        if stages:
            text+=report.table(path,SCENES[dataset]+' · 三种微调→Qwen27B',visible,
                [('微调方式',lambda r:methods[r['cascade_method']]),('阶段',lambda r:r['stage_name']),
                 ('提示数量',lambda r:r['prompt_budget_display'])])
            displayed.extend(stages)
        else:text+=f'### {SCENES[dataset]} · 三种微调→Qwen27B\n\n尚未产出新级联指标，进度见上方补跑记录。\n\n'
        for method in methods:
            group={r['input_variant']:r for r in stages if r['cascade_method']==method and eligible(r)}
            if 'baseline' not in group:continue
            a=group['baseline']
            audit=audits.get((dataset,method),{})
            if audit.get('filtered_deduplicated_identical') and audit.get('filtered_deduplicated_metrics_identical'):
                text+=(f'{methods[method]}：B1保留{audit["box_counts"]["filtered"]}框，'
                       f'B1D仍为{audit["box_counts"]["deduplicated"]}框，去重额外删除0框。'+
                       link('B1D原始评估',ROOT/group['deduplicated']['report'],path)+'。\n\n')
            for stage in ('filtered','deduplicated','refined'):
                if stage not in group:continue
                if stage=='deduplicated' and audit.get('filtered_deduplicated_identical') and audit.get('filtered_deduplicated_metrics_identical'):continue
                b=group[stage]
                stage_name=b['stage_name'] if stage=='filtered' else names[stage]
                text+=(f'{methods[method]} · {stage_name}相对B0：AP50变化{100*(b["AP50"]-a["AP50"]):+.2f}个百分点，'
                    f'P变化{100*(b["precision_0_5"]-a["precision_0_5"]):+.2f}，R变化{100*(b["recall_0_5"]-a["recall_0_5"]):+.2f}。\n\n')
    text+='<details>\n<summary>附录：未微调YOLOE级联与直接Qwen对照</summary>\n\n'
    for dataset in DATASETS:
        stages=[copy.deepcopy(r) for r in report.rows if r['dataset']==dataset and r['algorithm']=='OVD-MLLM' and not r.get('cascade_method')]
        stages.sort(key=lambda r:order.index(r['input_variant']))
        direct=qwen_representatives([r for r in report.rows if r['dataset']==dataset and r['model']=='qwen3.8-27b'
            and r['inference_shot_per_class'] in [1,10]])
        historical_visible=[]
        audit=audits.get((dataset,'pretrained'),{})
        for r in stages:
            r['stage_name']=names[r['input_variant']]
            if audit.get('filtered_deduplicated_identical') and audit.get('filtered_deduplicated_metrics_identical'):
                if r['input_variant']=='deduplicated':continue
                if r['input_variant']=='filtered':r['stage_name']='B1筛选 / B1D去重（无额外变化）'
            historical_visible.append(r)
        for r in direct:r['stage_name']='直接Qwen对照'
        pool=historical_visible+direct;displayed.extend(stages+direct)
        text+=report.table(path,SCENES[dataset]+' · 历史/直接对照',pool,
            [('处理阶段',lambda r:r['stage_name']),('参考数量',lambda r:r['prompt_budget_display']),
             ('关键输入',lambda r:('仅YOLOE' if r['input_variant'] in {'source','baseline'} else 'YOLOE候选→Qwen27B')
              if r['algorithm']=='OVD-MLLM' else f'{r["input_variant"]}; 负参考{r["negative_references"]}')])
    text+='</details>\n\n'
    text+=('三种微调级联与旧未微调级联分别报告，不能将旧模型的增益直接套用到微调模型。'
        '直接Qwen的输入和计算预算不同；筛选/框修正的收益须结合AP、召回及误检一起判断。\n')
    report.save(path,text,displayed)


def load_novel():
    base = OUT/'YOLOE/finetune/chache_val300/comparisons/novel_chache4_20261102'
    document = read(base/'results.json')
    if read(base/'verification.json')['status']!='passed':
        raise ValueError('Novel-class evidence not verified.')
    rows = []
    for original in document['results']:
        row = copy.deepcopy(original)
        source = ROOT/row['source']
        metric = read(source/'novel/metrics.json')
        if sha(source/'novel/metrics.json') != row['novel_metrics_sha256']:
            raise ValueError('Novel metrics changed.')
        row.update(run_dir=row['source'],report=str((source/'novel/evaluation.txt').relative_to(ROOT)),
            eligible_for_comparison=metric['evaluation_status']=='complete' and metric['evaluated_image_count']==300,
            status='完成',metric_evidence='逐图核验',counts='/'.join(str(metric[k]) for k in ['tp','fp','fn']))
        for key,target in [('mAP50','AP50'),('mAP75','AP75'),('mAP50_95','AP50_95'),
                ('precision','precision_0_5'),('recall','recall_0_5'),('f1','f1_0_5'),('f2','f2_0_5'),
                ('negative_error_rate','negative_error_rate')]:
            row[target] = metric[key]
        rows.append(row)
    return rows


def render_novel(report, path, rows):
    text=('<a id="novel-chache4"></a>\n\n## 二、Chache4新类别：保留SAVPE的提示扩展能力\n\n'
        '新类固定为Chache4，基础四类为Chache1/2/3与Qianyinche；四类微调后仅添加Chache4图框参考，不再更新权重。'
        '5shot训练图数为20/25，10shot为40/50，四类与五类条件均保留并更新SAVPE和backbone。'
        '共同88张dev排除Chache4正图，只按基础四类选权重。新类指标在完整300张测试图、30个Chache4真值框上统计；'
        '基础类图和背景图保留用于统计新类误检。新类单类AP不参加上节或全局五类mAP排序。\n\n')
    for shot in [5,10]:
        pool=[r for r in rows if r['shot']==shot]
        text+=report.table(path,f'{shot}shot · 新类Chache4',pool,
            [('条件',lambda r:r['label']),('训练图数',lambda r:r['training_image_count']),
             ('Chache4训练K',lambda r:r['novel_training_shot'])],
            metrics=METRICS+[('基础四类AP50↑','base_mAP50'),('全五类AP50↑','all_mAP50')])
    csv_path=DETAIL/'05_novel_chache4.csv'
    export_csv(csv_path,rows)
    text+=link('新类指标CSV（独立标签范围）',csv_path,path)+' · '
    text+=link('完整协议、训练配置与可视化',OUT/'YOLOE/finetune/chache_val300/comparisons/novel_chache4_20261102/README.md',path)+'\n\n'
    text+=('“新类”指本次微调未见，不保证原始预训练未见；参考标注范围经核验，但未标注同图物体仍需人工确认。'
        '这不是按新视频/新相机重划分的实验；只有一个held-out类别与一个seed，结论限于本次提示迁移。\n\n')
    return text


def finetune_advantages(rows, novel):
    def best(dataset,algorithm,savpe=None,budget=5):
        candidates=[r for r in rows if r['dataset']==dataset and r['algorithm']==algorithm
            and r['training_shot_per_class']==budget and eligible(r) and (savpe is None or r['savpe']==savpe)]
        return max(candidates,key=lambda r:r['AP50'])
    text=('<a id="yoloe-advantages"></a>\n\n## 三、如何说明YOLOE微调的优点\n\n'
        '**建议围绕“小样本任务适配”和“保留SAVPE后的新类别提示扩展”两点展开，分别给出证据。**\n\n'
        '**1. 已见类别：YOLOE具备小样本适配优势，但需要区分微调路线。** '
        '在每类5张、640输入的现有对照中：\n\n')
    for dataset in DATASETS:
        y8=best(dataset,'YOLOv8');fixed=best(dataset,'YOLOE','移除');vp=best(dataset,'YOLOE','训练')
        text+=(f'- {SCENES[dataset]}：固定头YOLOE的AP50为{pct(fixed["AP50"])}%，YOLOv8为{pct(y8["AP50"])}%，'
            f'高{100*(fixed["AP50"]-y8["AP50"]):.2f}个百分点；'
            f'保留并更新SAVPE的视觉提示YOLOE为{pct(vp["AP50"])}%，相对YOLOv8变化'
            f'{100*(vp["AP50"]-y8["AP50"]):+.2f}个百分点。\n')
    text+=('\n固定头成绩体现所测YOLOE方案在少量标注下的适配效果；该路线移除了SAVPE，'
        '不能用它的提升证明SAVPE有效。保留SAVPE的5shot路线在吊物上优于当前YOLOv8对照，在叉车上未胜出。'
        '这些是已测配置的结果；预训练与搜索预算不同，不能据此宣称所有条件下均优于YOLOv8。\n\n'
        '**2. 新类别扩展：保留SAVPE后，基础类微调权重可以直接配合新类参考使用。** '
        '四类微调阶段不使用Chache4监督；完成后冻结权重，只在推理时加入带框参考。'
        'SAVPE为新增类别编码视觉提示，无须再次针对Chache4执行梯度更新：\n\n')
    groups={shot:{r['condition']:r for r in novel if r['shot']==shot} for shot in (5,10)}
    for shot,group in groups.items():
        a,b=group['pretrained'],group['base']
        text+=(f'- {shot}shot：Chache4 AP50由原始权重的{pct(a["AP50"])}%提高到{pct(b["AP50"])}%，'
            f'增加{100*(b["AP50"]-a["AP50"]):.2f}个百分点；AP50:95由{pct(a["AP50_95"])}%提高到{pct(b["AP50_95"])}%。\n')
    text+=('\n这组结果支持的优点是：**基础任务微调后仍可通过新增参考扩展类别，并且本次新类AP高于原始YOLOE视觉提示。** '
        '这验证的是保留SAVPE的完整方案；backbone等参数也同时微调，不能把提升单独归因于SAVPE参数更新。'
        '新增类别仍需要5/10张带框参考，所以应表述为“无需新类再训练”，不能写成“无需新类标注”。'
        '当前YOLOv8对照采用固定类别输出，尚未做同一四类训练→新增类适配协议，因此没有YOLOv8新类迁移的直接数值对照。\n\n'
        '**3. 同时说明精度与召回的代价。** ')
    a,b=groups[5]['all'],groups[10]['all']
    base10=groups[10]['base']
    text+=(f'五类一起微调的Chache4 AP50为{pct(a["AP50"])}/{pct(b["AP50"])}%（5/10shot），'
        '仍高于四类微调后只加提示的结果。'
        f'10shot四类微调条件在score≥0.5时P={pct(base10["precision_0_5"])}%、R={pct(base10["recall_0_5"])}%，'
        f'TP/FP/FN={base10["counts"]}：误检减少，但30个真值中仍漏检20个。'
        '因此不能把AP提升等同于固定阈值召回提升，也不能声称已证明新视频、新相机或所有新类别上的泛化。\n\n'
        '**可用于总结的表述：**\n\n')
    g5,g10=groups[5],groups[10]
    text+=('> YOLOE微调的价值同时体现在小样本任务适配与视觉提示扩展能力上。'
        '已见类别对照中，固定头路线取得较好的少样本检测结果；保留SAVPE的路线则允许基础类微调完成后，'
        '通过新增带框参考检测本次微调未见的类别，无须再次更新权重。'
        f'在Chache4实验中，5shot和10shot提示的AP50分别达到{pct(g5["base"]["AP50"])}%和{pct(g10["base"]["AP50"])}%，'
        f'高于原始YOLOE的{pct(g5["pretrained"]["AP50"])}%和{pct(g10["pretrained"]["AP50"])}%。'
        '现有结果支持本场景中的提示迁移价值，但尚不足以证明保留SAVPE在所有任务上精度都优于YOLOv8；'
        '固定阈值召回与多类别、多seed泛化仍需进一步验证。\n')
    return text


def make_supervised(report):
    path,text = report.page('supervised',
        '保留JSTY定制网络FP32 ONNX结果，并列展示相同测试队列上的固定类别YOLOE与YOLOv8作为参考。'
        'JSTY输入960×1280，其原始训练预算与训练/测试重叠尚未绑定到该权重的可核验训练快照。'
        '分数可以并列阅读，不能据此声称在相同标注预算下优于fewshot。')
    displayed=[]
    for dataset in DATASETS:
        pool=[r for r in report.rows if r['dataset']==dataset and r['algorithm']=='JSTY-YOLO']
        for algorithm in ['YOLOE','YOLOv8']:
            pool += winners([r for r in report.rows if r['dataset']==dataset and r['algorithm']==algorithm
                             and r['training_shot_per_class']==10 and (algorithm=='YOLOv8' or r['savpe']=='移除')])
        displayed.extend(pool)
        text += report.table(path,SCENES[dataset],pool,
            [('模型/方案',lambda r:r['model']+' / '+(fine_name(r) if r['algorithm']!='JSTY-YOLO' else '定制监督')),
             ('训练/类',lambda r:r['training_budget_display']),('输入',lambda r:r['input_size'])])
    report.save(path,text,displayed)


def representatives(rows):
    groups = defaultdict(list)
    for row in rows:
        algorithm = row['algorithm']
        if algorithm=='Qwen':
            family = f'Qwen/{row["model"]}'
        elif row['primary_topic']=='finetune':
            if not row['training_shot_per_class']:
                # Negative control remains in the full fine-tuning table.
                continue
            family = f'{fine_name(row)} / 训练{row["training_shot_per_class"]}shot'
        elif algorithm=='OVD-MLLM':
            if row['input_variant'] not in {'filtered','deduplicated','refined'}:
                continue
            family = 'YOLOE→Qwen'+(' / '+{'head':'固定头','savpe_train':'更新SAVPE','savpe_frozen':'冻结SAVPE'}[row['cascade_method']]
                if row.get('cascade_method') else ' / 未微调')
        elif algorithm=='JSTY-YOLO':
            family = 'JSTY定制监督参考'
        else:
            family = algorithm+' / 未微调'
        groups[row['dataset'],family].append(row)
    selected=[]
    for (dataset,family),pool in groups.items():
        for row in winners(pool):
            row['family']=family
            row['representative_rule']='及'.join('AP50:95' if k=='AP50_95' else k for k in row['selected_by'])+'最高'
            selected.append(row)
    return selected


def global_parameters(row):
    model = row['model'].replace('qwen3.8-','Qwen-')
    if row['algorithm']=='Qwen':
        return f'{model}; {row["prompt_version"]}; {row["input_variant"]}; 正{row["inference_shot_per_class"]}/负{row["negative_references"]}'
    if row['algorithm']=='OVD-MLLM':
        stage={'filtered':'B1筛选','deduplicated':'B1D去重','refined':'B2框修正'}[row['input_variant']]
        method={'head':'固定头','savpe_train':'更新SAVPE','savpe_frozen':'冻结SAVPE'}.get(row.get('cascade_method'),'未微调')
        return f'{method}; 训练{row["training_shot_per_class"]}/类; {stage}; {row["prompt_budget_display"]}; 候选≥0.01'
    if row['algorithm']=='JSTY-YOLO':
        return 'FP32 ONNX; 960×1280; 监督预算未绑定'
    if row['primary_topic']=='finetune':
        augmentation = row.get('augmentation', {})
        hint = f'; mosaic={augmentation["mosaic"]},翻转={augmentation.get("fliplr",0)}' if augmentation.get('mosaic',0)>0 or augmentation.get('fliplr',0)>0 else ''
        return (f'训练{row["training_shot_per_class"]}/类; 提示{row["prompt_budget_display"]}; '
                f'{row["training_scope"]}/{row["training_protocol"] or "固定头"}; '
                f'lr={row["learning_rate"]}; 输入{row["input_size"]}; seed{row["seed"]}')+hint
    return f'{model}; 提示{row["prompt_budget_display"]}; {row["prompt_version"]}; 输入{row["input_size"] or "未记录"}'


def matched_qwen_pairs(rows):
    """Pair completed runs by prompt and cohort identity, never by their score."""
    models = ('qwen3.8-27b', 'qwen3.8-flash')
    fields = ('dataset', 'prompt_version', 'inference_shot_per_class', 'input_variant',
              'negative_references', 'support_seed', 'prompt_sha256', 'dataset_sha256')
    pools = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (row['algorithm']=='Qwen' and row['model'] in models and eligible(row)
                and row['inference_shot_per_class'] in (1,5,10)
                and row['input_variant'] in ('O+B','O+C') and row['negative_references']==0
                and row.get('prompt_sha256') and row.get('dataset_sha256')):
            pools[tuple(row.get(k) for k in fields)][row['model']].append(row)
    pairs=[]
    for identity, pool in sorted(pools.items()):
        if not all(model in pool for model in models):
            continue
        if any(len(pool[model])!=1 for model in models):
            raise ValueError(f'Ambiguous Qwen matched condition; define a repeat protocol: {identity}')
        pair=dict(zip(fields,identity))
        pair['models']={model: {key:pool[model][0][key] for key in
            ('run_dir','report','metric_evidence','metrics_sha256',*[key for _,key in METRICS])}
            for model in models}
        a,b=(pool[model][0] for model in models)
        pair['flash_minus_27b']={key:b[key]-a[key] if a[key] is not None and b[key] is not None else None
                                for _,key in METRICS}
        pairs.append(pair)
    return pairs


def export_qwen_pairs(pairs):
    """Keep the earlier matched-condition export available as supporting data."""
    flattened=[]
    for pair in pairs:
        record={k:v for k,v in pair.items() if k not in ('models','flash_minus_27b')}
        for model,prefix in [('qwen3.8-27b','27b'),('qwen3.8-flash','flash')]:
            for key,value in pair['models'][model].items():
                record[f'{prefix}_{key}_pct' if key in RATE_KEYS else f'{prefix}_{key}']=(
                    round(100*value,6) if key in RATE_KEYS and value is not None else value)
        for key,value in pair['flash_minus_27b'].items():
            record[f'flash_minus_27b_{key}_pp']=round(100*value,6) if value is not None else None
        flattened.append(record)
    path=DETAIL/'qwen_matched_conditions.csv'
    export_csv(path,flattened)
    return path


def conclusions(rows, novel):
    lines=['## 根据现有结果可得的结论','']
    for dataset in DATASETS:
        scene=SCENES[dataset]
        base=[r for r in rows if r['dataset']==dataset and r['primary_topic'] in {'qwen','models'} and eligible(r)]
        a=max(base,key=lambda r:r['AP50']);b=max(base,key=lambda r:r['AP50_95'])
        lines += [f'- {scene}未微调方法：AP50最高为{a["algorithm"]}/{a["model"]}，{pct(a["AP50"])}%；'
            f'AP50:95最高为{b["algorithm"]}/{b["model"]}，{pct(b["AP50_95"])}%。'
            '两项最优可以来自不同配置，详见子表，不将它们拼成一次实验。']
        for budget in [5,10]:
            def best(algorithm,savpe=None):
                return max([r for r in rows if r['dataset']==dataset and r['algorithm']==algorithm
                    and r['training_shot_per_class']==budget and eligible(r) and (savpe is None or r['savpe']==savpe)],key=lambda r:r['AP50'])
            y8=best('YOLOv8');head=best('YOLOE','移除');vp=best('YOLOE','训练')
            lines += [f'- {scene}训练{budget}shot/类：YOLOv8、固定头YOLOE、训练SAVPE的视觉提示YOLOE，'
                f'已测试最高AP50分别为{pct(y8["AP50"])}/{pct(head["AP50"])}/{pct(vp["AP50"])}%。'
                '这是各自实际配置的测试最高值，尺寸与搜索预算仍需查看，不能当作等预算统计显著结论。']
        cascade_groups=defaultdict(dict)
        for r in rows:
            if r['dataset']==dataset and r['algorithm']=='OVD-MLLM' and eligible(r):
                cascade_groups[r.get('cascade_method') or 'pretrained'][r['input_variant']]=r
        for method,stages in cascade_groups.items():
            if not {'baseline','refined'} <= set(stages):continue
            b0,b2=stages['baseline'],stages['refined']
            title={'pretrained':'未微调','head':'固定头','savpe_train':'更新SAVPE','savpe_frozen':'冻结SAVPE'}[method]
            lines += [f'- {scene}{title}YOLOE→Qwen：B2相对B0的AP50从{pct(b0["AP50"])}%到{pct(b2["AP50"])}%，'
                f'变化{100*(b2["AP50"]-b0["AP50"]):+.2f}个百分点；R从{pct(b0["recall_0_5"])}%到{pct(b2["recall_0_5"])}%。'
                '因此同时报告筛选收益与召回代价。']

    for shot in [5,10]:
        group={r['condition']:r for r in novel if r['shot']==shot}
        a,b=group['pretrained'],group['base']
        lines += [f'- Chache4新增类{shot}shot提示：四类微调后不再优化新类，AP50较原始权重'
            f'{pct(a["AP50"])}%→{pct(b["AP50"])}%，支持本场景的提示迁移价值；'
            f'score≥0.5召回为{pct(b["recall_0_5"])}%。此单类指标不参与上面的全类别排名。']
    lines += ['- 现有证据支持分别讨论“已见类检测精度”“固定阈值漏检/误检”“无需新类梯度更新的提示扩展”。'
        '不能用单一最好数字推出保留SAVPE在所有任务上优于YOLOv8。JSTY是训练预算不同的监督参考。',
        '- 未完成结果不用于排名；多数Qwen结果仅有指标元数据。输入调整、搜索最高值与单个新类均是现有测试集上的探索回顾，仍需独立视频/相机与更多seed验证泛化。','']
    return '\n'.join(lines)


def build(raw_rows):
    DETAIL.mkdir(parents=True,exist_ok=True)
    rows=[enrich(row) for row in raw_rows]
    if len({r['run_dir'] for r in rows})!=len(rows):
        raise ValueError('Duplicate experiment sources.')
    for row in rows:
        if row['metric_file'] and sha(ROOT/row['metric_file'])!=row['metrics_sha256']:
            raise ValueError(f'Stale source metrics: {row["run_dir"]}')
    report=Report(rows)
    historical,qwen_analysis=make_qwen(report)
    novel=load_novel()
    make_models(report);make_finetune(report,novel);make_cascade(report)
    make_supervised(report)
    selected=representatives(rows)
    page=OUT/'FINAL_COMPARISON.md'
    now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M')
    text=f'# 实验结论与全局比较\n\n更新：{now}（北京时间）。先按子任务分析，再在此比较关键代表方案。\n\n'
    for topic in FILES:
        text += '- '+link(TITLES[topic],DETAIL/FILES[topic],page)
        if topic=='finetune':
            text+=' · '+link('Chache4新类扩展',DETAIL/(FILES['finetune']+'#novel-chache4'),page)
            text+=' · '+link('YOLOE优势说明',DETAIL/(FILES['finetune']+'#yoloe-advantages'),page)
        text+='\n'
    text += '\n'+rules()+'\n\n'
    if historical:
        text += ('Qwen3.8-Max已找到8组旧吊物99张实验，单列在'+
            link('Qwen输入子表的Max历史部分',DETAIL/FILES['qwen'],page)+
            '；与当前300张队列不同，未纳入下面的同队列排名。\n\n')
    text += (f'原始全类别明细{len(rows)}条保留在'+link('全部运行明细',OUT/'ALL_RESULTS.md',page)+'；本页仅选取每个场景、每种方案/训练预算的'
        'AP50或AP50:95实际最高配置的并集。两项最优来自不同实验时列两行，每行全部指标来自同一次运行；不逐列拼接。'
        '†沿用子表的元数据证据限制。以下为已见类别的完整300张测试结果，按场景分别计算粗体。'
        '训练预算、输入尺寸、文字/视觉提示与搜索次数并未统一，JSTY为监督参考；粗体表示本表数值最高，不等于公平预算下的普遍优胜。\n\n')
    text += report.table(page,'已见类全局代表方案',selected,
        [('场景',lambda r:SCENES[r['dataset']]),('子方案',lambda r:link(r['family'],DETAIL/FILES[r['primary_topic']],page)),
         ('关键参数',global_parameters),('代表依据',lambda r:r['representative_rule']+('†' if r['metric_evidence']=='元数据†' else ''))],
        metrics=SHORT,counts=False,status=False,group=['dataset'])
    text += link('全局CSV',OUT/'FINAL_COMPARISON.csv',page)+' · '+link('来源与比较规则JSON',OUT/'FINAL_COMPARISON.json',page)+'\n\n'
    qwen_pairs=matched_qwen_pairs(rows)
    pair_csv=export_qwen_pairs(qwen_pairs)
    text+='## Qwen分析顺序：输入 → shot → 模型\n\n'
    text+=link('按三步阅读Qwen实验表',DETAIL/FILES['qwen'],page)+'：先用27B的5shot/10shot选择固定输入，再比较shot（叉车0/1/5/10，吊物0/1/5/10/20），最后固定10shot比较27B、Flash和Max（叉车O+C+B、吊物O+B+J）。\n\n'
    for dataset,selection in qwen_analysis['datasets'].items():
        text+=f'- {SCENES[dataset]}：当前固定{selection["selected_input"]}，在已完成5/10shot配对中平均AP50最高，为{pct(selection["candidates"][0]["mean_AP50"])}%。\n'
    text+=('\n后续阶段尚缺的shot与模型条件在子表逐项留位；第一阶段补齐后，固定输入按相同选择规则自动更新。'
        '全局代表方案仍按各算法已完成配置的最高值选择，可能使用其他输入；固定条件模型比较以Qwen子表第三步为准。\n\n')
    text+=link('其他已完成条件的27B/Flash配对CSV',pair_csv,page)+'（保留原12组配对供查阅）。\n\n' 
    text += conclusions(rows,novel)
    text += '\n更新：`python scripts/build_final_comparison.py`。更新完整明细及全部子表：`python scripts/summarize_all_algorithms.py`。只读取保存指标，不运行模型或API。\n'
    page.write_text(text,encoding='utf-8')
    export_csv(OUT/'FINAL_COMPARISON.csv',selected)
    coverage=Counter(primary(r) for r in rows)
    for row in rows:
        expected_page=FILES[row['primary_topic']]
        if row['run_dir'] not in report.members[expected_page]:
            raise ValueError(f'Run missing from its primary task: {row["run_dir"]}')
    manifest=dict(schema='industrial_fewshot.thematic_comparison.v1',updated=now,
        source_row_count=len(rows),global_row_count=len(selected),global_rows=selected,
        qwen_matched_pairs=qwen_pairs,qwen_analysis=qwen_analysis,
        primary_coverage=dict(coverage),novel_row_count=len(novel),
        primary_rows=rows,novel_rows=novel,historical_rows=historical,tables=report.tables,
        rules=dict(representatives='union of actual AP50 and AP50_95 winners per dataset/family/training budget',
                   finetune_selection='Per dataset and training budget, union of actual AP50 and AP50_95 winners '
                       'per method family and inference shot; all other test runs retained in the topic CSV',
                   model_comparison_qwen='qwen3.8-27b only; text_only at 0shot; O+C at 1/5/10shot; '
                       'no negative references, support seed 0; chache v2 and diaowu v1',
                   highlight='visible table eligible rows, separate datasets and novel scopes; displayed precision ties',
                   incomplete='visible but excluded from selection and highlighting',
                   global_scope='full-class seen-category metrics only; novel-class metrics remain separate',
                   metric_values='JSON uses raw rates 0..1; CSV rate columns end in _pct and use percent; Markdown shows percent'))
    (OUT/'FINAL_COMPARISON.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    index='# 按子任务阅读实验结果\n\n'+link('全局代表方案与结论',page,DETAIL/'README.md')+'\n\n'
    for topic in FILES:
        index+='- '+link(TITLES[topic],DETAIL/FILES[topic],DETAIL/'README.md')+' · '+link('CSV',DETAIL/Path(FILES[topic]).with_suffix('.csv'),DETAIL/'README.md')
        if topic=='finetune':
            index+=' · '+link('新类扩展',DETAIL/(FILES['finetune']+'#novel-chache4'),DETAIL/'README.md')
            index+=' · '+link('优势说明',DETAIL/(FILES['finetune']+'#yoloe-advantages'),DETAIL/'README.md')
        index+='\n'
    index += f'\n覆盖原完整明细{len(rows)}条独立运行，微调文档另设章节保留新类实验{len(novel)}组；不同主题可有必要的基线引用，但同一表内同一来源只出现一次。\n\n'
    if historical:index += f'另保留Max的{len(historical)}组旧99张结果，按独立队列展示。\n\n'
    index += link('原始全部运行明细',OUT/'ALL_RESULTS.md',DETAIL/'README.md')+'\n'
    (DETAIL/'README.md').write_text(index,encoding='utf-8')
    navigation=OUT/'README.md'
    nav=navigation.read_text(encoding='utf-8')
    entry='[全局代表方案与最终结论](FINAL_COMPARISON.md) · [按子任务比较](_shared/comparisons/final_results/README.md)'
    if entry not in nav:
        nav=nav.replace('\n\n','\n\n'+entry+'\n\n',1)
    nav=nav.replace('[两个300张测试集：全部算法、提示与微调设置总表]', '[全部运行明细（完整保留）]')
    nav=nav.replace('## 跨方法总表','## 按队列的原始索引')
    navigation.write_text(nav,encoding='utf-8')
    topic_indexes = {
        OUT/'Qwen/comparisons/README.md': ['qwen','models'],
        OUT/'GroundingDINO/comparisons/README.md': ['models'],
        OUT/'WeDetect/comparisons/README.md': ['models'],
        OUT/'PET-DINO/comparisons/README.md': ['models'],
        OUT/'YOLOE/comparisons/README.md': ['models'],
        OUT/'YOLOE/finetune/README.md': ['finetune'],
        OUT/'OVD-MLLM/comparisons/README.md': ['cascade'],
        OUT/'JSTY-YOLO/comparisons/README.md': ['supervised']}
    for index,topics in topic_indexes.items():
        if not index.is_file():
            continue
        entry='专题汇总（逐列加粗最优）：'+' · '.join(link(TITLES[t],DETAIL/FILES[t],index) for t in topics)
        content=index.read_text(encoding='utf-8').replace('[Qwen输入调整]',f'[{TITLES["qwen"]}]')
        if index==OUT/'YOLOE/finetune/README.md':
            content=re.sub(r'^专题汇总（逐列加粗最优）：.*$',entry,content,flags=re.MULTILINE)
        index.write_text(content,encoding='utf-8')
        if entry not in content:
            index.write_text(content.rstrip()+'\n\n'+entry+'\n',encoding='utf-8')
    flash_progress=OUT/'Qwen/comparisons/flash_supplement_20261103/README.md'
    if flash_progress.is_file():
        index=OUT/'Qwen/comparisons/README.md'
        entry=link('Flash补跑：叉车O+C与吊物O+B/O+C，各1/5/10shot',flash_progress,index)
        content=index.read_text(encoding='utf-8').replace('[Qwen输入调整]',f'[{TITLES["qwen"]}]')
        if index==OUT/'YOLOE/finetune/README.md':
            content=re.sub(r'^专题汇总（逐列加粗最优）：.*$',entry,content,flags=re.MULTILINE)
        index.write_text(content,encoding='utf-8')
        if entry not in content:index.write_text(content.rstrip()+'\n\n'+entry+'\n',encoding='utf-8')
    ledger=OUT/'ALL_RESULTS.md'
    content=ledger.read_text(encoding='utf-8')
    entry='[精简全局比较与结论](FINAL_COMPARISON.md) · [分子任务对照](_shared/comparisons/final_results/README.md)'
    if entry not in content:
        content=content.replace('\n\n','\n\n'+entry+'\n\n本页为原始完整运行明细，主题结论请从上方入口阅读。\n\n',1)
        ledger.write_text(content,encoding='utf-8')
    print(f'Thematic comparison: {len(rows)} source runs + {len(novel)} novel conditions; '
          f'{len(selected)} global representatives; {len(report.tables)} tables; coverage={dict(coverage)}')


def main():
    build(read(OUT/'ALL_RESULTS.json')['rows'])


if __name__=='__main__':
    main()

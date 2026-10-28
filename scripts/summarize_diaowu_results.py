"""Summarize saved results on the current diaowu cohort without invoking a model."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import io
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from industrial_fewshot.data.labelme import read_json, write_json
from industrial_fewshot.run_discovery import saved_runs
from summarize_chache_results import records, pct, VARIANTS


def collect(root):
    marker = read_json(root / 'dataset.json')
    manifest = read_json(Path(marker['dataset_manifest']))
    query_hashes = {s['relative_path']: s['image_sha256'] for s in manifest['samples']}
    gt_count = sum(len(s['ground_truth']) for s in manifest['samples'])
    rows = []
    for file in saved_runs(root):
        directory = file.parent
        run = read_json(file)
        target = run['targets'][0]
        metric_path = directory / 'metrics.json'
        metrics = read_json(metric_path) if metric_path.is_file() else {}
        data = records(directory)
        successes = sum(r.get('status') == 'success' for r in data)
        failures = sum(r.get('status') == 'failed' for r in data)
        counts = {name: c['ground_truth_count'] for name, c in metrics.get('per_class', {}).items()}
        complete = (run.get('dataset_sha256') == marker['dataset_sha256']
                    and metrics.get('evaluation_status') == 'complete'
                    and successes == len(data) == len(query_hashes)
                    and len({r['relative_path'] for r in data}) == len(query_hashes)
                    and {r['relative_path']: r.get('image_sha256') for r in data} == query_hashes
                    and counts == {'hanging object': gt_count} and failures == 0)
        exp = run.get('experiment', {})
        shot = exp.get('positive_count', 0)
        variant = exp.get('variant', 'text_only' if shot == 0 else 'unknown')
        version = directory.relative_to(root).parts[0]
        review_enabled = target.get('generation_settings', {}).get('enable_completeness_review', False)
        fallbacks = [dict(image=r['relative_path'], error=r.get('review_error') or 'review_not_performed')
                     for r in data if r.get('status') == 'success'
                     and review_enabled and (r.get('review_error') or not r.get('review_performed'))]
        scores = [o['confidence'] for r in data if r.get('status') == 'success' for o in r.get('objects', [])]
        points = metrics.get('score_threshold_results', {})
        p5 = points.get('0.5') or (metrics if metrics.get('settings', {}).get('score_threshold') == .5 else {})
        p7 = points.get('0.7', {})
        values = dict(AP50=metrics.get('mAP50'), AP50_95=metrics.get('mAP50_95'))
        for suffix, point in [('0_5', p5), ('0_7', p7)]:
            values.update({f'{k}_{suffix}': point.get(k) for k in ('precision', 'recall', 'f2', 'tp', 'fp', 'fn')})
        rows.append(dict(model=target['model'], prompt_version=version,
                         prompt_sha256=target.get('prompt_sha256'), shot=shot, input=variant,
                         input_name=exp.get('name', variant), negative_references=exp.get('negative_count', 0),
                         reference_images=exp.get('reference_image_count', 0), **values,
                         success_images=successes, failed_images=failures, total_images=run['image_count'],
                         ground_truth_count=sum(counts.values()), complete=complete,
                         status='完成' if complete else '未完整核验', review_enabled=review_enabled,
                         review_fallback_count=len(fallbacks), review_fallbacks=fallbacks,
                         min_score=min(scores) if scores else None, max_score=max(scores) if scores else None,
                         dataset_sha256=run.get('dataset_sha256'),
                         run_dir=directory.relative_to(root).as_posix(),
                         report=(directory / 'evaluation.txt').relative_to(root).as_posix(),
                         prompt_file=(directory / 'prompt.txt').relative_to(root).as_posix()))
    rows.sort(key=lambda r: (r['model'], r['prompt_version'], r['shot'],
                            VARIANTS.index(r['input']) if r['input'] in VARIANTS else 99,
                            r['negative_references'], r['run_dir']))
    return rows, manifest


def conclusion(rows):
    good = [r for r in rows if r['complete']]
    if not good:
        return '尚无通过当前数据集身份及完整性核验的结果，暂不比较优劣。'
    best = max(good, key=lambda r: r['AP50'])
    best_f2 = max(good, key=lambda r: r['f2_0_5'])
    best_p = max(good, key=lambda r: r['precision_0_5'])
    text = [f"在最新吊物验证集的完整实验中，AP50最高的是{best['model']}、{best['shot']}shot、"
            f"{best['input_name']}，AP50/AP50:95为{pct(best['AP50'])}%/{pct(best['AP50_95'])}%，"
            f"Precision/Recall/F2为{pct(best['precision_0_5'])}%/{pct(best['recall_0_5'])}%/{pct(best['f2_0_5'])}%，"
            f"对应TP/FP/FN={best['tp_0_5']}/{best['fp_0_5']}/{best['fn_0_5']}。"]
    pure = next((r for r in good if r['model'] == best['model'] and r['shot'] == 0), None)
    if pure:
        text.append(f"纯文字AP50已经达到{pct(pure['AP50'])}%；最佳视觉组合仅增加"
                    f"{(best['AP50'] - pure['AP50']) * 100:.2f}个百分点，"
                    f"TP增加{best['tp_0_5'] - pure['tp_0_5']}个、FP减少{pure['fp_0_5'] - best['fp_0_5']}个，"
                    '说明在这批图像和当前文字规则下，参考材料的额外收益有限。')
    text.append(f"重视召回时，{best_f2['shot']}shot {best_f2['input_name']}的Recall/F2最高，"
                f"为{pct(best_f2['recall_0_5'])}%/{pct(best_f2['f2_0_5'])}%；"
                f"重视少误报时，{best_p['shot']}shot {best_p['input_name']}的Precision最高"
                f"（{pct(best_p['precision_0_5'])}%、FP={best_p['fp_0_5']}），但Recall仅"
                f"{pct(best_p['recall_0_5'])}%。")
    def find(shot, variant, negatives):
        return next((r for r in good if r['model'] == best['model'] and r['shot'] == shot
                     and r['input'] == variant and r['negative_references'] == negatives), None)
    a, b, c = [find(shot, 'O+B', 0) for shot in [1, 5, 10]]
    if all([a, b, c]):
        text.append(f"同为原图＋带框图且无负例时，1/5/10shot AP50分别为"
                    f"{pct(a['AP50'])}%/{pct(b['AP50'])}%/{pct(c['AP50'])}%，"
                    '增加shot并非持续受益，10shot更偏保守且漏检更多。')
    pairs = [(find(5, v, 0), find(5, v, 5)) for v in VARIANTS[1:]]
    valid_pairs = [(a, b) for a, b in pairs if a and b]
    if valid_pairs:
        improved = sum(b['precision_0_5'] > a['precision_0_5'] for a, b in valid_pairs)
        text.append(f"5shot的{len(valid_pairs)}组负例对照中，加入负例有{improved}组提高Precision，"
                    '但Recall通常下降，AP收益也不一致；负例更适合控制误报，不能据此认定整体性能都改善。')
    if all(r['min_score'] is not None and r['min_score'] >= .7 for r in good):
        text.append('本批全部输出框分数均不低于0.7，因此阈值0.5和0.7的结果相同，调到0.7无法再去掉多余框。')
    text.append('目前主要短板仍是漏检和框定位；表中只有当前Qwen模型结果，其他算法尚无这版数据的可比运行。'
                '本次未改变文字prompt，不能从这些组推断文字版本本身的收益；旧版300张数据的目标分布与规则均不同，也不能直接比较AP。'
                '以上为固定验证集上的单轮结果，接近的AP还会受同分框排序及模型采样影响，不能视为显著优势。')
    return ''.join(text)


def main(root):
    rows, manifest = collect(root)
    samples = manifest['samples']
    positives = sum(bool(s['ground_truth']) for s in samples)
    gt_count = sum(len(s['ground_truth']) for s in samples)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    status_counts = dict(Counter(r['status'] for r in rows))
    fallback_total = sum(r['review_fallback_count'] for r in rows)
    prompt_count = len({r['prompt_sha256'] for r in rows})
    text = ['# 最新吊物验证集：全部运行结果总表', '',
            f'更新时间：{stamp}（北京时间）。共{len(rows)}组，完成{sum(r["complete"] for r in rows)}组；'
            f'成功图像推理合计{sum(r["success_images"] for r in rows)}次，逐图复核回退{fallback_total}次。', '',
            f'数据为当前diaowu_val300：{len(samples)}张，{positives}张有目标、{len(samples)-positives}张无目标，'
            f'{gt_count}个真值框。仅汇总同一运行数据指纹且逐张内容匹配的结果，不混入原99张或旧版300张实验。',
            'O=正例原图，C=目标裁剪图，B=带框原图，J=参考标注JSON；负例列为额外无目标参考图数量。'
            'shot按正例源图数量计，裁剪图可能一张源图产生多个；纯文字实际为0shot，即使历史父目录名含5shot。',
            f'本批保存的文字prompt共{prompt_count}种，均为当前允许少量遮挡、逐个目标判断的版本。'
            '这里的MLLM提示_v1是新数据命名空间中的版本号，不代表旧数据的严格遮挡版本；每行可点击prompt查看实际文字。',
            '指标均为百分数；P/R/F2及TP/FP/FN使用IoU=0.5，maxDets=100。AP读取原始完整分数的评估结果；'
            '0.5/0.7是输出框筛选阈值，对应两组P/R/F2，不重新截断AP计算。', '',
            '| # | 模型 | prompt | shot | 输入 | 负例图 | AP50 | AP50:95 | P/R/F2≥0.5 | P/R/F2≥0.7 | TP/FP/FN≥0.5 | 成功/总图 | 复核回退 | 状态/详情 |',
            '|---:|---|---|---:|---|---:|---:|---:|---|---|---|---|---:|---|']
    best = max((r['AP50'] for r in rows if r['complete']), default=None)
    for i, r in enumerate(rows, 1):
        triple = lambda t: '/'.join(pct(r[f'{k}_{t}']) for k in ('precision', 'recall', 'f2'))
        counts = '/'.join(str(r[f'{k}_0_5']) if r[f'{k}_0_5'] is not None else '—' for k in ('tp', 'fp', 'fn'))
        ap = pct(r['AP50'])
        if r['complete'] and r['AP50'] == best:
            ap = '**' + ap + '**'
        text.append('| ' + ' | '.join([str(i), r['model'], f"[v1]({r['prompt_file']})", str(r['shot']),
                    r['input'], str(r['negative_references']), ap, pct(r['AP50_95']), triple('0_5'), triple('0_7'),
                    counts, f"{r['success_images']}/{r['total_images']}", str(r['review_fallback_count']),
                    f"[{r['status']}]({r['report']})"]) + ' |')
    text += ['', '**总结归纳**', '', conclusion(rows), '', '**复核回退记录**', '',
             '所有图像均保留有效预测。以下第二遍复核遇到连接或服务错误，最终采用成功的首轮结果；'
             '不算推理失败，也不表示这几张完成了第二遍复核。', '']
    for i, r in enumerate(rows, 1):
        for failure in r['review_fallbacks']:
            feedback = f"{r['run_dir']}/feedback/{Path(failure['image']).stem}.json"
            error = str(failure['error']).replace('\n', ' ').replace('\r', ' ')
            text.append(f"- 表中第{i}组，`{failure['image']}`：{error}（[原始反馈]({feedback})）。")
    text += ['', '当前数据指纹：`' + read_json(root / 'dataset.json')['dataset_sha256'] + '`。',
             '导出：[CSV](results_summary.csv)（UTF-8 BOM，`_pct`列为百分数）；'
             '[JSON](results_summary.json)保留0～1原始指标、分数范围、指纹及来源。',
             '相关核查：[纯文字结果反差分析](text_result_sanity_check.md)、'
             '[5shot全部输入组合](MLLM提示_v1/qwen3.8-27b_5shot/comparison_all_modes.md)、'
             '[1/5/10shot对照](qwen27b_visual_shot_comparison.md)。',
             '更新命令：`.\\.venv-yoloe\\Scripts\\python.exe scripts/summarize_diaowu_results.py`。'
             '只读取已保存记录，不调用模型、不改写原始预测或指标。']
    (root / 'results_summary.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    percentages = ['AP50', 'AP50_95', *[f'{k}_{t}' for t in ('0_5', '0_7') for k in ('precision', 'recall', 'f2')]]
    exported = [{k + ('_pct' if k in percentages else ''):
                 (round(v * 100, 2) if v is not None else '') if k in percentages else v
                 for k, v in r.items() if k != 'review_fallbacks'} for r in rows]
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(exported[0]) if exported else [])
    writer.writeheader()
    writer.writerows(exported)
    (root / 'results_summary.csv').write_text(stream.getvalue(), encoding='utf-8-sig')
    write_json(root / 'results_summary.json', dict(generated_at=stamp, dataset=read_json(root / 'dataset.json'),
               row_count=len(rows), counts=status_counts, review_fallback_total=fallback_total,
               rows=rows, conclusion=conclusion(rows)))
    print(f'Runs: {len(rows)}; statuses: {status_counts}; review fallbacks: {fallback_total}; '
          f'report: {root / "results_summary.md"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/runs/diaowu_val300')
    main(parser.parse_args().root.resolve())

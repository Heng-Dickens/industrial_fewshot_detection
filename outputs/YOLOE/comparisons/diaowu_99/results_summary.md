# YOLOE / diaowu_99：实验结果总表

更新时间：2026-10-30 16:03:05（北京时间）。共 13 组独立实验；算法分布：YOLOE 13组。

当前数据集：99 张；真值框：hanging object=55。
数据指纹：`65dcc186b3d1d8cc2de51f6ca992bd5b629be8c987a111692306d386d5f4a22a`。

- 每个独立实验一行；五类合并实验使用合并指标，单类子运行作为来源保留，不重复当成五组实验。
- 所有率指标为百分数；P@0.5、R@0.5、F2@0.5分别单列，表示置信度≥0.5、按表中IoU阈值匹配；另保留分数阈值0.7的P/R/F2。AP读取完整候选评估值。缺失值为“—”，不是0。
- O=原图，C=目标裁剪，B=带框原图，J=标注JSON；shot读取实际实验设置，纯文字为0shot。
- 不同数据指纹、类别范围、不完整运行保留在表内，但不参与全数据集比较。仅元数据完成的运行不代表已核验逐图内容。
- 不同模型的预处理、候选下限、参考样本和提示词可能不同；比较前查看实际prompt、IoU/maxDets和原始配置。

[CSV表格](results_summary.csv) · [JSON完整字段](results_summary.json)

| # | 算法 | 模型 | prompt | shot/类 | 输入 | 负例 | AP50 | AP50:95 | P@0.5 | R@0.5 | F2@0.5 | P/R/F2≥0.7 | TP/FP/FN≥0.5 | 成功/总图 | 状态 | IoU/maxDets | 详情 |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 1 | YOLOE | yoloe-11l-seg | [文字提示_v1](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/yoloe-11l-seg_0shot/prompt.txt) | 0 | text_only | 0 | 4.60 | 2.43 | 0.00 | 0.00 | 0.00 | 0.00/0.00/0.00 | 0/0/55 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/yoloe-11l-seg_0shot/evaluation.txt) |
| 2 | YOLOE | yoloe-11l-seg | [文字提示_v1](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/yoloe-11l-seg_5shot_%E8%A3%81%E5%89%AA%E5%9B%BE0.75%2B%E6%96%87%E5%AD%970.25/prompt.txt) | 5 | 视觉+文字 | 0 | 24.91 | 14.65 | 0.00 | 0.00 | 0.00 | 0.00/0.00/0.00 | 0/0/55 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/yoloe-11l-seg_5shot_%E8%A3%81%E5%89%AA%E5%9B%BE0.75%2B%E6%96%87%E5%AD%970.25/evaluation.txt) |
| 3 | YOLOE | yoloe-11l-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_5shot/prompt.txt) | 5 | 视觉参考 | 0 | 54.27 | 29.56 | 82.35 | 50.91 | 55.12 | 82.61/34.55/39.09 | 28/6/27 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_5shot/evaluation.txt) |
| 4 | YOLOE | yoloe-11l-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_10shot/prompt.txt) | 10 | 视觉参考 | 0 | 60.50 | 31.91 | 75.61 | 56.36 | 59.39 | 81.82/49.09/53.36 | 31/10/24 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_10shot/evaluation.txt) |
| 5 | YOLOE | yoloe-11l-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_25shot/prompt.txt) | 25 | 视觉参考 | 0 | 62.64 | 32.50 | 70.45 | 56.36 | 58.71 | 81.08/54.55/58.37 | 31/13/24 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_25shot/evaluation.txt) |
| 6 | YOLOE | yoloe-v8l-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_1shot/prompt.txt) | 1 | 视觉参考 | 0 | 68.63 | 43.93 | 76.19 | 58.18 | 61.07 | 78.95/54.55/58.14 | 32/10/23 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_1shot/evaluation.txt) |
| 7 | YOLOE | yoloe-v8l-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_5shot/prompt.txt) | 5 | 视觉参考 | 0 | 69.15 | 43.65 | 75.51 | 67.27 | 68.77 | 83.33/54.55/58.59 | 37/12/18 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_5shot/evaluation.txt) |
| 8 | YOLOE | yoloe-v8l-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_10shot/prompt.txt) | 10 | 视觉参考 | 0 | 69.01 | 42.26 | 72.22 | 70.91 | 71.17 | 78.57/60.00/62.98 | 39/15/16 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_10shot/evaluation.txt) |
| 9 | YOLOE | yoloe-v8l-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_25shot/prompt.txt) | 25 | 视觉参考 | 0 | 69.38 | 41.75 | 68.97 | 72.73 | 71.94 | 78.57/60.00/62.98 | 40/18/15 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_25shot/evaluation.txt) |
| 10 | YOLOE | yoloe-v8s-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_1shot/prompt.txt) | 1 | 视觉参考 | 0 | 52.41 | 36.28 | 67.50 | 49.09 | 51.92 | 79.31/41.82/46.18 | 27/13/28 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_1shot/evaluation.txt) |
| 11 | YOLOE | yoloe-v8s-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_5shot/prompt.txt) | 5 | 视觉参考 | 0 | 56.65 | 37.69 | 72.97 | 49.09 | 52.53 | 88.89/43.64/48.58 | 27/10/28 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_5shot/evaluation.txt) |
| 12 | YOLOE | yoloe-v8s-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_10shot/prompt.txt) | 10 | 视觉参考 | 0 | 56.01 | 37.79 | 60.42 | 52.73 | 54.10 | 73.53/45.45/49.21 | 29/19/26 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_10shot/evaluation.txt) |
| 13 | YOLOE | yoloe-v8s-seg | [视觉提示](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_25shot/prompt.txt) | 25 | 视觉参考 | 0 | 56.64 | 38.43 | 62.00 | 56.36 | 57.41 | 76.47/47.27/51.18 | 31/19/24 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_25shot/evaluation.txt) |

## 核验与特殊情况

- 第1组 [yoloe-11l-seg / text_only](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/yoloe-11l-seg_0shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第2组 [yoloe-11l-seg / 视觉+文字](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/yoloe-11l-seg_5shot_%E8%A3%81%E5%89%AA%E5%9B%BE0.75%2B%E6%96%87%E5%AD%970.25/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第3组 [yoloe-11l-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_5shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第4组 [yoloe-11l-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_10shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第5组 [yoloe-11l-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-11l-seg_25shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第6组 [yoloe-v8l-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_1shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第7组 [yoloe-v8l-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_5shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第8组 [yoloe-v8l-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_10shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第9组 [yoloe-v8l-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8l-seg_25shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第10组 [yoloe-v8s-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_1shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第11组 [yoloe-v8s-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_5shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第12组 [yoloe-v8s-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_10shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第13组 [yoloe-v8s-seg / 视觉参考](../../experiments/diaowu_99/%E8%A7%86%E8%A7%89%E6%8F%90%E7%A4%BA/yoloe-v8s-seg_25shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。

CSV另含每类AP50、指纹、评估范围、复核开关、来源路径等字段；JSON包含单类子运行路径和记录核验统计。

更新：`python scripts/summarize_results.py --dataset diaowu_99`（在项目根目录执行）。
生成过程只读取已保存结果，不调用模型、不重算AP、不改写原始指标。

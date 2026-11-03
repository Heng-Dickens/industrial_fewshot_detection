# diaowu_val100：实验结果总表

更新时间：2026-10-30 15:54:19（北京时间）。共 5 组独立实验；算法分布：OVD-MLLM 5组。

当前数据集：100 张；真值框：hanging object=75。
数据指纹：`71de070a3d0733943b8ec4cadaea93bae33cd065ad6b7b0412bb448c6dedc148`。

- 每个独立实验一行；五类合并实验使用合并指标，单类子运行作为来源保留，不重复当成五组实验。
- 所有率指标为百分数；P@0.5、R@0.5、F2@0.5分别单列，表示置信度≥0.5、按表中IoU阈值匹配；另保留分数阈值0.7的P/R/F2。AP读取完整候选评估值。缺失值为“—”，不是0。
- O=原图，C=目标裁剪，B=带框原图，J=标注JSON；shot读取实际实验设置，纯文字为0shot。
- 不同数据指纹、类别范围、不完整运行保留在表内，但不参与全数据集比较。仅元数据完成的运行不代表已核验逐图内容。
- 不同模型的预处理、候选下限、参考样本和提示词可能不同；比较前查看实际prompt、IoU/maxDets和原始配置。

[CSV表格](results_summary.csv) · [JSON完整字段](results_summary.json)

| # | 算法 | 模型 | prompt | shot/类 | 输入 | 负例 | AP50 | AP50:95 | P@0.5 | R@0.5 | F2@0.5 | P/R/F2≥0.7 | TP/FP/FN≥0.5 | 成功/总图 | 状态 | IoU/maxDets | 详情 |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 1 | OVD-MLLM | yoloe-v8l-seg | — | 10 | baseline | 0 | 71.58 | 44.07 | 76.39 | 73.33 | 73.92 | 83.93/62.67/66.01 | 55/17/20 | 100/100 | 完成 | 0.5/100 | [查看](../../../OVD-MLLM/experiments/diaowu_val100/yoloe_v8l_10shot_qwen38_v1_20261030/baseline/evaluation.txt) |
| 2 | OVD-MLLM | yoloe-v8l-seg | — | 10 | source | 0 | 71.61 | 44.09 | 76.39 | 73.33 | 73.92 | 83.93/62.67/66.01 | 55/17/20 | 100/100 | 完成 | 0.5/100 | [查看](../../../OVD-MLLM/experiments/diaowu_val100/yoloe_v8l_10shot_qwen38_v1_20261030/source/evaluation.txt) |
| 3 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | deduplicated | 0 | 75.72 | 45.48 | 82.81 | 70.67 | 72.80 | 88.46/61.33/65.34 | 53/11/22 | 100/100 | 完成 | 0.5/100 | [查看](../../../OVD-MLLM/experiments/diaowu_val100/yoloe_v8l_10shot_qwen38_v1_20261030/deduplicated/evaluation.txt) |
| 4 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | filtered | 0 | 75.72 | 45.48 | 82.81 | 70.67 | 72.80 | 88.46/61.33/65.34 | 53/11/22 | 100/100 | 完成 | 0.5/100 | [查看](../../../OVD-MLLM/experiments/diaowu_val100/yoloe_v8l_10shot_qwen38_v1_20261030/filtered/evaluation.txt) |
| 5 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | refined | 0 | 73.42 | 45.83 | 81.25 | 69.33 | 71.43 | 86.54/60.00/63.92 | 52/12/23 | 100/100 | 完成 | 0.5/100 | [查看](../../../OVD-MLLM/experiments/diaowu_val100/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt) |

## 核验与特殊情况


CSV另含每类AP50、指纹、评估范围、复核开关、来源路径等字段；JSON包含单类子运行路径和记录核验统计。

更新：`python scripts/summarize_results.py --dataset diaowu_val100`（在项目根目录执行）。
生成过程只读取已保存结果，不调用模型、不重算AP、不改写原始指标。

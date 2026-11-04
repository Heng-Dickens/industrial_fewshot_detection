# YOLOE / diaowu_val300：实验结果总表

[微调模型的300张测试完整成绩（含P@0.5、R@0.5）](../../finetune/diaowu_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md)

更新时间：2026-11-03 21:15:52（北京时间）。共 3 组独立实验；算法分布：YOLOE 3组。

当前数据集：300 张；真值框：hanging object=350。
数据指纹：`470b98531e4a34810e7ee42bf250e39c97480e42427c3d84152e29e4d368f3fc`。

- 每个独立实验一行；五类合并实验使用合并指标，单类子运行作为来源保留，不重复当成五组实验。
- 所有率指标为百分数；P@0.5、R@0.5、F2@0.5分别单列，表示置信度≥0.5、按表中IoU阈值匹配；另保留分数阈值0.7的P/R/F2。AP读取完整候选评估值。缺失值为“—”，不是0。
- O=原图，C=目标裁剪，B=带框原图，J=标注JSON；shot读取实际实验设置，纯文字为0shot。
- 不同数据指纹、类别范围、不完整运行保留在表内，但不参与全数据集比较。仅元数据完成的运行不代表已核验逐图内容。
- 不同模型的预处理、候选下限、参考样本和提示词可能不同；比较前查看实际prompt、IoU/maxDets和原始配置。

[CSV表格](results_summary.csv) · [JSON完整字段](results_summary.json)

| # | 算法 | 模型 | prompt | shot/类 | 输入 | 负例 | AP50 | AP50:95 | P@0.5 | R@0.5 | F2@0.5 | P/R/F2≥0.7 | TP/FP/FN≥0.5 | 成功/总图 | 状态 | IoU/maxDets | 详情 |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 1 | YOLOE | yoloe-v8l-seg | [视觉提示](../../experiments/diaowu_val300/yoloe-v8l-seg_1shot/prompt.txt) | 1 | 视觉参考 | 0 | 53.03 | 32.56 | 84.36 | 43.14 | 47.82 | 89.71/34.86/39.71 | 151/28/199 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/yoloe-v8l-seg_1shot/evaluation.txt) |
| 2 | YOLOE | yoloe-v8l-seg | [视觉提示](../../experiments/diaowu_val300/yoloe-v8l-seg_5shot/prompt.txt) | 5 | 视觉参考 | 0 | 54.81 | 33.17 | 77.73 | 50.86 | 54.63 | 86.62/38.86/43.67 | 178/51/172 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/yoloe-v8l-seg_5shot/evaluation.txt) |
| 3 | YOLOE | yoloe-v8l-seg | [视觉提示](../../experiments/diaowu_val300/yoloe-v8l-seg_10shot/prompt.txt) | 10 | 视觉参考 | 0 | 55.96 | 33.76 | 74.21 | 53.43 | 56.60 | 83.08/46.29/50.78 | 187/65/163 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/yoloe-v8l-seg_10shot/evaluation.txt) |

## 核验与特殊情况


CSV另含每类AP50、指纹、评估范围、复核开关、来源路径等字段；JSON包含单类子运行路径和记录核验统计。

更新：`python scripts/summarize_results.py --dataset diaowu_val300`（在项目根目录执行）。
生成过程只读取已保存结果，不调用模型、不重算AP、不改写原始指标。

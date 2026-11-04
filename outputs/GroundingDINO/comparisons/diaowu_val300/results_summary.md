# GroundingDINO / diaowu_val300：实验结果总表

更新时间：2026-11-03 21:15:52（北京时间）。共 2 组独立实验；算法分布：GroundingDINO 2组。

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
| 1 | GroundingDINO | grounding-dino-base | [prompt_v1](../../experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v1/prompt.txt) | 0 | text_only | 0 | 0.49 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00/0.00/0.00 | 0/0/350 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v1/evaluation.txt) |
| 2 | GroundingDINO | grounding-dino-base | [prompt_v2](../../experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v2/prompt.txt) | 0 | text_only | 0 | 1.00 | 0.52 | 0.00 | 0.00 | 0.00 | 0.00/0.00/0.00 | 0/0/350 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v2/evaluation.txt) |

## 核验与特殊情况


CSV另含每类AP50、指纹、评估范围、复核开关、来源路径等字段；JSON包含单类子运行路径和记录核验统计。

更新：`python scripts/summarize_results.py --dataset diaowu_val300`（在项目根目录执行）。
生成过程只读取已保存结果，不调用模型、不重算AP、不改写原始指标。

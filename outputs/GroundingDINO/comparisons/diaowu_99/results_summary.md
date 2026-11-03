# GroundingDINO / diaowu_99：实验结果总表

更新时间：2026-10-30 16:03:05（北京时间）。共 2 组独立实验；算法分布：GroundingDINO 2组。

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
| 1 | GroundingDINO | grounding-dino-base | [文字提示_v1](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/grounding-dino-base_0shot/prompt.txt) | 0 | text_only | 0 | 2.01 | 1.75 | 0.00 | 0.00 | 0.00 | 0.00/0.00/0.00 | 0/95/55 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/grounding-dino-base_0shot/evaluation.txt) |
| 2 | GroundingDINO | grounding-dino-tiny | [文字提示_v1](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/grounding-dino-tiny_0shot/prompt.txt) | 0 | text_only | 0 | 5.33 | 3.03 | 0.00 | 0.00 | 0.00 | 0.00/0.00/0.00 | 0/0/55 | —/99 | 完成（元数据） | 0.5/100 | [查看](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/grounding-dino-tiny_0shot/evaluation.txt) |

## 核验与特殊情况

- 第1组 [grounding-dino-base / text_only](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/grounding-dino-base_0shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。
- 第2组 [grounding-dino-tiny / text_only](../../experiments/diaowu_99/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/grounding-dino-tiny_0shot/evaluation.txt)：本机无records.jsonl，未进行逐图内容核验。

CSV另含每类AP50、指纹、评估范围、复核开关、来源路径等字段；JSON包含单类子运行路径和记录核验统计。

更新：`python scripts/summarize_results.py --dataset diaowu_99`（在项目根目录执行）。
生成过程只读取已保存结果，不调用模型、不重算AP、不改写原始指标。

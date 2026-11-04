# OVD-MLLM / chache_val300：实验结果总表

更新时间：2026-11-03 22:01:28（北京时间）。共 20 组独立实验；算法分布：OVD-MLLM 20组。

当前数据集：300 张；真值框：Chache2=30、Chache1=40、Qianyinche=48、Chache4=30、Chache3=40。
数据指纹：`bf04a484ca78255b5d8a8208dac79a6ac350c0dae1b4aa696bb5d921140b8d2b`。

- 每个独立实验一行；五类合并实验使用合并指标，单类子运行作为来源保留，不重复当成五组实验。
- 所有率指标为百分数；P@0.5、R@0.5、F2@0.5分别单列，表示置信度≥0.5、按表中IoU阈值匹配；另保留分数阈值0.7的P/R/F2。AP读取完整候选评估值。缺失值为“—”，不是0。
- O=原图，C=目标裁剪，B=带框原图，J=标注JSON；shot读取实际实验设置，纯文字为0shot。
- 不同数据指纹、类别范围、不完整运行保留在表内，但不参与全数据集比较。仅元数据完成的运行不代表已核验逐图内容。
- 不同模型的预处理、候选下限、参考样本和提示词可能不同；比较前查看实际prompt、IoU/maxDets和原始配置。

[CSV表格](results_summary.csv) · [JSON完整字段](results_summary.json)

| # | 算法 | 模型 | prompt | shot/类 | 输入 | 负例 | AP50 | AP50:95 | P@0.5 | R@0.5 | F2@0.5 | P/R/F2≥0.7 | TP/FP/FN≥0.5 | 成功/总图 | 状态 | IoU/maxDets | 详情 |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| 1 | OVD-MLLM | yoloe-v8l-seg | — | 10 | baseline | 0 | 28.98 | 18.03 | 22.26 | 37.77 | 33.15 | 33.06/21.81/23.40 | 71/248/117 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/baseline/evaluation.txt) |
| 2 | OVD-MLLM | yoloe-v8l-seg | — | 10 | source | 0 | 29.00 | 18.04 | 22.26 | 37.77 | 33.15 | 33.06/21.81/23.40 | 71/248/117 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/source/evaluation.txt) |
| 3 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | deduplicated | 0 | 49.38 | 27.93 | 80.00 | 31.91 | 36.28 | 86.05/19.68/23.27 | 60/15/128 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/deduplicated/evaluation.txt) |
| 4 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | filtered | 0 | 49.31 | 27.92 | 80.00 | 31.91 | 36.28 | 86.05/19.68/23.27 | 60/15/128 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/filtered/evaluation.txt) |
| 5 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | refined | 0 | 52.20 | 27.86 | 80.00 | 31.91 | 36.28 | 86.05/19.68/23.27 | 60/15/128 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt) |
| 6 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | baseline | 0 | 80.94 | 70.11 | 78.65 | 80.32 | 79.98 | 82.56/75.53/76.84 | 151/41/37 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 7 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | deduplicated | 0 | 63.98 | 56.47 | 95.93 | 62.77 | 67.43 | 99.11/59.04/64.24 | 118/5/70 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 8 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | filtered | 0 | 63.98 | 56.47 | 95.93 | 62.77 | 67.43 | 99.11/59.04/64.24 | 118/5/70 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 9 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | refined | 0 | 62.06 | 46.46 | 93.50 | 61.17 | 65.71 | 96.43/57.45/62.50 | 115/8/73 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 10 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | source | 0 | 82.22 | 70.96 | 78.65 | 80.32 | 79.98 | 82.56/75.53/76.84 | 151/41/37 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 11 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | baseline | 0 | 58.99 | 46.07 | 42.24 | 68.09 | 60.66 | 50.92/59.04/57.22 | 128/175/60 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 12 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | baseline | 0 | 64.10 | 49.84 | 47.46 | 69.68 | 63.72 | 57.51/59.04/58.73 | 131/145/57 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 13 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | deduplicated | 0 | 42.27 | 33.29 | 73.39 | 42.55 | 46.46 | 79.35/38.83/43.25 | 80/29/108 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 14 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | deduplicated | 0 | 43.45 | 33.18 | 72.97 | 43.09 | 46.93 | 77.32/39.89/44.17 | 81/30/107 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 15 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | filtered | 0 | 42.27 | 33.29 | 73.39 | 42.55 | 46.46 | 79.35/38.83/43.25 | 80/29/108 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 16 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | filtered | 0 | 43.45 | 33.18 | 72.97 | 43.09 | 46.93 | 77.32/39.89/44.17 | 81/30/107 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 17 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | refined | 0 | 49.97 | 35.18 | 84.40 | 48.94 | 53.43 | 85.87/42.02/46.80 | 92/17/96 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 18 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | refined | 0 | 50.72 | 35.77 | 83.78 | 49.47 | 53.88 | 86.60/44.68/49.47 | 93/18/95 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 19 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | source | 0 | 59.02 | 46.11 | 42.24 | 68.09 | 60.66 | 50.92/59.04/57.22 | 128/175/60 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 20 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | source | 0 | 64.13 | 49.86 | 47.46 | 69.68 | 63.72 | 57.51/59.04/58.73 | 131/145/57 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |

## 核验与特殊情况

- 第1组 [yoloe-v8l-seg / baseline](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/baseline/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第2组 [yoloe-v8l-seg / source](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/source/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第3组 [yoloe-v8l-seg+qwen3.8-27b / deduplicated](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/deduplicated/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第4组 [yoloe-v8l-seg+qwen3.8-27b / filtered](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/filtered/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第5组 [yoloe-v8l-seg+qwen3.8-27b / refined](../../experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第6组 [yoloe→qwen3.8-27b / baseline](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第7组 [yoloe→qwen3.8-27b / deduplicated](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第8组 [yoloe→qwen3.8-27b / filtered](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第9组 [yoloe→qwen3.8-27b / refined](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/refined/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第10组 [yoloe→qwen3.8-27b / source](../../experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/source/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第11组 [yoloe→qwen3.8-27b / baseline](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第12组 [yoloe→qwen3.8-27b / baseline](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第13组 [yoloe→qwen3.8-27b / deduplicated](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第14组 [yoloe→qwen3.8-27b / deduplicated](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第15组 [yoloe→qwen3.8-27b / filtered](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第16组 [yoloe→qwen3.8-27b / filtered](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第17组 [yoloe→qwen3.8-27b / refined](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/refined/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第18组 [yoloe→qwen3.8-27b / refined](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/refined/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第19组 [yoloe→qwen3.8-27b / source](../../experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/source/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第20组 [yoloe→qwen3.8-27b / source](../../experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/source/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。

CSV另含每类AP50、指纹、评估范围、复核开关、来源路径等字段；JSON包含单类子运行路径和记录核验统计。

更新：`python scripts/summarize_results.py --dataset chache_val300`（在项目根目录执行）。
生成过程只读取已保存结果，不调用模型、不重算AP、不改写原始指标。

# OVD-MLLM / diaowu_val300：实验结果总表

更新时间：2026-11-03 21:15:52（北京时间）。共 20 组独立实验；算法分布：OVD-MLLM 20组。

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
| 1 | OVD-MLLM | yoloe-v8l-seg | — | 10 | baseline | 0 | 55.83 | 33.73 | 74.21 | 53.43 | 56.60 | 83.08/46.29/50.78 | 187/65/163 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/baseline/evaluation.txt) |
| 2 | OVD-MLLM | yoloe-v8l-seg | — | 10 | source | 0 | 55.96 | 33.76 | 74.21 | 53.43 | 56.60 | 83.08/46.29/50.78 | 187/65/163 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/source/evaluation.txt) |
| 3 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | deduplicated | 0 | 57.45 | 33.20 | 85.12 | 52.29 | 56.66 | 90.45/46.00/51.01 | 183/32/167 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/deduplicated/evaluation.txt) |
| 4 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | filtered | 0 | 57.45 | 33.20 | 85.12 | 52.29 | 56.66 | 90.45/46.00/51.01 | 183/32/167 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/filtered/evaluation.txt) |
| 5 | OVD-MLLM | yoloe-v8l-seg+qwen3.8-27b | — | 10 | refined | 0 | 58.36 | 36.44 | 84.19 | 51.71 | 56.04 | 89.33/45.43/50.38 | 181/34/169 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt) |
| 6 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | baseline | 0 | 91.33 | 71.37 | 93.36 | 80.29 | 82.60 | 93.38/72.57/75.96 | 281/20/69 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 7 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | deduplicated | 0 | 57.95 | 47.59 | 95.12 | 55.71 | 60.75 | 95.83/52.57/57.79 | 195/10/155 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 8 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | filtered | 0 | 57.95 | 47.59 | 95.12 | 55.71 | 60.75 | 95.83/52.57/57.79 | 195/10/155 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 9 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | refined | 0 | 56.72 | 42.99 | 93.66 | 54.86 | 59.81 | 94.79/52.00/57.16 | 192/13/158 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 10 | OVD-MLLM | yoloe→qwen3.8-27b | — | — | source | 0 | 92.85 | 72.32 | 93.36 | 80.29 | 82.60 | 93.38/72.57/75.96 | 281/20/69 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 11 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | baseline | 0 | 77.05 | 57.12 | 92.12 | 43.43 | 48.56 | 93.28/35.71/40.74 | 152/13/198 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 12 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | baseline | 0 | 77.44 | 56.02 | 93.12 | 42.57 | 47.76 | 92.70/36.29/41.31 | 149/11/201 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 13 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | deduplicated | 0 | 55.62 | 42.65 | 91.61 | 40.57 | 45.66 | 92.80/33.14/38.03 | 142/13/208 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 14 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | deduplicated | 0 | 55.09 | 42.11 | 94.59 | 40.00 | 45.22 | 93.94/35.43/40.47 | 140/8/210 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 15 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | filtered | 0 | 55.62 | 42.65 | 91.61 | 40.57 | 45.66 | 92.80/33.14/38.03 | 142/13/208 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 16 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | filtered | 0 | 55.09 | 42.11 | 94.59 | 40.00 | 45.22 | 93.94/35.43/40.47 | 140/8/210 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 17 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | refined | 0 | 55.51 | 39.36 | 91.61 | 40.57 | 45.66 | 92.80/33.14/38.03 | 142/13/208 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 18 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | refined | 0 | 54.85 | 39.67 | 93.92 | 39.71 | 44.90 | 93.18/35.14/40.14 | 139/9/211 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 19 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | source | 0 | 83.33 | 60.17 | 92.12 | 43.43 | 48.56 | 93.28/35.71/40.74 | 152/13/198 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 20 | OVD-MLLM | yoloe→qwen3.8-27b | — | 10 | source | 0 | 84.98 | 60.56 | 93.12 | 42.57 | 47.76 | 92.70/36.29/41.31 | 149/11/201 | 300/300 | 完成 | 0.5/100 | [查看](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |

## 核验与特殊情况

- 第1组 [yoloe-v8l-seg / baseline](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/baseline/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第2组 [yoloe-v8l-seg / source](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/source/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第3组 [yoloe-v8l-seg+qwen3.8-27b / deduplicated](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/deduplicated/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第4组 [yoloe-v8l-seg+qwen3.8-27b / filtered](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/filtered/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第5组 [yoloe-v8l-seg+qwen3.8-27b / refined](../../experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第6组 [yoloe→qwen3.8-27b / baseline](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第7组 [yoloe→qwen3.8-27b / deduplicated](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第8组 [yoloe→qwen3.8-27b / filtered](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第9组 [yoloe→qwen3.8-27b / refined](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/refined/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第10组 [yoloe→qwen3.8-27b / source](../../experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/source/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第11组 [yoloe→qwen3.8-27b / baseline](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第12组 [yoloe→qwen3.8-27b / baseline](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第13组 [yoloe→qwen3.8-27b / deduplicated](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第14组 [yoloe→qwen3.8-27b / deduplicated](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第15组 [yoloe→qwen3.8-27b / filtered](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第16组 [yoloe→qwen3.8-27b / filtered](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第17组 [yoloe→qwen3.8-27b / refined](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/refined/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第18组 [yoloe→qwen3.8-27b / refined](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/refined/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第19组 [yoloe→qwen3.8-27b / source](../../experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/source/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。
- 第20组 [yoloe→qwen3.8-27b / source](../../experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/source/evaluation.txt)：清单指纹与基准指纹口径不同，已按图像哈希及完整GT核对为同一数据集。

CSV另含每类AP50、指纹、评估范围、复核开关、来源路径等字段；JSON包含单类子运行路径和记录核验统计。

更新：`python scripts/summarize_results.py --dataset diaowu_val300`（在项目根目录执行）。
生成过程只读取已保存结果，不调用模型、不重算AP、不改写原始指标。

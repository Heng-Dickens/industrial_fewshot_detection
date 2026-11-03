# Qwen3.8-27B第二步shot补齐

本批仅新跑吊物O+B+J的1shot与20shot，各评估原300张测试集，共600张次。
叉车20shot按用户要求取消；叉车O+C+B的1shot已有完整300张结果，直接引用。
吊物20shot保留原10shot为前缀，另从原25张正参考中补选10张，seed0；无需从val补图。
参考与测试图像SHA无重叠。全部使用原有JPEG质量90、1280最长边和独立图片输入方式。

[参考来源与抽样清单](../../../../data/references/qwen_shot_extension_20261103/selection_provenance.json)

| 场景 | shot/类 | 输入 | 状态 | 成功/总图 | AP50 | AP50:95 | 结果 |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| diaowu | 1 | O+B+J | completed | 300/300 | 51.81 | 40.47 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-27b_1shot_第2次/原图+带框图+标注JSON_无负例/evaluation.txt) |
| diaowu | 20 | O+B+J | completed | 300/300 | 53.14 | 40.36 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-27b_20shot/原图+带框图+标注JSON_无负例/evaluation.txt) |

累计成功：600/600张次。

[最终Qwen表：第二步](../../../_shared/comparisons/final_results/01_qwen_inputs.md#step-2)
每组完成后自动刷新最终表、CSV与全局汇总；不完整结果不参与最优值。

命令：`python scripts/run_qwen27b_shot_supplement.py --prepare` → `--probe` → `--background`。
刷新进度：`python scripts/run_qwen27b_shot_supplement.py --status`。

叉车1shot已入表：AP50=50.00%，AP50:95=36.35%。[已有原始结果](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-27b_1shot/原图+裁剪图+带框图_无负例/evaluation.txt)。

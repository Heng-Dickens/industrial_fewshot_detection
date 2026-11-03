# Qwen3.8-27B输入消融补齐

仅补第一部分缺失/未完成条件；已有5shot与10shot成功实验保留。
叉车沿用v2，吊物沿用v1及完整性复核；无负例，seed0，各自固定300张测试图。
叉车1个worker、吊物3个worker，每个并发2张图；保留成功缓存，失败可恢复，不改代理全局路由。

| 场景 | shot/类 | 输入 | 状态 | 成功/总图 | AP50 | AP50:95 | 结果 |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| chache | 10 | O+J | completed | 300/300 | 70.82 | 45.58 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-27b_10shot_第2次/原图+标注JSON_无负例/evaluation.txt) |
| chache | 10 | O+B+J | completed | 300/300 | 77.20 | 50.22 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-27b_10shot_第3次/原图+带框图+标注JSON_无负例/evaluation.txt) |
| chache | 10 | O+C+J | completed | 300/300 | 74.37 | 49.68 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-27b_10shot_第4次/原图+裁剪图+标注JSON_无负例/evaluation.txt) |
| chache | 10 | O+C+B+J | completed | 300/300 | 75.60 | 51.15 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-27b_10shot_第5次/原图+裁剪图+带框图+标注JSON_无负例/evaluation.txt) |
| diaowu | 10 | O+J | completed | 300/300 | 53.00 | 40.94 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-27b_10shot_第2次/原图+标注JSON_无负例/evaluation.txt) |
| diaowu | 10 | O+B+J | completed | 300/300 | 56.75 | 43.71 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-27b_10shot_第4次/原图+带框图+标注JSON_无负例/evaluation.txt) |
| diaowu | 10 | O+C+J | completed | 300/300 | 53.94 | 40.71 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-27b_10shot_第3次/原图+裁剪图+标注JSON_无负例/evaluation.txt) |
| diaowu | 10 | O+C+B | completed | 300/300 | 51.76 | 41.17 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-27b_10shot_第5次/原图+裁剪图+带框图_无负例/evaluation.txt) |
| diaowu | 10 | O+C+B+J | completed | 300/300 | 54.56 | 41.60 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-27b_10shot_第6次/原图+裁剪图+带框图+标注JSON_无负例/evaluation.txt) |

累计成功：2700/2700张次。

[按输入→shot→模型阅读最终Qwen表](../../../_shared/comparisons/final_results/01_qwen_inputs.md)。
每组完成自动刷新完整明细、Qwen子表与全局代表方案；若最佳输入改变，后两阶段采用新输入并明确显示缺项。

命令：`python scripts/run_qwen27b_input_supplement.py --prepare`、`--probe`、`--background`；
查看/刷新进度：`python scripts/run_qwen27b_input_supplement.py --status`。

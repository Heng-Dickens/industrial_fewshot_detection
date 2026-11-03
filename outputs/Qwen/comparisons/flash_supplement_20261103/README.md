# Flash补齐：叉车O+C与吊物O+B/O+C

固定两个300张测试集、seed0、各1/5/10shot、无负例。共9组、2700张图像任务。
叉车沿用加长v2五类联合提示；吊物沿用原27B的提示、离地判定、逐框理由及完整性复核。
使用现有Claude Messages转发别名，响应模型必须严格为qwen3.8-flash；不修改服务全局路由。
叉车按shot顺序运行、并发2；吊物按三个shot并行、各并发2，总并发最多8个图像任务；成功记录缓存保留。

| 场景 | shot/类 | 输入 | 状态 | 成功/总图 | AP50 | AP50:95 | P | R | F2 | 结果 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| chache | 1 | O+C | completed | 300/300 | 52.66 | 40.33 | 50.81 | 66.49 | 62.63 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-flash_1shot_第3次/原图+裁剪图_无负例/evaluation.txt) |
| chache | 5 | O+C | completed | 300/300 | 73.18 | 50.65 | 64.71 | 81.91 | 77.78 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-flash_5shot_第2次/原图+裁剪图_无负例/evaluation.txt) |
| chache | 10 | O+C | completed | 300/300 | 77.96 | 54.72 | 69.53 | 86.17 | 82.23 | [查看](../../experiments/chache_val300/MLLM提示_v2/qwen3.8-flash_10shot_第2次/原图+裁剪图_无负例/evaluation.txt) |
| diaowu | 1 | O+B | completed | 300/300 | 57.45 | 45.44 | 83.02 | 62.86 | 66.07 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-flash_1shot/原图+带框图_无负例/evaluation.txt) |
| diaowu | 1 | O+C | completed | 300/300 | 56.21 | 45.31 | 81.48 | 62.86 | 65.87 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-flash_1shot_第2次/原图+裁剪图_无负例/evaluation.txt) |
| diaowu | 5 | O+B | completed | 300/300 | 57.28 | 45.71 | 83.27 | 62.57 | 65.84 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-flash_5shot/原图+带框图_无负例/evaluation.txt) |
| diaowu | 5 | O+C | completed | 300/300 | 57.19 | 45.70 | 84.17 | 62.29 | 65.70 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-flash_5shot_第2次/原图+裁剪图_无负例/evaluation.txt) |
| diaowu | 10 | O+B | completed | 300/300 | 58.06 | 46.05 | 84.77 | 62.00 | 65.52 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-flash_10shot/原图+带框图_无负例/evaluation.txt) |
| diaowu | 10 | O+C | completed | 300/300 | 57.78 | 44.80 | 84.27 | 64.29 | 67.49 | [查看](../../experiments/diaowu_val300/MLLM提示_v1/qwen3.8-flash_10shot_第2次/原图+裁剪图_无负例/evaluation.txt) |

累计成功：2700/2700张次（100.0%）；成功数按逐图记录去重，恢复时不重复计数。

运行：`python scripts/run_qwenflash_supplement.py --prepare` → `--probe` → `--background`。
查看/刷新进度：`python scripts/run_qwenflash_supplement.py --status`。
所有组完成后自动刷新Qwen子表、原始明细及全局比较；不完整结果不参与最优值排名。

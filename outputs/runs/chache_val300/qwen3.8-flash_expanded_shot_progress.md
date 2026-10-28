# 叉车加长版：Qwen3.8-27B / Flash 的1、5、10shot对比

状态：complete。共3组，每组300张，900张图像任务；已有完整结果经验证复用。

固定加长版v2、同300张验证图与五类共188个GT框。shot按每类计，原图＋带框图，无负例、无额外复核；参考按seed0嵌套选择。

| 模型 | shot/类 | 状态 | AP50 | AP50:95 | Precision@0.5 | Recall@0.5 | F2@0.5 | 结果 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| qwen3.8-flash | 1 | 完成（复用） | 56.31 | 42.42 | 50.21 | 63.83 | 60.54 | [evaluation](D:/Learn/VLM/industrial_fewshot_detection/outputs/runs/chache_val300/MLLM提示_v2/qwen3.8-flash_1shot_第2次/原图+带框图_无负例/evaluation.txt) |
| qwen3.8-flash | 5 | 完成（复用） | 69.21 | 47.56 | 62.61 | 79.26 | 75.25 | [evaluation](D:/Learn/VLM/industrial_fewshot_detection/outputs/runs/chache_val300/MLLM提示_v2/qwen3.8-flash_5shot/原图+带框图_无负例/evaluation.txt) |
| qwen3.8-flash | 10 | 完成 | 71.10 | 50.39 | 65.38 | 81.38 | 77.59 | [evaluation](D:/Learn/VLM/industrial_fewshot_detection/outputs/runs/chache_val300/MLLM提示_v2/qwen3.8-flash_10shot/原图+带框图_无负例/evaluation.txt) |

## qwen3.8-flash / 1shot

| 类别 | GT | TP | FP | FN | AP50 | Recall@0.5 | F2@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chache1 | 40 | 30 | 14 | 10 | 72.57 | 75.00 | 73.53 |
| Chache2 | 30 | 11 | 74 | 19 | 23.28 | 36.67 | 26.83 |
| Chache3 | 40 | 6 | 3 | 34 | 13.20 | 15.00 | 17.75 |
| Chache4 | 30 | 25 | 1 | 5 | 82.79 | 83.33 | 85.62 |
| Qianyinche | 48 | 48 | 27 | 0 | 89.69 | 100.00 | 89.89 |

## qwen3.8-flash / 5shot

| 类别 | GT | TP | FP | FN | AP50 | Recall@0.5 | F2@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chache1 | 40 | 33 | 7 | 7 | 77.74 | 82.50 | 82.50 |
| Chache2 | 30 | 18 | 26 | 12 | 39.74 | 60.00 | 54.88 |
| Chache3 | 40 | 21 | 9 | 19 | 43.74 | 52.50 | 55.26 |
| Chache4 | 30 | 30 | 31 | 0 | 94.24 | 100.00 | 82.87 |
| Qianyinche | 48 | 47 | 16 | 1 | 90.60 | 97.92 | 92.16 |

## qwen3.8-flash / 10shot

| 类别 | GT | TP | FP | FN | AP50 | Recall@0.5 | F2@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chache1 | 40 | 33 | 6 | 7 | 78.77 | 82.50 | 82.91 |
| Chache2 | 30 | 19 | 31 | 11 | 43.60 | 63.33 | 55.88 |
| Chache3 | 40 | 23 | 18 | 17 | 42.58 | 57.50 | 57.21 |
| Chache4 | 30 | 30 | 16 | 0 | 97.77 | 100.00 | 90.36 |
| Qianyinche | 48 | 48 | 10 | 0 | 92.77 | 100.00 | 96.00 |

各组同时保留score阈值0.5、0.7的结果；AP使用完整置信度排序。模型名称不符的响应不作为成功预测，也不计作该模型的实验结果。

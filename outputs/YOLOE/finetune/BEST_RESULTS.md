# 两个场景：各微调方式的测试推理最高成绩

本页汇总20261029搜索及其冻结SAVPE对照，微调训练预算均为每类10张；[每类5张训练的新对照](5SHOT_COMPARISON.md)另表保存。

此页是300张测试集推理成绩的最高值摘要，不是训练成绩。全部训练和测试已完成；两个场景各31项测试设置的完整推理结果见文末链接。AP50与AP50:95分别取已测试结果最高值；同一实验同时获得两项最高值时合并为一行，不同实验分别列出。

| 场景 | 方法 | 推理shot | 选取依据 | AP50 % | AP50:95 % | P@0.5 % | R@0.5 % | 实验 / seed |
|---|---|---:|---|---:|---:|---:|---:|---|
| 叉车 | YOLOE固定头微调 | 不适用 | 两项AP均最高 | 82.22 | 70.96 | 78.65 | 80.32 | `head_full_aug` / 0 |
| 叉车 | YOLOE视觉提示·训练SAVPE | 5 | 两项AP均最高 | 61.78 | 48.12 | 48.43 | 65.43 | `vp_cross_neck` / 0 |
| 叉车 | YOLOE视觉提示·训练SAVPE | 10 | 两项AP均最高 | 64.13 | 49.86 | 47.46 | 69.68 | `vp_cross_neck` / 0 |
| 叉车 | YOLOE视觉提示·冻结SAVPE | 5 | 最高AP50 | 56.65 | 43.59 | 44.76 | 68.09 | `vp_cross_neck_frozen_savpe` / 0 |
| 叉车 | YOLOE视觉提示·冻结SAVPE | 5 | 最高AP50:95 | 55.85 | 43.84 | 42.20 | 63.30 | `vp_cross_neck_frozen_savpe_seed1` / 1 |
| 叉车 | YOLOE视觉提示·冻结SAVPE | 10 | 两项AP均最高 | 59.02 | 46.11 | 42.24 | 68.09 | `vp_cross_neck_frozen_savpe_seed1` / 1 |
| 叉车 | YOLOv8-L预训练微调 | 不适用 | 两项AP均最高 | 79.26 | 66.96 | 97.54 | 63.30 | `YOLOv8-L微调v1` / 0 |
| 吊物 | YOLOE固定头微调 | 不适用 | 最高AP50 | 92.85 | 72.32 | 93.36 | 80.29 | `head_full_960_seed2` / 2 |
| 吊物 | YOLOE固定头微调 | 不适用 | 最高AP50:95 | 92.50 | 73.37 | 91.96 | 75.14 | `head_full_960_seed1` / 1 |
| 吊物 | YOLOE视觉提示·训练SAVPE | 5 | 两项AP均最高 | 81.13 | 60.74 | 92.31 | 34.29 | `vp_full_lr0.0001_seed1` / 1 |
| 吊物 | YOLOE视觉提示·训练SAVPE | 10 | 最高AP50 | 84.98 | 60.56 | 93.12 | 42.57 | `vp_full_lr0.0001` / 0 |
| 吊物 | YOLOE视觉提示·训练SAVPE | 10 | 最高AP50:95 | 82.68 | 62.00 | 92.99 | 41.71 | `vp_full_lr0.0001_seed1` / 1 |
| 吊物 | YOLOE视觉提示·冻结SAVPE | 5 | 两项AP均最高 | 81.24 | 58.25 | 94.93 | 37.43 | `vp_full_lr0.0001_frozen_savpe` / 0 |
| 吊物 | YOLOE视觉提示·冻结SAVPE | 10 | 两项AP均最高 | 83.33 | 60.17 | 92.12 | 43.43 | `vp_full_lr0.0001_frozen_savpe` / 0 |
| 吊物 | YOLOv8-L预训练微调 | 不适用 | 两项AP均最高 | 92.19 | 69.72 | 36.52 | 94.86 | `baseline_yolov8` / 0 |

[叉车完整推理表](chache_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md) · [吊物完整推理表](diaowu_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md) · [全部专题](../README.md)

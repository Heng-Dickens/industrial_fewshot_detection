# 定制监督网络参考

保留JSTY定制网络FP32 ONNX结果，并列展示相同测试队列上的固定类别YOLOE与YOLOv8作为参考。JSTY输入960×1280，其原始训练预算与训练/测试重叠尚未绑定到该权重的可核验训练快照。分数可以并列阅读，不能据此声称在相同标注预算下优于fewshot。

[全局比较](../../../FINAL_COMPARISON.md) · [子任务导航](README.md)

指标均为百分数。AP使用该运行保存的全部候选；P/R/F1/F2使用score≥0.5、IoU=0.5，TP/FP/FN为框数。箭头表示优化方向；每张表内逐列加粗最优值，按显示精度并列最优同时加粗。参数值不作最优标记。未完成、队列不符或类别范围不全的记录保留展示，但不参与最优值计算。†表示本机仅保存指标和元数据，尚无逐图记录可复核；仍可作为有此证据限制的结果对比。P或负图错误率单独最优不等于检测最好，需结合R和TP/FP/FN阅读。

### 叉车（五类）

| 模型/方案 | 训练/类 | 输入 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| jsty_chache_det5_960_1280_0_0_0_2_fp32 / 定制监督 | 监督/训练量未绑定 | 960×1280 | **93.68** | **82.54** | **71.53** | 90.64 | **82.45** | **86.35** | **83.97** | 1.59 | 155/16/33 | 完成 | [查看](../../../JSTY-YOLO/experiments/chache_val300/jsty_chache_det5_960_1280_0_0_0_2_fp32_val300_20261102/evaluation.txt) |
| yoloe-v8l-seg / YOLOE固定头／移除SAVPE | 10 | 640 | 82.22 | 79.41 | 70.96 | 78.65 | 80.32 | 79.47 | 79.98 | 19.84 | 151/41/37 | 完成 | [查看](../../../YOLOE/finetune/chache_val300/runs/yoloe_search_20261029_r2/head_full_aug/test_val300/fixed_head/evaluation.txt) |
| yolov8l / YOLOv8-L微调 | 10 | 640 | 79.26 | 73.47 | 66.96 | **97.54** | 63.30 | 76.77 | 68.08 | **0.00** | 119/3/69 | 完成 | [查看](../../../YOLOE/finetune/chache_val300/runs/yolov8_val100_v1/test_val300/fixed_head/evaluation.txt) |

### 吊物（单类）

| 模型/方案 | 训练/类 | 输入 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| jsty_hanging_object_det1_0_0_0_1_fp32 / 定制监督 | 监督/训练量未绑定 | 960×1280 | **98.69** | **98.55** | **87.37** | **99.04** | 88.86 | **93.67** | **90.72** | **3.33** | 311/3/39 | 完成 | [查看](../../../JSTY-YOLO/experiments/diaowu_val300/jsty_hanging_object_det1_0_0_0_1_fp32_val300_20261102/evaluation.txt) |
| yoloe-v8l-seg / YOLOE固定头／移除SAVPE | 10 | 960 | 92.85 | 83.34 | 72.32 | 93.36 | 80.29 | 86.33 | 82.60 | 21.11 | 281/20/69 | 完成 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/head_full_960_seed2/test_val300/fixed_head/evaluation.txt) |
| yoloe-v8l-seg / YOLOE固定头／移除SAVPE | 10 | 960 | 92.50 | 84.92 | 73.37 | 91.96 | 75.14 | 82.70 | 78.00 | 23.33 | 263/23/87 | 完成 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/head_full_960_seed1/test_val300/fixed_head/evaluation.txt) |
| yolov8l / YOLOv8-L微调 | 10 | 640 | 92.19 | 85.34 | 69.72 | 36.52 | **94.86** | 52.74 | 71.89 | 77.78 | 332/577/18 | 完成 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/baseline_yolov8/test_val300/fixed_head/evaluation.txt) |


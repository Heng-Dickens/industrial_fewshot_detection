# YOLOE微调、YOLOv8对照与新类别扩展

已见类别结果按场景、训练5shot/10shot分为四张表，之后展示Chache4新类别提示迁移，最后归纳YOLOE微调的优点与证据范围。每种微调方式、每档推理shot仅保留AP50或AP50:95最高的实际配置；两项最优来自同一实验时合并，来自不同实验时分别列出。每行全部指标来自同一次测试推理。

微调方式只分为固定头（移除SAVPE）、视觉提示（更新SAVPE）、视觉提示（冻结SAVPE）和YOLOv8微调。学习率、更新范围、同图/跨图、输入尺寸和seed均作为配置选择，不再逐项展开所有尝试。训练K与推理提示K分开列出，训练10shot后用5shot提示仍属于10shot微调。

[全局比较](../../../FINAL_COMPARISON.md) · [完整测试明细CSV](03_yoloe_finetuning.csv)

[已见类别微调对比](#seen-class) · [Chache4新类别扩展](#novel-chache4) · [如何说明YOLOE微调的优点](#yoloe-advantages)

指标为百分数，均来自各自300张测试集；P/R/F1/F2与TP/FP/FN采用score≥0.5、IoU=0.5。每张表逐列加粗最优值。原始YOLOE同shot结果作为未微调基线；未训练YOLOv8初始化与其他搜索记录收在完整CSV中。

<a id="seen-class"></a>

## 一、已见类别：各方式最佳微调结果

### 叉车（五类） · 训练每类5张

| 方法 | 训练K/提示K | 最佳配置 | 保留依据 | AP50↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 原始结果 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 原始YOLOE视觉提示 | 0/5 | 原始权重；输入640 | 未微调基线 | 27.26 | 16.24 | 25.00 | 35.11 | 29.20 | 32.48 | 22.22 | 66/198/122 | [查看](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_5shot/evaluation.txt) |
| YOLOE固定头／移除SAVPE | 5/不适用 | 输入640；full；lr=0.0001；seed=0 | 两项AP最高 | **60.30** | **48.16** | 67.81 | **52.66** | **59.28** | **55.12** | 19.05 | 99/47/89 | [查看](../../../YOLOE/finetune/chache_val300/runs/yolo_5shot_20261102/yoloe-head/test_val300/fixed_head/evaluation.txt) |
| YOLOE视觉提示／更新SAVPE | 5/5 | 输入640；full/同图；lr=0.0001；seed=0 | 两项AP最高 | 48.80 | 39.52 | 62.50 | 31.91 | 42.25 | 35.38 | **0.79** | 60/36/128 | [查看](../../../YOLOE/finetune/chache_val300/runs/yolo_5shot_20261102/yoloe-vp/test_val300/visual_5shot/evaluation.txt) |
| YOLOv8-L微调 | 5/不适用 | 输入640；full；lr=0.0001；seed=0 | 两项AP最高 | 54.06 | 39.45 | **75.21** | 46.81 | 57.70 | 50.63 | 3.17 | 88/29/100 | [查看](../../../YOLOE/finetune/chache_val300/runs/yolo_5shot_20261102/yolov8/test_val300/fixed_head/evaluation.txt) |

### 叉车（五类） · 训练每类10张

| 方法 | 训练K/提示K | 最佳配置 | 保留依据 | AP50↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 原始结果 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 原始YOLOE视觉提示 | 0/5 | 原始权重；输入640 | 未微调基线 | 27.26 | 16.24 | 25.00 | 35.11 | 29.20 | 32.48 | 22.22 | 66/198/122 | [查看](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_5shot/evaluation.txt) |
| 原始YOLOE视觉提示 | 0/10 | 原始权重；输入640 | 未微调基线 | 29.00 | 18.04 | 22.26 | 37.77 | 28.01 | 33.15 | 26.98 | 71/248/117 | [查看](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_10shot/evaluation.txt) |
| YOLOE固定头／移除SAVPE | 10/不适用 | 输入640；full；lr=0.0001；seed=0；mosaic=0.2/翻转=0.5 | 两项AP最高 | **82.22** | **70.96** | 78.65 | **80.32** | **79.47** | **79.98** | 19.84 | 151/41/37 | [查看](../../../YOLOE/finetune/chache_val300/runs/yoloe_search_20261029_r2/head_full_aug/test_val300/fixed_head/evaluation.txt) |
| YOLOE视觉提示／冻结SAVPE | 10/5 | 输入640；neck/跨图；lr=1e-05；seed=0 | AP50最高 | 56.65 | 43.59 | 44.76 | 68.09 | 54.01 | 61.66 | 22.22 | 128/158/60 | [查看](../../../YOLOE/finetune/chache_val300/runs/yoloe_chache_frozen_savpe_20261030/vp_cross_neck_frozen_savpe/test_val300/visual_5shot/evaluation.txt) |
| YOLOE视觉提示／冻结SAVPE | 10/5 | 输入640；neck/跨图；lr=1e-05；seed=1 | AP50:95最高 | 55.85 | 43.84 | 42.20 | 63.30 | 50.64 | 57.54 | 22.22 | 119/163/69 | [查看](../../../YOLOE/finetune/chache_val300/runs/yoloe_chache_frozen_savpe_20261030/vp_cross_neck_frozen_savpe_seed1/test_val300/visual_5shot/evaluation.txt) |
| YOLOE视觉提示／冻结SAVPE | 10/10 | 输入640；neck/跨图；lr=1e-05；seed=1 | 两项AP最高 | 59.02 | 46.11 | 42.24 | 68.09 | 52.14 | 60.66 | 22.22 | 128/175/60 | [查看](../../../YOLOE/finetune/chache_val300/runs/yoloe_chache_frozen_savpe_20261030/vp_cross_neck_frozen_savpe_seed1/test_val300/visual_10shot/evaluation.txt) |
| YOLOE视觉提示／更新SAVPE | 10/5 | 输入640；neck/跨图；lr=1e-05；seed=0 | 两项AP最高 | 61.78 | 48.12 | 48.43 | 65.43 | 55.66 | 61.13 | 19.05 | 123/131/65 | [查看](../../../YOLOE/finetune/chache_val300/runs/yoloe_search_20261029_r2/vp_cross_neck/test_val300/visual_5shot/evaluation.txt) |
| YOLOE视觉提示／更新SAVPE | 10/10 | 输入640；neck/跨图；lr=1e-05；seed=0 | 两项AP最高 | 64.13 | 49.86 | 47.46 | 69.68 | 56.47 | 63.72 | 20.63 | 131/145/57 | [查看](../../../YOLOE/finetune/chache_val300/runs/yoloe_search_20261029_r2/vp_cross_neck/test_val300/visual_10shot/evaluation.txt) |
| YOLOv8-L微调 | 10/不适用 | 输入640；full；lr=0.0001；seed=0 | 两项AP最高 | 79.26 | 66.96 | **97.54** | 63.30 | 76.77 | 68.08 | **0.00** | 119/3/69 | [查看](../../../YOLOE/finetune/chache_val300/runs/yolov8_val100_v1/test_val300/fixed_head/evaluation.txt) |

### 吊物（单类） · 训练每类5张

| 方法 | 训练K/提示K | 最佳配置 | 保留依据 | AP50↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 原始结果 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 原始YOLOE视觉提示 | 0/5 | 原始权重；输入640 | 未微调基线 | 54.81 | 33.17 | 77.73 | 50.86 | 61.49 | 54.63 | 22.22 | 178/51/172 | [查看](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_5shot/evaluation.txt) |
| YOLOE固定头／移除SAVPE | 5/不适用 | 输入640；full；lr=0.0001；seed=0 | 两项AP最高 | **60.87** | **44.93** | 88.60 | **57.71** | **69.90** | **62.04** | 17.78 | 202/26/148 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yolo_5shot_20261102/yoloe-head/test_val300/fixed_head/evaluation.txt) |
| YOLOE视觉提示／更新SAVPE | 5/5 | 输入640；full/同图；lr=0.0001；seed=0 | 两项AP最高 | 59.71 | 43.27 | **94.24** | 37.43 | 53.58 | 42.56 | **8.89** | 131/8/219 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yolo_5shot_20261102/yoloe-vp/test_val300/visual_5shot/evaluation.txt) |
| YOLOv8-L微调 | 5/不适用 | 输入640；full；lr=0.0001；seed=0 | 两项AP最高 | 39.65 | 26.71 | 78.79 | 37.14 | 50.49 | 41.53 | 30.00 | 130/35/220 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yolo_5shot_20261102/yolov8/test_val300/fixed_head/evaluation.txt) |

### 吊物（单类） · 训练每类10张

| 方法 | 训练K/提示K | 最佳配置 | 保留依据 | AP50↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 原始结果 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 原始YOLOE视觉提示 | 0/5 | 原始权重；输入640 | 未微调基线 | 54.81 | 33.17 | 77.73 | 50.86 | 61.49 | 54.63 | 22.22 | 178/51/172 | [查看](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_5shot/evaluation.txt) |
| 原始YOLOE视觉提示 | 0/10 | 原始权重；输入640 | 未微调基线 | 55.96 | 33.76 | 74.21 | 53.43 | 62.13 | 56.60 | 24.44 | 187/65/163 | [查看](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_10shot/evaluation.txt) |
| YOLOE固定头／移除SAVPE | 10/不适用 | 输入960；full；lr=0.0001；seed=2 | AP50最高 | **92.85** | 72.32 | 93.36 | 80.29 | **86.33** | **82.60** | 21.11 | 281/20/69 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/head_full_960_seed2/test_val300/fixed_head/evaluation.txt) |
| YOLOE固定头／移除SAVPE | 10/不适用 | 输入960；full；lr=0.0001；seed=1 | AP50:95最高 | 92.50 | **73.37** | 91.96 | 75.14 | 82.70 | 78.00 | 23.33 | 263/23/87 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/head_full_960_seed1/test_val300/fixed_head/evaluation.txt) |
| YOLOE视觉提示／冻结SAVPE | 10/5 | 输入640；full/同图；lr=0.0001；seed=0 | 两项AP最高 | 81.24 | 58.25 | **94.93** | 37.43 | 53.69 | 42.59 | **7.78** | 131/7/219 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_frozen_savpe_20261030/vp_full_lr0.0001_frozen_savpe/test_val300/visual_5shot/evaluation.txt) |
| YOLOE视觉提示／冻结SAVPE | 10/10 | 输入640；full/同图；lr=0.0001；seed=0 | 两项AP最高 | 83.33 | 60.17 | 92.12 | 43.43 | 59.03 | 48.56 | 14.44 | 152/13/198 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_frozen_savpe_20261030/vp_full_lr0.0001_frozen_savpe/test_val300/visual_10shot/evaluation.txt) |
| YOLOE视觉提示／更新SAVPE | 10/5 | 输入640；full/同图；lr=0.0001；seed=1 | 两项AP最高 | 81.13 | 60.74 | 92.31 | 34.29 | 50.00 | 39.22 | 11.11 | 120/10/230 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/vp_full_lr0.0001_seed1/test_val300/visual_5shot/evaluation.txt) |
| YOLOE视觉提示／更新SAVPE | 10/10 | 输入640；full/同图；lr=0.0001；seed=0 | AP50最高 | 84.98 | 60.56 | 93.12 | 42.57 | 58.43 | 47.76 | 12.22 | 149/11/201 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/vp_full_lr0.0001/test_val300/visual_10shot/evaluation.txt) |
| YOLOE视觉提示／更新SAVPE | 10/10 | 输入640；full/同图；lr=0.0001；seed=1 | AP50:95最高 | 82.68 | 62.00 | 92.99 | 41.71 | 57.59 | 46.89 | 12.22 | 146/11/204 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/vp_full_lr0.0001_seed1/test_val300/visual_10shot/evaluation.txt) |
| YOLOv8-L微调 | 10/不适用 | 输入640；full；lr=0.0001；seed=0 | 两项AP最高 | 92.19 | 69.72 | 36.52 | **94.86** | 52.74 | 71.89 | 77.78 | 332/577/18 | [查看](../../../YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/baseline_yolov8/test_val300/fixed_head/evaluation.txt) |

已见类别部分共27行最佳结果与基线；同名CSV保留70条完整测试记录及详细参数。

[叉车（五类）全部实验](../../../YOLOE/finetune/chache_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md) · [训练/dev搜索CSV](../../../YOLOE/finetune/chache_val300/comparisons/yoloe_search_20261029/all_trials.csv) · [SAVPE方法说明](../../../YOLOE/finetune/chache_val300/comparisons/yoloe_search_20261029/SAVPE_METHODS.md)

[吊物（单类）全部实验](../../../YOLOE/finetune/diaowu_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md) · [训练/dev搜索CSV](../../../YOLOE/finetune/diaowu_val300/comparisons/yoloe_search_20261029/all_trials.csv) · [SAVPE方法说明](../../../YOLOE/finetune/diaowu_val300/comparisons/yoloe_search_20261029/SAVPE_METHODS.md)

full表示更新backbone、neck与头；neck表示冻结backbone。以上为已测试配置的最高值回顾，训练预算相同时，输入尺寸和搜索次数仍可能不同。下节的新类单类AP与本节全类别mAP分开比较。

<a id="novel-chache4"></a>

## 二、Chache4新类别：保留SAVPE的提示扩展能力

新类固定为Chache4，基础四类为Chache1/2/3与Qianyinche；四类微调后仅添加Chache4图框参考，不再更新权重。5shot训练图数为20/25，10shot为40/50，四类与五类条件均保留并更新SAVPE和backbone。共同88张dev排除Chache4正图，只按基础四类选权重。新类指标在完整300张测试图、30个Chache4真值框上统计；基础类图和背景图保留用于统计新类误检。新类单类AP不参加上节或全局五类mAP排序。

### 5shot · 新类Chache4

| 条件 | 训练图数 | Chache4训练K | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | 基础四类AP50↑ | 全五类AP50↑ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 原始预训练＋视觉提示 | 0 | 0 | 33.68 | 26.87 | 23.58 | 29.73 | **73.33** | 42.31 | **56.70** | 17.04 | 25.66 | 27.26 | 22/52/8 | 完成 | [查看](../../../YOLOE/finetune/chache_val300/runs/novel_chache4_20261102/pretrained_5shot/test_val300/novel/evaluation.txt) |
| 四类微调→新增Chache4提示 | 20 | 0 | 58.22 | 47.54 | 42.28 | 43.90 | 60.00 | 50.70 | 55.90 | 8.15 | 35.85 | 40.33 | 18/23/12 | 完成 | [查看](../../../YOLOE/finetune/chache_val300/runs/novel_chache4_20261102/base_5shot/test_val300/novel/evaluation.txt) |
| 五类一起微调 | 25 | 5 | **66.82** | **63.88** | **58.31** | **45.95** | 56.67 | **50.75** | 54.14 | **7.41** | **36.40** | **42.49** | 17/20/13 | 完成 | [查看](../../../YOLOE/finetune/chache_val300/runs/novel_chache4_20261102/all_5shot/test_val300/novel/evaluation.txt) |

### 10shot · 新类Chache4

| 条件 | 训练图数 | Chache4训练K | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | 基础四类AP50↑ | 全五类AP50↑ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 原始预训练＋视觉提示 | 0 | 0 | 41.89 | 35.85 | 31.43 | 32.50 | **86.67** | 47.27 | 65.00 | 18.15 | 25.78 | 29.00 | 26/54/4 | 完成 | [查看](../../../YOLOE/finetune/chache_val300/runs/novel_chache4_20261102/pretrained_10shot/test_val300/novel/evaluation.txt) |
| 四类微调→新增Chache4提示 | 40 | 0 | 59.04 | 37.85 | 34.59 | **90.91** | 33.33 | 48.78 | 38.17 | **0.37** | 40.91 | 44.53 | 10/1/20 | 完成 | [查看](../../../YOLOE/finetune/chache_val300/runs/novel_chache4_20261102/base_10shot/test_val300/novel/evaluation.txt) |
| 五类一起微调 | 50 | 10 | **79.06** | **76.93** | **70.19** | 68.97 | 66.67 | **67.80** | **67.11** | 3.33 | **45.27** | **52.03** | 20/9/10 | 完成 | [查看](../../../YOLOE/finetune/chache_val300/runs/novel_chache4_20261102/all_10shot/test_val300/novel/evaluation.txt) |

[新类指标CSV（独立标签范围）](05_novel_chache4.csv) · [完整协议、训练配置与可视化](../../../YOLOE/finetune/chache_val300/comparisons/novel_chache4_20261102/README.md)

“新类”指本次微调未见，不保证原始预训练未见；参考标注范围经核验，但未标注同图物体仍需人工确认。这不是按新视频/新相机重划分的实验；只有一个held-out类别与一个seed，结论限于本次提示迁移。

<a id="yoloe-advantages"></a>

## 三、如何说明YOLOE微调的优点

**建议围绕“小样本任务适配”和“保留SAVPE后的新类别提示扩展”两点展开，分别给出证据。**

**1. 已见类别：YOLOE具备小样本适配优势，但需要区分微调路线。** 在每类5张、640输入的现有对照中：

- 叉车（五类）：固定头YOLOE的AP50为60.30%，YOLOv8为54.06%，高6.24个百分点；保留并更新SAVPE的视觉提示YOLOE为48.80%，相对YOLOv8变化-5.26个百分点。
- 吊物（单类）：固定头YOLOE的AP50为60.87%，YOLOv8为39.65%，高21.22个百分点；保留并更新SAVPE的视觉提示YOLOE为59.71%，相对YOLOv8变化+20.06个百分点。

固定头成绩体现所测YOLOE方案在少量标注下的适配效果；该路线移除了SAVPE，不能用它的提升证明SAVPE有效。保留SAVPE的5shot路线在吊物上优于当前YOLOv8对照，在叉车上未胜出。这些是已测配置的结果；预训练与搜索预算不同，不能据此宣称所有条件下均优于YOLOv8。

**2. 新类别扩展：保留SAVPE后，基础类微调权重可以直接配合新类参考使用。** 四类微调阶段不使用Chache4监督；完成后冻结权重，只在推理时加入带框参考。SAVPE为新增类别编码视觉提示，无须再次针对Chache4执行梯度更新：

- 5shot：Chache4 AP50由原始权重的33.68%提高到58.22%，增加24.54个百分点；AP50:95由23.58%提高到42.28%。
- 10shot：Chache4 AP50由原始权重的41.89%提高到59.04%，增加17.15个百分点；AP50:95由31.43%提高到34.59%。

这组结果支持的优点是：**基础任务微调后仍可通过新增参考扩展类别，并且本次新类AP高于原始YOLOE视觉提示。** 这验证的是保留SAVPE的完整方案；backbone等参数也同时微调，不能把提升单独归因于SAVPE参数更新。新增类别仍需要5/10张带框参考，所以应表述为“无需新类再训练”，不能写成“无需新类标注”。当前YOLOv8对照采用固定类别输出，尚未做同一四类训练→新增类适配协议，因此没有YOLOv8新类迁移的直接数值对照。

**3. 同时说明精度与召回的代价。** 五类一起微调的Chache4 AP50为66.82/79.06%（5/10shot），仍高于四类微调后只加提示的结果。10shot四类微调条件在score≥0.5时P=90.91%、R=33.33%，TP/FP/FN=10/1/20：误检减少，但30个真值中仍漏检20个。因此不能把AP提升等同于固定阈值召回提升，也不能声称已证明新视频、新相机或所有新类别上的泛化。

**可用于总结的表述：**

> YOLOE微调的价值同时体现在小样本任务适配与视觉提示扩展能力上。已见类别对照中，固定头路线取得较好的少样本检测结果；保留SAVPE的路线则允许基础类微调完成后，通过新增带框参考检测本次微调未见的类别，无须再次更新权重。在Chache4实验中，5shot和10shot提示的AP50分别达到58.22%和59.04%，高于原始YOLOE的33.68%和41.89%。现有结果支持本场景中的提示迁移价值，但尚不足以证明保留SAVPE在所有任务上精度都优于YOLOv8；固定阈值召回与多类别、多seed泛化仍需进一步验证。

# YOLOE→Qwen分阶段处理

比较三种YOLOE微调获得候选后，再交由Qwen3.8-27B筛选、去重和修正框的效果。每个场景分别选训练10shot的固定头（移除SAVPE）、视觉提示（更新SAVPE）、视觉提示（冻结SAVPE）各一组；视觉提示推理使用10shot，Qwen统一仅每类1shot，seed0、无负例。原始未微调级联与直接Qwen结果保留在附录。

[全局比较](../../../FINAL_COMPARISON.md) · [子任务导航](README.md)

指标均为百分数。AP使用该运行保存的全部候选；P/R/F1/F2使用score≥0.5、IoU=0.5，TP/FP/FN为框数。箭头表示优化方向；每张表内逐列加粗最优值，按显示精度并列最优同时加粗。参数值不作最优标记。未完成、队列不符或类别范围不全的记录保留展示，但不参与最优值计算。†表示本机仅保存指标和元数据，尚无逐图记录可复核；仍可作为有此证据限制的结果对比。P或负图错误率单独最优不等于检测最好，需结合R和TP/FP/FN阅读。

[三种微调级联：6组补跑进度与选中权重](../../../OVD-MLLM/comparisons/finetuned_yoloe_qwen_20261103/README.md)

S0保留原始逐图预测，B0仅保留score≥0.01、每类最多100框的候选；B1只删除明确非目标，uncertain保留；B1D仅按Qwen明确给出的同类同实例关系且IoU≥0.5去重；B2只改坐标，候选集合、类别和原始分数固定，不补框。没有候选的图片不调用Qwen。候选score下限与P/R工作阈值0.5不同。S0→B0反映截断影响，B0→后续阶段反映Qwen处理影响。选择依据为现有测试AP50最高的实际配置，尺寸、seed和训练范围可能不同，不能当作等预算独立验证。

**相同阶段合并展示：**已逐图核验6组B1与B1D的全部300条预测完全相同，独立去重阶段额外删除0框；表中合为“B1筛选 / B1D去重（无额外变化）”，不算两次独立改进。原先相邻两行相同是实际输出未变，重复列出没有增加信息。B1D是本地后处理，未再次调用Qwen。本批去重关联多数涉及筛选已删除的框，其余关联不满足同类或IoU门限，因此没有额外删除。S0与B0仍分别保留：删去score<0.01的低分框会影响AP，而P/R使用score≥0.5，因此P/R可能完全相同。[逐图核验、框数和SHA256](04_yoloe_qwen_stage_audit.json) · [所有阶段完整CSV](04_yoloe_qwen.csv)。

### 叉车（五类） · 三种微调→Qwen27B

| 微调方式 | 阶段 | 提示数量 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 固定头／移除SAVPE | S0 原始YOLOE | YOLOE 固定头 / Qwen 0 | **82.22** | **79.41** | **70.96** | 78.65 | **80.32** | **79.47** | **79.98** | 19.84 | 151/41/37 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 固定头／移除SAVPE | B0 候选截断 | YOLOE 固定头 / Qwen 0 | 80.94 | 78.39 | 70.11 | 78.65 | **80.32** | **79.47** | **79.98** | 19.84 | 151/41/37 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 固定头／移除SAVPE | B1筛选 / B1D去重（无额外变化） | YOLOE 固定头 / Qwen 1 | 63.98 | 63.98 | 56.47 | **95.93** | 62.77 | 75.88 | 67.43 | **0.00** | 118/5/70 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 固定头／移除SAVPE | B2 框修正 | YOLOE 固定头 / Qwen 1 | 62.06 | 51.77 | 46.46 | 93.50 | 61.17 | 73.95 | 65.71 | **0.00** | 115/8/73 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 视觉提示／更新SAVPE | S0 原始YOLOE | YOLOE 10 / Qwen 0 | 64.13 | 54.88 | 49.86 | 47.46 | 69.68 | 56.47 | 63.72 | 20.63 | 131/145/57 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 视觉提示／更新SAVPE | B0 候选截断 | YOLOE 10 / Qwen 0 | 64.10 | 54.86 | 49.84 | 47.46 | 69.68 | 56.47 | 63.72 | 20.63 | 131/145/57 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 视觉提示／更新SAVPE | B1筛选 / B1D去重（无额外变化） | YOLOE 10 / Qwen 1 | 43.45 | 36.01 | 33.18 | 72.97 | 43.09 | 54.18 | 46.93 | **0.00** | 81/30/107 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 视觉提示／更新SAVPE | B2 框修正 | YOLOE 10 / Qwen 1 | 50.72 | 36.75 | 35.77 | 83.78 | 49.47 | 62.21 | 53.88 | **0.00** | 93/18/95 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 视觉提示／冻结SAVPE | S0 原始YOLOE | YOLOE 10 / Qwen 0 | 59.02 | 49.76 | 46.11 | 42.24 | 68.09 | 52.14 | 60.66 | 22.22 | 128/175/60 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 视觉提示／冻结SAVPE | B0 候选截断 | YOLOE 10 / Qwen 0 | 58.99 | 49.73 | 46.07 | 42.24 | 68.09 | 52.14 | 60.66 | 22.22 | 128/175/60 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 视觉提示／冻结SAVPE | B1筛选 / B1D去重（无额外变化） | YOLOE 10 / Qwen 1 | 42.27 | 36.62 | 33.29 | 73.39 | 42.55 | 53.87 | 46.46 | **0.00** | 80/29/108 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 视觉提示／冻结SAVPE | B2 框修正 | YOLOE 10 / Qwen 1 | 49.97 | 37.28 | 35.18 | 84.40 | 48.94 | 61.95 | 53.43 | **0.00** | 92/17/96 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |

固定头／移除SAVPE：B1保留137框，B1D仍为137框，去重额外删除0框。[B1D原始评估](../../../OVD-MLLM/experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)。

固定头／移除SAVPE · B1筛选 / B1D去重（无额外变化）相对B0：AP50变化-16.96个百分点，P变化+17.29，R变化-17.55。

固定头／移除SAVPE · B2 框修正相对B0：AP50变化-18.89个百分点，P变化+14.85，R变化-19.15。

视觉提示／更新SAVPE：B1保留190框，B1D仍为190框，去重额外删除0框。[B1D原始评估](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)。

视觉提示／更新SAVPE · B1筛选 / B1D去重（无额外变化）相对B0：AP50变化-20.65个百分点，P变化+25.51，R变化-26.60。

视觉提示／更新SAVPE · B2 框修正相对B0：AP50变化-13.38个百分点，P变化+36.32，R变化-20.21。

视觉提示／冻结SAVPE：B1保留203框，B1D仍为203框，去重额外删除0框。[B1D原始评估](../../../OVD-MLLM/experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)。

视觉提示／冻结SAVPE · B1筛选 / B1D去重（无额外变化）相对B0：AP50变化-16.72个百分点，P变化+31.15，R变化-25.53。

视觉提示／冻结SAVPE · B2 框修正相对B0：AP50变化-9.03个百分点，P变化+42.16，R变化-19.15。

### 吊物（单类） · 三种微调→Qwen27B

| 微调方式 | 阶段 | 提示数量 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 固定头／移除SAVPE | S0 原始YOLOE | YOLOE 固定头 / Qwen 0 | **92.85** | **83.34** | **72.32** | 93.36 | **80.29** | **86.33** | **82.60** | 21.11 | 281/20/69 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 固定头／移除SAVPE | B0 候选截断 | YOLOE 固定头 / Qwen 0 | 91.33 | 82.04 | 71.37 | 93.36 | **80.29** | **86.33** | **82.60** | 21.11 | 281/20/69 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 固定头／移除SAVPE | B1筛选 / B1D去重（无额外变化） | YOLOE 固定头 / Qwen 1 | 57.95 | 52.41 | 47.59 | **95.12** | 55.71 | 70.27 | 60.75 | 10.00 | 195/10/155 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 固定头／移除SAVPE | B2 框修正 | YOLOE 固定头 / Qwen 1 | 56.72 | 49.36 | 42.99 | 93.66 | 54.86 | 69.19 | 59.81 | 10.00 | 192/13/158 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 视觉提示／更新SAVPE | S0 原始YOLOE | YOLOE 10 / Qwen 0 | 84.98 | 68.10 | 60.56 | 93.12 | 42.57 | 58.43 | 47.76 | 12.22 | 149/11/201 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 视觉提示／更新SAVPE | B0 候选截断 | YOLOE 10 / Qwen 0 | 77.44 | 62.26 | 56.02 | 93.12 | 42.57 | 58.43 | 47.76 | 12.22 | 149/11/201 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 视觉提示／更新SAVPE | B1筛选 / B1D去重（无额外变化） | YOLOE 10 / Qwen 1 | 55.09 | 45.90 | 42.11 | 94.59 | 40.00 | 56.22 | 45.22 | **8.89** | 140/8/210 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 视觉提示／更新SAVPE | B2 框修正 | YOLOE 10 / Qwen 1 | 54.85 | 44.84 | 39.67 | 93.92 | 39.71 | 55.82 | 44.90 | **8.89** | 139/9/211 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 视觉提示／冻结SAVPE | S0 原始YOLOE | YOLOE 10 / Qwen 0 | 83.33 | 66.73 | 60.17 | 92.12 | 43.43 | 59.03 | 48.56 | 14.44 | 152/13/198 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/source/evaluation.txt) |
| 视觉提示／冻结SAVPE | B0 候选截断 | YOLOE 10 / Qwen 0 | 77.05 | 63.89 | 57.12 | 92.12 | 43.43 | 59.03 | 48.56 | 14.44 | 152/13/198 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/baseline/evaluation.txt) |
| 视觉提示／冻结SAVPE | B1筛选 / B1D去重（无额外变化） | YOLOE 10 / Qwen 1 | 55.62 | 46.74 | 42.65 | 91.61 | 40.57 | 56.24 | 45.66 | 14.44 | 142/13/208 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/filtered/evaluation.txt) |
| 视觉提示／冻结SAVPE | B2 框修正 | YOLOE 10 / Qwen 1 | 55.51 | 44.62 | 39.36 | 91.61 | 40.57 | 56.24 | 45.66 | 14.44 | 142/13/208 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |

固定头／移除SAVPE：B1保留228框，B1D仍为228框，去重额外删除0框。[B1D原始评估](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)。

固定头／移除SAVPE · B1筛选 / B1D去重（无额外变化）相对B0：AP50变化-33.38个百分点，P变化+1.77，R变化-24.57。

固定头／移除SAVPE · B2 框修正相对B0：AP50变化-34.61个百分点，P变化+0.30，R变化-25.43。

视觉提示／更新SAVPE：B1保留228框，B1D仍为228框，去重额外删除0框。[B1D原始评估](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)。

视觉提示／更新SAVPE · B1筛选 / B1D去重（无额外变化）相对B0：AP50变化-22.35个百分点，P变化+1.47，R变化-2.57。

视觉提示／更新SAVPE · B2 框修正相对B0：AP50变化-22.59个百分点，P变化+0.79，R变化-2.86。

视觉提示／冻结SAVPE：B1保留231框，B1D仍为231框，去重额外删除0框。[B1D原始评估](../../../OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt)。

视觉提示／冻结SAVPE · B1筛选 / B1D去重（无额外变化）相对B0：AP50变化-21.43个百分点，P变化-0.51，R变化-2.86。

视觉提示／冻结SAVPE · B2 框修正相对B0：AP50变化-21.54个百分点，P变化-0.51，R变化-2.86。

<details>
<summary>附录：未微调YOLOE级联与直接Qwen对照</summary>

### 叉车（五类） · 历史/直接对照

| 处理阶段 | 参考数量 | 关键输入 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| S0 原始YOLOE | YOLOE 10 / Qwen 0 | 仅YOLOE | 29.00 | 17.54 | 18.04 | 22.26 | 37.77 | 28.01 | 33.15 | 26.98 | 71/248/117 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/source/evaluation.txt) |
| B0 候选截断 | YOLOE 10 / Qwen 0 | 仅YOLOE | 28.98 | 17.53 | 18.03 | 22.26 | 37.77 | 28.01 | 33.15 | 26.98 | 71/248/117 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/baseline/evaluation.txt) |
| B1 Qwen筛选 | YOLOE 10 / Qwen 1 | YOLOE候选→Qwen27B | 49.31 | 27.16 | 27.92 | **80.00** | 31.91 | 45.63 | 36.28 | **0.00** | 60/15/128 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/filtered/evaluation.txt) |
| B1D 去重 | YOLOE 10 / Qwen 1 | YOLOE候选→Qwen27B | 49.38 | 27.17 | 27.93 | **80.00** | 31.91 | 45.63 | 36.28 | **0.00** | 60/15/128 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/deduplicated/evaluation.txt) |
| B2 框修正 | YOLOE 10 / Qwen 1 | YOLOE候选→Qwen27B | 52.20 | 25.19 | 27.86 | **80.00** | 31.91 | 45.63 | 36.28 | **0.00** | 60/15/128 | 完成 | [查看](../../../OVD-MLLM/experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt) |
| 直接Qwen对照 | 1 | O+C+B+J; 负参考0 | 53.41 | 35.49 | 35.46 | 63.92 | 65.96 | 64.92 | 65.54 | 7.14 | 124/70/64 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 直接Qwen对照 | 1 | O+C; 负参考0 | 51.67 | 39.13 | 36.80 | 56.28 | 64.36 | 60.05 | 62.56 | 18.25 | 121/94/67 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 直接Qwen对照 | 10 | O+B+J; 负参考0 | **77.20** | 49.37 | 50.22 | 76.96 | 83.51 | **80.10** | **82.11** | 11.11 | 157/47/31 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot_%E7%AC%AC3%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 直接Qwen对照 | 10 | O+C; 负参考0 | 74.48 | **56.63** | **53.36** | 74.18 | **84.04** | 78.80 | 81.87 | 15.08 | 158/55/30 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

### 吊物（单类） · 历史/直接对照

| 处理阶段 | 参考数量 | 关键输入 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| S0 原始YOLOE | YOLOE 10 / Qwen 0 | 仅YOLOE | 55.96 | 38.28 | 33.76 | 74.21 | 53.43 | 62.13 | 56.60 | 24.44 | 187/65/163 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/source/evaluation.txt) |
| B0 候选截断 | YOLOE 10 / Qwen 0 | 仅YOLOE | 55.83 | 38.28 | 33.73 | 74.21 | 53.43 | 62.13 | 56.60 | 24.44 | 187/65/163 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/baseline/evaluation.txt) |
| B1筛选 / B1D去重（无额外变化） | YOLOE 10 / Qwen 1 | YOLOE候选→Qwen27B | 57.45 | 36.08 | 33.20 | 85.12 | 52.29 | 64.78 | 56.66 | **22.22** | 183/32/167 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/filtered/evaluation.txt) |
| B2 框修正 | YOLOE 10 / Qwen 1 | YOLOE候选→Qwen27B | **58.36** | 40.78 | 36.44 | 84.19 | 51.71 | 64.07 | 56.04 | **22.22** | 181/34/169 | 完成 | [查看](../../../OVD-MLLM/experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt) |
| 直接Qwen对照 | 1 | O+B; 负参考1 | 53.40 | 48.15 | 42.02 | 87.39 | 59.43 | 70.75 | 63.49 | 32.22 | 208/30/142 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 直接Qwen对照 | 10 | O+B+J; 负参考0 | 56.75 | **51.21** | **43.71** | **89.54** | **61.14** | **72.67** | **65.28** | 27.78 | 214/25/136 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC4%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

</details>

三种微调级联与旧未微调级联分别报告，不能将旧模型的增益直接套用到微调模型。直接Qwen的输入和计算预算不同；筛选/框修正的收益须结合AP、召回及误检一起判断。

# Qwen输入、shot与模型对比

按实验顺序组织为三个阶段：输入对比按场景和shot分为4张表，shot对比和模型对比各按场景分为2张表，正文共8张表。先只用27B选择输入，再只用27B比较shot，最后比较27B、Flash、Max。O=参考原图，C=目标裁剪图，B=带框参考图，J=标注JSON。0shot没有视觉参考，但仍输入测试查询图。

[全局比较](../../../FINAL_COMPARISON.md) · [子任务导航](README.md)

指标均为百分数。AP使用该运行保存的全部候选；P/R/F1/F2使用score≥0.5、IoU=0.5，TP/FP/FN为框数。箭头表示优化方向；每张表内逐列加粗最优值，按显示精度并列最优同时加粗。参数值不作最优标记。未完成、队列不符或类别范围不全的记录保留展示，但不参与最优值计算。†表示本机仅保存指标和元数据，尚无逐图记录可复核；仍可作为有此证据限制的结果对比。P或负图错误率单独最优不等于检测最好，需结合R和TP/FP/FN阅读。

主线统一负参考=0、support seed=0；叉车使用提示v2，吊物使用v1，测试集均为各自固定300张。缺少的条件显示待补跑，不用其他输入、其他shot或旧测试集结果代填。已有其他条件与Max旧结果收在折叠附录，CSV仍保留全部真实运行。

- [第一步：比较27B的输入方式](#step-1)
- [第二步：固定输入比较shot](#step-2)
- [第三步：固定10shot比较模型](#step-3)

[27B输入消融缺项补跑进度与新增结果](../../../Qwen/comparisons/input_supplement_20261103/README.md)

[第二步1shot/20shot补跑进度与参考来源](../../../Qwen/comparisons/shot_supplement_20261103/README.md)

<a id="step-1"></a>

## 第一步：27B的5shot与10shot输入对比

本阶段仅比较qwen3.8-27b。每个场景分别列5shot与10shot两张表，共4张；每张表内加粗最优指标，比较相同shot下输入组成的影响。后续只固定一种输入：在5shot与10shot均已完成的候选中，选择两档AP50算术平均最高者；平均AP50相同再比较平均AP50:95。此规则不会把两档各自最好的不同输入拼成shot曲线。只有5shot完成的输入保留，但须补齐10shot才能参与这项选择。

### 叉车（五类） · 27B输入对比 · 5shot/类

| 输入 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| O+B | 69.22 | 48.15 | 45.83 | 74.37 | 78.72 | 76.49 | 77.81 | 9.52 | 148/51/40 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C | 75.00 | 50.62 | 50.17 | 67.38 | **83.51** | 74.58 | 79.70 | 30.16 | 157/76/31 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+J | 66.28 | 37.65 | 42.21 | 65.91 | 77.13 | 71.08 | 74.59 | 20.63 | 145/75/43 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+B+J | 67.57 | 38.85 | 42.80 | 74.36 | 77.13 | 75.72 | 76.56 | **5.56** | 145/50/43 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+J | 74.18 | 49.08 | 49.31 | 69.20 | 82.45 | 75.24 | 79.41 | 23.81 | 155/69/33 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B | **75.07** | **50.63** | 50.28 | **77.78** | 81.91 | **79.79** | **81.05** | **5.56** | 154/44/34 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B+J | 74.23 | 49.71 | **50.39** | 75.61 | 82.45 | 78.88 | 80.98 | 11.11 | 155/50/33 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

### 叉车（五类） · 27B输入对比 · 10shot/类

| 输入 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| O+B | 75.30 | 53.05 | 51.40 | 80.20 | **84.04** | 82.08 | 83.25 | 6.35 | 158/39/30 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C | 74.48 | **56.63** | **53.36** | 74.18 | **84.04** | 78.80 | 81.87 | 15.08 | 158/55/30 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+J | 70.82 | 42.15 | 45.58 | 78.06 | 81.38 | 79.69 | 80.70 | 6.35 | 153/43/35 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+B+J | **77.20** | 49.37 | 50.22 | 76.96 | 83.51 | 80.10 | 82.11 | 11.11 | 157/47/31 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot_%E7%AC%AC3%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+J | 74.37 | 48.89 | 49.68 | 76.59 | 83.51 | 79.90 | 82.03 | 10.32 | 157/48/31 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot_%E7%AC%AC4%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B | 75.62 | 53.76 | 51.37 | 80.61 | **84.04** | 82.29 | 83.33 | **5.56** | 158/38/30 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B+J | 75.60 | 53.38 | 51.15 | **81.03** | **84.04** | **82.51** | **83.42** | **5.56** | 158/37/30 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot_%E7%AC%AC5%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

**叉车（五类）后续固定O+C+B**。已完成两档的候选平均AP50：O+C+B=75.35%；O+C+B+J=74.91%；O+C=74.74%；O+C+J=74.27%；O+B+J=72.39%；O+B=72.26%；O+J=68.55%。所选输入5shot/10shot的AP50分别为75.07/75.62%，平均75.35%。

### 吊物（单类） · 27B输入对比 · 5shot/类

| 输入 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| O+B | 53.35 | 49.60 | 42.23 | 89.08 | 58.29 | 70.47 | 62.62 | **26.67** | 204/25/146 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C | 52.96 | 46.01 | 40.19 | 87.71 | 59.14 | 70.65 | 63.26 | 31.11 | 207/29/143 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+J | 51.98 | 47.30 | 40.06 | 87.66 | 58.86 | 70.43 | 63.00 | 32.22 | 206/29/144 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+B+J | **54.42** | 48.86 | **42.36** | 88.46 | 59.14 | 70.89 | 63.34 | 28.89 | 207/27/143 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+J | 52.23 | 48.12 | 40.69 | 87.66 | 58.86 | 70.43 | 63.00 | 32.22 | 206/29/144 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B | 53.89 | **49.74** | 42.14 | **89.57** | 58.86 | **71.03** | 63.19 | **26.67** | 206/24/144 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B+J | 52.76 | 47.19 | 40.50 | 87.03 | **59.43** | 70.63 | **63.45** | 34.44 | 208/31/142 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

### 吊物（单类） · 27B输入对比 · 10shot/类

| 输入 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| O+B | 50.70 | 48.45 | 40.57 | 88.58 | 55.43 | 68.19 | 59.91 | 25.56 | 194/25/156 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C | 53.81 | 48.98 | 42.65 | **90.09** | 57.14 | 69.93 | 61.65 | **24.44** | 200/22/150 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+J | 53.00 | 46.71 | 40.94 | 87.93 | 58.29 | 70.10 | 62.50 | 30.00 | 204/28/146 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+B+J | **56.75** | **51.21** | **43.71** | 89.54 | **61.14** | **72.67** | **65.28** | 27.78 | 214/25/136 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC4%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+J | 53.94 | 45.76 | 40.71 | 88.09 | 59.14 | 70.77 | 63.30 | 28.89 | 207/28/143 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC3%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B | 51.76 | 47.97 | 41.17 | 89.14 | 56.29 | 69.00 | 60.76 | 25.56 | 197/24/153 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC5%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B+J | 54.56 | 48.33 | 41.60 | 88.89 | 59.43 | 71.23 | 63.65 | 27.78 | 208/26/142 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC6%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

**吊物（单类）后续固定O+B+J**。已完成两档的候选平均AP50：O+B+J=55.58%；O+C+B+J=53.66%；O+C=53.38%；O+C+J=53.08%；O+C+B=52.82%；O+J=52.49%；O+B=52.03%。所选输入5shot/10shot的AP50分别为54.42/56.75%，平均55.58%。

这是现有测试结果的探索性选择；两个场景的七种输入均已补齐5shot与10shot。正式确认输入方案宜使用独立验证集，不能把按测试集选出的最高值当作独立验证结论。

<a id="step-2"></a>

## 第二步：固定输入比较27B的shot（叉车至10shot，吊物至20shot）

本阶段仍只比较qwen3.8-27b，每个场景一张表。采用第一步选定的输入，提示版本、负参考数与seed保持一致。0shot是同版本文字提示基线，不含O/C/B/J参考；其他shot使用固定的视觉输入。叉车比较0/1/5/10shot，吊物比较0/1/5/10/20shot。ΔAP50相对本场景0shot计算，单位为百分点；尚缺的条件保留空位，不假定增加shot必然提升。

吊物20shot保留原10shot为前缀，再从原25张正参考池中补选10张，seed0；参考与300张测试图像SHA无重叠。参考使用原有JPEG质量90、最长边1280与独立图片格式。叉车20shot按用户要求取消，叉车1shot直接引用已有完整结果。[吊物20shot参考来源与抽样清单](../../../../data/references/qwen_shot_extension_20261103/selection_provenance.json)。

### 叉车（五类） · 27B · 固定O+C+B

| shot/类 | 输入 | ΔAP50/百分点 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | text_only | +0.00 | 44.59 | 33.40 | 30.27 | 44.86 | 51.06 | 47.76 | 49.69 | 29.37 | 96/118/92 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/evaluation.txt) |
| 1 | O+C+B | +5.41 | 50.00 | 39.45 | 36.35 | 56.59 | 61.70 | 59.03 | 60.61 | 14.29 | 116/89/72 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 5 | O+C+B | +30.48 | 75.07 | 50.63 | 50.28 | 77.78 | 81.91 | 79.79 | 81.05 | **5.56** | 154/44/34 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 10 | O+C+B | +31.03 | **75.62** | **53.76** | **51.37** | **80.61** | **84.04** | **82.29** | **83.33** | **5.56** | 158/38/30 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

已完成AP50序列：0shot 44.59% → 1shot 50.00% → 5shot 75.07% → 10shot 75.62%。4档shot均已完成。

### 吊物（单类） · 27B · 固定O+B+J

| shot/类 | 输入 | ΔAP50/百分点 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | text_only | +0.00 | 52.20 | 46.28 | 40.96 | 87.39 | 57.43 | 69.31 | 61.66 | 28.89 | 201/29/149 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/evaluation.txt) |
| 1 | O+B+J | -0.39 | 51.81 | 47.12 | 40.47 | 83.81 | 59.14 | 69.35 | 62.84 | 42.22 | 207/40/143 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 5 | O+B+J | +2.22 | 54.42 | 48.86 | 42.36 | 88.46 | 59.14 | 70.89 | 63.34 | 28.89 | 207/27/143 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 10 | O+B+J | +4.55 | **56.75** | **51.21** | **43.71** | **89.54** | **61.14** | **72.67** | **65.28** | **27.78** | 214/25/136 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC4%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 20 | O+B+J | +0.94 | 53.14 | 46.12 | 40.36 | 88.03 | 58.86 | 70.55 | 63.04 | 30.00 | 206/28/144 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_20shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

已完成AP50序列：0shot 52.20% → 1shot 51.81% → 5shot 54.42% → 10shot 56.75% → 20shot 53.14%。5档shot均已完成。

<a id="step-3"></a>

## 第三步：固定10shot，比较27B、Flash、Max

到本阶段才比较模型。每个场景一张表，只比较10shot/类的三个模型。固定叉车O+C+B、吊物O+B+J，在同场景、同shot、同版本、无负例和seed0条件下比较三个模型。不为某个模型单独改用其最优输入；不把缺失Max替换成旧99张结果。每张表至少两个模型有完整可比结果才加粗最优值，仅有一个模型时不判断模型优胜。

[Flash与Max的10shot补跑队列（第二步当前批次完成后启动）](../../../Qwen/comparisons/model10_supplement_20261103/README.md)

### 叉车（五类） · 固定O+C+B · 模型对比（10shot）

| shot/类 | 模型 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 10 | qwen3.8-27b | 75.62 | 53.76 | 51.37 | **80.61** | 84.04 | **82.29** | 83.33 | **5.56** | 158/38/30 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 10 | qwen3.8-flash | **78.24** | **57.15** | **53.53** | 74.44 | **88.30** | 80.78 | **85.13** | 25.40 | 166/57/22 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_10shot_%E7%AC%AC3%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 10 | qwen3.8-max | 76.42 | 43.46 | 43.64 | 76.33 | 84.04 | 80.00 | 82.38 | 10.32 | 158/49/30 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-max_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

### 吊物（单类） · 固定O+B+J · 模型对比（10shot）

| shot/类 | 模型 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 10 | qwen3.8-27b | 56.75 | 51.21 | 43.71 | 89.54 | 61.14 | **72.67** | 65.28 | 27.78 | 214/25/136 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC4%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 10 | qwen3.8-flash | **56.78** | 50.43 | 44.17 | 82.22 | **63.43** | 71.61 | **66.47** | 47.78 | 222/48/128 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_10shot_%E7%AC%AC3%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 10 | qwen3.8-max | 56.04 | **53.11** | **44.38** | **97.51** | 56.00 | 71.14 | 61.21 | **2.22** | 196/5/154 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

叉车（五类）固定O+C+B：三个模型10shot均已完成。

吊物（单类）固定O+B+J：三个模型10shot均已完成。

Max的旧吊物99张结果单列在附录，不代替当前300张结果；其他输入的Flash结果也保留在附录。

<details>
<summary>附录A：其他输入、负例、提示版本和Flash已完成/历史尝试</summary>

以下条件保留原始指标，不参与上述固定条件主线；按shot、负参考数和提示版本分别计算粗体。

### 叉车（五类） · 其他条件

| 模型 | shot/类 | 输入 | 负参考 | 版本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| qwen3.8-27b | 0 | text_only | 0 | MLLM提示_v1 | **47.85** | **34.38** | **33.83** | **50.72** | **55.85** | **53.16** | **54.74** | **30.16** | 105/102/83 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/evaluation.txt) |
| qwen3.8-27b | 1 | O+B | 0 | MLLM提示_v2 | 50.77 | 36.08 | 34.89 | 60.50 | 64.36 | 62.37 | 63.55 | 11.90 | 121/79/67 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 1 | O+B | 0 | MLLM提示_v2 | **56.31** | **47.78** | **42.42** | 50.21 | 63.83 | 56.21 | 60.54 | 38.89 | 120/119/68 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_1shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 1 | O+B | 0 | MLLM提示_v2 | — | — | — | — | — | — | — | — | —/—/— | 未产出评估指标 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/run.json) |
| qwen3.8-27b | 1 | O+C | 0 | MLLM提示_v2 | 51.67 | 39.13 | 36.80 | 56.28 | 64.36 | 60.05 | 62.56 | 18.25 | 121/94/67 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 1 | O+C | 0 | MLLM提示_v2 | 52.66 | 44.01 | 40.33 | 50.81 | **66.49** | 57.60 | 62.63 | 39.68 | 125/121/63 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_1shot_%E7%AC%AC3%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 1 | O+C+B+J | 0 | MLLM提示_v2 | 53.41 | 35.49 | 35.46 | **63.92** | 65.96 | **64.92** | **65.54** | **7.14** | 124/70/64 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 5 | O+B | 0 | MLLM提示_v1 | **65.46** | **44.97** | **44.22** | **72.96** | **76.06** | **74.48** | **75.42** | **9.52** | 143/53/45 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 5 | O+B | 0 | MLLM提示_v2 | 69.21 | 51.13 | 47.56 | 62.61 | 79.26 | 69.95 | 75.25 | **34.92** | 149/89/39 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 5 | O+C | 0 | MLLM提示_v2 | **73.18** | **55.10** | **50.65** | **64.71** | **81.91** | **72.30** | **77.78** | 35.71 | 154/84/34 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_5shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 10 | O+B | 0 | MLLM提示_v2 | 71.10 | 51.63 | 50.39 | 65.38 | 81.38 | 72.51 | 77.59 | 31.75 | 153/81/35 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 10 | O+C | 0 | MLLM提示_v2 | **77.96** | **57.89** | **54.72** | **69.53** | **86.17** | **76.96** | **82.23** | **30.16** | 162/71/26 | 完成 | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_10shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 10 | O+C+B+J | 0 | MLLM提示_v2 | 74.92 | 52.61 | 50.19 | 79.49 | 82.45 | 80.94 | 81.84 | 5.56 | 155/40/33 | 未完成/失败† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

### 吊物（单类） · 其他条件

| 模型 | shot/类 | 输入 | 负参考 | 版本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| qwen3.8-27b | 1 | O+B | 0 | MLLM提示_v1 | 51.41 | 47.36 | 40.33 | 84.90 | 59.43 | 69.92 | 63.22 | 40.00 | 208/37/142 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 1 | O+B | 0 | MLLM提示_v1 | **57.45** | 51.62 | **45.44** | 83.02 | **62.86** | **71.54** | **66.07** | 45.56 | 220/45/130 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 1 | O+C | 0 | MLLM提示_v1 | 52.34 | 45.94 | 39.83 | **86.01** | 59.71 | 70.49 | 63.60 | **37.78** | 209/34/141 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 1 | O+C | 0 | MLLM提示_v1 | 56.21 | **52.07** | 45.31 | 81.48 | **62.86** | 70.97 | 65.87 | 48.89 | 220/50/130 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_1shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 1 | O+B | 1 | MLLM提示_v1 | **53.40** | **48.15** | **42.02** | **87.39** | **59.43** | **70.75** | **63.49** | **32.22** | 208/30/142 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 1 | O+C | 1 | MLLM提示_v1 | 51.47 | 46.21 | 40.31 | 84.84 | 59.14 | 69.70 | 62.96 | 38.89 | 207/37/143 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E5%90%AB1%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 5 | O+B | 0 | MLLM提示_v1 | **57.28** | **53.31** | **45.71** | 83.27 | **62.57** | 71.45 | **65.84** | 43.33 | 219/44/131 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 5 | O+C | 0 | MLLM提示_v1 | 57.19 | 51.55 | 45.70 | **84.17** | 62.29 | **71.59** | 65.70 | **40.00** | 218/41/132 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_5shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 5 | O+B | 5 | MLLM提示_v1 | 53.35 | 48.29 | 42.49 | 89.78 | 57.71 | 70.26 | 62.15 | 25.56 | 202/23/148 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 5 | O+B+J | 5 | MLLM提示_v1 | **55.13** | 48.99 | **42.68** | 90.00 | **59.14** | **71.38** | **63.50** | 25.56 | 207/23/143 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 5 | O+C | 5 | MLLM提示_v1 | 53.84 | 48.50 | 42.50 | 91.40 | 57.71 | 70.75 | 62.31 | 21.11 | 202/19/148 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 5 | O+C+B | 5 | MLLM提示_v1 | 52.96 | **50.59** | 42.32 | **91.71** | 56.86 | 70.19 | 61.53 | **18.89** | 199/18/151 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 5 | O+C+B+J | 5 | MLLM提示_v1 | 54.00 | 48.51 | 42.30 | 89.91 | 58.57 | 70.93 | 62.96 | 25.56 | 205/23/145 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 5 | O+C+J | 5 | MLLM提示_v1 | 53.83 | 47.84 | 41.04 | 91.36 | 57.43 | 70.53 | 62.04 | 21.11 | 201/19/149 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 5 | O+J | 5 | MLLM提示_v1 | 54.17 | 48.73 | 42.31 | 89.52 | 58.57 | 70.81 | 62.92 | 26.67 | 205/24/145 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E5%90%AB5%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 10 | O+B | 0 | MLLM提示_v1 | **58.06** | **53.01** | **46.05** | **84.77** | 62.00 | 71.62 | 65.52 | **38.89** | 217/39/133 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-flash | 10 | O+C | 0 | MLLM提示_v1 | 57.78 | 50.02 | 44.80 | 84.27 | **64.29** | **72.93** | **67.49** | 41.11 | 225/42/125 | 完成 | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_10shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 10 | O+B | 10 | MLLM提示_v1 | 51.54 | **48.03** | 40.84 | 91.30 | 54.00 | 67.86 | 58.81 | 18.89 | 189/18/161 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E5%90%AB10%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| qwen3.8-27b | 10 | O+C | 10 | MLLM提示_v1 | **52.75** | 47.49 | **41.90** | **92.79** | **55.14** | **69.18** | **60.01** | **16.67** | 193/15/157 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E5%90%AB10%E5%BC%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

</details>

<details>
<summary>附录B：Qwen3.8-Max旧吊物99张结果</summary>

Max有8组旧99张、55个真值框的结果，7组完成、1组未完成。队列不同，仅在此单独展示，不参与当前300张模型比较。目录中的0shot文字条件按实际输入标记。

### 旧吊物99张 · qwen3.8-max

| 输入 | shot/类 | 负参考 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| text_only | 0 | 0 | 38.61 | 38.61 | 33.87 | **100.00** | 38.18 | 55.26 | 43.57 | — | 21/0/34 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/evaluation.txt) |
| O+B | 5 | 0 | 49.50 | 47.27 | 38.46 | **100.00** | 49.09 | 65.85 | 54.66 | — | 27/0/28 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+B+J | 5 | 0 | 47.52 | 43.03 | 36.19 | **100.00** | 47.27 | 64.20 | 52.85 | — | 26/0/29 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C | 5 | 0 | **56.44** | **48.16** | **40.51** | 96.88 | **56.36** | **71.26** | **61.51** | — | 31/1/24 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B | 5 | 0 | 45.98 | 39.64 | 32.82 | 96.30 | 47.27 | 63.41 | 52.63 | — | 26/1/29 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+B+J | 5 | 0 | 38.17 | 33.96 | 28.59 | 91.67 | 40.00 | 55.70 | 45.08 | — | 22/2/33 | 未完成/失败† | [查看](../../../Qwen/experiments/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+C+J | 5 | 0 | 50.15 | 37.30 | 34.82 | 90.32 | 50.91 | 65.12 | 55.78 | — | 28/3/27 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| O+J | 5 | 0 | 41.58 | 37.93 | 31.98 | **100.00** | 41.82 | 58.97 | 47.33 | — | 23/0/32 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_99/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_5shot/%E5%8E%9F%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |

[旧99张全部模型明细](../../../Qwen/comparisons/diaowu_99/results_summary.md)

</details>

[Flash补跑原始记录](../../../Qwen/comparisons/flash_supplement_20261103/README.md) · [完整真实运行CSV（含附录，不含待补跑占位）](01_qwen_inputs.csv)

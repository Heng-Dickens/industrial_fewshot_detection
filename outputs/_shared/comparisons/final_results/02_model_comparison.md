# Qwen与开放词汇模型对比

对比Qwen3.8-27B、GroundingDINO、WeDetect、原始YOLOE和PET-DINO，不含任务微调。按提示shot=0/1/5/10分别成表；0shot包含文字提示，1/5/10shot为视觉参考。本页Qwen仅用27B：0shot固定文字提示，1/5/10shot固定O+C（参考原图+目标裁剪图），均无负例、support seed=0；叉车统一提示v2，吊物统一提示v1。按固定条件引用已有结果，不按最高指标更换输入或提示版本。其余方法保留所有已有设置。

[全局比较](../../../FINAL_COMPARISON.md) · [子任务导航](README.md)

指标均为百分数。AP使用该运行保存的全部候选；P/R/F1/F2使用score≥0.5、IoU=0.5，TP/FP/FN为框数。箭头表示优化方向；每张表内逐列加粗最优值，按显示精度并列最优同时加粗。参数值不作最优标记。未完成、队列不符或类别范围不全的记录保留展示，但不参与最优值计算。†表示本机仅保存指标和元数据，尚无逐图记录可复核；仍可作为有此证据限制的结果对比。P或负图错误率单独最优不等于检测最好，需结合R和TP/FP/FN阅读。

提示类型按实际保存文本标注：简单提示是基础或细分类名；复杂提示是增加属性、结构或外观定义的描述。吊物v2虽列为复杂提示，实际仅为属性限定短语，不等同于Qwen的长篇场景指令。下表的提示版本可点击查看完整文本，同名版本的WeDetect/GroundingDINO使用相同文本。本页所需27B的两个场景×四档shot共8项均已有完成结果，直接补入，不重复请求模型。同shot仅对齐正参考张数，输入类型、模型规模、预训练和提示复杂度仍不同。

### 叉车（五类） · 0shot

| 算法/模型 | 提示类型 | 关键输入/提示文本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| GroundingDINO/grounding-dino-base | 简单提示（基础类别名；例：Chache1=叉车） | [prompt_v1](../../../GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v1/prompt.txt) / text_only / 负参考0 | 10.59 | 9.07 | 8.31 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/188 | 完成 | [查看](../../../GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v1/evaluation.txt) |
| GroundingDINO/grounding-dino-base | 简单提示（细分类名；例：Chache1=平衡重叉车） | [prompt_v2](../../../GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v2/prompt.txt) / text_only / 负参考0 | 10.49 | 9.04 | 8.28 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/188 | 完成 | [查看](../../../GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v2/evaluation.txt) |
| GroundingDINO/grounding-dino-base | 复杂提示（简短结构描述；例：有护顶架和普通货叉的座驾叉车） | [prompt_v3](../../../GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v3/prompt.txt) / text_only / 负参考0 | 9.28 | 7.66 | 7.15 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/188 | 完成 | [查看](../../../GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v3/evaluation.txt) |
| GroundingDINO/grounding-dino-base | 复杂提示（详细外观定义，包含结构、驾驶姿态、部件与颜色） | [prompt_v4](../../../GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v4/prompt.txt) / text_only / 负参考0 | 3.79 | 3.01 | 2.88 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/188 | 完成 | [查看](../../../GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v4/evaluation.txt) |
| Qwen/qwen3.8-27b | 复杂文字提示（场景定义与检测规则） | [MLLM提示_v2](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/prompt.txt) / text_only / 负参考0 | **44.59** | **33.40** | **30.27** | 44.86 | **51.06** | **47.76** | **49.69** | 29.37 | 96/118/92 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/evaluation.txt) |
| WeDetect/wedetect_base | 简单提示（基础类别名；例：Chache1=叉车） | [prompt_v1](../../../WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v1/prompt.txt) / text_only / 负参考0 | 18.03 | 14.50 | 13.17 | 28.00 | 18.62 | 22.36 | 19.95 | 1.59 | 35/90/153 | 完成 | [查看](../../../WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v1/evaluation.txt) |
| WeDetect/wedetect_base | 简单提示（细分类名；例：Chache1=平衡重叉车） | [prompt_v2](../../../WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v2/prompt.txt) / text_only / 负参考0 | 19.06 | 14.45 | 13.27 | 20.00 | 12.77 | 15.58 | 13.76 | 7.14 | 24/96/164 | 完成 | [查看](../../../WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v2/evaluation.txt) |
| WeDetect/wedetect_base | 复杂提示（简短结构描述；例：有护顶架和普通货叉的座驾叉车） | [prompt_v3](../../../WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v3/prompt.txt) / text_only / 负参考0 | 20.70 | 17.40 | 15.65 | 32.79 | 10.64 | 16.06 | 12.30 | 6.35 | 20/41/168 | 完成 | [查看](../../../WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v3/evaluation.txt) |
| WeDetect/wedetect_base | 复杂提示（详细外观定义，包含结构、驾驶姿态、部件与颜色） | [prompt_v4](../../../WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v4/prompt.txt) / text_only / 负参考0 | 35.10 | 29.88 | 27.56 | **56.36** | 16.49 | 25.51 | 19.21 | 0.79 | 31/24/157 | 完成 | [查看](../../../WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v4/evaluation.txt) |

### 叉车（五类） · 1shot

| 算法/模型 | 提示类型 | 关键输入/提示文本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PET-DINO/pet_dino_swin_t | 视觉提示 | [视觉提示](../../../PET-DINO/experiments/chache_val300/pet_dino_swin_t_1shot/prompt.txt) / 视觉参考 / 负参考0 | 19.49 | 13.86 | 14.67 | 43.55 | 14.36 | 21.60 | 16.58 | **8.73** | 27/35/161 | 完成 | [查看](../../../PET-DINO/experiments/chache_val300/pet_dino_swin_t_1shot/evaluation.txt) |
| Qwen/qwen3.8-27b | 视觉提示（O+C） | [MLLM提示_v2](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/prompt.txt) / O+C / 负参考0 | **51.67** | **39.13** | **36.80** | **56.28** | **64.36** | **60.05** | **62.56** | 18.25 | 121/94/67 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| YOLOE/yoloe-v8l-seg | 视觉提示 | [视觉提示](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_1shot/prompt.txt) / 视觉参考 / 负参考0 | 28.14 | 11.33 | 15.45 | 52.87 | 24.47 | 33.45 | 27.41 | 17.46 | 46/41/142 | 完成 | [查看](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_1shot/evaluation.txt) |

### 叉车（五类） · 5shot

| 算法/模型 | 提示类型 | 关键输入/提示文本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PET-DINO/pet_dino_swin_t | 视觉提示 | [视觉提示](../../../PET-DINO/experiments/chache_val300/pet_dino_swin_t_5shot/prompt.txt) / 视觉参考 / 负参考0 | 20.33 | 13.97 | 14.40 | **100.00** | 1.06 | 2.11 | 1.33 | **0.00** | 2/0/186 | 完成 | [查看](../../../PET-DINO/experiments/chache_val300/pet_dino_swin_t_5shot/evaluation.txt) |
| Qwen/qwen3.8-27b | 视觉提示（O+C） | [MLLM提示_v2](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/prompt.txt) / O+C / 负参考0 | **75.00** | **50.62** | **50.17** | 67.38 | **83.51** | **74.58** | **79.70** | 30.16 | 157/76/31 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| YOLOE/yoloe-v8l-seg | 视觉提示 | [视觉提示](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_5shot/prompt.txt) / 视觉参考 / 负参考0 | 27.26 | 15.01 | 16.24 | 25.00 | 35.11 | 29.20 | 32.48 | 22.22 | 66/198/122 | 完成 | [查看](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_5shot/evaluation.txt) |

### 叉车（五类） · 10shot

| 算法/模型 | 提示类型 | 关键输入/提示文本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PET-DINO/pet_dino_swin_t | 视觉提示 | [视觉提示](../../../PET-DINO/experiments/chache_val300/pet_dino_swin_t_10shot/prompt.txt) / 视觉参考 / 负参考0 | 22.54 | 15.07 | 15.97 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/188 | 完成 | [查看](../../../PET-DINO/experiments/chache_val300/pet_dino_swin_t_10shot/evaluation.txt) |
| Qwen/qwen3.8-27b | 视觉提示（O+C） | [MLLM提示_v2](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/prompt.txt) / O+C / 负参考0 | **74.48** | **56.63** | **53.36** | **74.18** | **84.04** | **78.80** | **81.87** | 15.08 | 158/55/30 | 完成（元数据）† | [查看](../../../Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| YOLOE/yoloe-v8l-seg | 视觉提示 | [视觉提示](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_10shot/prompt.txt) / 视觉参考 / 负参考0 | 29.00 | 17.54 | 18.04 | 22.26 | 37.77 | 28.01 | 33.15 | 26.98 | 71/248/117 | 完成 | [查看](../../../YOLOE/experiments/chache_val300/yoloe-v8l-seg_10shot/evaluation.txt) |

### 吊物（单类） · 0shot

| 算法/模型 | 提示类型 | 关键输入/提示文本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| GroundingDINO/grounding-dino-base | 简单提示（类别名：吊物） | [prompt_v1](../../../GroundingDINO/experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v1/prompt.txt) / text_only / 负参考0 | 0.49 | 0.28 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/350 | 完成 | [查看](../../../GroundingDINO/experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v1/evaluation.txt) |
| GroundingDINO/grounding-dino-base | 复杂提示（属性限定短语：被吊起的圆筒吊物；不是长篇指令） | [prompt_v2](../../../GroundingDINO/experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v2/prompt.txt) / text_only / 负参考0 | 1.00 | 0.56 | 0.52 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/350 | 完成 | [查看](../../../GroundingDINO/experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v2/evaluation.txt) |
| Qwen/qwen3.8-27b | 复杂文字提示（场景定义与检测规则） | [MLLM提示_v1](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/prompt.txt) / text_only / 负参考0 | **52.20** | **46.28** | **40.96** | **87.39** | **57.43** | **69.31** | **61.66** | 28.89 | 201/29/149 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E4%BB%85%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA/evaluation.txt) |
| WeDetect/wedetect_base | 简单提示（类别名：吊物） | [文字提示_v1](../../../WeDetect/experiments/diaowu_val300/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/wedetect_base_0shot/prompt.txt) / text_only / 负参考0 | 1.17 | 1.00 | 0.86 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/350 | 完成 | [查看](../../../WeDetect/experiments/diaowu_val300/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/wedetect_base_0shot/evaluation.txt) |
| WeDetect/wedetect_base | 复杂提示（属性限定短语：被吊起的圆筒吊物；不是长篇指令） | [文字提示_v2](../../../WeDetect/experiments/diaowu_val300/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v2/wedetect_base_0shot/prompt.txt) / text_only / 负参考0 | 0.07 | 0.07 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/350 | 完成 | [查看](../../../WeDetect/experiments/diaowu_val300/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v2/wedetect_base_0shot/evaluation.txt) |

### 吊物（单类） · 1shot

| 算法/模型 | 提示类型 | 关键输入/提示文本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PET-DINO/pet_dino_swin_t | 视觉提示 | [视觉提示](../../../PET-DINO/experiments/diaowu_val300/pet_dino_swin_t_1shot/prompt.txt) / 视觉参考 / 负参考0 | 19.00 | 14.85 | 13.22 | **100.00** | 2.29 | 4.47 | 2.84 | **0.00** | 8/0/342 | 完成 | [查看](../../../PET-DINO/experiments/diaowu_val300/pet_dino_swin_t_1shot/evaluation.txt) |
| Qwen/qwen3.8-27b | 视觉提示（O+C） | [MLLM提示_v1](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/prompt.txt) / O+C / 负参考0 | 52.34 | **45.94** | **39.83** | 86.01 | **59.71** | **70.49** | **63.60** | 37.78 | 209/34/141 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_1shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| YOLOE/yoloe-v8l-seg | 视觉提示 | [视觉提示](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_1shot/prompt.txt) / 视觉参考 / 负参考0 | **53.03** | 37.01 | 32.56 | 84.36 | 43.14 | 57.09 | 47.82 | 14.44 | 151/28/199 | 完成 | [查看](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_1shot/evaluation.txt) |

### 吊物（单类） · 5shot

| 算法/模型 | 提示类型 | 关键输入/提示文本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PET-DINO/pet_dino_swin_t | 视觉提示 | [视觉提示](../../../PET-DINO/experiments/diaowu_val300/pet_dino_swin_t_5shot/prompt.txt) / 视觉参考 / 负参考0 | 10.01 | 4.96 | 5.57 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/350 | 完成 | [查看](../../../PET-DINO/experiments/diaowu_val300/pet_dino_swin_t_5shot/evaluation.txt) |
| Qwen/qwen3.8-27b | 视觉提示（O+C） | [MLLM提示_v1](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/prompt.txt) / O+C / 负参考0 | 52.96 | **46.01** | **40.19** | **87.71** | **59.14** | **70.65** | **63.26** | 31.11 | 207/29/143 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_5shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| YOLOE/yoloe-v8l-seg | 视觉提示 | [视觉提示](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_5shot/prompt.txt) / 视觉参考 / 负参考0 | **54.81** | 37.52 | 33.17 | 77.73 | 50.86 | 61.49 | 54.63 | 22.22 | 178/51/172 | 完成 | [查看](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_5shot/evaluation.txt) |

### 吊物（单类） · 10shot

| 算法/模型 | 提示类型 | 关键输入/提示文本 | AP50↑ | AP75↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | TP/FP/FN | 状态/证据 | 原始结果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PET-DINO/pet_dino_swin_t | 视觉提示 | [视觉提示](../../../PET-DINO/experiments/diaowu_val300/pet_dino_swin_t_10shot/prompt.txt) / 视觉参考 / 负参考0 | 15.17 | 8.82 | 9.11 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0/0/350 | 完成 | [查看](../../../PET-DINO/experiments/diaowu_val300/pet_dino_swin_t_10shot/evaluation.txt) |
| Qwen/qwen3.8-27b | 视觉提示（O+C） | [MLLM提示_v1](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/prompt.txt) / O+C / 负参考0 | 53.81 | **48.98** | **42.65** | **90.09** | **57.14** | **69.93** | **61.65** | 24.44 | 200/22/150 | 完成（元数据）† | [查看](../../../Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| YOLOE/yoloe-v8l-seg | 视觉提示 | [视觉提示](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_10shot/prompt.txt) / 视觉参考 / 负参考0 | **55.96** | 38.28 | 33.76 | 74.21 | 53.43 | 62.13 | 56.60 | 24.44 | 187/65/163 | 完成 | [查看](../../../YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_10shot/evaluation.txt) |

GroundingDINO和WeDetect目前是文字0shot；1/5/10shot表中未填造它们的视觉提示结果。低置信度检测器在score≥0.5时可能没有输出，但AP仍非零，不能把AP和固定阈值下的P/R当作同一个量。

# 实验结论与全局比较

更新：2026-11-04 09:00（北京时间）。先按子任务分析，再在此比较关键代表方案。

- [Qwen输入、shot与模型对比](_shared/comparisons/final_results/01_qwen_inputs.md)
- [Qwen与开放词汇模型对比](_shared/comparisons/final_results/02_model_comparison.md)
- [YOLOE微调与YOLOv8对照](_shared/comparisons/final_results/03_yoloe_finetuning.md) · [Chache4新类扩展](_shared/comparisons/final_results/03_yoloe_finetuning.md#novel-chache4) · [YOLOE优势说明](_shared/comparisons/final_results/03_yoloe_finetuning.md#yoloe-advantages)
- [YOLOE→Qwen分阶段处理](_shared/comparisons/final_results/04_yoloe_qwen.md)
- [定制监督网络参考](_shared/comparisons/final_results/06_supervised_reference.md)

指标均为百分数。AP使用该运行保存的全部候选；P/R/F1/F2使用score≥0.5、IoU=0.5，TP/FP/FN为框数。箭头表示优化方向；每张表内逐列加粗最优值，按显示精度并列最优同时加粗。参数值不作最优标记。未完成、队列不符或类别范围不全的记录保留展示，但不参与最优值计算。†表示本机仅保存指标和元数据，尚无逐图记录可复核；仍可作为有此证据限制的结果对比。P或负图错误率单独最优不等于检测最好，需结合R和TP/FP/FN阅读。

Qwen3.8-Max已找到8组旧吊物99张实验，单列在[Qwen输入子表的Max历史部分](_shared/comparisons/final_results/01_qwen_inputs.md)；与当前300张队列不同，未纳入下面的同队列排名。

原始全类别明细201条保留在[全部运行明细](ALL_RESULTS.md)；本页仅选取每个场景、每种方案/训练预算的AP50或AP50:95实际最高配置的并集。两项最优来自不同实验时列两行，每行全部指标来自同一次运行；不逐列拼接。†沿用子表的元数据证据限制。以下为已见类别的完整300张测试结果，按场景分别计算粗体。训练预算、输入尺寸、文字/视觉提示与搜索次数并未统一，JSTY为监督参考；粗体表示本表数值最高，不等于公平预算下的普遍优胜。

### 已见类全局代表方案

| 场景 | 子方案 | 关键参数 | 代表依据 | AP50↑ | AP50:95↑ | P↑ | R↑ | F1↑ | F2↑ | 负图错误率↓ | 原始结果 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 叉车（五类） | [GroundingDINO / 未微调](_shared/comparisons/final_results/02_model_comparison.md) | grounding-dino-base; 提示0; prompt_v1; 输入未记录 | AP50及AP50:95最高 | 10.59 | 8.31 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | [查看](GroundingDINO/experiments/chache_val300/grounding-dino-base_0shot_prompt_v1/evaluation.txt) |
| 叉车（五类） | [PET-DINO / 未微调](_shared/comparisons/final_results/02_model_comparison.md) | pet_dino_swin_t; 提示10; 视觉提示; 输入未记录 | AP50及AP50:95最高 | 22.54 | 15.97 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | [查看](PET-DINO/experiments/chache_val300/pet_dino_swin_t_10shot/evaluation.txt) |
| 叉车（五类） | [WeDetect / 未微调](_shared/comparisons/final_results/02_model_comparison.md) | wedetect_base; 提示0; prompt_v4; 输入未记录 | AP50及AP50:95最高 | 35.10 | 27.56 | 56.36 | 16.49 | 25.51 | 19.21 | 0.79 | [查看](WeDetect/experiments/chache_val300/wedetect_base_0shot_prompt_v4/evaluation.txt) |
| 叉车（五类） | [YOLOE / 未微调](_shared/comparisons/final_results/02_model_comparison.md) | yoloe-v8l-seg; 提示10; 视觉提示; 输入640 | AP50及AP50:95最高 | 29.00 | 18.04 | 22.26 | 37.77 | 28.01 | 33.15 | 26.98 | [查看](YOLOE/experiments/chache_val300/yoloe-v8l-seg_10shot/evaluation.txt) |
| 叉车（五类） | [YOLOE固定头／移除SAVPE / 训练5shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练5/类; 提示不适用; full/固定头; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 60.30 | 48.16 | 67.81 | 52.66 | 59.28 | 55.12 | 19.05 | [查看](YOLOE/finetune/chache_val300/runs/yolo_5shot_20261102/yoloe-head/test_val300/fixed_head/evaluation.txt) |
| 叉车（五类） | [YOLOE视觉提示／更新SAVPE / 训练5shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练5/类; 提示5; full/same_image; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 48.80 | 39.52 | 62.50 | 31.91 | 42.25 | 35.38 | 0.79 | [查看](YOLOE/finetune/chache_val300/runs/yolo_5shot_20261102/yoloe-vp/test_val300/visual_5shot/evaluation.txt) |
| 叉车（五类） | [YOLOE视觉提示／冻结SAVPE / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示10; neck/cross_image; lr=1e-05; 输入640; seed1 | AP50及AP50:95最高 | 59.02 | 46.11 | 42.24 | 68.09 | 52.14 | 60.66 | 22.22 | [查看](YOLOE/finetune/chache_val300/runs/yoloe_chache_frozen_savpe_20261030/vp_cross_neck_frozen_savpe_seed1/test_val300/visual_10shot/evaluation.txt) |
| 叉车（五类） | [YOLOE固定头／移除SAVPE / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示不适用; full/固定头; lr=0.0001; 输入640; seed0; mosaic=0.2,翻转=0.5 | AP50及AP50:95最高 | 82.22 | 70.96 | 78.65 | 80.32 | 79.47 | 79.98 | 19.84 | [查看](YOLOE/finetune/chache_val300/runs/yoloe_search_20261029_r2/head_full_aug/test_val300/fixed_head/evaluation.txt) |
| 叉车（五类） | [YOLOE视觉提示／更新SAVPE / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示10; neck/cross_image; lr=1e-05; 输入640; seed0 | AP50及AP50:95最高 | 64.13 | 49.86 | 47.46 | 69.68 | 56.47 | 63.72 | 20.63 | [查看](YOLOE/finetune/chache_val300/runs/yoloe_search_20261029_r2/vp_cross_neck/test_val300/visual_10shot/evaluation.txt) |
| 叉车（五类） | [YOLOv8-L微调 / 训练5shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练5/类; 提示不适用; full/固定头; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 54.06 | 39.45 | 75.21 | 46.81 | 57.70 | 50.63 | 3.17 | [查看](YOLOE/finetune/chache_val300/runs/yolo_5shot_20261102/yolov8/test_val300/fixed_head/evaluation.txt) |
| 叉车（五类） | [YOLOv8-L微调 / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示不适用; full/固定头; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 79.26 | 66.96 | **97.54** | 63.30 | 76.77 | 68.08 | **0.00** | [查看](YOLOE/finetune/chache_val300/runs/yolov8_val100_v1/test_val300/fixed_head/evaluation.txt) |
| 叉车（五类） | [Qwen/qwen3.8-27b](_shared/comparisons/final_results/01_qwen_inputs.md) | Qwen-27b; MLLM提示_v2; O+B+J; 正10/负0 | AP50最高 | 77.20 | 50.22 | 76.96 | 83.51 | 80.10 | 82.11 | 11.11 | [查看](Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot_%E7%AC%AC3%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 叉车（五类） | [Qwen/qwen3.8-27b](_shared/comparisons/final_results/01_qwen_inputs.md) | Qwen-27b; MLLM提示_v2; O+C; 正10/负0 | AP50:95最高† | 74.48 | 53.36 | 74.18 | 84.04 | 78.80 | 81.87 | 15.08 | [查看](Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-27b_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 叉车（五类） | [Qwen/qwen3.8-flash](_shared/comparisons/final_results/01_qwen_inputs.md) | Qwen-flash; MLLM提示_v2; O+C+B; 正10/负0 | AP50最高 | 78.24 | 53.53 | 74.44 | **88.30** | 80.78 | **85.13** | 25.40 | [查看](Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_10shot_%E7%AC%AC3%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 叉车（五类） | [Qwen/qwen3.8-flash](_shared/comparisons/final_results/01_qwen_inputs.md) | Qwen-flash; MLLM提示_v2; O+C; 正10/负0 | AP50:95最高 | 77.96 | 54.72 | 69.53 | 86.17 | 76.96 | 82.23 | 30.16 | [查看](Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-flash_10shot_%E7%AC%AC2%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 叉车（五类） | [Qwen/qwen3.8-max](_shared/comparisons/final_results/01_qwen_inputs.md) | Qwen-max; MLLM提示_v2; O+C+B; 正10/负0 | AP50及AP50:95最高 | 76.42 | 43.64 | 76.33 | 84.04 | 80.00 | 82.38 | 10.32 | [查看](Qwen/experiments/chache_val300/MLLM%E6%8F%90%E7%A4%BA_v2/qwen3.8-max_10shot/%E5%8E%9F%E5%9B%BE%2B%E8%A3%81%E5%89%AA%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 叉车（五类） | [YOLOE→Qwen / 未微调](_shared/comparisons/final_results/04_yoloe_qwen.md) | 未微调; 训练0/类; B2框修正; YOLOE 10 / Qwen 1; 候选≥0.01 | AP50最高 | 52.20 | 27.86 | 80.00 | 31.91 | 45.63 | 36.28 | **0.00** | [查看](OVD-MLLM/experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt) |
| 叉车（五类） | [YOLOE→Qwen / 未微调](_shared/comparisons/final_results/04_yoloe_qwen.md) | 未微调; 训练0/类; B1D去重; YOLOE 10 / Qwen 1; 候选≥0.01 | AP50:95最高 | 49.38 | 27.93 | 80.00 | 31.91 | 45.63 | 36.28 | **0.00** | [查看](OVD-MLLM/experiments/chache_val300/yoloe_v8l_10shot_qwen38_v1_20261030/deduplicated/evaluation.txt) |
| 叉车（五类） | [YOLOE→Qwen / 冻结SAVPE](_shared/comparisons/final_results/04_yoloe_qwen.md) | 冻结SAVPE; 训练10/类; B2框修正; YOLOE 10 / Qwen 1; 候选≥0.01 | AP50及AP50:95最高 | 49.97 | 35.18 | 84.40 | 48.94 | 61.95 | 53.43 | **0.00** | [查看](OVD-MLLM/experiments/chache_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 叉车（五类） | [YOLOE→Qwen / 固定头](_shared/comparisons/final_results/04_yoloe_qwen.md) | 固定头; 训练10/类; B1D去重; YOLOE 固定头 / Qwen 1; 候选≥0.01 | AP50及AP50:95最高 | 63.98 | 56.47 | 95.93 | 62.77 | 75.88 | 67.43 | **0.00** | [查看](OVD-MLLM/experiments/chache_val300/finetuned_head_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 叉车（五类） | [YOLOE→Qwen / 更新SAVPE](_shared/comparisons/final_results/04_yoloe_qwen.md) | 更新SAVPE; 训练10/类; B2框修正; YOLOE 10 / Qwen 1; 候选≥0.01 | AP50及AP50:95最高 | 50.72 | 35.77 | 83.78 | 49.47 | 62.21 | 53.88 | **0.00** | [查看](OVD-MLLM/experiments/chache_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/refined/evaluation.txt) |
| 叉车（五类） | [JSTY定制监督参考](_shared/comparisons/final_results/06_supervised_reference.md) | FP32 ONNX; 960×1280; 监督预算未绑定 | AP50及AP50:95最高 | **93.68** | **71.53** | 90.64 | 82.45 | **86.35** | 83.97 | 1.59 | [查看](JSTY-YOLO/experiments/chache_val300/jsty_chache_det5_960_1280_0_0_0_2_fp32_val300_20261102/evaluation.txt) |
| 吊物（单类） | [GroundingDINO / 未微调](_shared/comparisons/final_results/02_model_comparison.md) | grounding-dino-base; 提示0; prompt_v2; 输入未记录 | AP50及AP50:95最高 | 1.00 | 0.52 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | [查看](GroundingDINO/experiments/diaowu_val300/grounding-dino-base_0shot_prompt_v2/evaluation.txt) |
| 吊物（单类） | [PET-DINO / 未微调](_shared/comparisons/final_results/02_model_comparison.md) | pet_dino_swin_t; 提示1; 视觉提示; 输入未记录 | AP50及AP50:95最高 | 19.00 | 13.22 | **100.00** | 2.29 | 4.47 | 2.84 | **0.00** | [查看](PET-DINO/experiments/diaowu_val300/pet_dino_swin_t_1shot/evaluation.txt) |
| 吊物（单类） | [WeDetect / 未微调](_shared/comparisons/final_results/02_model_comparison.md) | wedetect_base; 提示0; 文字提示_v1; 输入未记录 | AP50及AP50:95最高 | 1.17 | 0.86 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | [查看](WeDetect/experiments/diaowu_val300/%E6%96%87%E5%AD%97%E6%8F%90%E7%A4%BA_v1/wedetect_base_0shot/evaluation.txt) |
| 吊物（单类） | [YOLOE / 未微调](_shared/comparisons/final_results/02_model_comparison.md) | yoloe-v8l-seg; 提示10; 视觉提示; 输入640 | AP50及AP50:95最高 | 55.96 | 33.76 | 74.21 | 53.43 | 62.13 | 56.60 | 24.44 | [查看](YOLOE/experiments/diaowu_val300/yoloe-v8l-seg_10shot/evaluation.txt) |
| 吊物（单类） | [YOLOE固定头／移除SAVPE / 训练5shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练5/类; 提示不适用; full/固定头; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 60.87 | 44.93 | 88.60 | 57.71 | 69.90 | 62.04 | 17.78 | [查看](YOLOE/finetune/diaowu_val300/runs/yolo_5shot_20261102/yoloe-head/test_val300/fixed_head/evaluation.txt) |
| 吊物（单类） | [YOLOE视觉提示／更新SAVPE / 训练5shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练5/类; 提示5; full/same_image; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 59.71 | 43.27 | 94.24 | 37.43 | 53.58 | 42.56 | 8.89 | [查看](YOLOE/finetune/diaowu_val300/runs/yolo_5shot_20261102/yoloe-vp/test_val300/visual_5shot/evaluation.txt) |
| 吊物（单类） | [YOLOE视觉提示／冻结SAVPE / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示10; full/same_image; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 83.33 | 60.17 | 92.12 | 43.43 | 59.03 | 48.56 | 14.44 | [查看](YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_frozen_savpe_20261030/vp_full_lr0.0001_frozen_savpe/test_val300/visual_10shot/evaluation.txt) |
| 吊物（单类） | [YOLOE固定头／移除SAVPE / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示不适用; full/固定头; lr=0.0001; 输入960; seed2 | AP50最高 | 92.85 | 72.32 | 93.36 | 80.29 | 86.33 | 82.60 | 21.11 | [查看](YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/head_full_960_seed2/test_val300/fixed_head/evaluation.txt) |
| 吊物（单类） | [YOLOE固定头／移除SAVPE / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示不适用; full/固定头; lr=0.0001; 输入960; seed1 | AP50:95最高 | 92.50 | 73.37 | 91.96 | 75.14 | 82.70 | 78.00 | 23.33 | [查看](YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/head_full_960_seed1/test_val300/fixed_head/evaluation.txt) |
| 吊物（单类） | [YOLOE视觉提示／更新SAVPE / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示10; full/same_image; lr=0.0001; 输入640; seed0 | AP50最高 | 84.98 | 60.56 | 93.12 | 42.57 | 58.43 | 47.76 | 12.22 | [查看](YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/vp_full_lr0.0001/test_val300/visual_10shot/evaluation.txt) |
| 吊物（单类） | [YOLOE视觉提示／更新SAVPE / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示10; full/same_image; lr=0.0001; 输入640; seed1 | AP50:95最高 | 82.68 | 62.00 | 92.99 | 41.71 | 57.59 | 46.89 | 12.22 | [查看](YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/vp_full_lr0.0001_seed1/test_val300/visual_10shot/evaluation.txt) |
| 吊物（单类） | [YOLOv8-L微调 / 训练5shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练5/类; 提示不适用; full/固定头; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 39.65 | 26.71 | 78.79 | 37.14 | 50.49 | 41.53 | 30.00 | [查看](YOLOE/finetune/diaowu_val300/runs/yolo_5shot_20261102/yolov8/test_val300/fixed_head/evaluation.txt) |
| 吊物（单类） | [YOLOv8-L微调 / 训练10shot](_shared/comparisons/final_results/03_yoloe_finetuning.md) | 训练10/类; 提示不适用; full/固定头; lr=0.0001; 输入640; seed0 | AP50及AP50:95最高 | 92.19 | 69.72 | 36.52 | **94.86** | 52.74 | 71.89 | 77.78 | [查看](YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/baseline_yolov8/test_val300/fixed_head/evaluation.txt) |
| 吊物（单类） | [Qwen/qwen3.8-27b](_shared/comparisons/final_results/01_qwen_inputs.md) | Qwen-27b; MLLM提示_v1; O+B+J; 正10/负0 | AP50及AP50:95最高 | 56.75 | 43.71 | 89.54 | 61.14 | 72.67 | 65.28 | 27.78 | [查看](Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-27b_10shot_%E7%AC%AC4%E6%AC%A1/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 吊物（单类） | [Qwen/qwen3.8-flash](_shared/comparisons/final_results/01_qwen_inputs.md) | Qwen-flash; MLLM提示_v1; O+B; 正10/负0 | AP50及AP50:95最高 | 58.06 | 46.05 | 84.77 | 62.00 | 71.62 | 65.52 | 38.89 | [查看](Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-flash_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 吊物（单类） | [Qwen/qwen3.8-max](_shared/comparisons/final_results/01_qwen_inputs.md) | Qwen-max; MLLM提示_v1; O+B+J; 正10/负0 | AP50及AP50:95最高 | 56.04 | 44.38 | 97.51 | 56.00 | 71.14 | 61.21 | 2.22 | [查看](Qwen/experiments/diaowu_val300/MLLM%E6%8F%90%E7%A4%BA_v1/qwen3.8-max_10shot/%E5%8E%9F%E5%9B%BE%2B%E5%B8%A6%E6%A1%86%E5%9B%BE%2B%E6%A0%87%E6%B3%A8JSON_%E6%97%A0%E8%B4%9F%E4%BE%8B/evaluation.txt) |
| 吊物（单类） | [YOLOE→Qwen / 未微调](_shared/comparisons/final_results/04_yoloe_qwen.md) | 未微调; 训练0/类; B2框修正; YOLOE 10 / Qwen 1; 候选≥0.01 | AP50及AP50:95最高 | 58.36 | 36.44 | 84.19 | 51.71 | 64.07 | 56.04 | 22.22 | [查看](OVD-MLLM/experiments/diaowu_val300/yoloe_v8l_10shot_qwen38_v1_20261030/refined/evaluation.txt) |
| 吊物（单类） | [YOLOE→Qwen / 冻结SAVPE](_shared/comparisons/final_results/04_yoloe_qwen.md) | 冻结SAVPE; 训练10/类; B1D去重; YOLOE 10 / Qwen 1; 候选≥0.01 | AP50及AP50:95最高 | 55.62 | 42.65 | 91.61 | 40.57 | 56.24 | 45.66 | 14.44 | [查看](OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_frozen_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 吊物（单类） | [YOLOE→Qwen / 固定头](_shared/comparisons/final_results/04_yoloe_qwen.md) | 固定头; 训练10/类; B1D去重; YOLOE 固定头 / Qwen 1; 候选≥0.01 | AP50及AP50:95最高 | 57.95 | 47.59 | 95.12 | 55.71 | 70.27 | 60.75 | 10.00 | [查看](OVD-MLLM/experiments/diaowu_val300/finetuned_head_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 吊物（单类） | [YOLOE→Qwen / 更新SAVPE](_shared/comparisons/final_results/04_yoloe_qwen.md) | 更新SAVPE; 训练10/类; B1D去重; YOLOE 10 / Qwen 1; 候选≥0.01 | AP50及AP50:95最高 | 55.09 | 42.11 | 94.59 | 40.00 | 56.22 | 45.22 | 8.89 | [查看](OVD-MLLM/experiments/diaowu_val300/finetuned_savpe_train_10shot_qwen27b_1shot_20261103/deduplicated/evaluation.txt) |
| 吊物（单类） | [JSTY定制监督参考](_shared/comparisons/final_results/06_supervised_reference.md) | FP32 ONNX; 960×1280; 监督预算未绑定 | AP50及AP50:95最高 | **98.69** | **87.37** | 99.04 | 88.86 | **93.67** | **90.72** | 3.33 | [查看](JSTY-YOLO/experiments/diaowu_val300/jsty_hanging_object_det1_0_0_0_1_fp32_val300_20261102/evaluation.txt) |

[全局CSV](FINAL_COMPARISON.csv) · [来源与比较规则JSON](FINAL_COMPARISON.json)

## Qwen分析顺序：输入 → shot → 模型

[按三步阅读Qwen实验表](_shared/comparisons/final_results/01_qwen_inputs.md)：先用27B的5shot/10shot选择固定输入，再比较shot（叉车0/1/5/10，吊物0/1/5/10/20），最后固定10shot比较27B、Flash和Max（叉车O+C+B、吊物O+B+J）。

- 叉车（五类）：当前固定O+C+B，在已完成5/10shot配对中平均AP50最高，为75.35%。
- 吊物（单类）：当前固定O+B+J，在已完成5/10shot配对中平均AP50最高，为55.58%。

后续阶段尚缺的shot与模型条件在子表逐项留位；第一阶段补齐后，固定输入按相同选择规则自动更新。全局代表方案仍按各算法已完成配置的最高值选择，可能使用其他输入；固定条件模型比较以Qwen子表第三步为准。

[其他已完成条件的27B/Flash配对CSV](_shared/comparisons/final_results/qwen_matched_conditions.csv)（保留原12组配对供查阅）。

## 根据现有结果可得的结论

- 叉车（五类）未微调方法：AP50最高为Qwen/qwen3.8-flash，78.24%；AP50:95最高为Qwen/qwen3.8-flash，54.72%。两项最优可以来自不同配置，详见子表，不将它们拼成一次实验。
- 叉车（五类）训练5shot/类：YOLOv8、固定头YOLOE、训练SAVPE的视觉提示YOLOE，已测试最高AP50分别为54.06/60.30/48.80%。这是各自实际配置的测试最高值，尺寸与搜索预算仍需查看，不能当作等预算统计显著结论。
- 叉车（五类）训练10shot/类：YOLOv8、固定头YOLOE、训练SAVPE的视觉提示YOLOE，已测试最高AP50分别为79.26/82.22/64.13%。这是各自实际配置的测试最高值，尺寸与搜索预算仍需查看，不能当作等预算统计显著结论。
- 叉车（五类）未微调YOLOE→Qwen：B2相对B0的AP50从28.98%到52.20%，变化+23.22个百分点；R从37.77%到31.91%。因此同时报告筛选收益与召回代价。
- 叉车（五类）冻结SAVPEYOLOE→Qwen：B2相对B0的AP50从58.99%到49.97%，变化-9.03个百分点；R从68.09%到48.94%。因此同时报告筛选收益与召回代价。
- 叉车（五类）固定头YOLOE→Qwen：B2相对B0的AP50从80.94%到62.06%，变化-18.89个百分点；R从80.32%到61.17%。因此同时报告筛选收益与召回代价。
- 叉车（五类）更新SAVPEYOLOE→Qwen：B2相对B0的AP50从64.10%到50.72%，变化-13.38个百分点；R从69.68%到49.47%。因此同时报告筛选收益与召回代价。
- 吊物（单类）未微调方法：AP50最高为Qwen/qwen3.8-flash，58.06%；AP50:95最高为Qwen/qwen3.8-flash，46.05%。两项最优可以来自不同配置，详见子表，不将它们拼成一次实验。
- 吊物（单类）训练5shot/类：YOLOv8、固定头YOLOE、训练SAVPE的视觉提示YOLOE，已测试最高AP50分别为39.65/60.87/59.71%。这是各自实际配置的测试最高值，尺寸与搜索预算仍需查看，不能当作等预算统计显著结论。
- 吊物（单类）训练10shot/类：YOLOv8、固定头YOLOE、训练SAVPE的视觉提示YOLOE，已测试最高AP50分别为92.19/92.85/84.98%。这是各自实际配置的测试最高值，尺寸与搜索预算仍需查看，不能当作等预算统计显著结论。
- 吊物（单类）未微调YOLOE→Qwen：B2相对B0的AP50从55.83%到58.36%，变化+2.53个百分点；R从53.43%到51.71%。因此同时报告筛选收益与召回代价。
- 吊物（单类）冻结SAVPEYOLOE→Qwen：B2相对B0的AP50从77.05%到55.51%，变化-21.54个百分点；R从43.43%到40.57%。因此同时报告筛选收益与召回代价。
- 吊物（单类）固定头YOLOE→Qwen：B2相对B0的AP50从91.33%到56.72%，变化-34.61个百分点；R从80.29%到54.86%。因此同时报告筛选收益与召回代价。
- 吊物（单类）更新SAVPEYOLOE→Qwen：B2相对B0的AP50从77.44%到54.85%，变化-22.59个百分点；R从42.57%到39.71%。因此同时报告筛选收益与召回代价。
- Chache4新增类5shot提示：四类微调后不再优化新类，AP50较原始权重33.68%→58.22%，支持本场景的提示迁移价值；score≥0.5召回为60.00%。此单类指标不参与上面的全类别排名。
- Chache4新增类10shot提示：四类微调后不再优化新类，AP50较原始权重41.89%→59.04%，支持本场景的提示迁移价值；score≥0.5召回为33.33%。此单类指标不参与上面的全类别排名。
- 现有证据支持分别讨论“已见类检测精度”“固定阈值漏检/误检”“无需新类梯度更新的提示扩展”。不能用单一最好数字推出保留SAVPE在所有任务上优于YOLOv8。JSTY是训练预算不同的监督参考。
- 未完成结果不用于排名；多数Qwen结果仅有指标元数据。输入调整、搜索最高值与单个新类均是现有测试集上的探索回顾，仍需独立视频/相机与更多seed验证泛化。

更新：`python scripts/build_final_comparison.py`。更新完整明细及全部子表：`python scripts/summarize_all_algorithms.py`。只读取保存指标，不运行模型或API。

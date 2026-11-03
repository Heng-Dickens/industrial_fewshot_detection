# GroundingDINO 对齐实验

Frozen 300-image cohorts. YOLOE uses exactly the PET-DINO 1/5/10-shot supports, boxes and seed=0. GroundingDINO uses verbatim WeDetect Chinese prompts. Every class queries every image; class-labelled predictions are merged without cross-class NMS. Candidate floor=0.001, NMS IoU=0.7, maxDets=100/class, evaluation IoU=0.5 and score=0.5. Model-native preprocessing and feature aggregation are retained. Fresh inference; no training/API calls.

| 实验 | mAP50 | mAP50:95 | P@0.5 | R@0.5 | 对齐基线 AP50 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 简短类别词 | 10.585% | 8.305% | 0.00% | 0.00% | 18.028% | [grounding-dino-base_0shot_prompt_v1](grounding-dino-base_0shot_prompt_v1/evaluation.txt) |
| 具体类别名 | 10.494% | 8.277% | 0.00% | 0.00% | 19.063% | [grounding-dino-base_0shot_prompt_v2](grounding-dino-base_0shot_prompt_v2/evaluation.txt) |
| 短描述 | 9.279% | 7.146% | 0.00% | 0.00% | 20.696% | [grounding-dino-base_0shot_prompt_v3](grounding-dino-base_0shot_prompt_v3/evaluation.txt) |
| 长描述 | 3.789% | 2.876% | 0.00% | 0.00% | 35.101% | [grounding-dino-base_0shot_prompt_v4](grounding-dino-base_0shot_prompt_v4/evaluation.txt) |

YOLOE 对齐 PET-DINO；GroundingDINO 对齐 WeDetect。每次实验使用完整300张图片；chache覆盖五类。
YOLOE 使用640方形输入及 normalized_mean 聚合；GroundingDINO保留本地800/1333预处理和text_threshold=0.25。
提示词逐字对齐，不作翻译。模型架构、预处理、分词器及特征聚合仍各自独立。

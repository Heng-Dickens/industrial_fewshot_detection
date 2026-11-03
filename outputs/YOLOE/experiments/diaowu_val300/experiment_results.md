# YOLOE 对齐实验

Frozen 300-image cohorts. YOLOE uses exactly the PET-DINO 1/5/10-shot supports, boxes and seed=0. GroundingDINO uses verbatim WeDetect Chinese prompts. Every class queries every image; class-labelled predictions are merged without cross-class NMS. Candidate floor=0.001, NMS IoU=0.7, maxDets=100/class, evaluation IoU=0.5 and score=0.5. Model-native preprocessing and feature aggregation are retained. Fresh inference; no training/API calls.

| 实验 | mAP50 | mAP50:95 | P@0.5 | R@0.5 | 对齐基线 AP50 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 1-shot | 53.035% | 32.557% | 84.36% | 43.14% | 18.996% | [yoloe-v8l-seg_1shot](yoloe-v8l-seg_1shot/evaluation.txt) |
| 5-shot | 54.810% | 33.169% | 77.73% | 50.86% | 10.007% | [yoloe-v8l-seg_5shot](yoloe-v8l-seg_5shot/evaluation.txt) |
| 10-shot | 55.956% | 33.762% | 74.21% | 53.43% | 15.169% | [yoloe-v8l-seg_10shot](yoloe-v8l-seg_10shot/evaluation.txt) |

YOLOE 对齐 PET-DINO；GroundingDINO 对齐 WeDetect。每次实验使用完整300张图片；chache覆盖五类。
YOLOE 使用640方形输入及 normalized_mean 聚合；GroundingDINO保留本地800/1333预处理和text_threshold=0.25。
提示词逐字对齐，不作翻译。模型架构、预处理、分词器及特征聚合仍各自独立。

# 两个场景：YOLOv8与YOLOE的5shot训练及推理对照

训练集：叉车每类5张、共25张；吊物5张。两任务分别使用原100张dev选best、300张test评估。
训练seed与参考集seed均为0；640输入、FP32、batch=4、AdamW lr=1e-4、最多100epoch、patience=10。三种微调均更新backbone；SAVPE方案使用same_image全量训练，固定SAVPE的BN运行统计，权重仍参与训练。
未训练YOLOv8使用目标类别数重新构造检测模型，迁移全部形状兼容的预训练张量，仅目标分类输出随机初始化；优化步数为0。它是随机目标分类输出的负对照，不能解释为YOLOv8的语义零样本能力。原始YOLOE一行复用已保存的同一300张、seed=0、5shot视觉提示结果。

| 场景 | 方法 | 训练shot/类 | 推理参考shot/类 | AP50 % | AP50:95 % | P@0.5 % | R@0.5 % | F2@0.5 % | TP/FP/FN | 详情 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| chache | YOLOv8-L改目标分类输出、不训练 | 0 | 不适用 | 0.58 | 0.47 | 0.00 | 0.00 | 0.00 | 0/13/188 | [评估](chache_val300/runs/yolo_5shot_20261102/yolov8-untrained/test_val300/fixed_head/evaluation.txt) · [示例](chache_val300/runs/yolo_5shot_20261102/yolov8-untrained/test_val300/fixed_head/preview.jpg) |
| chache | YOLOv8-L 5shot微调 | 5 | 不适用 | 54.06 | 39.45 | 75.21 | 46.81 | 50.63 | 88/29/100 | [评估](chache_val300/runs/yolo_5shot_20261102/yolov8/test_val300/fixed_head/evaluation.txt) · [示例](chache_val300/runs/yolo_5shot_20261102/yolov8/test_val300/fixed_head/preview.jpg) |
| chache | YOLOE固定头 5shot微调 | 5 | 不适用 | 60.30 | 48.16 | 67.81 | 52.66 | 55.12 | 99/47/89 | [评估](chache_val300/runs/yolo_5shot_20261102/yoloe-head/test_val300/fixed_head/evaluation.txt) · [示例](chache_val300/runs/yolo_5shot_20261102/yoloe-head/test_val300/fixed_head/preview.jpg) |
| chache | YOLOE保留并训练SAVPE 5shot微调 | 5 | 5 | 48.80 | 39.52 | 62.50 | 31.91 | 35.38 | 60/36/128 | [评估](chache_val300/runs/yolo_5shot_20261102/yoloe-vp/test_val300/visual_5shot/evaluation.txt) · [示例](chache_val300/runs/yolo_5shot_20261102/yoloe-vp/test_val300/visual_5shot/preview.jpg) |
| chache | 原始YOLOE视觉提示 | 0 | 5 | 27.26 | 16.24 | 25.00 | 35.11 | 32.48 | 66/198/122 | [评估](../experiments/chache_val300/yoloe-v8l-seg_5shot/evaluation.txt) |
| diaowu | YOLOv8-L改目标分类输出、不训练 | 0 | 不适用 | 0.31 | 0.15 | 0.00 | 0.00 | 0.00 | 0/0/350 | [评估](diaowu_val300/runs/yolo_5shot_20261102/yolov8-untrained/test_val300/fixed_head/evaluation.txt) · [示例](diaowu_val300/runs/yolo_5shot_20261102/yolov8-untrained/test_val300/fixed_head/preview.jpg) |
| diaowu | YOLOv8-L 5shot微调 | 5 | 不适用 | 39.65 | 26.71 | 78.79 | 37.14 | 41.53 | 130/35/220 | [评估](diaowu_val300/runs/yolo_5shot_20261102/yolov8/test_val300/fixed_head/evaluation.txt) · [示例](diaowu_val300/runs/yolo_5shot_20261102/yolov8/test_val300/fixed_head/preview.jpg) |
| diaowu | YOLOE固定头 5shot微调 | 5 | 不适用 | 60.87 | 44.93 | 88.60 | 57.71 | 62.04 | 202/26/148 | [评估](diaowu_val300/runs/yolo_5shot_20261102/yoloe-head/test_val300/fixed_head/evaluation.txt) · [示例](diaowu_val300/runs/yolo_5shot_20261102/yoloe-head/test_val300/fixed_head/preview.jpg) |
| diaowu | YOLOE保留并训练SAVPE 5shot微调 | 5 | 5 | 59.71 | 43.27 | 94.24 | 37.43 | 42.56 | 131/8/219 | [评估](diaowu_val300/runs/yolo_5shot_20261102/yoloe-vp/test_val300/visual_5shot/evaluation.txt) · [示例](diaowu_val300/runs/yolo_5shot_20261102/yoloe-vp/test_val300/visual_5shot/preview.jpg) |
| diaowu | 原始YOLOE视觉提示 | 0 | 5 | 54.81 | 33.17 | 77.73 | 50.86 | 54.63 | 178/51/172 | [评估](../experiments/diaowu_val300/yoloe-v8l-seg_5shot/evaluation.txt) |

所有AP来自保存的完整候选；P/R/F2为score≥0.5、IoU≥0.5。固定头联合多类查询；SAVPE逐类查询后合并，查询与NMS竞争方式不同。每个结果均保留逐图预测、原始指标与权重身份；本轮为单训练seed实验。

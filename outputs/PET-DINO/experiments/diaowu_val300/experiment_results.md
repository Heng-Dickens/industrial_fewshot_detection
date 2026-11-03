# PET-DINO 视觉提示实验

PET-DINO Swin-T Visual-G, seed=0, frozen nested supports per class; average the official per-support visual embeddings. Each class independently queries all 300 images, including negatives. Merge class-labelled predictions without cross-class NMS. No query GT is used for support encoding or class routing. No training.

| shot / 类 | 图片数 | mAP50 | mAP50:95 | P@0.5 | R@0.5 | 失败 | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 300 | 18.996% | 13.224% | 100.00% | 2.29% | 0 | [pet_dino_swin_t_1shot](pet_dino_swin_t_1shot/evaluation.txt) |
| 5 | 300 | 10.007% | 5.571% | 0.00% | 0.00% | 0 | [pet_dino_swin_t_5shot](pet_dino_swin_t_5shot/evaluation.txt) |
| 10 | 300 | 15.169% | 9.108% | 0.00% | 0.00% | 0 | [pet_dino_swin_t_10shot](pet_dino_swin_t_10shot/evaluation.txt) |

AP 使用全部保存候选分数；P/R 使用 0.5 阈值。分数下限 0.001，NMS=0.7，每类最多100框。
1/5/10-shot 为每类参考图数量；固定 seed=0，支持集逐级包含，且与验证图片内容不重合。

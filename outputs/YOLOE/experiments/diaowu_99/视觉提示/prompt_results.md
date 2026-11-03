# 提示词与结果对照

每行对应一个已保存的运行。文字取自该次运行的快照或历史文件，不读取当前 YAML。
AP 使用保存的全部候选分数；Precision / Recall 使用表中评估阈值。不同数据、参考材料和阈值应结合各次记录解读。

| 结果目录 | 提示类型 | 实际文字摘要 | AP50 | AP50:95 | Precision / Recall | 阈值 | 状态 | 提示词 |
|---|---|---|---:|---:|---:|---:|---|---|
| [yoloe-11l-seg_10shot](yoloe-11l-seg_10shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 60.50% | 31.91% | 75.61% / 56.36% | 0.5 | complete | [prompt.txt](yoloe-11l-seg_10shot/prompt.txt) |
| [yoloe-11l-seg_25shot](yoloe-11l-seg_25shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 62.64% | 32.50% | 70.45% / 56.36% | 0.5 | complete | [prompt.txt](yoloe-11l-seg_25shot/prompt.txt) |
| [yoloe-11l-seg_5shot](yoloe-11l-seg_5shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 54.27% | 29.56% | 82.35% / 50.91% | 0.5 | complete | [prompt.txt](yoloe-11l-seg_5shot/prompt.txt) |
| [yoloe-v8l-seg_10shot](yoloe-v8l-seg_10shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 69.01% | 42.26% | 72.22% / 70.91% | 0.5 | complete | [prompt.txt](yoloe-v8l-seg_10shot/prompt.txt) |
| [yoloe-v8l-seg_1shot](yoloe-v8l-seg_1shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 68.63% | 43.93% | 76.19% / 58.18% | 0.5 | complete | [prompt.txt](yoloe-v8l-seg_1shot/prompt.txt) |
| [yoloe-v8l-seg_25shot](yoloe-v8l-seg_25shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 69.38% | 41.75% | 68.97% / 72.73% | 0.5 | complete | [prompt.txt](yoloe-v8l-seg_25shot/prompt.txt) |
| [yoloe-v8l-seg_5shot](yoloe-v8l-seg_5shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 69.15% | 43.65% | 75.51% / 67.27% | 0.5 | complete | [prompt.txt](yoloe-v8l-seg_5shot/prompt.txt) |
| [yoloe-v8s-seg_10shot](yoloe-v8s-seg_10shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 56.01% | 37.79% | 60.42% / 52.73% | 0.5 | complete | [prompt.txt](yoloe-v8s-seg_10shot/prompt.txt) |
| [yoloe-v8s-seg_1shot](yoloe-v8s-seg_1shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 52.41% | 36.28% | 67.50% / 49.09% | 0.5 | complete | [prompt.txt](yoloe-v8s-seg_1shot/prompt.txt) |
| [yoloe-v8s-seg_25shot](yoloe-v8s-seg_25shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 56.64% | 38.43% | 62.00% / 56.36% | 0.5 | complete | [prompt.txt](yoloe-v8s-seg_25shot/prompt.txt) |
| [yoloe-v8s-seg_5shot](yoloe-v8s-seg_5shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 56.65% | 37.69% | 72.97% / 49.09% | 0.5 | complete | [prompt.txt](yoloe-v8s-seg_5shot/prompt.txt) |

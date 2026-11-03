# 提示词与结果对照

每行对应一个已保存的运行。文字取自该次运行的快照或历史文件，不读取当前 YAML。
AP 使用保存的全部候选分数；Precision / Recall 使用表中评估阈值。不同数据、参考材料和阈值应结合各次记录解读。

| 结果目录 | 提示类型 | 实际文字摘要 | AP50 | AP50:95 | Precision / Recall | 阈值 | 状态 | 提示词 |
|---|---|---|---:|---:|---:|---:|---|---|
| [yoloe-v8l-seg_10shot](yoloe-v8l-seg_10shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 55.96% | 33.76% | 74.21% / 53.43% | 0.5 | complete | [prompt.txt](yoloe-v8l-seg_10shot/prompt.txt) |
| [yoloe-v8l-seg_1shot](yoloe-v8l-seg_1shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 53.03% | 32.56% | 84.36% / 43.14% | 0.5 | complete | [prompt.txt](yoloe-v8l-seg_1shot/prompt.txt) |
| [yoloe-v8l-seg_5shot](yoloe-v8l-seg_5shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 54.81% | 33.17% | 77.73% / 50.86% | 0.5 | complete | [prompt.txt](yoloe-v8l-seg_5shot/prompt.txt) |

# 提示词与结果对照

每行对应一个已保存的运行。文字取自该次运行的快照或历史文件，不读取当前 YAML。
AP 使用保存的全部候选分数；Precision / Recall 使用表中评估阈值。不同数据、参考材料和阈值应结合各次记录解读。

| 结果目录 | 提示类型 | 实际文字摘要 | AP50 | AP50:95 | Precision / Recall | 阈值 | 状态 | 提示词 |
|---|---|---|---:|---:|---:|---:|---|---|
| [jsty_hanging_object_det1_0_0_0_1_fp32](jsty_hanging_object_det1_0_0_0_1_fp32/evaluation.txt) | 无文字（监督检测） | 不使用文字提示 | 93.21% | 83.18% | 86.67% / 94.55% | 0.5 | complete | [prompt.txt](jsty_hanging_object_det1_0_0_0_1_fp32/prompt.txt) |

# 提示词与结果对照

每行对应一个已保存的运行。文字取自该次运行的快照或历史文件，不读取当前 YAML。
AP 使用保存的全部候选分数；Precision / Recall 使用表中评估阈值。不同数据、参考材料和阈值应结合各次记录解读。

| 结果目录 | 提示类型 | 实际文字摘要 | AP50 | AP50:95 | Precision / Recall | 阈值 | 状态 | 提示词 |
|---|---|---|---:|---:|---:|---:|---|---|
| [wedetect_base_0shot](wedetect_base_0shot/evaluation.txt) | 文字 | 吊物 | 1.17% | 0.86% | 0.00% / 0.00% | 0.5 | complete | [prompt.txt](wedetect_base_0shot/prompt.txt) |

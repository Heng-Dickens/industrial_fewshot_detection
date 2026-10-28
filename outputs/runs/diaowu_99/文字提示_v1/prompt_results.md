# 提示词与结果对照

每行对应一个已保存的运行。文字取自该次运行的快照或历史文件，不读取当前 YAML。
AP 使用保存的全部候选分数；Precision / Recall 使用表中评估阈值。不同数据、参考材料和阈值应结合各次记录解读。

| 结果目录 | 提示类型 | 实际文字摘要 | AP50 | AP50:95 | Precision / Recall | 阈值 | 状态 | 提示词 |
|---|---|---|---:|---:|---:|---:|---|---|
| [grounding-dino-base_0shot](grounding-dino-base_0shot/evaluation.txt) | 文字 | a large hollow cylindrical load suspended from a yellow lifting hook by two taut slings spread to opposite sides, including the hook and slings. | 2.01% | 1.75% | 0.00% / 0.00% | 0.5 | complete | [prompt.txt](grounding-dino-base_0shot/prompt.txt) |
| [grounding-dino-tiny_0shot](grounding-dino-tiny_0shot/evaluation.txt) | 文字 | a large hollow cylindrical load suspended from a yellow lifting hook by two taut slings spread to opposite sides, including the hook and slings. | 5.33% | 3.03% | 0.00% / 0.00% | 0.5 | complete | [prompt.txt](grounding-dino-tiny_0shot/prompt.txt) |
| [yoloe-11l-seg_0shot](yoloe-11l-seg_0shot/evaluation.txt) | 文字 | a large hollow cylindrical load suspended from a yellow lifting hook by two taut slings spread to opposite sides, including the hook and slings. | 4.60% | 2.43% | 0.00% / 0.00% | 0.5 | complete | [prompt.txt](yoloe-11l-seg_0shot/prompt.txt) |
| [yoloe-11l-seg_5shot_裁剪图0.75+文字0.25](yoloe-11l-seg_5shot_裁剪图0.75+文字0.25/evaluation.txt) | 裁剪视觉+文字 | a large hollow cylindrical load suspended from a yellow lifting hook by two taut slings spread to opposite sides, including the hook and slings. | 24.91% | 14.65% | 0.00% / 0.00% | 0.5 | complete | [prompt.txt](yoloe-11l-seg_5shot_裁剪图0.75+文字0.25/prompt.txt) |

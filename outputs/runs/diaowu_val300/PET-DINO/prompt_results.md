# 提示词与结果对照

每行对应一个已保存的运行。文字取自该次运行的快照或历史文件，不读取当前 YAML。
AP 使用保存的全部候选分数；Precision / Recall 使用表中评估阈值。不同数据、参考材料和阈值应结合各次记录解读。

| 结果目录 | 提示类型 | 实际文字摘要 | AP50 | AP50:95 | Precision / Recall | 阈值 | 状态 | 提示词 |
|---|---|---|---:|---:|---:|---:|---|---|
| [pet_dino_swin_t_10shot](pet_dino_swin_t_10shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 15.17% | 9.11% | 0.00% / 0.00% | 0.5 | complete | [prompt.txt](pet_dino_swin_t_10shot/prompt.txt) |
| [pet_dino_swin_t_1shot](pet_dino_swin_t_1shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 19.00% | 13.22% | 100.00% / 2.29% | 0.5 | complete | [prompt.txt](pet_dino_swin_t_1shot/prompt.txt) |
| [pet_dino_swin_t_5shot](pet_dino_swin_t_5shot/evaluation.txt) | 视觉参考 | 不使用文字提示 | 10.01% | 5.57% | 0.00% / 0.00% | 0.5 | complete | [prompt.txt](pet_dino_swin_t_5shot/prompt.txt) |

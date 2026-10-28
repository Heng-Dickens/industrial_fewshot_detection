# 提示词与结果对照

每行对应一个已保存的运行。文字取自该次运行的快照或历史文件，不读取当前 YAML。
AP 使用保存的全部候选分数；Precision / Recall 使用表中评估阈值。不同数据、参考材料和阈值应结合各次记录解读。

| 结果目录 | 提示类型 | 实际文字摘要 | AP50 | AP50:95 | Precision / Recall | 阈值 | 状态 | 提示词 |
|---|---|---|---:|---:|---:|---:|---|---|
| [wedetect_base_0shot_prompt_v1](wedetect_base_0shot_prompt_v1/evaluation.txt) | 文字 | 叉车 | 18.03% | 13.17% | 28.00% / 18.62% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v1/Chache1/prompt.txt) |
| [wedetect_base_0shot_prompt_v1](wedetect_base_0shot_prompt_v1/evaluation.txt) | 文字 | 搬运车 | 18.03% | 13.17% | 28.00% / 18.62% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v1/Chache2/prompt.txt) |
| [wedetect_base_0shot_prompt_v1](wedetect_base_0shot_prompt_v1/evaluation.txt) | 文字 | 堆高车 | 18.03% | 13.17% | 28.00% / 18.62% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v1/Chache3/prompt.txt) |
| [wedetect_base_0shot_prompt_v1](wedetect_base_0shot_prompt_v1/evaluation.txt) | 文字 | 叉车 | 18.03% | 13.17% | 28.00% / 18.62% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v1/Chache4/prompt.txt) |
| [wedetect_base_0shot_prompt_v1](wedetect_base_0shot_prompt_v1/evaluation.txt) | 文字 | 牵引车 | 18.03% | 13.17% | 28.00% / 18.62% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v1/Qianyinche/prompt.txt) |
| [wedetect_base_0shot_prompt_v2](wedetect_base_0shot_prompt_v2/evaluation.txt) | 文字 | 平衡重叉车 | 19.06% | 13.27% | 20.00% / 12.77% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v2/Chache1/prompt.txt) |
| [wedetect_base_0shot_prompt_v2](wedetect_base_0shot_prompt_v2/evaluation.txt) | 文字 | 电动托盘搬运车 | 19.06% | 13.27% | 20.00% / 12.77% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v2/Chache2/prompt.txt) |
| [wedetect_base_0shot_prompt_v2](wedetect_base_0shot_prompt_v2/evaluation.txt) | 文字 | 站驾式堆高车 | 19.06% | 13.27% | 20.00% / 12.77% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v2/Chache3/prompt.txt) |
| [wedetect_base_0shot_prompt_v2](wedetect_base_0shot_prompt_v2/evaluation.txt) | 文字 | 带属具的叉车 | 19.06% | 13.27% | 20.00% / 12.77% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v2/Chache4/prompt.txt) |
| [wedetect_base_0shot_prompt_v2](wedetect_base_0shot_prompt_v2/evaluation.txt) | 文字 | 电动牵引车 | 19.06% | 13.27% | 20.00% / 12.77% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v2/Qianyinche/prompt.txt) |
| [wedetect_base_0shot_prompt_v3](wedetect_base_0shot_prompt_v3/evaluation.txt) | 文字 | 有护顶架和普通货叉的座驾叉车 | 20.70% | 15.65% | 32.79% / 10.64% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v3/Chache1/prompt.txt) |
| [wedetect_base_0shot_prompt_v3](wedetect_base_0shot_prompt_v3/evaluation.txt) | 文字 | 无门架的低位双叉搬运车 | 20.70% | 15.65% | 32.79% / 10.64% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v3/Chache2/prompt.txt) |
| [wedetect_base_0shot_prompt_v3](wedetect_base_0shot_prompt_v3/evaluation.txt) | 文字 | 有高门架的站驾式堆高车 | 20.70% | 15.65% | 32.79% / 10.64% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v3/Chache3/prompt.txt) |
| [wedetect_base_0shot_prompt_v3](wedetect_base_0shot_prompt_v3/evaluation.txt) | 文字 | 装有黄色大型属具的叉车 | 20.70% | 15.65% | 32.79% / 10.64% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v3/Chache4/prompt.txt) |
| [wedetect_base_0shot_prompt_v3](wedetect_base_0shot_prompt_v3/evaluation.txt) | 文字 | 没有门架和货叉的座驾牵引车 | 20.70% | 15.65% | 32.79% / 10.64% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v3/Qianyinche/prompt.txt) |
| [wedetect_base_0shot_prompt_v4](wedetect_base_0shot_prompt_v4/evaluation.txt) | 文字 | 普通座驾叉车，驾驶位上方有护顶架，驾驶员坐着，前方有高门架和普通成对细长货叉；车身多为绿色或红色。 | 35.10% | 27.56% | 56.36% / 16.49% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v4/Chache1/prompt.txt) |
| [wedetect_base_0shot_prompt_v4](wedetect_base_0shot_prompt_v4/evaluation.txt) | 文字 | 低位搬运车，车体小而低，驾驶员站着，前方伸出贴近地面的长双叉，无高起升门架和护顶架。 | 35.10% | 27.56% | 56.36% / 16.49% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v4/Chache2/prompt.txt) |
| [wedetect_base_0shot_prompt_v4](wedetect_base_0shot_prompt_v4/evaluation.txt) | 文字 | 紧凑起重机车，车体较小，驾驶员站着，前方有明显竖直高门架及下部货叉，无驾驶位护顶架。 | 35.10% | 27.56% | 56.36% / 16.49% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v4/Chache3/prompt.txt) |
| [wedetect_base_0shot_prompt_v4](wedetect_base_0shot_prompt_v4/evaluation.txt) | 文字 | 带大型专用属具的座驾叉车，驾驶员坐着，带护顶架和粗壮门架，前端装有醒目的黄色宽大搬运属具；车身通常为橙色。 | 35.10% | 27.56% | 56.36% / 16.49% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v4/Chache4/prompt.txt) |
| [wedetect_base_0shot_prompt_v4](wedetect_base_0shot_prompt_v4/evaluation.txt) | 文字 | 单座位牵引车，车体低矮紧凑、外形圆润，带座椅和靠背，驾驶员坐着，无起升门架和成对货叉；车身通常为绿色。 | 35.10% | 27.56% | 56.36% / 16.49% | 0.5 | complete | [prompt.txt](wedetect_base_0shot_prompt_v4/Qianyinche/prompt.txt) |

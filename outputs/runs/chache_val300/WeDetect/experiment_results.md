# WeDetect 实验结果

每行是一次完整实验的合并指标；五类提示词、单类指标及预测位于对应实验目录。

| 实验 | 提示词 | 图片数 | mAP50 | mAP50:95 | P@0.5 | R@0.5 |
|---|---|---:|---:|---:|---:|---:|
| [wedetect_base_0shot_prompt_v1](wedetect_base_0shot_prompt_v1/evaluation.txt) | 简短类别词 | 300 | 18.03% | 13.17% | 28.00% | 18.62% |
| [wedetect_base_0shot_prompt_v2](wedetect_base_0shot_prompt_v2/evaluation.txt) | 具体类别名 | 300 | 19.06% | 13.27% | 20.00% | 12.77% |
| [wedetect_base_0shot_prompt_v3](wedetect_base_0shot_prompt_v3/evaluation.txt) | 短描述 | 300 | 20.70% | 15.65% | 32.79% | 10.64% |
| [wedetect_base_0shot_prompt_v4](wedetect_base_0shot_prompt_v4/evaluation.txt) | 长描述 | 300 | 35.10% | 27.56% | 56.36% | 16.49% |

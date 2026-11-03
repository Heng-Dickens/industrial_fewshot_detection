# 微调方法速查

| 方法 / 范围 | Backbone | Neck | 检测头 | SAVPE | 推理参考图 |
|---|---|---|---|---|---|
| 固定头head | 冻结 | 冻结 | 仅分类输出层训练 | 移除 | 无 |
| 固定头neck | 冻结 | 训练 | 训练 | 移除 | 无 |
| 固定头full | 训练 | 训练 | 训练 | 移除 | 无 |
| VP savpe | 冻结 | 冻结 | 其余部分冻结 | 训练 | 5/10shot |
| VP head | 冻结 | 冻结 | 训练 | 训练 | 5/10shot |
| VP neck | 冻结 | 训练 | 训练 | 训练 | 5/10shot |
| VP full | 训练 | 训练 | 训练 | 训练 | 5/10shot |
| VP head + freeze_savpe | 冻结 | 冻结 | 其余部分训练 | 保留并冻结 | 5/10shot |
| VP neck + freeze_savpe | 冻结 | 训练 | 其余部分训练 | 保留并冻结 | 5/10shot |
| VP full + freeze_savpe | 训练 | 训练 | 其余部分训练 | 保留并冻结 | 5/10shot |
| YOLOv8-L全量 | 训练 | 训练 | 训练 | 原架构没有 | 无 |

same_image/cross_image表示训练参考来自当前图/另一训练图，不改变上述范围。仅固定BN统计不等于冻结SAVPE权重；freeze_savpe同时固定全部参数和BN统计。full遵守DFL固定投影等上游例外。

[全部测试推理结果与最高值](ALL_EXPERIMENTS.md)、[逐实验参数和dev记录](all_trials.csv)、[详细机制](../../../../../../docs/guides/YOLOE_FINETUNING_METHODS.md)。

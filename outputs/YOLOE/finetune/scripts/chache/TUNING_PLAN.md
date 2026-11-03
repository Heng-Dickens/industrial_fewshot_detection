# 叉车YOLOE微调调整方案

2026-10-29检查了三组配置、CSV、可训练参数清单、best/last权重及现有测试指标。
本次仅审计、增加配置和回归检查，没有重新训练、重新跑300张测试或修改既有实验结果。
这些调整是待验证的候选，不保证YOLOE超过YOLOv8。

## 当前结果说明了什么

所有微调组均用每类10张，共50张训练图。原始YOLOE视觉基线没有本任务的梯度更新。
以下为已有300张测试的百分数，P/R为micro、IoU=0.5、score≥0.5。

| 方法 | 实际可训练参数 | 推理提示shot/类 | AP50 | AP50:95 | P | R |
|---|---:|---:|---:|---:|---:|---:|
| 原始YOLOE | 0（本任务） | 10 | 29.00 | 18.04 | 22.26 | 37.77 |
| YOLOE固定头 | 3,855 | 不适用 | 12.15 | 8.18 | 26.47 | 4.79 |
| YOLOE视觉提示微调 | 5,562,336 | 5 | 26.50 | 16.12 | 31.63 | 16.49 |
| YOLOE视觉提示微调 | 5,562,336 | 10 | 28.56 | 18.10 | 31.20 | 20.74 |
| YOLOv8-L微调 | 43,633,679 | 不适用 | 79.26 | 66.96 | 97.54 | 63.30 |

数据来源：[固定头测试](../../chache_val300/runs/yoloe_head_val100_v1/test_val300/test_summary.csv)、
[视觉提示测试](../../chache_val300/runs/yoloe_vp_10shot_val100_v1/test_val300/test_summary.csv)、
[YOLOv8测试](../../chache_val300/runs/yolov8_val100_v1/test_val300/test_summary.csv)、
[原始视觉基线](../../../experiments/chache_val300/yoloe-v8l-seg_10shot/metrics.json)。

- 固定头只更新三个输出卷积的weight/bias，定位分支和特征提取都被冻结；YOLOv8则几乎全参数训练。这不是相同训练范围的比较。
- 固定头dev AP50从12.44%到20.26%，最高记录在第100轮；定位损失没有改善。可以提高分类器学习率，也应补一组全参数YOLOE。
- VP的最高**记录**在第1轮：dev AP50=35.29%、AP50:95=20.84%；第21轮降至31.32%/18.73%，第30轮为31.21%/18.49%。训练分类损失仍下降，跨图收益却没有增加。
- VP的10shot测试AP接近原始视觉基线，主要表现为没有获得明显收益，score≥0.5时召回更低。单改推理阈值可能改变P/R，不能修复AP。
- 权重检查确认SAVPE有变化；其他大量被冻结张量保持一致。SAVPE的BN运行统计变化明显，例如一个running_var相对L2变化约46%。这是待排查因素，不能据此断言BN就是唯一原因。

固定头一次多类查询，视觉提示逐类查询；NMS竞争方式和预训练来源也不同，不能把全部差异归因于某一个模块。

## 先修正验证频率

旧args.yaml三组均为val_interval=20，patience=10。
当前上游训练器会在跳过验证时继续使用旧fitness，并在fitness等于best_fitness时保存best；因此未验证的后续轮次可能覆盖原来的best。
早停也会读取重复成绩。训练结束又会重写checkpoint的epoch/history字段，不能从这些字段完整恢复此前被覆盖的权重。
所以“CSV第1轮最好”不等于现存best.pt精确对应第1轮，当前测试指标仍然是现存权重的真实成绩。

训练入口现在在配置未显式给出时默认val_interval=1；以下所有候选也显式设为1。
本次未修改third_party源码、原始config.json或已保存训练记录。
以后若手动设为大于1，需要同时修正上游保存/早停逻辑，否则会重新遇到这个问题。

## 候选配置与运行顺序

所有配置位于本任务presets/并继承本任务config.json，不影响吊物。
训练图片、支持清单、imgsz=640、FP32、AdamW、batch=8、nbs=8、weight_decay=5e-4及原有轻度增强保持一致。
full候选改变了训练范围，head-only候选改变了学习率与最大训练预算，报告时分别命名。

| 顺序/用途 | 配置 | 方法与strategy | lr0 | 最多epoch | patience | SAVPE BN统计 |
|---|---|---|---:|---:|---:|---|
| 固定类别性能优先 | [head_full_lr1e4.json](presets/head_full_lr1e4.json) | yoloe-head / full | 1e-4 | 100 | 20 | 不适用 |
| 保留仅线性输出头的附加对照 | [head_only_lr1e3.json](presets/head_only_lr1e3.json) | yoloe-head / head | 1e-3 | 200 | 30 | 不适用 |
| VP逐轮验证控制组 | [vp_control_lr1e4.json](presets/vp_control_lr1e4.json) | yoloe-vp | 1e-4 | 50 | 10 | 更新 |
| VP低学习率 | [vp_lr1e5.json](presets/vp_lr1e5.json) | yoloe-vp | 1e-5 | 50 | 10 | 更新 |
| VP小批量BN对照 | [vp_lr1e5_bn_frozen.json](presets/vp_lr1e5_bn_frozen.json) | yoloe-vp | 1e-5 | 50 | 10 | 固定运行均值/方差 |

VP低学习率与BN对照只差一个BN选项，可分别判断两者贡献。
BN固定只是module.eval()，不冻结其可训练weight/bias，不冻结其余SAVPE权重。
回调在每个batch前执行，避免被上游每轮model.train()重新开启统计更新；开关记录在finetune_config.json的adaptation字段。
现有默认行为仍为不启用该选项。

为什么head增大学习率、VP反而降低：前者只学3,855个参数且末轮仍改善；后者早期验证记录最好，需要先减少对原始视觉表征的扰动。
官方固定头脚本也使用AdamW、lr0=1e-3，但它是COCO大batch配方，不能把官方VP大规模训练的16e-3直接搬到50张训练图。
参考：[固定头脚本](https://github.com/THU-MIG/yoloe/blob/40cd606cabdbe2b566d6f14a6b162c89206e9a1b/train_pe.py)、
[视觉提示脚本](https://github.com/THU-MIG/yoloe/blob/40cd606cabdbe2b566d6f14a6b162c89206e9a1b/train_vp.py)。

## 怎么执行

在industrial-fewshot环境、项目根目录执行。以下命令**只打印计划**：

```bash
python outputs/YOLOE/finetune/scripts/chache/train.py \
  --config outputs/YOLOE/finetune/scripts/chache/presets/head_full_lr1e4.json \
  --method yoloe-head --strategy full --name yoloe_head_full_lr1e4_v2

python outputs/YOLOE/finetune/scripts/chache/train.py \
  --config outputs/YOLOE/finetune/scripts/chache/presets/head_only_lr1e3.json \
  --method yoloe-head --strategy head --name yoloe_head_only_lr1e3_v2

python outputs/YOLOE/finetune/scripts/chache/train.py \
  --config outputs/YOLOE/finetune/scripts/chache/presets/vp_control_lr1e4.json \
  --method yoloe-vp --name yoloe_vp_control_lr1e4_v2

python outputs/YOLOE/finetune/scripts/chache/train.py \
  --config outputs/YOLOE/finetune/scripts/chache/presets/vp_lr1e5.json \
  --method yoloe-vp --name yoloe_vp_lr1e5_v2

python outputs/YOLOE/finetune/scripts/chache/train.py \
  --config outputs/YOLOE/finetune/scripts/chache/presets/vp_lr1e5_bn_frozen.json \
  --method yoloe-vp --name yoloe_vp_lr1e5_bn_frozen_v2
```

确认计划后，选择运行时空闲GPU，给所需命令追加 `--execute --device GPU编号 --batch 8` 才训练。
所有候选从config中的原始预训练权重新开；不从上一轮best继续，不覆盖v1结果。
可先用新名字追加--epochs 2核对数据、可训练层和正常退出，但两轮不能判断哪个学习率更好。
不必一次跑完所有候选：只关心固定类检测先跑full；关心视觉提示就先跑VP控制组，再低学习率与BN对照。

仅用100张dev的AP50:95选择候选与checkpoint；VP训练器dev使用10shot训练原型、联合多类查询，最终测试则逐类查询。
如果需要专门按5shot效果选参数，应另在同一100张dev上固定5shot和相同逐类查询流程，不能看300张test来选择。
候选确定后，再对VP用evaluate.py --prompt-shots 5 10复核；固定头不接受视觉提示数量。
由于300张成绩已经查看过，下一轮结果应注明为后续探索实验；更强的最终结论可再用未查看的新留出集确认。

## 参数调整仍无收益时

1. 固定头full若明显改善，主要瓶颈是此前可训练范围，而非仅学习率；仍保留head-only结果说明这个差异。
2. VP若低学习率/固定BN仍接近原始基线，下一步需要改变训练方式：在训练池内做不同图的support/query配对，避免只适应“同图框提示”。当前代码尚未实现该方式。
3. 也可以研究保留视觉提示的局部解冻，但必须明确层范围和损失；当前--strategy full仅支持固定头，不能直接用于VP，也不能把SAVPE删除后再称为视觉提示微调。
4. 再考虑扩充训练池中的独立场景与难负例；这会改变fewshot预算，必须单独记录，不把dev/test图加入训练。
5. 若640到960的分辨率对照或增强调整确有必要，作为后续独立变量做，不同时改学习率、分辨率、冻结范围和增强。

当前没有证据保证YOLOE一定能达到YOLOv8的79.26% AP50。固定类别监督检测中YOLOv8更好可以是合理实验结果；视觉提示的价值还需要由更换参考类别、支持数量或跨场景泛化实验来衡量。

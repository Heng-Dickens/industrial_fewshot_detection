# 叉车微调任务

## 新增：冻结整个SAVPE的成对对照

已加入原搜索全部7组可行的head/neck/full视觉提示配置，保持原训练参数，只启用`adaptation.freeze_savpe=true`，SAVPE参数及BN统计同时固定，仍参与提示编码和梯度传递。只训练SAVPE的配置冻结后无可学习参数，因此不重复训练。

[全部微调总表](../../chache_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md)、[新增实验状态与配对结果](../../chache_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md)。

公共入口支持以下配置组合；单独设置`freeze_savpe_bn_stats`不会冻结SAVPE权重：

```json
{
  "adaptation": {
    "scope": "full",
    "loss": "detection",
    "protocol": "same_image",
    "freeze_savpe": true,
    "freeze_savpe_bn_stats": true
  }
}
```

执行入口是`python scripts/run_yoloe_frozen_savpe.py --scene chache --gpus GPU编号列表`，当前实验已完成并精简；该命令或`--summarize-only`只刷新最终摘要，不重建过程目录。新训练请用本任务train.py及新的--name，旧运行不重复执行。原搜索运行ID、候选及成绩保持原记录；新增结果合并进入上面的总表。


固定头yoloe-head移除SAVPE；视觉提示yoloe-vp的savpe/head/neck/full范围均保留并训练SAVPE，推理仍使用参考图。详见[SAVPE方法与逐实验分类](../../../../../docs/guides/YOLOE_FINETUNING_METHODS.md)。

三组v1结果的诊断、下一轮参数与可执行命令见 [TUNING_PLAN.md](TUNING_PLAN.md)。
候选配置独立保存在本任务presets/，未改原config.json。

2026-10-29已执行多GPU系统搜索：48组主配置，随后对固定头/视觉提示各自的dev优胜配置复跑seed=1、2。
进度和最终结果见 [搜索总表](../../chache_val300/comparisons/yoloe_search_20261029/README.md)。
训练每类10张；视觉提示的5/10shot指推理参考数量。候选按100张dev选出，再测试300张。
调度器 [run_yoloe_optimization_sweep.py](../../../../../scripts/run_yoloe_optimization_sweep.py) 为每个进程设置独立GPU可见范围，单卡独立运行，不使用DDP。

48组中按dev选出的两种配置已提供独立预设。需要从原始权重新开复现实验时，替换GPU编号，并使用新名字：

```bash
python outputs/YOLOE/finetune/scripts/chache/train.py \
  --config outputs/YOLOE/finetune/scripts/chache/presets/search_head_full_batch4.json \
  --method yoloe-head --strategy full --batch 4 --device GPU编号 \
  --name search_head_full_batch4_reproduce --execute

python outputs/YOLOE/finetune/scripts/chache/train.py \
  --config outputs/YOLOE/finetune/scripts/chache/presets/search_vp_cross_neck.json \
  --method yoloe-vp --batch 2 --device GPU编号 \
  --name search_vp_cross_neck_reproduce --execute
```

视觉提示预设训练SAVPE及neck/head，骨干前10层冻结；采用box/cls/DFL检测损失，冻结SAVPE BN运行统计。
训练时每类从训练池抽取另一张支持图（排除当前query batch的图片），dev/test推理再用冻结5/10张支持图的归一化平均原型。
配置从本机已准备的同一50张训练图/100张dev读取，不会拿300张test参与训练。

本目录只保存叉车任务的配置、数据准备入口、训练入口和操作说明。
[config.json](config.json) 是独立完整配置，不继承另一个任务的参数。
通用实现位于 [共用训练实现](../common/README.md)。
已完成固定头、视觉提示微调、冻结SAVPE及YOLOv8对照；当前汇总64项训练记录、31项300张测试推理设置，见搜索总表。

## 当前配置

| 项目 | 配置 |
|---|---|
| 类别数 | 5 |
| 支持集 | `data/references/chache`，10shot，support-seed=0 |
| 训练图片 | 50张原图 |
| 验证集 | `data/raw/chache_val100/images`，100张，70正/30负，76框 |
| 最终测试 | `data/raw/chache_val300`，300张 |
| 训练数据输出 | `data/finetune/chache_10shot_val100/dataset.yaml` |
| 训练结果输出 | `outputs/YOLOE/finetune/chache_val300/runs/` |
| 参数 | epochs=100，imgsz=640，FP32，AdamW，lr0=1e-4 |

验证集只用于早停和选best，不更新权重、不提取视觉提示原型；原300张不参与模型选择。
图片内容哈希已检查无交集，不代表完全不同视频来源；仍需留意同源连续帧。
数据输出和结果路径沿用原配置，此次目录重组不会搬动数据或已有结果。

## 第一步：准备数据

在项目根目录和已有环境中执行。先检查计划：

```bash
conda activate industrial-fewshot
python outputs/YOLOE/finetune/scripts/chache/prepare_data.py
```

计划应显示train=50、val=100，自动使用当前目录旁的config.json，不用另外填写dev路径。
检查参考样本、类别和目录后，生成独立数据文件：

```bash
python outputs/YOLOE/finetune/scripts/chache/prepare_data.py --write
```

已有输出目录不会覆盖；换新输出目录后，在本任务配置中同步修改data。
需要5shot时同时调整本任务配置的shot和data，再准备对应数据，不能仅修改实验名称。

## 第二步：查看三组训练计划

```bash
python outputs/YOLOE/finetune/scripts/chache/train.py --method yoloe-head
python outputs/YOLOE/finetune/scripts/chache/train.py --method yoloe-vp
python outputs/YOLOE/finetune/scripts/chache/train.py --method yolov8
```

没有 `--execute` 时只读配置、显示计划，不导入torch/模型、不创建训练目录。
三种方法的区别见 [公共对照方案](../common/COMPARISON_PLAN.md)。

## 第三步：决定开始时先做2轮检查

先自行选择空闲GPU和适合的batch（不得超过50张训练图片），替换下面的占位符。
样例只支持单卡，GPU编号按当前Python环境可见设备解释。

```bash
python outputs/YOLOE/finetune/scripts/chache/train.py --method yoloe-vp \
  --execute --device GPU编号 --batch 批量大小 \
  --epochs 2 --name yoloe_vp_10shot_val100_smoke_v1
```

另外两组分别换成 `--method yoloe-head` 和 `--method yolov8`，并使用不同实验名字。
核对训练/验证数量、实际可训练参数、损失非零、SAVPE梯度及参数更新、最终验证和进程正常退出。
2026-10-29搜索已通过8组真实2轮检查，覆盖固定头neck/full、640/960/1024分辨率、SAVPE/full和跨图视觉提示训练。
更改结构、精度或显存预算后仍需重新检查。

通过后去掉 `--epochs 2` 并换正式名字，从原始预训练权重新开100轮实验，不接着短流程权重训练。
训练样例不支持DDP；不自动决定设备、下载权重或安装依赖。
VP训练意外中断后，可在原命令追加`--resume`，只接受原运行自己的`weights/last.pt`和完全一致的训练配置、数据哈希。
恢复优化器/EMA/epoch；上游不保存RNG及早停器完整历史，因此不承诺与不中断训练逐位一致。

## 权重与结果

- YOLOE两组用 `weights/yoloe/yoloe-v8l-seg.pt` 初始化。
- 固定头方法还需 `weights/yoloe/mobileclip_blt.pt`；SAVPE方法不调用文本编码器。
- YOLOv8方法用 `weights/yolov8/yolov8l.pt` 初始化，不从随机YAML开始。
- 各组从自己的原始预训练checkpoint重新开始，不能顺序接力训练。
- 保存 `finetune_config.json`、`trainable_parameters.json`、`args.yaml`、`results.csv`、`weights/best.pt`。
- 训练器成绩是100张验证集成绩，不能当成300张最终测试成绩。

## 第四步：微调后测试

新增 [evaluate.py](evaluate.py)，读取**实际训练记录和权重结构**，不从文件夹名字猜方法。
`train.py` 现在要求显式填写 `--method`，防止取名为vp却执行默认固定头。

原目录名 `yoloe_vp_10shot_val100_v1` 的训练实际为 `yoloe-head`，已更名为 `yoloe_head_val100_v1`，best权重不含SAVPE且检测头已融合。
其300张测试已完成，见 [可随Git同步的测试报告](../../chache_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md)。
该模型训练每类10张；推理无参考图，5shot/10shot**视觉提示**不适用。
复现时使用新输出目录，避免覆盖已完成测试：

```bash
python outputs/YOLOE/finetune/scripts/chache/evaluate.py \
  --run-dir outputs/YOLOE/finetune/chache_val300/runs/yoloe_head_val100_v1 \
  --output outputs/YOLOE/finetune/chache_val300/runs/yoloe_head_val100_v1/test_val300_repeat \
  --execute --device 5
```

真正用 `--method yoloe-vp` 训练完成后，下面一条命令同时测5shot/10shot参考图：

```bash
python outputs/YOLOE/finetune/scripts/chache/evaluate.py \
  --run-dir outputs/YOLOE/finetune/chache_val300/runs/实际的视觉提示训练目录 \
  --prompt-shots 5 10 --execute --device GPU编号
```

这里5/10只改变推理参考数量，不改变已经完成的训练预算。
支持图取本次训练池中的冻结子集，不能使用dev/test图；每类独立查询后合并，与已有视觉基线流程一致。
10shot训练可测5和10；5shot训练池不能用此协议额外加入未参与训练的5张支持图。
默认读取dev选出的best，不根据测试结果切换best/last。无 `--execute` 时只显示计划。
默认输出至训练目录的 `test_val300/`，保存总表、逐类指标、0.5/0.7阈值、预测和溯源信息。
与未微调视觉基线比较时注意：固定头一次多类查询，视觉提示逐类查询，不能只归因为微调效果。

边界和后续步骤见 [公共方案](../common/COMPARISON_PLAN.md)。
本目录代码和配置可随Git同步；data/和训练权重按现有规则留在本机。

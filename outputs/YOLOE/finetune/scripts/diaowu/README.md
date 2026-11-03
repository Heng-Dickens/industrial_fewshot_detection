# 吊物微调任务

## 新增：冻结整个SAVPE的成对对照

已加入原搜索全部7组可行的head/neck/full视觉提示配置，保持原训练参数，只启用`adaptation.freeze_savpe=true`，SAVPE参数及BN统计同时固定，仍参与提示编码和梯度传递。只训练SAVPE的配置冻结后无可学习参数，因此不重复训练。

[全部微调总表](../../diaowu_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md)、[新增实验状态与配对结果](../../diaowu_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md)。

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

执行入口是`python scripts/run_yoloe_frozen_savpe.py --scene diaowu --gpus GPU编号列表`，当前实验已完成并精简；该命令或`--summarize-only`只刷新最终摘要，不重建过程目录。新训练请用本任务train.py及新的--name，旧运行不重复执行。原搜索运行ID、候选及成绩保持原记录；新增结果合并进入上面的总表。


固定头yoloe-head移除SAVPE；视觉提示yoloe-vp的savpe/head/neck/full范围均保留并训练SAVPE，推理仍使用参考图。详见[SAVPE方法与逐实验分类](../../../../../docs/guides/YOLOE_FINETUNING_METHODS.md)。

本目录只保存吊物任务的配置、数据准备入口、训练入口和操作说明。
[config.json](config.json) 是独立完整配置，不继承另一个任务的参数。
通用实现位于 [../yoloe_finetune_common/](../common/README.md)。
2026-10-29已完成数据准备、9组真实2轮检查和与叉车同范围的多GPU搜索。
独立结果见 [吊物总表](../../diaowu_val300/comparisons/yoloe_search_20261029/README.md)。

本轮包含48组搜索、3组基础对照和4组dev优胜配置的种子复跑。3组基础对照为YOLOE固定头、YOLOE视觉提示和YOLOv8-L预训练权重微调。
原始搜索的55组训练/验证和19项测试已完成；加上冻结SAVPE等对照，当前汇总64项训练记录、31项测试推理设置。三种子平均test AP50/AP50:95：固定头90.46/70.69%，视觉5shot 76.79/56.36%，视觉10shot 79.34/58.21%；YOLOv8单次基线92.19/69.72%。
固定头dev优胜配置使用960输入，视觉提示和YOLOv8为640；完整标准差、参数和阈值工作点见独立总表。
基础对照不参与48组排名；按独立100张dev选每种YOLOE方法前三名，第一名复跑训练seed=1、2，支持集仍固定seed=0。
原始搜索测试13个checkpoint、19项推理设置；后续冻结SAVPE等对照已纳入当前31项完整推理表。视觉提示测5/10shot；训练池始终每类10张。

```bash
# 查看单组训练计划，不加载模型：
python outputs/YOLOE/finetune/scripts/diaowu/train.py --method yoloe-vp

# 中断后先确认原进程已退出，再继续已有实验：
python scripts/run_yoloe_optimization_sweep.py --scene diaowu --resume --execute --gpus 0 1 2 3 4 5 6 7

# 更新独立总表；未完成时可加 --partial 查看阶段性归档。
python scripts/summarize_yoloe_optimization.py --stamp yoloe_diaowu_search_20261029
```

每张GPU独立一项FP32实验，不使用DDP。命令、状态、完整日志在`outputs/_shared/runtime/yoloe_diaowu_search_20261029/`；
权重与逐图预测在`outputs/YOLOE/finetune/diaowu_val300/runs/yoloe_diaowu_search_20261029/`。
后台收尾程序会在全部实验结束后自动归档、统计和绘图；控制台日志为本次runtime目录下`finalizer.log`。

按dev选出的配置已保存为本目录的`presets/`，从原始预训练权重新开复现实验可执行：

```bash
python outputs/YOLOE/finetune/scripts/diaowu/train.py \
  --config outputs/YOLOE/finetune/scripts/diaowu/presets/search_head_full_960.json \
  --method yoloe-head --strategy full --batch 4 --device GPU编号 \
  --name head_full_960_reproduce --execute

python outputs/YOLOE/finetune/scripts/diaowu/train.py \
  --config outputs/YOLOE/finetune/scripts/diaowu/presets/search_vp_full_lr1e4.json \
  --method yoloe-vp --batch 4 --device GPU编号 \
  --name vp_full_lr1e4_reproduce --execute
```

视觉提示优胜配置保留SAVPE，全量训练、检测box/cls/DFL损失、冻结SAVPE BN运行统计，训练采用同图bbox提示；推理时使用冻结5/10shot参考图。
它与叉车的跨图neck优胜配置不同。不要使用叉车的权重或配置进行吊物结果复现。

## 当前配置

| 项目 | 配置 |
|---|---|
| 类别数 | 1 |
| 支持集 | `data/references/diaowu`，10shot，support-seed=0 |
| 训练图片 | 10张原图 |
| 验证集 | `data/raw/diaowu_val100/images`，100张，70正/30负，75框 |
| 最终测试 | `data/raw/diaowu_val300`，300张 |
| 训练数据输出 | `data/finetune/diaowu_10shot_val100/dataset.yaml` |
| 训练结果输出 | `outputs/YOLOE/finetune/diaowu_val300/runs/` |
| 参数 | epochs=100，imgsz=640，FP32，AdamW，lr0=1e-4 |

验证集只用于早停和选best，不更新权重、不提取视觉提示原型；原300张不参与模型选择。
图片内容哈希已检查无交集，不代表完全不同视频来源；仍需留意同源连续帧。
数据输出和结果路径沿用原配置，此次目录重组不会搬动数据或已有结果。

## 第一步：准备数据

在项目根目录和已有环境中执行。先检查计划：

```bash
conda activate industrial-fewshot
python outputs/YOLOE/finetune/scripts/diaowu/prepare_data.py
```

计划应显示train=10、val=100，自动使用当前目录旁的config.json，不用另外填写dev路径。
检查参考样本、类别和目录后，生成独立数据文件：

```bash
python outputs/YOLOE/finetune/scripts/diaowu/prepare_data.py --write
```

已有输出目录不会覆盖。本轮已生成上述数据目录；仅需查看时使用无`--write`的计划命令。
需要5shot时同时调整本任务配置的shot和data，再准备对应数据，不能仅修改实验名称。

## 第二步：查看三组训练计划

```bash
python outputs/YOLOE/finetune/scripts/diaowu/train.py --method yoloe-head
python outputs/YOLOE/finetune/scripts/diaowu/train.py --method yoloe-vp
python outputs/YOLOE/finetune/scripts/diaowu/train.py --method yolov8
```

没有 `--execute` 时只读配置、显示计划，不导入torch/模型、不创建训练目录。
三种方法的区别见 [公共对照方案](../common/COMPARISON_PLAN.md)。

## 第三步：决定开始时先做2轮检查

先自行选择空闲GPU和适合的batch（不得超过10张训练图片），替换下面的占位符。
样例只支持单卡，GPU编号按当前Python环境可见设备解释。

```bash
python outputs/YOLOE/finetune/scripts/diaowu/train.py --method yoloe-vp \
  --execute --device GPU编号 --batch 批量大小 \
  --epochs 2 --name yoloe_vp_10shot_val100_smoke_v1
```

另外两组分别换成 `--method yoloe-head` 和 `--method yolov8`，并使用不同实验名字。
核对训练/验证数量、实际可训练参数、损失非零、SAVPE梯度及参数更新、最终验证和进程正常退出。
本轮9组检查已通过，覆盖三种方法、固定头neck/full、640/960/1024尺寸、SAVPE与跨图训练，并完成最终验证与正常退出。

通过后去掉 `--epochs 2` 并换正式名字，从原始预训练权重新开100轮实验，不接着短流程权重训练。
训练样例不支持DDP；VP原运行可追加`--resume`从自身last.pt恢复优化器/EMA/epoch，要求配置和数据指纹一致。
上游不恢复完整RNG与早停历史，因此不承诺逐位连续。入口不自动下载权重或安装依赖。

## 权重与结果

- YOLOE两组用 `weights/yoloe/yoloe-v8l-seg.pt` 初始化。
- 固定头方法还需 `weights/yoloe/mobileclip_blt.pt`；SAVPE方法不调用文本编码器。
- YOLOv8方法用 `weights/yolov8/yolov8l.pt` 初始化，不从随机YAML开始。
- 各组从自己的原始预训练checkpoint重新开始，不能顺序接力训练。
- 保存 `finetune_config.json`、`trainable_parameters.json`、`args.yaml`、`results.csv`、`weights/best.pt`。
- 训练器成绩是100张验证集成绩，不能当成300张最终测试成绩。

## 第四步：微调后测试

完成真正的 `--method yoloe-vp` 训练后，使用本任务入口评估同一权重的两组参考数量：

```bash
python outputs/YOLOE/finetune/scripts/diaowu/evaluate.py \
  --run-dir outputs/YOLOE/finetune/diaowu_val300/runs/实际的视觉提示训练目录 \
  --prompt-shots 5 10 --execute --device GPU编号
```

5shot/10shot在这里表示**推理参考图数量**，训练池与权重保持不变。
只取训练池中已有的冻结参考图，不使用dev或test图；默认读取best，固定300张测试、conf=0.001、NMS=0.7。
固定头 `yoloe-head` 和 `yolov8` 去掉 `--prompt-shots`，只评估一次，推理不需要参考图。
没有 `--execute` 时只显示计划；结果保存在本训练目录的 `test_val300/`，拒绝覆盖已有目录。
正式300张测试按上述队列在dev选参和种子复跑之后执行。完成状态与指标以独立总表为准。
边界见 [公共方案](../common/COMPARISON_PLAN.md)。
本目录代码和配置可随Git同步；data/和训练权重按现有规则留在本机。

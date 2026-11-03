# YOLOE固定头、视觉提示与YOLOv8-L微调对照

叉车与吊物均已完成训练、独立dev选择及300张test评估。方法细节与逐实验分类见[SAVPE指南](../../../../../docs/guides/YOLOE_FINETUNING_METHODS.md)；本文件说明执行协议。

## 当前三条路线

| 路线 | 入口 | 训练范围 | SAVPE | 推理参考 |
|---|---|---|---|---|
| A：YOLOE固定头 | `--method yoloe-head --strategy head/neck/full` | 线性输出层 / neck+固定检测头 / backbone+neck+固定检测头 | 训练前移除 | 无 |
| B：YOLOE视觉提示 | `--method yoloe-vp`；配置`adaptation.scope=savpe/head/neck/full` | 从仅SAVPE到backbone+neck+检测头+SAVPE | 各范围均保留并训练 | 冻结训练支持集5/10shot |
| C：YOLOv8-L | `--method yolov8` | backbone+neck+普通检测头 | 原架构没有 | 无 |

A/B同从YOLOE检测结构迁入原始预训练权重；A额外生成类别文本embedding并融合固定头。C从官方YOLOv8-L检测预训练权重初始化。各实验从自己的原始checkpoint开始，不接力使用另一方法的微调权重。所有full范围遵守DFL固定投影等上游冻结例外。

原始未微调YOLOE配相同参考图作为B的基线。去掉分割输出不等于去掉SAVPE；BN运行统计固定也不等于冻结SAVPE卷积和BN仿射参数。

## 数据与选择协议

叉车五类、50张训练图；吊物单类、10张训练图，均为每类10shot、support-seed=0。各自100张独立dev和300张test，两场景不合成一个模型。

1. 每个任务的prepare_data.py读取本任务config.json，检查冻结参考清单、类别、图像与标注哈希。
2. 生成train/val列表与YOLO检测标签；检查train/dev/test内容哈希不交叉。内容无交集不等于视频序列独立。
3. 显式选择方法、训练范围、训练seed、尺寸、GPU和batch，从原始权重初始化。
4. 每轮只用dev验证并早停；原生best按0.1 AP50+0.9 AP50:95保存，再统一推理评价。
5. 按dev AP50:95选每种YOLOE方法前三名，各第一名补训练seed=1、2；候选先冻结，再测test。
6. VP在同一权重上分别用冻结5/10shot支持子集提取原型。改变推理shot不改变训练预算。

两个任务各执行48项YOLOE配置搜索及4项种子复跑；吊物另含3项基础训练。原始搜索为叉车52项训练/15项测试、吊物55项训练/19项测试，均已完成；加入冻结SAVPE等对照后，当前每个场景各汇总64项训练记录、31项测试推理设置。

- [叉车参数与执行入口](../chache/README.md)、[结果总表](../../chache_val300/comparisons/yoloe_search_20261029/README.md)。
- [吊物参数与执行入口](../diaowu/README.md)、[结果总表](../../diaowu_val300/comparisons/yoloe_search_20261029/README.md)。

## B路线的训练选项

`adaptation.scope`控制训练范围；`protocol=same_image/cross_image`控制提示来自当前训练图还是另一张训练支持图；`loss=classification/detection`控制分类项或box/cls/DFL检测损失。这些是不同维度，都不删除SAVPE。

cross_image排除当前整个query batch的图片，支持来源始终是train。SupportVPValidator也只从train提取类别原型，dev真值只用于评分。每轮重新提取支持原型，避免复用旧权重的embedding。

叉车dev优胜VP为`vp_cross_neck`：640，neck+head+SAVPE，跨图，检测损失。吊物dev优胜VP为`vp_full_lr0.0001`：640，全范围含SAVPE，同图，检测损失。两者均固定SAVPE BN运行统计。

最终推理先用微调后的SAVPE编码支持图片和框，再按类归一化平均。固定支持集可缓存原型；无需每张query重新编码支持图。切换参考集时重新提取。

## 操作与复现

任务入口无`--execute`时只显示计划，不导入模型或训练。先核对计划，再以新名字做2轮检查；正式运行从原始权重新开。单卡独立运行、FP32，不使用DDP。VP支持从同一中断运行的last.pt恢复，检查配置和数据指纹，但不保证恢复完整RNG和早停历史。

公共config的100轮/640/AdamW/lr0=1e-4是基础对照起点；搜索中的范围、学习率、epoch、batch、尺寸和增强按各自配置记录，不能把不同预算说成完全相同条件。固定头优胜配置中叉车用640、吊物用960，“全量”不要求960输入。

评估读取实际`method`与checkpoint结构并验证训练/支持/测试身份。固定头和YOLOv8直接推理；VP使用声明的训练支持池，默认conf=0.001、NMS IoU=0.7、每图每类最多100框，结果写入本运行test_val300。报告AP采用项目101点class-aware AP，P/R默认置信度0.5、IoU0.5；pycocotools AP另存，不能混写。

所有报告、元数据及CSV放[comparisons索引](../../../../README.md)，各方法 experiments 中的原始预测和跨方法数据集总表保留原提示推理实验。100张dev分数不能填进300张test表。YOLOE搜索预算与YOLOv8单次基线不同；已见类结果不证明未见类开放泛化。

只重建已有结果表，不运行模型：

```bash
python scripts/summarize_yoloe_optimization.py --stamp yoloe_search_20261029_r2
python scripts/summarize_yoloe_optimization.py --stamp yoloe_diaowu_search_20261029
```

## 新增冻结SAVPE条件

`adaptation.freeze_savpe=true`可用于VP的head/neck/full范围，固定整个SAVPE的参数与BN统计，其余选定模块训练，推理继续使用视觉提示。两个场景各有7组与原搜索匹配的配置；dev第一名复跑训练seed=1、2，前三名及复跑权重分别测5/10shot。详见[叉车新增实验](../../chache_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md)及[吊物新增实验](../../diaowu_val300/comparisons/yoloe_search_20261029/ALL_EXPERIMENTS.md)。

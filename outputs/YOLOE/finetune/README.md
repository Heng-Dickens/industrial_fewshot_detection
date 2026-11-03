# 两场景微调入口

[本轮：YOLOv8与YOLOE的5shot训练及推理对照](5SHOT_COMPARISON.md)

[SAVPE方法与逐实验分类](../../../docs/guides/YOLOE_FINETUNING_METHODS.md)明确区分固定头与视觉提示微调。

两个任务的配置、入口和执行说明分别保存。共用的只是训练实现，不在叉车任务中保存吊物配置，也没有任务之间的配置继承。
两任务均提供微调后测试入口，视觉提示可测5/10shot参考图。系统搜索和基础对照结果分别保存：

- [叉车微调搜索总表](chache_val300/comparisons/yoloe_search_20261029/README.md)。
- [吊物微调搜索总表](diaowu_val300/comparisons/yoloe_search_20261029/README.md)。

各报告记录当前执行状态、独立dev排名、测试成绩和训练种子统计，两个场景各用自己的数据与配置。

```text
outputs/YOLOE/finetune/
  scripts/
    chache/       # 叉车配置、训练、评估入口
    diaowu/       # 吊物配置、训练、评估入口
    common/       # 共用训练实现
  chache_val300/
    runs/         # 训练日志、权重及测试输出
    comparisons/  # 微调方法和参数对照
  diaowu_val300/
    runs/
    comparisons/
```

| 任务 | 配置 | 训练/验证/测试 | 操作说明 |
|---|---|---|---|
| 叉车五类 | [config.json](scripts/chache/config.json) | 50/100/300张 | [README](scripts/chache/README.md) |
| 吊物单类 | [config.json](scripts/diaowu/config.json) | 10/100/300张 | [README](scripts/diaowu/README.md) |

两任务均保留10shot、support-seed=0、epochs=100，以及原有数据/输出路径。
每个任务的配置是独立完整文件；修改叉车配置不再影响吊物。
三种方法均可通过各任务的train.py选择：`yoloe-head`、`yoloe-vp`、`yolov8`。

## 常用命令

在项目根目录执行；下面两条只显示训练计划，不加载模型：

```bash
python outputs/YOLOE/finetune/scripts/chache/train.py --method yoloe-vp
python outputs/YOLOE/finetune/scripts/diaowu/train.py --method yoloe-vp
```

查看数据准备计划：

```bash
python outputs/YOLOE/finetune/scripts/chache/prepare_data.py
python outputs/YOLOE/finetune/scripts/diaowu/prepare_data.py
```

按任务README确认后，数据准备加 `--write` 才会写文件，训练加 `--execute --device GPU编号 --batch 批量大小` 才会执行。
公共实现直接运行时必须显式传 `--config`，不会默认选择某个场景。

训练必须显式选择 `--method`，目录名字不能决定训练方式。
微调后使用各任务 `evaluate.py --run-dir 具体训练目录` 查看测试计划。
保留SAVPE的 `yoloe-vp` 权重可加 `--prompt-shots 5 10 --execute --device GPU编号` 一次测两组。
固定头只能直接检测，不支持更换推理提示图。详见各任务README。

方法、验证用途与当前未实现部分见 [三种方法对照方案](scripts/common/COMPARISON_PLAN.md)。

[Chache4新类别：四类微调后加提示 / 五类微调 / 预训练，5与10shot](chache_val300/comparisons/novel_chache4_20261102/README.md)

专题汇总（逐列加粗最优）：[YOLOE微调与YOLOv8对照](../../_shared/comparisons/final_results/03_yoloe_finetuning.md)

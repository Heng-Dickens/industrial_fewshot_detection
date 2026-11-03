# 微调公共实现

这里只保存两任务共同使用的代码和方法说明，不保存任何任务的默认配置。

- [叉车任务](../chache/README.md)：独立config.json及入口。
- [吊物任务](../diaowu/README.md)：独立config.json及入口。
- [总入口](../../README.md)。
- [三种方法对照](COMPARISON_PLAN.md)。

prepare_data.py、train.py、evaluate.py接收任务入口传入的default_config。
直接运行公共脚本时必须传 `--config /实际任务/config.json`；不默认选叉车或吊物。
finetune_config.py负责读取配置和检查类别顺序，vp_trainer.py仅在真正执行SAVPE训练时导入。

原目录重组前文件已备份在项目 `.planning/finetune_layout_backup_20261029/`。
evaluate.py读取实际训练方法、权重结构、冻结测试清单和训练池。
固定头直接评估一次；SAVPE权重按5shot/10shot提取原型并逐类查询。
叉车与吊物的固定头、视觉提示微调及YOLOv8基础对照均已完成测试，分场景保存结果。
训练入口要求显式 `--method`；训练范围、SAVPE状态及推理方式见[SAVPE指南](../../../../../docs/guides/YOLOE_FINETUNING_METHODS.md)。

## YOLOv8未训练目标检测头对照

`train.py --method yolov8 --initialize-only --execute`只重建目标类别检测模型并迁移兼容预训练张量，保存`weights/initialized.pt`；不会创建优化过程。目标分类输出随机初始化，其余兼容张量逐项核验一致，元数据标记`initialized_no_training`、优化步数0。

评估时指定`evaluate.py --checkpoint initialized --execute`。这与COCO原类别推理、只改类别名称、按COCO类别映射到目标类别均不同，不能作为模型具备新类别语义识别能力的证据。

本轮可复现的多GPU调度与状态查看入口为`python scripts/run_yolo_5shot_comparison.py`；完整配置在`outputs/_shared/runtime/yolo_5shot_comparison_20261102/configs/`，结果见[两场景5shot对照](../../5SHOT_COMPARISON.md)。

## Chache4作为微调未见类别

`python scripts/run_yoloe_novel_chache4.py`提供四类微调后加入Chache4提示、五类一起微调、原始权重提示的5shot/10shot对照。
仅该实验通过显式`class_protocol.type=heldout_visual_class_v1`允许四类训练，类别顺序仍严格校验。
两种训练均使用排除Chache4正图后的共同88张验证集，按基础四类效果选权重。
原有`evaluate.py`保留“推理参考必须属于训练池”的约束；新增参考的例外只存在于新实验入口，明确核验Chache4参考未进入四类训练和验证。
新类、基础类、全五类指标分别保存；[实验协议与结果](../../chache_val300/comparisons/novel_chache4_20261102/README.md)。

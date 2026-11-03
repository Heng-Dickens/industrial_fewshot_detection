# JSTY-YOLO 实验

自定义 YOLOv26_v8 监督检测网络的ONNX评估，按叉车、吊物及历史数据队列分别保存。
ONNX是推理文件格式；监督训练预算与视觉提示shot分别记录。两场景FP32 ONNX的300张测试结果已加入全部算法总表。

[模型来源和结构说明](../../docs/JSTY_MODEL_GUIDE.md) · [原始运行配置](../../configs/specific/jsty_hanging_object.yaml)

[返回全部方法](../README.md)

| 数据集 | 原始实验 | 结果对照 |
|---|---|---|
| chache_val300 | [实验目录](experiments/chache_val300/) | [统一表](comparisons/chache_val300/results_summary.md) |
| diaowu_99 | [实验目录](experiments/diaowu_99/) | [统一表](comparisons/diaowu_99/results_summary.md) |
| diaowu_val300 | [实验目录](experiments/diaowu_val300/) | [统一表](comparisons/diaowu_val300/results_summary.md) |

## 专题比较

- [comparisons / prompt_results](comparisons/prompt_results.md)

跨方法核查：[对齐基线](../_shared/comparisons/aligned_baselines_20261027/report.md) · [定位精度与四项误差≤20%](../_shared/comparisons/iou85_box20_20261028/README.md)

# JSTY 吊物监督检测模型

原目录 `D:\Work\JSTongYong\result\hanging_object_det1_0_0_0_1` 的7个模型及配套文件已移入项目 weights/jsty_hanging_object（本地路径 `../weights/jsty_hanging_object`）。迁移前后逐文件 SHA256 一致，来源、大小及哈希保存于 IMPORT.manifest.json（本地路径 `../weights/jsty_hanging_object/IMPORT.manifest.json`）。3.5GB 的验证数据压缩包仍在原目录。

## 模型与配置

| 文件 | 用途 |
|---|---|
| `jsty_hanging_object_det1_0_0_0_1_fp32.onnx` | 当前默认推理模型，ONNX Runtime CPU |
| `jsty_hanging_object_det1_0_0_0_1.pt` | 原 PyTorch 权重，保留；依赖原训练工程的自定义 `C2PSA_DA` 层 |
| `jsty_hanging_object_det1_0_0_0_1_int8.onnx`、`*_int8.tflite` | 原量化版本，保留；本次99张完整验证使用 FP32 |
| `*_config.py` | 原始部署配置，原样存档；其中 Linux 数据路径不再使用 |
| `*_result.png`、`train_result.png` | 原模型结果图 |

模型来自自定义 Ultralytics `YOLOv26_v8` 结构，采用 YOLOv8 风格 DFL 检测头。原部署 Python 写 `num_classes=2`，但 `.pt` 和 FP32 ONNX 的元数据均只有 `0: hanging object`；ONNX 实际每层65通道，即64个框分布通道加1个类别通道。新适配器使用实际1类，并在加载时校验元数据及输入输出形状，避免错误解码。

运行入口为 [configs/specific/jsty_hanging_object.yaml](../configs/specific/jsty_hanging_object.yaml)，模型与阈值设置集中在 [configs/specific/jsty_hanging_object.yaml](../configs/specific/jsty_hanging_object.yaml)。配置按数据集、模型、实验顺序覆盖，因此模型评估阈值不会被数据集默认值改回0。

| 参数 | 当前值及说明 |
|---|---|
| `yolo_onnx.weights` | 项目内 FP32 ONNX 路径 |
| `input_size` | `[960, 1280]`，顺序为高、宽，与固定模型输入一致 |
| `class_names`、`target_class_id` | `[hanging object]`、`0`，与模型及目标标注名称一致 |
| `strides`、`reg_max` | `[8,16,32]`、`16`，三层原始 DFL 检测头 |
| `confidence` | `0.2`，沿用原部署的候选框分数阈值；修改后重新推理 |
| `nms_iou`、`max_detections` | `0.65`、`300`，模型端 NMS 与候选框上限 |
| `providers`、`intra_op_threads` | CPU、4线程；设备不可用会明确报错 |
| `evaluation.score_threshold` | `0.2`，只筛选固定工作点统计，修改后可复用框重算 |
| `evaluation.iou_threshold`、`max_detections_per_image` | `0.5`、`100`，与统一评估报告口径对应 |

预处理为 OpenCV 解码、BGR转RGB、等比例双线性缩放、居中填充114、除以255、NCHW float32。输出经 DFL softmax 距离解码、类别 sigmoid、NMS，再取消填充及缩放，保留原图浮点像素坐标。此模型不接收文字或参考图提示，不进行二次语言模型复核。

## 运行及输出

当前项目环境已具备 ONNX Runtime 和 OpenCV。新环境可安装 `python -m pip install -e ".[yolo-onnx]"`。

```powershell
# 在项目根目录运行；仅检查配置、数据和缓存，不加载模型
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/jsty_hanging_object.yaml --dry-run

# 本地检测、导出及评估
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/jsty_hanging_object.yaml

# 忽略成功缓存，全部重新推理
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/jsty_hanging_object.yaml --force
```

默认使用 `configs/common/diaowu.yaml` 中的99张固定查询图。参考目录仅用于执行共同测试集的内容去重规则，原图、框标注、文字参考和待测真值均不会作为模型提示。`train` 仍表示推理加评估，不更新权重；`--mode test` 表示仅推理及导出。

结果直接保存在 `outputs/runs/diaowu_99/监督检测/jsty_hanging_object_det1_0_0_0_1_fp32_监督检测/`，重复运行使用同级 `_第2次` 后缀。目录内有 `evaluation.txt`、`metrics.json`、`run.json`、`summary.json`、`config.resolved.json`、`records.jsonl`、逐图 `predictions/` 与 `feedback/`。终端输出共同的 COCO AP/AR、分类表与 VOC 风格统计表。反馈包含框及分数，逐框自然语言依据标记为不支持。

修改模型文件、解码参数、推理阈值或图像会使对应缓存失效；仅修改评估阈值会复用框并重新计算指标。修改部署存档 Python 不会改变实际参数，需编辑当前模型 YAML。只对原部署存档做修改不会改变当前推理。

## 2026-10-22 完整验证

99张图片均为实际本地新推理，0张复用、0次失败。该组55个真实目标，IoU=0.5、分数阈值0.2：TP=54、FP=10、FN=1，Precision=84.38%、Recall=98.18%、F2=95.07%；COCO AP50约96.6%、AP50:95约86.1%。精确值以保存的 `metrics.json` 为准。

这些数值验证当前模型在项目测试图上的输出。模型是监督训练模型，其训练集与当前查询集是否重叠尚未核验；当前99张也不是独立选择工作阈值的验证集。不得把该模型标为0-shot或声称已证明相对 few-shot 方法的公平泛化优势。当前候选阈值为原部署的0.2；需要统一AP候选范围时，可另设较低推理阈值（例如0.001）后重新检测，并单独记录该实验。

已有结果可交给相同的 `evaluate`、比较及可视化工具，无需再次加载 ONNX：

```powershell
python main.py evaluate outputs/runs/diaowu_99/监督检测/jsty_hanging_object_det1_0_0_0_1_fp32_监督检测
python scripts/compare_models.py --run outputs/runs/diaowu_99/监督检测/jsty_hanging_object_det1_0_0_0_1_fp32_监督检测 --run outputs/runs/diaowu_99/视觉提示/yoloe-v8s-seg_5shot --output-dir outputs/comparisons/jsty_vs_yoloe
```

比较程序会在同一图片、真值及评估阈值上重算工作点指标；运行时记录的原始工作点可以不同。查看比较报告中的实际 `evaluation`，避免把重新评估的数值与原表混用。

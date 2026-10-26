# YOLOE 与 VLM 辅助标注对比

纯文字提示模式使用 `configs/specific/yoloe_text.yaml`，不输入参考图特征；配置、依赖和运行方式见 [YOLOE 文字模式指南](YOLOE_TEXT_GUIDE.md)。本文其余内容介绍原有视觉提示模式。

裁剪视觉特征 0.75＋文字特征 0.25 的 5shot 实验使用 `configs/specific/yoloe_fusion_5shot.yaml`，见[融合模式指南](YOLOE_FUSION_GUIDE.md)。

统一入口为 `main.py --config ...`。YOLOE 读取当前整理好的参考集，默认使用与 VLM 相同的 **5 张正样本原图**及各自的 JSON 目标框，在其余图片上执行本地检测。`train` 表示检测并评估，`test` 表示只检测，均不更新模型权重。

## 目录与来源

```text
third_party/yoloe/                 官方源码，保留上游目录结构与 LICENSE
  UPSTREAM.json                    下载来源、固定提交和源码压缩包 SHA256
third_party/yoloe-main.zip          官方源码存档
.venv-yoloe/                       本项目运行环境
weights/yoloe/yoloe-v8s-seg.pt      预训练权重，需下载成功后才可真实检测
src/industrial_fewshot/detectors/yoloe_visual.py          参考框读取、特征聚合和本地检测适配
configs/common/diaowu_yoloe_config.json    当前吊物场景配置
scripts/setup_yoloe.ps1               环境与权重准备
tool/compare_detectors.py           已保存的 YOLOE/VLM 结果统一比较
```

官方仓库：[THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)。本次使用提交 `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`，通过 GitHub 官方源码压缩包下载，源码存档 SHA256 为 `0d998dcec991d14879a3d16556f6c062999bede38a248519c08ee20afda45144`。这是包含上游源码的快照，未包含 Git 历史。项目适配放在上游目录之外。

## 当前环境与权重

安装入口现支持视觉、文字和融合模式；新建环境默认不继承全局site-packages，已有环境复用可用的PyTorch。最新分模式命令见[模型安装说明](guides/MODEL_INSTALLATION.md)。下文记录了当前机器最初创建环境的方式。

已在项目内建立 `.venv-yoloe`，继承本机 `D:\anaconda3\envs\yoloenv` 的 Python 3.10、CPU PyTorch 和 torchvision，新增依赖安装在该项目环境内。原有 yoloenv 的 ultralytics 安装保持原状。适配器在导入时明确选择 `third_party/yoloe`，防止误用其它项目或新版 PyPI ultralytics。

当前本地已存在 `weights/yoloe/yoloe-v8s-seg.pt`，并已有 `outputs/runs/diaowu_99/视觉提示/yoloe-v8s-seg_5shot/` 检测记录。是否重新加载模型执行检测取决于实际推理参数和成功缓存；全部命中时仅导出记录并重新计算评估指标。

重新建立环境（当前机器已完成，无需重复）：

```powershell
.\scripts\setup_yoloe.ps1 -InstallDependencies
```

其它机器可通过 `-Python "你的Python3.10或3.11路径"` 指定基础解释器。新环境的PyTorch构建通过 `-TorchBuild cpu/cu118/cu121/cu124` 选择；已有目标环境中可用的PyTorch会保留，基础环境不接收pip安装。

下载作者发布的 [yoloe-v8s-seg.pt](https://huggingface.co/jameslahm/yoloe/blob/main/yoloe-v8s-seg.pt)：

```powershell
.\scripts\setup_yoloe.ps1 -DownloadWeights
# 官方 README 提到的可选镜像
.\scripts\setup_yoloe.ps1 -DownloadWeights -WeightsEndpoint https://hf-mirror.com
```

也可把已下载的同一权重放到 `weights/yoloe/yoloe-v8s-seg.pt`。官方文件大小为 `31,135,890` 字节，SHA256 为 `ac2b90ed23011495a3e86d89caeb3432a15129cac8d849ba121293c8fc1e0536`。下载脚本校验 SHA256 后才把 `.part` 更名为 `.pt`。只读规划中的 `local_model.runtime_ready` 表示源码/权重文件是否存在，不代替完整模型加载检查。

此视觉提示路径不调用文字编码器，因此无需为了它额外下载 MobileCLIP 权重；官方完整训练、文字提示等功能仍按上游 requirements 安装各自的依赖。

## 参考图的使用方式

参考目录沿用 `data/references/diaowu`，通过 `manifest.json` 匹配正样本原图与标注，不依赖已删除的旧混合目录。

1. 用与 VLM 相同的 seed 和文件 ID 顺序选择前 K 张正样本，默认 K=5。
2. 从对应 LabelMe JSON 读取目标类别的矩形框；多边形取包围框。非目标类别的框不作为当前类的提示。
3. 使用上游 `YOLOEVPSegPredictor` 的 `return_vpe` 接口提取每张原图的视觉特征。同一图内的同类框按官方实现合成提示区域。
4. 每张图的特征 L2 归一化，对 K 张图等权平均，再 L2 归一化，得到一个类别原型；通过 `set_classes` 固定后在所有待测图上复用。
5. 待测全图只输入图像字节，不输入其真值、正负标签、参考框或其它待测答案。

其中第 4 步是本项目明确采用的多参考图聚合策略，不宣称它是官方论文的独立多样本基准协议。与 VLM 的 O+J 组比较时，二者利用同一批参考原图和框信息，但模型处理机制不同。

YOLOE 视觉接口接收图像及空间框/掩码，不将长篇自然语言规则、原始 JSON 文本、已画框的像素图或无框负样本作为同一种提示机制处理。当前基线不使用负样本原型、文字融合或额外的负例过滤器；负样本留在待测集中检验误标。若以后增加负例拒识，应作为另一个明确命名的方法比较。

当前 seed=0 的 1／5／10shot 优先使用最早运行的历史参考图；5shot 与 VLM 完全一致，为 `0000090.jpg`、`0000189.jpg`、`0001320.jpg`、`0001352.jpg`、`0003939.jpg`。固定清单、正负样本及配对素材见 支持集说明（本地路径 `../data/references/diaowu/README.md`）。修改 seed 或张数后以实际运行元数据为准。

## 场景配置

`configs/common/diaowu_yoloe_config.json` 中：

| 字段 | 默认值与用途 |
| --- | --- |
| `backend` | `yoloe`，使用统一入口的本地检测分支 |
| `labels` | 与 VLM 相同的目标类别名；description 记录目标定义，不作为文字模型输入 |
| `reference_image_dir` | 当前 prepared 数据目录 |
| `yoloe.positive_count` | 5，可改为 1、10、25 等可用张数 |
| `yoloe.selection_seed` | 0，与 VLM experiment_suite.selection_seed 对齐 |
| `yoloe.weights` | 本地预训练模型文件；更换权重同时改变缓存指纹 |
| `yoloe.device` | 当前为 cpu；GPU 环境改为 `0` 或 `cuda:0` |
| `yoloe.torch_threads` | 当前 CPU 线程数为 4 |
| `yoloe.imgsz` | 640，必须为 32 的倍数 |
| `yoloe.confidence` | 0.001，模型保留候选框的最低分数，决定可用于 AP 的候选范围 |
| `yoloe.nms_iou` | 0.7，模型端 NMS 阈值，与评估匹配 IoU 不同 |
| `yoloe.max_detections` | 100，模型端每张图片最大检测数 |
| `evaluation` | 与现有 VLM 使用相同的评估设置，含 IoU、score_threshold、代理 loss 权重等 |
| `missing_ground_truth` | 当前为 negative，已确认无同名标注的图片也参与误标评估 |
| `output_suffix` | `_yoloe_annotations`，默认输出为输入目录的同级目录 |

不要为 YOLOE 打开 VLM 的 `experiment_suite` 或 `ablation`，也无需配置 base_url/API key/prompt_file。它使用本地权重，串行复用一个模型，不发起 VLM 请求。后续换场景只需修改独立配置。

使用 `specific/visual_1shot.yaml`、`visual_5shot.yaml`、`visual_10shot.yaml`、`visual_25shot.yaml` 时，先加载 `common/diaowu.yaml`，再加载 `common/diaowu_yoloe_config.json`，最后加载专用文件。因此检测参数、`evaluation`、参考样本 seed 以共享JSON为准；shot 数由专用文件明确指定；待测数据路径和固定清单由共享数据配置的 `dataset` 部分指定。修改共享JSON可同时影响各shot实验；启动日志显示最终生效的评估及推理参数，运行目录中的 `config.resolved.json` 保存合并配置。

`yoloe.confidence` 是模型输出候选框的最低置信度，修改它会重新推理；`evaluation.score_threshold` 是对保存的框计算固定阈值 precision／recall／F2 时的筛选阈值，修改它会复用框并重算指标。COCO AP 扫描全部候选分数，不随这个固定评估阈值截断。

## 运行

在项目根目录运行，只读检查不需要 torch 或权重；正式检测使用已准备的项目 Python：

```powershell
.\.venv-yoloe\Scripts\python.exe main.py --config configs/common/diaowu_yoloe_config.json --dry-run

# 权重就绪后：检测并评估
.\.venv-yoloe\Scripts\python.exe main.py --config configs/common/diaowu_yoloe_config.json

# 仅检测，不计算指标
.\.venv-yoloe\Scripts\python.exe main.py --config configs/common/diaowu_yoloe_config.json --mode test
```

参考池与待测图片按原始文件 SHA256 去重，和 VLM 组使用相同规则。当前 `0000090.jpg` 与参考池重叠，统一排除后是 **99 张待测图**。参考原始的 50 个例子保持完整；参考目标框只从参考 JSON 读取。

输出在配置的结果根目录下按实际模型和 shot 数命名，重复运行追加 `_第2次` 等后缀；目录内直接保存与 VLM 相同格式的 `run.json`、`summary.json`、`records.jsonl`、`metrics.json` 和 `predictions/*.json`。保留 YOLOE 浮点像素坐标，不经 VLM 的 0~1000 整数坐标量化。`predictions/*.json` 只保留置信度不低于 `evaluation.score_threshold` 的框（供 LabelMe 人工查看的最终标注）；`records.jsonl` 与所有指标计算仍使用 conf=0.001 保留的全部候选，保证 AP 排名完整。

当前1/5/10/25-shot实验按 `outputs/runs/<测试数据集>/视觉提示/<模型与shot>/` 保存，文字和联合模式归入对应 `文字提示_vN`，详见 [结果目录说明](guides/RESULT_LAYOUT.md)。模型目录内直接保存 `evaluation.txt`、`feedback/` 等结果，终端使用共同的COCO AP/AR和VOC统计格式。YOLOE的反馈记录框、分数并标明不支持自然语言理由，不会生成MLLM的逐框文字解释。旧YOLOE输出根中的缓存仍可读取。

重新运行相同输入自动复用成功结果；更换权重、源码、有效参考框/原图或检测参数会使对应缓存失效。只改评估阈值或真值时自动复用检测结果。失败不会被缓存为成功，下一次会重试。模型初始化/参考编码时间与逐图时间分别记录；缓存命中不计作新检测计时。

`outputs/runs/.inference_cache/` 是按输入内容索引的成功检测缓存，内部散列名用于精确匹配。更早历史数据在 `outputs/archive/runs_legacy/`，Ultralytics运行设置在 `outputs/runtime/yoloe_settings/`。各数据集的模型目录内，`predictions/` 为逐图 LabelMe 标注，`feedback/` 为逐图框和分数，`records.jsonl` 为完整推理记录，`run.json` 为运行元数据，`summary.json` 为汇总，`metrics.json` 为结构化指标，`evaluation.txt` 为可读报告。MLLM在模型／shot下按参考材料分组；YOLOE单组实验直接保存这些文件。

## 比较已有结果

```powershell
.\.venv-yoloe\Scripts\python.exe tool/compare_detectors.py `
  --yoloe-run "YOLOE的单次运行目录" `
  --vlm-run "VLM的ablation目录或单组运行目录" `
  --evaluation-config configs/common/diaowu_yoloe_config.json `
  --output-dir outputs/comparisons/yoloe_vs_vlm
```

可以多次传 `--vlm-run`。比较工具仅读取已有结果，核对逐图指纹，在同一套当前真值和阈值下重算所有方法的 Precision、Recall、F1、mAP50、mAP50_95、整图正确率、负样本误报/错误率和代理 loss。推理失败留在准确率分母中。输出 `detector_comparison.csv` 和 `detector_comparison.json`，另含模型、参考数、是否与 YOLOE 使用相同参考集、可用的历史逐图耗时。

默认拒绝待测图片清单不同的结果。若比较历史 100 张 VLM 结果与新 99 张结果，可显式增加 `--intersection`，只对共同图片重新计算并记录各组排除清单，不删除或修改旧结果，也不重新请求模型。图片被修改或记录缺少 SHA256 时拒绝给出已核验的比较；未完成的推理记录也不能遗漏后参加比较。

准确率比较优先检查 mAP、召回与负样本错误率。候选框保留阈值和最终评估阈值需要分开记录；相同数值的置信度不代表两种模型校准一致，可用已有结果在统一评估阈值下重复比较，阈值的选择应使用独立验证集。代理 loss 仅为当前工程评估值，不是两种模型的训练损失。

本机 YOLOE 是 CPU 推理，当前 VLM 是远端服务；历史计时包括网络、预处理、复核/重试等各自开销，且旧记录可能没有时间。因此这些数据只能反映各自部署下的耗时，不能直接声称相同硬件的 FPS。实际优劣要等两边在同一测试集合上得到结果后再判断。

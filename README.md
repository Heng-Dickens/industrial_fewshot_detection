# Industrial Few-shot Detection

工业场景少样本目标检测对比实验：MLLM、YOLOE 视觉／文字提示、GroundingDINO 本地文本检测。项目在原工作目录中重组，并原地改名为 `D:\Learn\VLM\industrial_fewshot_detection`，没有创建另一份项目副本。

本仓库只同步代码、配置、提示词、安装脚本和使用文档。`data/`、`outputs/`、`weights/`、`third_party/`、虚拟环境、真实 `.env` 和 `*.local.yaml` 不进入 Git。服务器通过固定版本的导入脚本恢复第三方源码；数据与模型单独准备。首次同步、Linux 环境安装和服务器路径配置见[Git 与服务器部署说明](docs/guides/GIT_SERVER_DEPLOYMENT.md)。下文涉及的数据及历史结果是本地运行资产，克隆仓库不会附带。

```text
configs/common/          共享数据、YOLOE视觉默认参数、新场景模板
configs/specific/        可直接选择的模型、实验和辅助工具配置
data/raw/                外部原始数据位置说明，不复制监控原图
data/processed/diaowu/   COCO annotations.json、参考样本及准备报告
data/splits/diaowu/      固定 train/val/test 清单，含图片指纹和本地评估真值
data/supports/diaowu/    1/5/10/25 shot 支持集清单
src/industrial_fewshot/  数据、检测器、提示词、后处理、评估、缓存和运行器
scripts/                 准备、运行、评估、比较、稳定性分析及可视化入口
tests/                   单元、集成及数据工具测试
outputs/                 runs/、comparisons/、reports/
third_party/yoloe/       官方 YOLOE 源码；UPSTREAM.json 记录来源及版本
third_party/pet_dino/    官方 PET-DINO 源码，独立视觉／文字检测
third_party/wedetect/    官方 WeDetect 源码，纯 PyTorch 文字检测
weights/                 本地模型权重
docs/                    迁移说明及详细指南
main.py                  统一入口；保留原 --config 用法
```

## 使用

在项目根目录执行，先安装基础依赖：

```powershell
python -m pip install -e .
python scripts/prepare_dataset.py --config configs/specific/visual_5shot.yaml
python main.py --config configs/specific/mllm_ablation.yaml --dry-run
```

准备数据只写项目派生清单和 COCO，不调用模型、不修改原始图片或人工 JSON。已有 50 个参考例子放在 `data/processed/diaowu/references`，含 25 正例和 25 负例、正例原图／JSON／局部裁剪／彩色框全图。

`configs/specific/` 可直接选择：

| 配置 | 实验 |
|---|---|
| `mllm_text_only.yaml` | 当前 MLLM，只给文字 prompt |
| `mllm_fewshot.yaml` | 5 正例的 O+C+B+J，加 5 负例 |
| `mllm_ablation.yaml` | 按当前开关运行；当前为 5 正例的原图＋JSON、无负例，加文字基线 |
| `visual_1shot.yaml`、`visual_5shot.yaml`、`visual_10shot.yaml`、`visual_25shot.yaml` | YOLOE，对应数量的原图及框标注 |
| `jsty_hanging_object.yaml` | JSTY 吊物监督检测模型，FP32 ONNX，本地运行，统一标注与评估输出 |
| `zero_shot.yaml` | GroundingDINO 本地文本检测 |
| `pet_dino_5shot.yaml`、`pet_dino_text.yaml` | PET-DINO Swin-T：5shot 视觉／0shot 文字，默认 val300；见[环境与运行说明](docs/guides/PET_DINO_WEDETECT.md) |
| `wedetect_text.yaml` | WeDetect Base 中文文字检测，默认 val300；使用独立模型环境 |
| `groundingdino_base.yaml` | GroundingDINO-Base 本地文字检测，使用保存的Tiny实验描述和评估阈值 |
| `yoloe_text.yaml` | YOLOE-11L 纯文字提示、0shot，与最新 GroundingDINO 使用相同描述 |
| `yoloe_fusion_5shot.yaml` | YOLOE-11L，5shot 目标裁剪视觉特征 0.75＋文字特征 0.25，见[融合说明](docs/YOLOE_FUSION_GUIDE.md) |

O=参考原图，C=参考目标裁剪，B=在参考原图画框，J=参考标注 JSON。参考样本按 seed 固定并嵌套选取；负例数量与正例原图数量相等。YOLOE 从参考原图和框提取视觉原型，不支持同样的负例提示。GroundingDINO 不接收参考图，但共享参考池排除规则和固定测试清单。

实际运行命令（运行前补齐对应权重及服务）：

```powershell
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/mllm_ablation.yaml
python main.py --config configs/specific/mllm_fewshot.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/visual_5shot.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_text.yaml
# 在安装了 ovd 依赖的 Python 环境中运行
python main.py --config configs/specific/zero_shot.yaml
```

三个 MLLM 实验 YAML 都是独立完整配置，模型名以各文件开头的 `model` 为准；本次目录整理保留了各文件的实际设置。当前 CC Switch Claude Messages 调用方式和凭据规则保持原样，不修改 Codex 或 CC Switch 的共享配置。模型、协议和接口都会在启动日志中显示。YOLOE视觉模式共享 `configs/common/diaowu_yoloe_config.json` 的模型及默认参数；YOLOE文字、GroundingDINO和JSTY模型参数已合并到对应的 `configs/specific` 入口，公共数据仍在 `configs/common/diaowu.yaml`。相对路径属于声明它的文件；运行保存脱敏后的 `config.resolved.json`。各配置的修改位置和兼容说明见 [配置说明](configs/README.md)。

第一条命令按 YAML 的 `experiment_suite` 开关运行；全部七种材料及正负例开关均启用时为 15 组。项目结果按 `outputs/runs/<测试数据集>/<提示词版本>/<实际模型与shot>/<参考组合>/` 保存，各组直接记录指标、逐图记录、`predictions/` 与 `feedback/`。数据集和文字内容改变时独立归档，重复运行使用同级 `_第2次` 等后缀保护历史，详见 [结果目录说明](docs/guides/RESULT_LAYOUT.md)。HTTP 401／403／404 会终止整轮实验；连接失败按 `max_attempts` 重试，耗尽后停止。Ctrl+C 会取消并发请求并退出（退出码 130），已落盘记录和成功缓存保留。

YOLOE 的1/5/10/25-shot也按数据集和提示类型保存，例如原99张测试集的视觉5-shot位于 `outputs/runs/diaowu_99/视觉提示/yoloe-v8s-seg_5shot/`。模型目录内直接保存结果，终端同样显示COCO AP/AR及VOC统计；YOLOE的feedback包含检测框与分数，不提供MLLM式的自然语言逐框依据。

迁入的 JSTY 吊物模型运行命令为 `.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/jsty_hanging_object.yaml`。它使用本地 ONNX Runtime，模型参数在 `configs/specific/jsty_hanging_object.yaml`，结果直接写入模型名称加 `_监督检测` 的目录。模型实际为1类，已适配三层原始DFL输出、浮点坐标、缓存及全部评估工具；参见 [JSTY模型说明](docs/JSTY_MODEL_GUIDE.md)。

## 公平比较及结果

提示词与结果对应关系见 `outputs/runs/prompt_results.md`。每次运行保存实际 `prompt.txt` 和 `prompt.json`；GroundingDINO、YOLOE文字和融合模式保存实际文字，YOLOE视觉模式和监督模型说明各自的输入方式。历史提示词按各次保存的证据恢复，操作说明见 [提示词记录](docs/guides/PROMPT_RECORDS.md)。

当前 `00000` 目录有 100 张图，与参考池重复的 `0000090.jpg` 在所有方法中统一排除，固定测试集为 99 张。当前配置按已确认的数据约定将缺失 JSON 视为空目标负例；其他场景应自行明确 `missing_ground_truth`。本目录属于一个视频分组，暂时采用 train=0、val=0、test=99，避免把相邻帧随机分到训练和测试两侧。调整数据范围和分组比例后重新准备清单。

`train` 是沿用的命令名称，执行推理及检测评估，不进行参数训练或反向传播；`test` 只推理。评估包含 precision、recall、F1、mAP50／75／50:95、代理检测 loss、整图准确率与负例误标率。推理失败始终计入整图错误和负例错误的分母，评估标记为 incomplete。代理 loss 不是模型训练 loss；AP 使用共同的 bbox 匹配和 101 点插值实现，不包含 COCO crowd／ignore 等完整协议。

成功预测按实际模型请求和图片内容复用。仅重新运行、改变评估阈值或项目目录改名不会主动重跑相同 MLLM 请求；失败项会重试。只有显式 `--force` 才跳过成功缓存。新实验写入 `outputs/runs/`，旧结果在 `outputs/archive/runs_legacy/`，外部 `00000_annotations` 历史缓存仍会读取。原配置未指定输出目录时，仍按此前约定输出到输入目录的同级 `<目录名>_annotations`。

```powershell
# RUN_DIR 填实际单次运行目录，不填整个模型输出根目录
python scripts/evaluate.py RUN_DIR
python scripts/compare_models.py --run MLLM_RUN --run YOLOE_RUN --run GROUNDINGDINO_RUN --output-dir outputs/comparisons/diaowu
python scripts/visualize_predictions.py --run-dir RUN_DIR --output-dir outputs/reports/previews
python scripts/analyze_stability.py --comparison COMPARISON_SEED0_JSON --comparison COMPARISON_SEED1_JSON --output outputs/reports/stability.json
```

比较程序只读取保存的预测，核验同一图像字节和当前真值后重算指标，不调用模型。错误图片明细在各运行 `metrics.json` 的 `incorrect_images` 和 `negative_false_positive_images`。稳定性分析要求不同 support seed、相同 query 集和评估设置；只有一个 seed 时不报告标准差。原图 GT 只用于本地评估／可视化，检测器边界只接收当前图片，不接收待测真值。

终端默认显示 COCO AP／AR 摘要、分类 AP／Recall／F2 及 VOC 风格检测统计表，并保存文本报告；完整结构化输出可用 `--json`。指标口径及文件位置见 [评估报告说明](docs/guides/EVALUATION_OUTPUT.md)。

MLLM 还会保存每个框的可见依据与覆盖范围说明，写入 `objects[].reason`、LabelMe 框描述及逐图 `feedback/*.json`／`*.txt`。首轮、复核和失败响应均可追溯，详见 [模型反馈说明](docs/guides/MODEL_FEEDBACK.md)，其中说明了新增理由要求对旧缓存的影响。

## 模型环境

四类模型的独立安装入口、CPU/CUDA选择和检查命令见[模型安装说明](docs/guides/MODEL_INSTALLATION.md)。分别使用 `scripts/setup_yoloe.ps1`、`setup_groundingdino.ps1`、`setup_pet_dino.ps1`、`setup_wedetect.ps1`；不带操作开关只打印计划，`-All` 执行完整准备，`-All -Plan` 仅预览。

YOLOE 使用 `third_party/yoloe` 官方代码，可通过 `scripts/setup_yoloe.ps1 -InstallDependencies` 创建独立 `.venv-yoloe`。`-DownloadWeights` 下载官方权重并校验 SHA256。当前源码、CPU 环境和本地权重已有，`--dry-run` 的 `runtime_ready` 表示源码和权重是否存在，不代替实际模型加载检查。

GroundingDINO 采用 Transformers 的本地 checkpoint API，建议单独的 Python 3.10／3.11 环境安装 `pip install -e '.[ovd]'`，已固定 Transformers 4.46.3。将 [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) 的完整模型、配置及 processor/tokenizer 文件放入 `weights/grounding-dino-tiny`。运行只从本地加载，缺失模型会明确报错。接口依据 [官方 Transformers 文档](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/grounding-dino)。目前尚未安装并执行真实 GroundingDINO，不能据离线模拟测试给出模型优劣结论。

```powershell
.\.venv-yoloe\Scripts\python.exe -m unittest discover -s tests -t . -q
```

迁移位置、旧命令兼容及环境路径修复见 [docs/MIGRATION.md](docs/MIGRATION.md)。详细说明见 [主流程](docs/guides/MAIN_GUIDE.md)、[参考实验](docs/guides/REFERENCE_SUITE_GUIDE.md)、[YOLOE](docs/YOLOE_GUIDE.md)、[标注复核](docs/guides/ANNOTATION_REVIEW_GUIDE.md) 和 [参考数据整理](docs/guides/PREPARE_REFERENCE_DATA.md)。可运行 `python scripts/check_links.py` 检查 README 与 docs 中的本地文件链接。

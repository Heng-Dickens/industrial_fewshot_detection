# 场景配置驱动的辅助标注

项目只保留一条主流程：`main.py --config 某个场景.json`。新场景或目标的模型、类别、参考图和提示词均在配置中设置，不再复制 Python 标注脚本。

```powershell
python main.py --config configs/specific/diaowu_annotation_config.json --dry-run
python main.py --config configs/specific/diaowu_annotation_config.json --mode test
python main.py --config configs/specific/diaogou_annotation_config.json --mode train
```

`test` 执行标注，`train` 执行标注并对照原有 LabelMe JSON 评估，预测始终保存在独立输出目录。当前 train 不更新权重，loss 是明确标记的检测代理损失。配置和指标详情见 [MAIN_GUIDE.md](MAIN_GUIDE.md)。

没有指定 `output_dir`（或设为 null/空字符串）时，输出根目录默认为输入目录的同级目录，名称为“输入目录名 + `_annotations`”，例如 `00000_annotations`；后缀可用 `output_suffix` 配置。显式配置输出目录或使用 `--output-dir` 可指定位置，每次运行仍在其中创建独立运行目录。

## 代码职责

| 文件 | 唯一职责 |
| --- | --- |
| `main.py` | 命令入口 |
| `config.py`、`configuration.py` | 读取场景配置、组合 YAML/JSON、校验字段、解析路径、加载提示词 |
| `runner.py` | 扫描数据、并发调度、重试和复核、导出结果、选择 train/test |
| `prompts/references.py` | 正例全图、裁剪正例、负例的准备与多图消息组装 |
| `data/supports.py` | 正例原图/裁剪图/带框图/JSON 配对、固定样本选择及张数实验矩阵 |
| `detectors/yoloe_visual.py` | 官方 YOLOE 本地视觉参考检测、五图特征聚合，接入统一缓存/导出/指标 |
| `detectors/mllm.py` | 统一模型请求、可选悬吊确认过滤；共享响应协议位于 `postprocess/schema.py` |
| `data/labelme.py` | 真值读取与校验、预测 LabelMe 结构、JSON 原子写入 |
| `evaluation/detection_metrics.py` | mAP、Precision、Recall、F1、IoU 和检测代理 loss |
| `evaluation/evaluator.py` | 从已保存的完整推理记录补算指标，共用同一套评估逻辑，不调用模型 |
| `cache.py` | 按推理输入和图片内容自动复用成功结果，避免重复消耗 token |
| `experiments.py` | 在相同数据和模型设置下执行参考材料与张数对比，共用主流程，汇总指标与基线差值 |

除根目录的 `main.py` 外，表中模块均位于 `src/industrial_fewshot/`。吊物、吊钩和通用目标共用这些模块。对象判定规则放在 `prompts/templates/`，不再嵌入独立的目标 Python 文件。

当前 `configs/specific/diaowu_ablation_config.json` 默认运行 15 组参考实验：7 种正样本材料组合（包含 O+C+B，不包含仅 O），每组提供 5 张正样本原图，分别测试不加负样本与增加 5 张负样本，再加仅 prompt 基线。可通过配置启用 1/5/10/25 张的任意档位。待测真值始终只在本地评估，各组使用同一批不与参考集重叠的待测图片。说明见 [REFERENCE_SUITE_GUIDE.md](REFERENCE_SUITE_GUIDE.md)；旧四组模式见 [ABLATION_GUIDE.md](ABLATION_GUIDE.md)。

YOLOE 基线使用 `configs/common/diaowu_yoloe_config.json`，源码保留在 `third_party/yoloe`，默认读取同样的 5 张正样本原图及 JSON 框。模型权重就绪后运行 `.venv-yoloe/Scripts/python.exe main.py --config configs/specific/visual_5shot.yaml`，使用同一套 LabelMe 导出与评估。环境、权重状态和 YOLOE/VLM 比较工具见 [YOLOE_GUIDE.md](../YOLOE_GUIDE.md)。

## 测试职责

```text
tests/
  support.py                 # 统一样例工厂与假客户端，无网络调用
  unit/                      # 配置、参考输入、模型响应、LabelMe、指标
  integration/               # 场景切换和完整 train/test 流程
  tools/                     # tool/ 下转换、复核、清理等独立工具
```

运行全部测试或单独运行一组：

```powershell
python -m unittest discover -t . -s tests
python -m unittest discover -t . -s tests/unit
python -m unittest discover -t . -s tests/integration
python -m unittest discover -t . -s tests/tools
```

## 旧入口的替代方式

原来的 `diaowu_annotate.py`、`diaogou_annotate.py` 及两份 `*_batch_folders.py` 已合并移除。用 `main.py --config` 选择原来的场景配置；`recursive: true` 会扫描所选图片目录及全部子目录，需要处理整批数字文件夹时把 `image_dir` 指向它们的父目录。

旧流程中的“原图旁写标注、跳过已有 JSON、强制清理旧标注、逐文件夹检查点”不再混入主流程。当前入口为所选全部图片生成或复用预测，并将本次完整结果写入新的运行目录，保护评估真值。模型、提示词、参考图和图片不变时，普通命令默认复用已成功的推理；仅新增、变化或失败的图片请求模型。使用 `--force` 才明确绕过缓存重跑。

推理已完成、只有评估失败时，在同一 Python 环境补齐 `numpy` 和 `scipy` 后重跑原命令，成功推理会自动复用。也可用 `python main.py --evaluate-only "原运行目录"` 指定一次旧运行补算。详细规则见 [MAIN_GUIDE.md](MAIN_GUIDE.md)。

原 `annotate/test.py` 是单图 API 演示，已移到 `examples/chat_image.py`，不会被当作测试或自动发起请求。依赖统一在项目根目录 `requirements.txt`。

## 独立工具

以下功能与新场景标注不重复，仍保留在 `tool/`：

- `annotation_tools.py`：现有汇总记录的可视化和 LabelMe/VOC/YOLO/COCO 格式转换。
- `review_existing_annotations.py`：复核并修正已有框，说明见 [ANNOTATION_REVIEW_GUIDE.md](ANNOTATION_REVIEW_GUIDE.md)。
- `clean_empty_annotations.py`：清理历史空标注。
- `prepare_reference_data.py`：读取图片及同名 LabelMe JSON，整理正负样本、原图、JSON、分类目标裁剪图及多类别彩色框全图；使用说明见 [PREPARE_REFERENCE_DATA.md](PREPARE_REFERENCE_DATA.md)。
- `setup_yoloe.ps1`：准备项目内 YOLOE 环境和校验后的官方预训练权重。
- `compare_detectors.py`：读取已有 VLM/YOLOE 结果，核对图片指纹，在同一批图片、真值和阈值下重算比较，不调用模型。
- `batch_annotate.py`：通用批量推理工具及共享响应协议，供上述工具复用；场景任务使用 main。

旧版长文档和开发记录保存在 `docs/history/`，不再作为当前运行指南。

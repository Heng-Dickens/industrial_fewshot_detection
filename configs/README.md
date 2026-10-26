# 配置说明

配置只分两类：`common/` 放共享设置，`specific/` 放可直接选择的专用配置。不再单独划分 models、datasets、experiments、legacy、credentials。

```text
configs/
  common/
    diaowu.yaml                 共享数据目录、标签、测试集和准备参数
    diaowu_val300.yaml          从JSTY验证数据包抽取的300张验证图片及标注
    chache_val300.yaml          五类叉车/牵引车场景的300张验证图片及标注
    diaowu_yoloe_config.json     YOLOE视觉模式的共享模型、推理和评估默认值
    scene_template.json         新场景配置模板
  specific/
    visual_1shot.yaml            YOLOE视觉提示，1张参考原图＋框
    visual_5shot.yaml            YOLOE视觉提示，5张参考原图＋框
    visual_10shot.yaml           YOLOE视觉提示，10张参考原图＋框
    visual_25shot.yaml           YOLOE视觉提示，25张参考原图＋框
    yoloe_text.yaml              YOLOE纯文字提示，0shot
    yoloe_fusion_5shot.yaml       5shot裁剪视觉0.75＋文字0.25
    zero_shot.yaml               GroundingDINO文字检测
    pet_dino_5shot.yaml           PET-DINO Swin-T，5张独立视觉参考，val300
    pet_dino_text.yaml            PET-DINO Swin-T，英文文字检测，val300
    wedetect_text.yaml            WeDetect Base，中文文字检测，val300
    groundingdino_base.yaml      GroundingDINO-Base，沿用保存的Tiny文字与评估设置
    diaowu_val300_benchmark.yaml  300张吊物验证集全算法复验任务表（排除Qwen-Max）
    jsty_hanging_object.yaml     JSTY监督检测，FP32 ONNX
    mllm_text_only.yaml          MLLM文字基线
    mllm_fewshot.yaml            MLLM参考图实验
    mllm_ablation.yaml           MLLM参考输入对比实验
    qwen_mllm.yaml               原独立单模式配置
    *.json                      原场景和辅助工具的专用配置
```

## 修改哪里

| 要修改的内容 | 文件 |
|---|---|
| 共享数据、标签、固定测试集和参考集位置 | [common/diaowu.yaml](common/diaowu.yaml) |
| 新的300张验证集 | [common/diaowu_val300.yaml](common/diaowu_val300.yaml)，抽样及格式说明见数据集说明（本地路径 `../data/raw/diaowu_val300/README.md`） |
| 叉车五类别300张验证集 | [common/chache_val300.yaml](common/chache_val300.yaml)，原始YOLO及转换格式见数据集说明（本地路径 `../data/raw/chache_val300/README.md`） |
| YOLOE视觉模式的模型权重、候选阈值和评估默认值 | [common/diaowu_yoloe_config.json](common/diaowu_yoloe_config.json)，同时供1/5/10/25shot使用 |
| YOLOE视觉模式的参考张数 | 对应 `specific/visual_*shot.yaml` 的 `yoloe.positive_count` |
| YOLOE文字描述与权重 | [specific/yoloe_text.yaml](specific/yoloe_text.yaml) |
| 裁剪＋文字的描述、两路权重与shot数 | [specific/yoloe_fusion_5shot.yaml](specific/yoloe_fusion_5shot.yaml) |
| GroundingDINO模型与描述 | [specific/zero_shot.yaml](specific/zero_shot.yaml) |
| PET-DINO视觉提示与文字检测 | [specific/pet_dino_5shot.yaml](specific/pet_dino_5shot.yaml)、[specific/pet_dino_text.yaml](specific/pet_dino_text.yaml)，独立环境及下载见[接入说明](../docs/guides/PET_DINO_WEDETECT.md) |
| WeDetect模型与中文类别名 | [specific/wedetect_text.yaml](specific/wedetect_text.yaml) |
| GroundingDINO-Base独立实验 | [specific/groundingdino_base.yaml](specific/groundingdino_base.yaml) |
| JSTY ONNX模型与检测参数 | [specific/jsty_hanging_object.yaml](specific/jsty_hanging_object.yaml) |
| MLLM模型、转发地址、提示词和实验组合 | 对应 `specific/mllm_*.yaml`；模型以各文件中的 `model` 为准 |

YOLOE文字、GroundingDINO、JSTY的模型参数已合并到各自专用配置中，不需要再追踪一层模型文件。YOLOE视觉模式保留一份共享JSON，便于修改一次后应用到全部shot配置。专用文件可以覆盖共享默认值；相对路径始终属于声明它的文件。

最终输出默认按 `evaluation.score_threshold` 过滤；`evaluation.score_thresholds: [0.5, 0.7]` 让同一次推理额外导出两档框和 P/R/F2 表，不增加模型调用。未填写列表时默认也是这两档。默认标注位于 `predictions/`，0.7 额外标注位于 `predictions_score_0.7/`（主阈值为 0.5 时）。标准 AP 始终使用完整候选框，详见 [结果输出说明](../docs/guides/EVALUATION_OUTPUT.md)。

三个 `mllm_*.yaml` 仍是独立完整配置，不继承 `qwen_mllm.yaml`。CC Switch协议、地址、模型名、API key读取规则等保持各文件原有设置；目录整理不会切换供应商或模型。可选凭据说明见 [API凭据](../docs/guides/API_CREDENTIALS.md)。

## 运行

在项目根目录选择一个专用配置：

```powershell
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/visual_5shot.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_text.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_fusion_5shot.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/zero_shot.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/mllm_ablation.yaml
```

添加 `--dry-run` 只检查配置、数据和缓存，不调用模型。`train` 表示推理并评估，不更新权重；`test` 表示仅推理。结果位置、命名、提示词记录和缓存规则沿用各实验当前设置，参见 [结果输出说明](../docs/guides/EVALUATION_OUTPUT.md) 和 [prompt对照说明](../docs/guides/PROMPT_RECORDS.md)。

`outputs/runs` 现在按“测试数据集 / 提示词版本 / 模型与 shot”保存；具体规则见 [结果目录说明](../docs/guides/RESULT_LAYOUT.md)。吊物轻度遮挡新版使用 [specific/mllm_val300_partial_occlusion.yaml](specific/mllm_val300_partial_occlusion.yaml)，明确为 Qwen3.8-27B、val300、5-shot，不会覆盖旧提示词的基线结果。

`specific/diaowu_val300_benchmark.yaml` 是批量任务表，使用 `scripts/run_val300_benchmark.py --prepare` 检查，`--run` 执行或恢复，`--status` 查看进度，`--stop` 停止。它包含8个本地方法和Qwen27B/Flash共23个参考输入组，统一300张数据；参考方法为5-shot，原生文字/监督方法保留其模式。结果说明见 val300批次说明（本地路径 `../outputs/runs/diaowu_val300/README.md`）。

旧命令中的 `configs/experiments/...`、`configs/legacy/...`、`configs/models/...`、`configs/datasets/...` 和更早的 `config/...` 仍通过入口路径映射解析到新文件；无需保留重复目录或占位文件。新命令统一使用上面的两类目录。历史运行记录保留当时的配置路径，不回写或改动原始实验结果。

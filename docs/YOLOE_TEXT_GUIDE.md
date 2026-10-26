# YOLOE 纯文字提示检测

在项目根目录执行：

```powershell
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_text.yaml
```

模型与文字在 `configs/specific/yoloe_text.yaml` 中直接配置。使用现有的 `yoloe-11l-seg.pt`，通过 MobileCLIP-B(LT) 编码文字，再执行 YOLOE 的 RepRTA 对齐；不读取参考图来生成特征，不使用视觉提示框、不做微调。待测图片仍正常经过图像检测网络。

初始文字与最新 GroundingDINO 实验逐字一致：

```text
a large hollow cylindrical load suspended from a yellow lifting hook by two taut slings spread to opposite sides, including the hook and slings.
```

这是一条目标类别描述，完整编码为一个类别，不会按逗号拆成多个类别。输出标注保持数据集的 `hanging object` 标签。MobileCLIP 上下文最多 77 tokens；当前描述为 33 tokens。超过限制会报错，避免静默截断导致 prompt 记录与实际输入不符。

`yoloe_text` 是独立 backend。原 `visual_5shot.yaml` 等配置仍然使用视觉提示。文字模式的 `max_reference_examples/crops/negatives` 必须全部为 0，不能配置 `positive_count`。保留参考池仅用于排除与已有支持集重复的测试图片，以便与其他方法使用相同测试队列；参考池不是模型输入。标签 description 只用于记录，实际编码的是 `yoloe_text.text_prompt`。

结果直接保存到 `outputs/runs/diaowu_99/文字提示_v1/yoloe-11l-seg_0shot/`，重跑使用 `_第2次` 等同级目录。包含 `prompt.txt`、`prompt.json`、`run.json`、`config.resolved.json`、`records.jsonl`、`predictions/`、`feedback/`、`metrics.json`、`evaluation.txt`，并自动更新 `outputs/runs/prompt_results.md`。终端继续输出 COCO AP/AR 和逐类统计。反馈保存检测框和分数，不生成自然语言框选理由。

模型保留候选分数阈值为 0.001，NMS IoU 为 0.7，输入大小为 640；评估 IoU 为 0.5、工作点 score_threshold 为 0.5，与最新保存的 GroundingDINO 评估一致。AP 使用候选框分数排序，工作点 precision/recall 则使用评估阈值，两者含义不同。

文字、检测权重、文字编码器权重、适配代码、上游检测代码、依赖版本均进入缓存签名。修改文字或权重会重新推理；只改评估阈值可复用已有框。`--force` 明确强制重新推理。`--dry-run` 只检查配置、输入和缓存，不加载任何文字编码器或检测模型。

## 在其他机器准备环境

先按 [YOLOE 视觉模式指南](YOLOE_GUIDE.md) 建立可运行的项目环境和检测权重，然后在项目根目录执行：

```powershell
.\.venv-yoloe\Scripts\python.exe -m pip install -r requirements-yoloe-text.txt
.\.venv-yoloe\Scripts\python.exe scripts/download_yoloe_text_encoder.py
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_text.yaml --dry-run
```

下载脚本从官方 Apple 地址获取 `weights/yoloe/mobileclip_blt.pt`，校验文件长度及官方 ETag 对应的 MD5，并记录 SHA256 和来源到同目录的 manifest 文件。首次下载约 599 MB；正常推理只读取本地文件，不自动安装依赖或下载模型。不需要 CC Switch 或 API key。

本项目固定的是 [THU-MIG YOLOE 源码](https://github.com/THU-MIG/yoloe)，需要 `.pt` 文字编码器，不使用新版 Ultralytics 文档中的 `.ts` 文件。编码遵循官方 `YOLOEModel.get_text_pe`：文字特征 L2 归一化，reshape，再调用检测头的 `get_tpe`，最后 `set_classes`。适配器显式传入文字编码器路径，避免依赖当前工作目录，也不改动固定的上游源码。

## 当前机器验证（2026-10-22）

已安装依赖、下载并校验文字编码器，使用上述配置实际运行全部 99 张固定测试图，99 张成功、0 张失败、0 张复用。描述、图片顺序、数据与真值指纹、评估参数均与当前保留的 GroundingDINO 精简描述记录一致。

实际输出位于 `outputs/runs/diaowu_99/文字提示_v1/yoloe-11l-seg_0shot/`：AP50 为 4.60%，AP50:95 为 2.43%。在既定工作点 score>=0.5 下没有框通过筛选，因此 Precision/Recall 为 0；这表示当前描述和模型的检测效果有限，不是推理请求失败。此处没有为了得到更好结果而调整文字或阈值。

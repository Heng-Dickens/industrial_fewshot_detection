# PET-DINO 与 WeDetect

源码、配置和统一检测适配器已加入当前项目。尚未下载新模型权重、安装独立模型环境或验证真实推理；离线测试验证的是项目接口。

| 后端 | 官方源码及本地位置 | 固定 commit | 已接入模式 |
|---|---|---|---|
| `pet_dino` | [PET_DINO](https://github.com/fuweifuvtoo/PET_DINO)，`third_party/pet_dino` | `7830a462f9a320f34293cce0aabdda1256d9dc15` | 独立参考图＋框的 Visual-G；英文 Text |
| `wedetect` | [WeDetect](https://github.com/WeChatCV/WeDetect)，`third_party/wedetect` | `42c34c6742a22f26280bb0e59f95a701baee1a32` | 官方纯 PyTorch 双塔检测，中文类别文字 |

上游源码未修改。`UPSTREAM.json` 记录来源、版本及下载包 SHA256。PET-DINO 的 LICENSE 随源码保留；WeDetect 上游未提供独立 LICENSE 文件，原始 README 中声明代码和模型使用 GPLv3。重新导入脚本为 `scripts/import_prompt_detectors.py`，不会覆盖已有源码。

WeDetect-Ref、WeDetect-Uni 的源码也已完整导入；当前统一后端只接普通 WeDetect，不能使用 Ref/Uni 权重。其他训练、检索等入口仍按原仓库说明执行。

## 实验配置

| 配置 | 输入 | 测试集 |
|---|---|---|
| `configs/specific/pet_dino_5shot.yaml` | 5 张独立参考原图及其框，Visual-G | diaowu_val300 |
| `configs/specific/pet_dino_text.yaml` | 与当前 GroundingDINO 相同的英文描述，0shot | diaowu_val300 |
| `configs/specific/wedetect_text.yaml` | 中文类别名“吊物”，0shot | diaowu_val300 |

```powershell
# 只检查配置、300 张数据、支持样本及缺失文件
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/pet_dino_5shot.yaml --dry-run
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/pet_dino_text.yaml --dry-run
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/wedetect_text.yaml --dry-run

# 准备好以下环境和权重后执行；初次可追加 --limit 1
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/pet_dino_5shot.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/pet_dino_text.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/wedetect_text.yaml
```

PET-DINO 的 `positive_count: 5` 使用已有 `5shot/seed_0.json` 中同一组支持样本。每张参考图通过官方接口提取该类 embedding，再按参考图取均值形成原型；不使用待测图的真值生成提示。`max_reference_*: 0` 关闭 MLLM 材料拼装，PET 的视觉样本单独准备。1shot/10shot 修改 `pet_dino.positive_count` 即可。

WeDetect 普通文字接口使用中文类别名，PET 文字模式使用英文描述，两者输入并不完全相同。实际文字和支持图/框保存在 `prompt.txt`、`prompt.json`，便于对照效果。

## 独立环境和权重

已补齐专用安装入口 `scripts/setup_pet_dino.ps1`、`scripts/setup_wedetect.ps1`。可先用 `-All -Plan` 检查步骤，再去掉 `-Plan` 执行。完整参数及分场景命令见[模型安装说明](MODEL_INSTALLATION.md)；下方手工命令保留供排查使用。

模型由配置的 `worker_python` 在独立子进程运行，不修改现有 `.venv-yoloe`。PET-DINO 使用自带的修改版 MMDetection；WeDetect 使用官方 `deploy/test_coco_pytorch.py`，不依赖 MMDetection/MMCV。

WeDetect 环境准备示例，使用 Python 3.10：

```powershell
.\.venv-yoloe\Scripts\python.exe -m venv .venv-wedetect
.\.venv-wedetect\Scripts\python.exe -m pip install -r requirements-wedetect.txt
.\.venv-wedetect\Scripts\python.exe scripts/download_prompt_detector_weights.py --backend wedetect
```

PET-DINO 还需要带编译算子的 MMCV，`mmcv-lite` 不够。以下为准备顺序示例，尚未在本机安装验证；MMCV wheel 必须匹配 Python、PyTorch 和 CUDA：

```powershell
.\.venv-yoloe\Scripts\python.exe -m venv .venv-pet_dino
.\.venv-pet_dino\Scripts\python.exe -m pip install "numpy==1.23.5" "torch==2.1.2" "torchvision==0.16.2"
.\.venv-pet_dino\Scripts\python.exe -m pip install openmim
.\.venv-pet_dino\Scripts\python.exe -m mim install "mmcv==2.1.0"
.\.venv-pet_dino\Scripts\python.exe -m pip install -r requirements-pet-dino.txt
.\.venv-pet_dino\Scripts\python.exe scripts/download_prompt_detector_weights.py --backend pet_dino
```

不要安装 PyPI 的 `mmdet` 替换 PET-DINO 自带源码。有 GPU 时先选择匹配驱动的 torch/torchvision 组合，再准备 MMCV，并在 YAML 中设置 `device: cuda:0`；当前默认 `cpu`。如 Windows 没有适配的 MMCV wheel，可在 Linux/WSL 中准备环境并运行整个实验入口，将 `worker_python` 改成相应 `.venv-pet_dino/bin/python`；Windows 入口不能直接启动 Linux ELF 解释器。

下载脚本使用 [PET-DINO 官方权重](https://huggingface.co/fuweifu/PET-DINO) 和 [WeDetect 官方权重](https://huggingface.co/fushh7/WeDetect)。PET 的 BERT 使用 `google-bert/bert-base-uncased`。WeDetect 的训练后文本权重已包含在检测 checkpoint 中，词表及配置从导入的官方源码复制。已有检测权重不会被覆盖。

```text
weights/pet_dino/pet_dino_swin_t.pth
weights/pet_dino/bert-base-uncased/...
weights/wedetect/wedetect_base.pth
weights/wedetect/xlm-roberta-base/...

outputs/runs/diaowu_val300/
  视觉提示/pet_dino_swin_t_5shot/
  文字提示_vN/pet_dino_swin_t_0shot/
  文字提示_vM/wedetect_base_0shot/
```

更换 WeDetect Tiny/Large 时，同时修改 `variant`、`weights`；Large 还需将 `language_model_dir` 改为 `xlm-roberta-large`。下载命令追加 `--variant tiny` 或 `--variant large`。

模型目录直接保存 `evaluation.txt`、`metrics.json`、`run.json`、`records.jsonl`、提示快照和标注。终端沿用 COCO AP/AR、分类 AP/Recall/F2、VOC 表。模型 `confidence: 0.001` 保留用于排序 AP 的候选框；导出和点指标使用 0.5，并额外给出 0.7 结果。两种检测器不产生自然语言逐框理由。

缓存指纹包含实际源码、适配器、权重、语言文件、参数及参考图和框。改文字/权重会重推理，改评估阈值可复用框。`local_model.missing_assets` 列出缺失环境/文件；`runtime_ready` 是准备检查，不代替模型加载和编译算子检查。新配置未加入正在运行的旧 31 组批次。

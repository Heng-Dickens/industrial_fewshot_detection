# 各模型独立安装与检查

在项目根目录运行。**不带操作开关只打印计划**；`-All` 执行源码、环境、依赖、权重及检查，追加 `-Plan` 只预览，绝不创建环境或执行下载/安装。

| 模型 | 专用入口 | 环境 | 支持模式 |
|---|---|---|---|
| YOLOE | `scripts/setup_yoloe.ps1` | `.venv-yoloe` | visual/text/fusion，v8/11系列s/m/l |
| GroundingDINO | `scripts/setup_groundingdino.ps1` | `.venv-groundingdino` | tiny/base，Transformers API |
| PET-DINO | `scripts/setup_pet_dino.ps1` | `.venv-pet_dino` | Swin-T视觉参考/文字 |
| WeDetect | `scripts/setup_wedetect.ps1` | `.venv-wedetect` | 普通tiny/base/large，官方纯PyTorch实现 |

## 分别安装

```powershell
# YOLOE视觉、文字、0.75视觉＋0.25文字融合
.\scripts\setup_yoloe.ps1 -Mode visual -Model yoloe-v8s-seg.pt -All
.\scripts\setup_yoloe.ps1 -Mode visual -Model yoloe-v8l-seg.pt -All
.\scripts\setup_yoloe.ps1 -Mode text -All
.\scripts\setup_yoloe.ps1 -Mode fusion -All

# GroundingDINO Tiny/Base
.\scripts\setup_groundingdino.ps1 -Variant tiny -All
.\scripts\setup_groundingdino.ps1 -Variant base -All

# PET-DINO：两种模式共享同一模型和权重
.\scripts\setup_pet_dino.ps1 -Mode visual -All
.\scripts\setup_pet_dino.ps1 -Mode text -Check

# WeDetect：默认Base，也可分别下载Tiny/Large
.\scripts\setup_wedetect.ps1 -Variant base -All
.\scripts\setup_wedetect.ps1 -Variant tiny -All
.\scripts\setup_wedetect.ps1 -Variant large -All
```

以上任一安装命令追加 `-Plan` 即可预览，例如：

```powershell
.\scripts\setup_wedetect.ps1 -Variant base -All -Plan
```

基础Python要求3.10或3.11，默认选择现有 `.venv-yoloe`，否则使用命令行的 `python`；可用 `-Python 'D:\路径\python.exe'` 指定。基础解释器只负责启动和创建环境，pip始终在所选模型环境运行。新环境不继承全局site-packages，已有环境不重建；YOLOE/GroundingDINO已有可导入的PyTorch会保留。重新安装仍可能补装或调整其他目标依赖，应在没有实验使用对应环境时执行。

YOLOE视觉默认v8s，文字/融合默认11L，后两者额外安装MobileCLIP依赖并下载文字编码器。GroundingDINO使用项目已适配的Transformers本地checkpoint API，不需要编译原GroundingDINO仓库。PET使用自带修改版MMDetection源码，不安装PyPI标准mmdet；WeDetect使用 `deploy/test_coco_pytorch.py`，不需要mmdet/MMCV和Ref分支的LLM依赖。

## CPU、CUDA与PET-DINO算子

默认 `-TorchBuild cpu`。YOLOE、GroundingDINO、WeDetect支持cpu/cu118/cu121/cu124；PET支持cpu/cu118/cu121。

```powershell
.\scripts\setup_wedetect.ps1 -Variant base -TorchBuild cu124 -All
.\scripts\setup_pet_dino.ps1 -TorchBuild cu121 -All
```

新环境采用以下组合：

| 模型 | torch / torchvision | 其他关键版本 |
|---|---|---|
| YOLOE | 2.5.1 / 0.20.1 | 文字路径open-clip-torch 2.24.0、timm 0.9.5 |
| GroundingDINO | 2.5.1 / 0.20.1 | Transformers 4.46.3 |
| PET-DINO | 2.1.2 / 0.16.2 | NumPy 1.23.5、MMCV 2.1.0、Transformers 4.46.3 |
| WeDetect | 2.5.1 / 0.20.1 | Transformers 4.57.1 |

PET的编译版MMCV默认只安装匹配的wheel，并同时约束NumPy。没有对应Python/平台/CUDA的wheel会明确报错停止，不隐式源码编译，也不使用mmcv-lite代替。支持指定已准备好的匹配wheel：

```powershell
.\scripts\setup_pet_dino.ps1 -TorchBuild cu121 -MMCVWheel 'D:\packages\mmcv-2.1.0-cp310-cp310-win_amd64.whl' -InstallDependencies -Check
```

具体兼容性需要本机安装及算子检查确认。官方说明：[MMCV安装](https://github.com/open-mmlab/mmcv/blob/v2.1.0/docs/en/get_started/installation.md)、[PyTorch历史版本](https://pytorch.org/get-started/previous-versions/)。CUDA推理还需要兼容驱动、硬件和YAML的 `device: cuda:0`；安装构建不会自动修改实验配置。

## 单独执行及检查

| 参数 | 执行范围 |
|---|---|
| `-ImportSource` | 准备指定模型固定版本源码；GroundingDINO实现由依赖包提供，无此开关 |
| `-InstallDependencies` | 准备源码、创建或复用目标环境、安装依赖 |
| `-DownloadWeights` | 创建或复用目标环境、准备轻量下载客户端、下载所选模型，不安装整套推理依赖 |
| `-Check` | 离线核对文件、依赖导入、源码来源、torchvision NMS及PET编译算子导入，失败退出非零 |
| `-Check -LoadModel` | 检查通过后实际加载模型，不做图像推理或精度评估 |
| `-All -Plan` | 只打印全流程计划，无文件、安装或网络操作 |
| `-WeightsEndpoint URL` | 本轮HF_ENDPOINT，如 `https://hf-mirror.com`，不改全局配置 |

```powershell
.\scripts\setup_yoloe.ps1 -Mode text -Check -LoadModel
.\scripts\setup_groundingdino.ps1 -Variant base -Check -LoadModel
.\scripts\setup_pet_dino.ps1 -Mode text -Check -LoadModel
.\scripts\setup_wedetect.ps1 -Variant base -Check -LoadModel
```

PET的加载检查使用共同模型的Text路径，不借用待测图真值生成视觉提示。Visual-G的支持图编码仍需实际单图实验验证。所有模型的加载检查都不代表精度验收。

源码固定revision、已有不同版本拒绝覆盖。下载记录官方revision，YOLOE/PET/WeDetect检测权重核对官方大小和SHA256。任一步失败立即停止后续步骤。脚本不删除环境或历史结果，不改API/CC Switch设置。

## 安装后的执行命令

```powershell
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/visual_5shot.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_text.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/yoloe_fusion_5shot.yaml
.\.venv-groundingdino\Scripts\python.exe main.py --config configs/specific/zero_shot.yaml
.\.venv-groundingdino\Scripts\python.exe main.py --config configs/specific/groundingdino_base.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/pet_dino_5shot.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/pet_dino_text.yaml
.\.venv-yoloe\Scripts\python.exe main.py --config configs/specific/wedetect_text.yaml
```

初次实测可追加 `--limit 1`。PET/WeDetect主流程使用项目环境，YAML的 `worker_python` 启动各自模型环境；worker环境不包含整个项目的共享运行依赖。

安装其他型号不会自动修改YAML。WeDetect Large还需将配置的 `variant`、`weights`、`language_model_dir` 同时调整，使用 `xlm-roberta-large`；Tiny/Base使用 `xlm-roberta-base`。详细输入规则见[PET/WeDetect说明](PET_DINO_WEDETECT.md)、[YOLOE文字指南](../YOLOE_TEXT_GUIDE.md)、[融合指南](../YOLOE_FUSION_GUIDE.md)。

旧 `tool/setup_yoloe.ps1` 和YOLOE的 `-InstallDependencies`/`-DownloadWeights` 用法保留；旧 `scripts/download_groundingdino.ps1` 已转发到下载专用流程，并新增Base支持。

Linux可直接使用同一Python实现：

```bash
python scripts/install_models.py --backend wedetect --variant base --all --plan
python scripts/install_models.py --backend pet_dino --mode visual --torch-build cu121 --all
```

Linux环境为 `bin/python`，PET/WeDetect的配置 `worker_python` 也需按系统调整。数据场景由实验配置决定，切换吊物/叉车数据无需重复安装相同模型依赖，多类别推理适配需另行验证。

本次仅整理安装脚本并进行离线验证，没有执行新的大规模安装、下载或实验。

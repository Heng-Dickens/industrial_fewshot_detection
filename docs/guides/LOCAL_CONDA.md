# 本机 Linux / Conda 运行与测试

项目目录：`/home/yuheng/Work/industrial_fewshot_detection`。
本机使用 CentOS 7、TITAN V；Python 统一采用 3.10，模型依赖分别隔离。

| Conda 环境 | 用途 | PyTorch / CUDA |
|---|---|---|
| `industrial-fewshot` | 主流程、MLLM 客户端、YOLOE 视觉/文字/融合、JSTY ONNX | 2.5.1 / 12.4 |
| `industrial-groundingdino` | Transformers GroundingDINO | 2.5.1 / 12.4 |
| `industrial-pet-dino` | PET-DINO worker、MMCV 2.1.0 编译算子 | 2.1.2 / 12.1 |
| `industrial-wedetect` | WeDetect worker | 2.5.1 / 12.4 |

2026-10-26 本机验证结果：四个环境的 `pip check`、模型接口导入，以及 8 张 TITAN V
的 CUDA 矩阵运算/NMS 均通过；PET-DINO 的 MMCV deformable attention 在全部 8 张 GPU
上执行通过。项目现有测试共 353 项，352 项通过、1 项跳过。环境初次检查记录位于
`outputs/environment_setup/*-runtime-check.log` 和 `outputs/environment_setup/project-tests.log`。
随后下载了 PET-DINO Swin-T 和 WeDetect Base 官方权重及语言文件；两个检测权重均通过
官方 SHA256 校验，两个模型均通过离线实际加载检查。检查记录位于
`outputs/environment_setup/{pet_dino,wedetect}-model-load.log`。

2026-10-27：PET-DINO 文字模式和 5-shot Visual-G 均通过本机 GPU 真实单图推理，
每种模式成功 1 张、失败 0 张、缓存复用 0 张。5-shot 使用冻结清单的五张独立参考图。
结果位于 `outputs/runtime/pet_dino_smoke_20261027/{text,5shot}/`，日志位于
`outputs/environment_setup/pet-dino-{text,5shot}-smoke.log`。这只是运行验证；该张图在
0.5 阈值下两种模式均漏检，不能据此判断完整数据集效果。WeDetect 也已完成五类特种车辆对比。

PET-DINO 已准备好本机配置 `configs/specific/pet_dino_5shot.local.yaml` 和
`configs/specific/pet_dino_text.local.yaml`。前者使用只调整路径的冻结支持清单副本
`data/references/diaowu/linux.local/5shot/seed_0.json`；原始清单、样本顺序和内容哈希保持不变。
在项目目录使用主环境运行，worker 会自动使用 PET-DINO 环境：

```bash
conda activate industrial-fewshot
python main.py --config configs/specific/pet_dino_5shot.local.yaml --limit 1
python main.py --config configs/specific/pet_dino_text.local.yaml --limit 1
```

删除 `--limit 1` 即对 `diaowu_val300` 完整验证集运行推理与评估。上面的 smoke 记录仅对应单图验证。

两个完整验证集的 1/5/10-shot Visual-G 批量入口如下。主程序使用 `industrial-fewshot`，
模型 worker 使用 `industrial-pet-dino`，不调用远程 API、不训练模型。

```bash
python scripts/run_pet_dino_shots.py --phase prepare
python scripts/run_pet_dino_shots.py --phase complete
python scripts/run_pet_dino_shots.py --phase status
```

默认使用 8 张 GPU；可通过 `--devices cuda:0,cuda:1` 限定。六组结果分布在
`outputs/runs/{diaowu_val300,chache_val300}/PET-DINO/`；chache 的 shot 数为每类数量。
原始支持清单保持不变，本机派生清单仅覆盖路径，并校验原有样本 ID、图像及标注哈希。
重复运行同一计划会跳过已完成任务，部分成功记录按输入指纹复用。

2026-10-27 完整批次已完成：两个场景各 300 张，chache 覆盖全部五类；六组实验、
18 项单类任务、5400 条成功记录，失败数为 0。最终数据/支持数量/输出链接核验保存在
`outputs/runtime/pet_dino_shots_20261027/verification.json`。

## 环境检查（不需要权重、数据或 API Key）

```bash
source /home/yuheng/miniconda3/etc/profile.d/conda.sh
conda activate industrial-fewshot
cd /home/yuheng/Work/industrial_fewshot_detection

# 检查四个环境的依赖、全部可见 GPU 的矩阵运算/NMS，以及 PET 的 CUDA attention。
python scripts/setup_conda_linux.py --verify

# 检查主流程依赖、命令行入口并运行仓库已有单元/集成测试。
python scripts/setup_conda_linux.py --test-project

# 单独运行现有测试：
python -m unittest discover -s tests -t . -q
```

单独检查某个模型：`python scripts/setup_conda_linux.py --verify --backend pet_dino`。
可选后端为 `yoloe`、`groundingdino`、`pet_dino`、`wedetect`。
检查日志和实际安装版本位于 `outputs/environment_setup/`；安装过程查看
`python scripts/setup_conda_linux.py --status`。

## 实际模型推理前的准备

环境检查不会下载模型或运行真实图像推理。本仓库不附带模型权重、测试数据、
支持集或 API Key。模型权重可在对应环境中使用原有下载脚本准备：

```bash
conda activate industrial-fewshot
python scripts/download_yoloe_weights.py --model yoloe-v8l-seg.pt
python scripts/download_yoloe_weights.py --model yoloe-11l-seg.pt
python scripts/download_yoloe_text_encoder.py

conda activate industrial-groundingdino
python scripts/download_groundingdino.py --variant tiny
# groundingdino_base.yaml 使用 base，按需另下：
# python scripts/download_groundingdino.py --variant base

conda activate industrial-pet-dino
python scripts/download_prompt_detector_weights.py --backend pet_dino

conda activate industrial-wedetect
python scripts/download_prompt_detector_weights.py --backend wedetect --variant base
```

JSTY 使用项目外的私有 ONNX 文件；MLLM 需要在 `.env` 或进程环境中设置与配置匹配的
API 凭据，并填写服务器可访问的接口地址。安装环境不更改已有服务或凭据。

原实验配置含 `D:/...` 图片路径和 Windows worker 路径。实际运行需新建
`configs/specific/<实验名称>.local.yaml`，使用 `extends` 继承原配置，覆盖本机路径。
不要直接修改公共实验的原数据划分。需要覆盖的数据项包括：

- `dataset.image_dir`、`dataset.ground_truth_dir`、`dataset.manifest_file`；
- 视觉模式需要 `reference_image_dir` 和 `support_manifest_dir`；
- 使用 GPU 时，将相应模型配置的 `device` 改为 `cuda:0`；
- PET-DINO 的 `pet_dino.worker_python` 改为
  `/home/yuheng/miniconda3/envs/industrial-pet-dino/bin/python`；
- WeDetect 的 `wedetect.worker_python` 改为
  `/home/yuheng/miniconda3/envs/industrial-wedetect/bin/python`。

仅推理新图片、不使用已有 manifest/支持集时，可参考下面的 GroundingDINO 单图配置。
保存为 `configs/specific/groundingdino_smoke.local.yaml`，将 `/实际/测试图片目录` 换成
已有图片目录；相应模型权重需先下载：

```yaml
extends: zero_shot.yaml
mode: test
dataset:
  image_dir: /实际/测试图片目录
  ground_truth_dir: null
  manifest_file: null
  recursive: false
reference_image_dir: null
support_manifest_dir: null
groundingdino:
  device: cuda:0
  exclude_reference_overlap: false
```

```bash
conda activate industrial-groundingdino
python main.py --config configs/specific/groundingdino_smoke.local.yaml --dry-run
python main.py --config configs/specific/groundingdino_smoke.local.yaml --limit 1
```

PET-DINO/WeDetect 用 `industrial-fewshot` 运行主程序，配置中的 `worker_python` 自动调用各自
环境。先用 `--dry-run` 检查数据与文件，再用 `--limit 1` 验证真实推理。已有真值且需要检测
指标时使用 `mode: train`；项目中此名称表示推理加评估，不会训练模型。

## 安装复现

```bash
# 固定版本的官方源码，不会覆盖其他版本的既有目录。
python scripts/import_prompt_detectors.py --backend all

# 在独立 Conda 环境中安装；不会向 base 安装模型包。
/home/yuheng/miniconda3/bin/python scripts/setup_conda_linux.py
```

安装器为 CentOS 7 使用 Conda 的 ONNX Runtime 1.17.3，避免新版 PyPI wheel 的 glibc
要求；保留 NumPy 1.x、OpenCV <4.12 和 setuptools <81，兼容原模型代码。
PET-DINO 必须使用编译版 MMCV，不能用 mmcv-lite 替代。
PyTorch 构建参见[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)，
MMCV wheel 参见[官方 cu121/torch2.1 索引](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)。

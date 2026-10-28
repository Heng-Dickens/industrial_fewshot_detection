# 代码仓库与服务器部署

项目根目录是独立 Git 仓库，不使用上一级 `D:\Learn\VLM` 的仓库。所有命令均在项目根目录执行；已配置 `origin` 的机器无需重复添加远程地址。

## 同步范围

| 进入 Git | 单独准备，不进入 Git |
|---|---|
| `main.py`、`src/`（含全部文字 prompt） | `data/` 全部图片、标注、manifest、支持集 |
| `configs/common/`、`configs/specific/` 的公共配置及示例 | `outputs/runtime/`、`outputs/environment_setup/`、缓存、日志、旧归档及根目录 `runs/` |
| `outputs/runs/` 所有算法的报告、总表、指标、提示词和运行配置快照 | 逐图 `predictions*/`、`feedback/`、`records.jsonl`、含原始记录的 `summary.json` |
| `outputs/comparisons/` 的专题报告、诊断摘要、索引和对比图 | 原始数据图片与完整逐图推理产物 |
| `scripts/` 安装、下载、数据准备、推理和评估入口 | `weights/` 模型和语言编码器文件 |
| `requirements*.txt`、`pyproject.toml` | `third_party/`，按固定 commit 在服务器重新导入 |
| 测试、兼容工具、示例、操作文档 | `.venv*/`、`.planning/`、IDE 设置、临时文件 |
| `.env.example` 空模板 | 真实 `.env`、`*.local.yaml/json`、私钥及历史结果分析文档 |

忽略规则只控制 Git，不删除本地资产。根目录使用显式允许列表；新建根目录文件如需同步，必须同时在 `.gitignore` 中放行。不要用 `git add -f` 强行加入数据或密钥。

结果规则适用于全部后端，包括 MLLM/Qwen、PET-DINO、GroundingDINO、WeDetect、YOLOE，以及按历史提示词版本保存的实验目录。JSON采用文件名允许列表：`metrics.json`、`run.json`、`config.resolved.json`、`prompt*.json`、`dataset.json`、`results_summary.json`、`method.json`、`experiment.json`、`experiment_results.json`。已有结果不会因修改 `.gitignore` 自动上传，仍需暂存、提交和推送：

```bash
git add .gitignore outputs/README.md outputs/runs/ outputs/comparisons/
git diff --cached --stat
git commit -m "Sync results from all detector backends"
git push origin main
```

提交前同时检查已有暂存变更。其他机器执行 `git pull --ff-only` 后即可查看报告和总表；若要重算指标，还需单独准备图片、真值和完整预测记录。

第三方源码由 `scripts/import_prompt_detectors.py` 固定版本：YOLOE `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`、PET-DINO `7830a462f9a320f34293cce0aabdda1256d9dc15`、WeDetect `42c34c6742a22f26280bb0e59f95a701baee1a32`。安装时保留上游许可证，下载的源码不反向提交到本仓库。

## 本机首次推送及后续更新

先在 GitHub、GitLab 或其他 Git 服务建立空仓库，避免预先创建不同的 README 历史。下面的 `<REMOTE_URL>` 替换为实际 SSH/HTTPS Git 地址；只设置一次 origin。

```powershell
Set-Location D:\Learn\VLM\industrial_fewshot_detection
git rev-parse --show-toplevel
git status --short
git ls-files data runs weights third_party .env
# 上一条应没有输出。
git remote add origin <REMOTE_URL>
git push -u origin main
```

后续改代码：

```powershell
git add .
git diff --cached --stat
git diff --cached
git commit -m "Describe the inference change"
git push
```

初次本地提交如提示身份未配置，可仅对本仓库设置 `git config user.name "你的名字"` 和 `git config user.email "你的邮箱"`，再执行 `git commit -m "Initialize inference project"`。不要为了初始化本项目修改全局身份或上级仓库。

## Linux 服务器克隆与基础环境

```bash
git clone <REMOTE_URL> industrial_fewshot_detection
cd industrial_fewshot_detection
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python main.py --help
```

建议用 Python 3.10/3.11。只运行 MLLM 客户端时基础环境即可；本地模型按需安装独立环境。Windows 的 `.venv` 不能复制到 Linux 使用。

## 本地模型安装

下列命令会下载源码、依赖和权重；只执行需要的模型。执行前可追加 `--plan` 查看完整步骤。`--torch-build` 按服务器驱动选择，CPU 使用 `cpu`；并非所有驱动都能用示例 CUDA 版本。

```bash
# PET-DINO：Swin-T，视觉5shot；包含编译算子的 MMCV，不能换成 mmcv-lite。
.venv/bin/python scripts/install_models.py --backend pet_dino --mode visual --torch-build cu121 --all

# WeDetect：普通 Base 文字模型，非 WeDetect-Ref/Uni。
.venv/bin/python scripts/install_models.py --backend wedetect --variant base --torch-build cu124 --all

# 按需安装其他后端；无需全部执行。
.venv/bin/python scripts/install_models.py --backend yoloe --mode visual --torch-build cu124 --all
.venv/bin/python scripts/install_models.py --backend groundingdino --variant base --torch-build cu124 --all
```

安装脚本按系统使用 `.venv-<backend>/bin/python`。若 PET-DINO 没有匹配的 MMCV wheel，按[安装说明](MODEL_INSTALLATION.md)提供 `--mmcv-wheel /path/to/matching.whl`。JSTY 私有 ONNX 权重需单独放到服务器，没有公开下载脚本。

## 数据、模型及本地配置

Git 传输代码及选定的实验报告和指标。服务器需另行准备测试图片、对应 LabelMe 真值、测试 manifest，以及视觉提示用的原图、框标注和 support manifests。保留相同目录结构和文件内容才能复现实验；manifest/支持集也是数据，不随 Git 分发。可以由共享磁盘提供，也可以之后单独传输。

新增了路径模板和两个 Linux 模型示例，保留现有 `common` / `specific` 两层结构：

```bash
cp configs/common/server_paths.example.yaml configs/common/server_paths.local.yaml
cp configs/specific/server_pet_dino_5shot.example.yaml configs/specific/server_pet_dino_5shot.local.yaml
cp configs/specific/server_wedetect_text.example.yaml configs/specific/server_wedetect_text.local.yaml
```

编辑 `server_paths.local.yaml` 中所有 `/srv/...` 路径，并将两个模型 `.local.yaml` 的最后一个 `extends` 从 `server_paths.example.yaml` 改为 `server_paths.local.yaml`。`reference_image_dir` 指向准备后的支持图/标注根目录，`support_manifest_dir` 指向对应的1/5/10shot清单目录，两者可以不同。示例保持吊物单类别定义，不能直接把它们用于五类叉车数据；本项目当前 PET-DINO / WeDetect 统一适配器仍是单类别接口。

服务器没有 CUDA 时，将示例 `device` 改为 `cpu`。PET-DINO / WeDetect 的 `worker_python` 必须使用 Linux 的 `bin/python`，不能沿用原配置中的 Windows `Scripts/python.exe`。

基础环境负责统一流程及评估，模型环境在独立子进程执行：

```bash
.venv/bin/python main.py --config configs/specific/server_pet_dino_5shot.local.yaml --dry-run
.venv/bin/python main.py --config configs/specific/server_wedetect_text.local.yaml --dry-run

# 先做单图验证，再移除 --limit 1 跑完整测试集。
.venv/bin/python main.py --config configs/specific/server_pet_dino_5shot.local.yaml --limit 1
.venv/bin/python main.py --config configs/specific/server_wedetect_text.local.yaml --limit 1
```

`--dry-run` 仍需要图片和支持集；检查通过不代表编译算子、CUDA和模型加载一定通过。原实验配置中的 `D:/...` 路径必须通过本地配置覆盖。仅传 `--image-dir` 不会同步修改真值、manifest或参考路径。

## MLLM 服务与凭据

本机配置的 `http://127.0.0.1:15721/v1` 指服务器自身，不会连接到 Windows 上的 CC Switch。服务器使用 MLLM 时，在被忽略的 `.local.yaml` 中单独配置服务器可达的 `base_url`、匹配的 `api_protocol`、实际模型和凭据环境变量。不要复制本机全局 CC Switch 或 Codex 配置。

`.env.example` 可以提交，真实密钥仅写服务器 `.env` 或进程环境变量；让配置的 `api_key_env` 与变量名一致。详细格式见[凭据说明](API_CREDENTIALS.md)。本地 PET-DINO、WeDetect、YOLOE、GroundingDINO 推理不需要 MLLM API Key。

## 服务器更新

```bash
git pull --ff-only
.venv/bin/python -m pip install -e .
```

服务器使用 `.local.yaml` 保存路径和设备配置，更新公共配置时这些本地文件不会被 Git 覆盖。Git 更新会同步已提交的实验报告和指标；被忽略的原始预测、数据和缓存需单独传输或在本机重新生成。不要在服务器执行 `git clean -fdx`，它会删除被忽略的数据、权重及环境。

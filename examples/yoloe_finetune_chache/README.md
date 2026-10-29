# YOLOE 两场景少样本检测微调样例

**2026-10-29当前入口：[两场景100张验证集配置](VAL100_CONFIGURATION.md)。** 已接入 `data/raw/chache_val100` 与 `data/raw/diaowu_val100`，默认沿用10shot，原300张保留测试。叉车使用 `config.json`，吊物使用 `config_diaowu.json`；下方早期5shot命令仅说明方法，实际执行以新文档为准，避免准备5shot却用10shot配置训练。

**三组对照及逐条执行命令见 [COMPARISON_PLAN.md](COMPARISON_PLAN.md)**：YOLOE固定检测头微调、YOLOE视觉提示编码器SAVPE微调、YOLOv8-L预训练权重微调。统一使用 `train.py --method yoloe-head / yoloe-vp / yolov8`，默认只打印计划。下面的编号解释以默认 `yoloe-head` 为例。

本目录为编写好的样例，**本次未运行数据转换、模型加载、测试或训练**，尚未经过运行验证。
适配项目内 `third_party/yoloe` 固定版本 `40cd606cabdbe2b566d6f14a6b162c89206e9a1b`。
使用现有固定支持集；当前叉车每类10张共50张、吊物10张，开发验证集均为新提供的100张。

这是更新权重的检测训练。项目原 `main.py --mode train` 仅推理并评估，不能用于本样例。
本样例只覆盖数据准备与 FP32 微调；未实现断点恢复、DDP、统一评估适配或 PTQ 部署流程。
不修改上游源码、原始参考数据、现有实验结果。

## 文件与每一步的作用

| 文件 | 作用 |
|---|---|
| [config.json](config.json) | 数据/源码/权重路径、固定类别顺序、语义类别描述及训练参数 |
| [config_diaowu.json](config_diaowu.json) | 继承公共训练参数，覆盖吊物单类和对应100张验证集路径 |
| [finetune_config.py](finetune_config.py) | 两场景配置加载、继承与类别顺序检查 |
| [prepare_data.py](prepare_data.py) | 选择原有5shot清单，校验原图与标注哈希，转换YOLO框，准备独立开发验证集及清单/YAML |
| [train.py](train.py) | 加载原始权重，生成类别嵌入，建立检测模型，设置冻结层，保存参数后训练 |
| [vp_trainer.py](vp_trainer.py) | SAVPE训练适配和仅从训练支持集生成视觉原型的开发验证器，仅yoloe-vp执行时导入 |

1. 数据准备读取 `data/references/chache/<类别>/5shot/seed_0.json`，保持提示基线的实际样本一致。
2. 通过参考池 `manifest.json` 找到原图及原标注，将 LabelMe rectangle/polygon 转成归一化包围框。
   ID顺序为 `Chache1, Chache2, Chache3, Chache4, Qianyinche`，分别对应0～4。
3. 开发验证集由 `--dev-dir` 指定：每张原图旁有同名 LabelMe JSON，覆盖五类，允许显式空标注负样本。
   缺标注、未知类别、越界框直接报错；不把缺少标注当成负样本。
4. 检查训练、开发验证和原300张评测图片的内容哈希，拒绝重复或交叉。
   原评测manifest仅读取图片哈希用于排除，不用其真值训练或选权重。
5. 只有 `prepare_data.py --write` 才复制原图、写入YOLO标签，再生成 `train.txt`、`val.txt`、
   `dataset.yaml` 和 `preparation.json`。源文件不改；目标目录已存在时拒绝覆盖。
6. 训练脚本默认只打印计划。只有显式 `--execute` 才导入模型库、初始化模型和创建训练目录。
7. 正式执行时，强制使用项目内 YOLOE；用原始分割checkpoint的语义头编码五类英文描述，
   得到类别嵌入。数据集类别名保持原来的五个ID名称，不把 `Chache1` 本身当作语义描述。
8. 建立同尺度 `yoloe-v8l` 检测模型，迁入原 `yoloe-v8l-seg.pt` 的匹配权重；没有mask训练任务。
9. 默认 `head` 策略复用官方 `train_pe.py` 的冻结方式，只训练融合后的三个分类输出卷积。
   `--strategy full` 才进行全参数微调；DFL固定层仍按上游规则冻结。
   `YOLOEPETrainer` 会把类别嵌入融合为固定类别检测头，样例不会训练SAVPE视觉提示编码器。
10. 保存类别嵌入、输入文件哈希、训练计划及实际可训练参数清单；通过独立开发验证集保存 `best.pt`。

## 接下来按顺序做

以下命令均为**供你之后手动执行的步骤，本次没有执行**。命令从项目根目录运行。

### 1. 先检查图片、标注和开发验证集

准备一份独立的开发验证集，例如：

```text
/实际路径/chache_dev/
  camera_a/frame_001.jpg
  camera_a/frame_001.json
  camera_b/frame_002.jpg
  camera_b/frame_002.json
```

每张图片中五类目标应完整标注，不要只标出作为提示的那个框。
转换脚本无法发现画面中漏标的目标，需要人工检查。
哈希只能防完全相同图片，不能防同视频相邻帧；你还需要按相机/视频/序列检查来源隔离。
不要自动从原300张评测集抽取开发验证图，或把同一张图放进train和val。
如果从现有每类10张池中拿出2张做验证，训练最多是8shot，不能仍记为10shot。

### 2. 选择已安装YOLOE的环境，先查看数据准备计划

```bash
conda activate industrial-fewshot
python examples/yoloe_finetune_chache/prepare_data.py \
  --shot 5 --support-seed 0 \
  --dev-dir /实际路径/chache_dev
```

这一步读取图片和标注并检查哈希、尺寸、类别及重叠，不复制图片、不写标签、不加载YOLOE。
应显示train=25张及你实际提供的val张数；数据存在问题时先修复，不继续训练。

确认后，再加 `--write`：

```bash
python examples/yoloe_finetune_chache/prepare_data.py \
  --shot 5 --support-seed 0 \
  --dev-dir /实际路径/chache_dev \
  --write
```

输出结构：

```text
data/finetune/chache_5shot/
  images/train/   images/val/
  labels/train/   labels/val/
  train.txt       val.txt
  dataset.yaml
  preparation.json
```

图像按内容哈希命名，避免同名覆盖。清单与YAML使用当前机器的实际路径；跨机器时重新准备数据。
修改或重新准备时使用新的 `--output`，不要覆盖旧版本；训练时通过 `--data` 指定新YAML。

### 3. 阅读配置与训练计划

```bash
python examples/yoloe_finetune_chache/train.py
```

默认命令只打印JSON计划，连torch和Ultralytics也不会导入。
先检查 [config.json](config.json)：

| 设置 | 当前样例值与原因 |
|---|---|
| 类别描述 | 五类英文描述是按现有类别定义写的初始化文本，可人工修订，必须维持类别顺序 |
| 模型 | v8-L，沿用原基线权重；每次独立实验重新从预训练初始化 |
| 精度 | FP32，`amp=false`、`half=false` |
| 输入 | 640，与原基线一致 |
| 学习率 | AdamW，`lr0=1e-4`，是待验证的起点，不是已验证最优参数 |
| 轮数 | 保留你当前配置的100轮；独立开发验证集挑选best，patience=10 |
| 增强 | 关闭Mosaic/MixUp/Copy-Paste，轻度平移缩放 |
| 小样本更新 | `nbs=batch`，避免隐式按64张累积；关闭上游最少100步的warmup |
| GPU/batch | 不替你选择，执行时必须明确传入；本样例只支持单卡 |

如果缺少YOLOE权重或文本编码器，可在需要时自行执行已有下载脚本：

```bash
python scripts/download_yoloe_weights.py --model yoloe-v8l-seg.pt
python scripts/download_yoloe_text_encoder.py
```

微调需要环境中的MobileCLIP/CLIP依赖，不能仅以之前视觉提示推理成功判定文本初始化环境就绪。
样例编码文本时临时从 `weights/yoloe/` 查找固定版本要求的 `mobileclip_blt.pt`，之后恢复工作目录。
本次没有下载、安装或更新依赖。

### 4. 你决定开始时，先运行2轮检查

先自行用 `nvidia-smi` 确认空闲GPU，再把下面的 `GPU编号` 和 `批量大小` 替换为实际数值。
FP32 v8-L显存需求需实测；batch不能超过本次训练图片数。此处不预设占用哪张卡。

```bash
python examples/yoloe_finetune_chache/train.py \
  --execute --device GPU编号 --batch 批量大小 \
  --epochs 2 --name head_5shot_seed0_smoke_v1
```

这条命令会真正初始化、训练和验证。检查日志：加载的是预期权重、类别数=5、训练/验证数量正确、
损失正常、实际可训练层仅为三个分类输出卷积，并且训练最终正常退出。
样例未验证运行成功；如果本地依赖或上游接口有问题，先修复检查流程，不直接扩大训练。

### 5. 再从原始预训练权重新开正式实验

```bash
python examples/yoloe_finetune_chache/train.py \
  --execute --device GPU编号 --batch 批量大小 \
  --name head_5shot_seed0_v1
```

新名字很重要：它与2轮检查分别保存，不接着检查权重训练，也不覆盖已有结果。
训练输出位于 `outputs/finetuning/chache_yoloe/<name>/`：

- `finetune_config.json`：数据/权重/编码器哈希、冻结策略、配置及运行状态。
- `class_embeddings.pt`：仅yoloe-head保存的、按类别ID排序的初始化文本嵌入。
- `trainable_parameters.json`：训练器实际创建并冻结模型后的可训练张量名称。
- `args.yaml`、`results.csv`、`weights/best.pt`、`weights/last.pt`：上游训练产物。

如果要做全参数对照，重新从原始预训练权重启动新实验，显式使用
`--strategy full --name full_5shot_seed0_v1`，并通过独立开发验证集选择合适的学习率和轮数。
这仍是FP32训练，不是QAT。

### 6. 训练结束后接入固定类别评估，再汇总

本样例不执行现有300张的最终评估，也不把训练器内部验证成绩冒充总表里的测试成绩。
下一项实现工作是给 `best.pt` 增加固定五类检测推理入口，导出本项目的 records/metrics 格式。
不要直接把best.pt塞进原视觉提示配置：PETrainer已经融合类别，不应再次提取视觉原型/set_classes。

最终评估保持原300张清单和标签不变，报告 AP50、AP50:95、各类AP、0.5/0.7阈值的P/R/F2及TP/FP/FN。
显式对齐预处理、候选下限、NMS、每类检测数量与类别映射。
现有叉车提示基线是五次单类查询后合并，本微调是五类联合检测；结果对比中必须说明这项差异，
或补充流程一致的未微调基线后分析训练收益。

建议正式评估目录为 `outputs/runs/chache_val300/YOLOE-Finetuned/<实验>/`，
正确生成 `run.json`、`metrics.json`、`evaluation.txt` 后，再用
`python scripts/summarize_results.py --dataset chache_val300` 更新总表。
后续有部署需求时再独立安排FP32导出与PTQ INT8及同集配对评估。

## Git保存范围

本样例的源码、配置和说明位于 `examples/`，可随代码仓库同步。
转换后的图片/标签在 `data/`，训练产物在 `outputs/finetuning/`，按当前规则留在本机。
最终统一评估的报告及指标放进 `outputs/runs/` 后才进入现有结果同步范围；权重需单独保存。

官方参考：[检测头冻结示例](../../third_party/yoloe/train_pe.py)、
[本地PE检测训练器](../../third_party/yoloe/ultralytics/models/yolo/yoloe/train_pe.py)。

# YOLOE两种微调与YOLOv8-L预训练微调对照方案

**2026-10-29数据配置更新：[两场景100张验证集执行步骤](VAL100_CONFIGURATION.md)。** 当前默认10shot，叉车训练50张/验证100张/测试300张，吊物训练10张/验证100张/测试300张。两场景共用三种方法，吊物配置为 `config_diaowu.json`；训练器与SAVPE验证器已支持单类。下方5shot命令保留为原方案示例，按当前配置执行时请使用新文档的命令。

状态：方案、统一入口和SAVPE适配代码已编写；**没有运行数据转换、模型加载、样例、测试或训练**。
以下命令是后续执行步骤，不能视为已验证可运行。当前样例仅支持单卡FP32，不支持DDP或断点恢复。

## 1. 三个实验回答什么问题

| 代号 / --method | 初始化权重 | 更新参数 | 推理是否提供参考图 | 研究问题 |
|---|---|---|---|---|
| A：`yoloe-head` | 原始 `yoloe-v8l-seg.pt`，迁入检测结构 | 融合类别后的三个分类输出卷积 | 否，固定五类 | 固定类别头适配能否改善当前任务 |
| B：`yoloe-vp` | 相同原始 `yoloe-v8l-seg.pt`，迁入检测结构 | 只更新SAVPE，其余参数冻结 | 是，固定支持图+框提取视觉原型 | 视觉提示编码器适配能否提高跨图检测效果 |
| C：`yolov8` | 官方检测预训练 `yolov8l.pt` | 全参数微调，DFL固定层按上游规则保留冻结 | 否，固定五类 | 相同小样本标注预算下普通YOLO的监督基线 |

另保留D：已有的未微调YOLOE视觉提示1/5/10shot成绩，作为B的主要参考。
如果要精确归因B相对D的收益，还需让二者使用同一种查询/候选合并与预处理流程。

A、B、C的训练样本、开发验证图片和最终测试图片相同；不代表它们有相同预训练历史或可训练参数量。
YOLOE与YOLOv8的预训练数据不同，A/B仅训练小模块而C全量训练。因此此实验比较的是三种方法，
不能把全部差异归因于某个单独结构。可选扩展为C增加冻结主体的版本，或为A增加全参数版本；不要混入三组主结果。

## 2. 共用的数据约定

先做5shot：五类每类5张原图，共25张，沿用 `data/references/chache/<类别>/5shot/seed_0.json`。
训练原图可以同时作为B最终推理的参考图；A、C不再使用推理参考图。

| 集合 | 用途 | 是否允许进入优化器/视觉原型 |
|---|---|---|
| A_train：固定25张支持原图 | 三组训练，B提取参考原型 | 允许 |
| B_dev：另行提供，覆盖五类和合理负例 | 早停、选best、选择参数 | 不能用于训练或提取参考原型 |
| C_test：原 `chache_val300` 300张 | 最终与已有实验统一比较 | 不能训练、选best、提取参考原型或挑阈值 |

注意：这里集合的A/B/C是数据用途，不是上表中的方法代号。
不能用test表现反复挑选模型。按相机/视频/序列检查隔离，脚本的哈希检查只能拒绝完全相同图片。
原图、裁剪图、画框图是同一个源样本，不能当成三个独立shot。
所有原图应标全五类中的实际目标，缺JSON不自动当负样本。

三组共同使用一次数据准备的 `dataset.yaml`、`preparation.json` 及实际清单，不分别重抽样。
之后开展1/10shot时重新准备对应清单，分别为5/50张；1shot训练更不稳定，建议最后做。

## 3. 当前参数与训练机制

公共配置为 [config.json](config.json)。保留了你已修改的 `epochs=100`，没有改回30。
共同起点：imgsz=640、FP32、AdamW、lr0=1e-4、patience=10、轻度平移缩放、关闭Mosaic/MixUp/Copy-Paste。
批量和GPU必须由你根据当前资源显式选择；同组对照尽量保持相同batch。
采用 `nbs=batch` 和关闭warmup，避免25张小样本下隐式按64张累积及最低100步预热。
这些是实验起点，并非验证过的最优参数。先固定同一设置比较，再按相同开发集调参预算分别优化。

默认训练seed=0；稳定后可复制配置设置1、2进行重复训练，数据支持清单仍固定support-seed=0。
只改训练seed，不重新抽支持图片。记录均值、标准差、实际训练轮数及可训练参数量。

### A：固定检测头

使用 `YOLOEPETrainer`；用五类语义描述初始化类别嵌入，再融合固定分类头，仅更新指定分类输出层。
需要本地 `mobileclip_blt.pt` 和对应依赖。它不需要推理参考图，也不是视觉提示模块微调。
原有 `--strategy full` 仍可作为额外YOLOE全参数实验，不属于本次主方案。

### B：保留视觉提示的SAVPE微调

[vp_trainer.py](vp_trainer.py) 提供本地适配，保留SAVPE、不融合固定类别头，只让SAVPE参数参与优化。
当前实现采用**同图框条件训练、跨图支持集验证**：

1. 训练：从当前训练图片的框生成空间提示，在该训练图像上计算视觉检测分类损失。
2. 开发验证：每轮从固定训练图片重新提取各类视觉原型，然后检测独立dev图片；dev真值只用于打分。
3. 原型聚合：每张支持图的特征归一化，按类别等权平均，再归一化；与现有基线的聚合原则一致。
4. 冻结策略与官方SAVPE训练方向一致，但当前适配保持五类全局ID，使用普通YOLODataset，
   不调用官方大规模训练依赖的全局文本特征缓存，不调用默认LVIS参考数据。
5. 本地默认 `TVPDetectLoss` 假设“文本通道+视觉通道”，只有五个视觉通道时会得到零视觉损失。
   适配器显式改用五类的 `v8DetectionLoss`，沿用 `load_vp=True` 时只优化分类项的上游行为；
   box/DFL可记录，但回归分支不更新。需要在短流程中确认SAVPE存在非零梯度和参数更新。

**它不是已实现的跨图support/query episodic训练，也不是原官方大规模联合训练的完整复现。**
如果同图训练未改善跨图效果，下一阶段才考虑在训练池内部构造不同图的support/query训练对。
不能用最终300张query及其标注构建训练对。

B保存的模型仍含SAVPE，可重新用参考原图+框生成原型；开发验证可能把训练集原型缓存在模型中，
最终推理时应显式重新提取并记录本次实际使用的支持清单，而不是依赖checkpoint缓存。

### C：YOLOv8-L普通监督微调

从本地 `weights/yolov8/yolov8l.pt` 载入官方检测预训练权重，不从YAML随机初始化。
训练时上游按数据集将80类输出适配到5类，迁移匹配的主体权重并更新全模型。
此组不用MobileCLIP，不需要推理fewshot图片；必须与A/B使用同一份25张训练图片。
入口统一选择项目内固定的Ultralytics代码，避免三个方法混用不同安装包。

## 4. 执行顺序

所有命令在项目根目录执行。本次没有执行下列命令。

### 第一步：准备数据和权重

在已经安装该项目依赖的环境中：

```bash
conda activate industrial-fewshot
python examples/yoloe_finetune_chache/prepare_data.py \
  --shot 5 --support-seed 0 --dev-dir /实际路径/chache_dev
```

检查计划、类别、图片及序列隔离，确认后再生成文件：

```bash
python examples/yoloe_finetune_chache/prepare_data.py \
  --shot 5 --support-seed 0 --dev-dir /实际路径/chache_dev --write
```

默认生成 `data/finetune/chache_5shot/dataset.yaml`。目录已存在时不会覆盖；准备新版本使用新 `--output`，
三组训练都传入相同的 `--data /实际新路径/dataset.yaml`。

确认以下本地文件存在，不在训练过程中临时更换权重：

```text
weights/yoloe/yoloe-v8l-seg.pt
weights/yoloe/mobileclip_blt.pt        # 仅A需要
weights/yolov8/yolov8l.pt              # C需要，官方检测预训练权重
```

已有YOLOE下载工具为 `scripts/download_yoloe_weights.py` 和 `scripts/download_yoloe_text_encoder.py`。
YOLOv8-L权重请从 [Ultralytics官方模型入口](https://docs.ultralytics.com/models/yolov8/) 下载并放到配置指定位置。
本次没有下载或安装任何依赖、权重。

### 第二步：仅查看三组计划

```bash
python examples/yoloe_finetune_chache/train.py --method yoloe-head
python examples/yoloe_finetune_chache/train.py --method yoloe-vp
python examples/yoloe_finetune_chache/train.py --method yolov8
```

没有 `--execute` 时只读取配置和显示计划，不加载模型、不创建训练目录。

### 第三步：各运行2轮短流程检查

先自行确认空闲GPU，替换命令中的 `GPU编号` 和 `批量大小`，三条命令按顺序运行。
这是单卡样例；GPU是当前Python进程可见的编号，不能直接套用其他容器的编号。

```bash
python examples/yoloe_finetune_chache/train.py --method yoloe-head \
  --execute --device GPU编号 --batch 批量大小 \
  --epochs 2 --name yoloe_head_5shot_seed0_smoke_v1

python examples/yoloe_finetune_chache/train.py --method yoloe-vp \
  --execute --device GPU编号 --batch 批量大小 \
  --epochs 2 --name yoloe_vp_5shot_seed0_smoke_v1

python examples/yoloe_finetune_chache/train.py --method yolov8 \
  --execute --device GPU编号 --batch 批量大小 \
  --epochs 2 --name yolov8l_5shot_seed0_smoke_v1
```

检查：预训练权重确实迁入，训练25张，dev数量正确，类别顺序=5，实际冻结层符合各组方案，
损失有限且非恒零、应训练的参数有梯度并更新，最终验证与进程退出正常。
B尤其需要确认只更新SAVPE，视觉原型的来源为train，查询图不带视觉真值提示。
未完成这些运行检查前，不能宣称样例已经验证可用。

### 第四步：新开三组正式实验

短流程通过后，分别从各自原始预训练权重新开实验，不接着2轮检查权重训练：

```bash
python examples/yoloe_finetune_chache/train.py --method yoloe-head \
  --execute --device GPU编号 --batch 批量大小 --name yoloe_head_5shot_seed0_v1

python examples/yoloe_finetune_chache/train.py --method yoloe-vp \
  --execute --device GPU编号 --batch 批量大小 --name yoloe_vp_5shot_seed0_v1

python examples/yoloe_finetune_chache/train.py --method yolov8 \
  --execute --device GPU编号 --batch 批量大小 --name yolov8l_5shot_seed0_v1
```

三个输出目录均位于 `outputs/finetuning/chache_yoloe/<name>/`，已有目录拒绝覆盖。
查看 `finetune_config.json`、`trainable_parameters.json`、`args.yaml`、`results.csv` 及 `weights/best.pt`。
A还会保存 `class_embeddings.pt`。这一路径目前被Git忽略，权重需要单独备份。

## 5. 训练后的评测与汇总安排

三组最终统一评测入口**尚未实现**。现阶段不要把训练器的dev指标填进原300张总表。
下一步的评测适配应完成：

1. A与C：直接加载固定五类best.pt，在原300张测试图上推理；不调用视觉原型接口。
2. B：加载仍带SAVPE的best.pt，用原25张支持原图及框重新提取原型，再在300张测试图上推理。
3. D：用原始YOLOE权重、同一25张支持图、同一查询流程重新评测，作为B的严格流程对照。
4. 同一图像预处理、类别映射、IoU/NMS/候选数量策略；保持原项目conf=0.001、NMS=0.7，
   对每类最多100框的约定进行显式适配。不能把五类联合模型全图max_det=100当作每类100。
5. AP使用所有保留候选，P/R/F2分别统计score=0.5与0.7，注明合并micro及每类macro口径。
6. 保存逐类AP50、AP50:95、TP/FP/FN、推理耗时，以及实际支持清单和模型哈希。
7. 原有D是五次单类查询后合并；A/C是五类联合预测。若B也采用联合预测，必须另保留同流程D，
   不能把由查询方式变化带来的差异全部称为微调收益。

最终结果分别归档到 `outputs/runs/chache_val300/YOLOE-Head-FT/`、`YOLOE-SAVPE-FT/`、
`YOLOv8-FT/`，按现有格式生成 `run.json`、`metrics.json`、`evaluation.txt` 后再更新总表。
仅改变文件夹名不足以完成算法归类，评测适配还需写入正确的backend/method元数据。
微调成功后若要部署，再安排FP32导出、PTQ INT8和同集配对评估；本方案不执行量化。

## 6. 先做哪些

第一轮建议只做5shot、固定seed=0的A/B/C，加现有D作参考；关注SAVPE微调是否真正改善跨图效果。
第二轮扩展1/10shot及训练seed重复实验。
若要论证对新目标的开放类别能力，需要保留未参加微调的类别，不能只根据同五类成绩下结论。

参考：[官方检测头微调](https://github.com/THU-MIG/yoloe/blob/40cd606cabdbe2b566d6f14a6b162c89206e9a1b/train_pe.py)、
[官方SAVPE冻结方案](https://github.com/THU-MIG/yoloe/blob/40cd606cabdbe2b566d6f14a6b162c89206e9a1b/train_vp.py)、
[YOLOv8官方文档](https://docs.ultralytics.com/models/yolov8/)。

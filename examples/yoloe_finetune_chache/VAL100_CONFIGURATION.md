# 两场景100张验证集：当前微调配置与执行步骤

更新日期：2026-10-29。配置和代码已调整，未运行数据转换或训练。
当前沿用最近一次选择的10shot、support-seed=0；原有epochs=100及其他训练超参数保留。

## 当前划分

| 场景 | 配置 | 训练图片 | 验证图片 | 最终测试图片 |
|---|---|---:|---|---|
| 叉车五类 | [config.json](config.json) | 每类10张，共50张 | `data/raw/chache_val100/images`，100张/76框 | `data/raw/chache_val300`，300张 |
| 吊物单类 | [config_diaowu.json](config_diaowu.json) | 10张 | `data/raw/diaowu_val100/images`，100张/75框 | `data/raw/diaowu_val300`，300张 |

两套新验证集均为70张有目标、30张无目标；原图旁均有同名LabelMe JSON。
本次只读核验了原图及JSON哈希、支持清单哈希、训练/验证/测试图片内容交集，结果无交集。
内容不重合不等于视频序列隔离，同源视频不同帧仍可能相似。
验证集用于每轮打分、early stopping和选择best.pt，不更新参数、不提取推理视觉原型。
SAVPE只用对应场景的训练参考图提取视觉原型。原300张不参与选权重或挑参数。
原始100张和300张目录不改写，转换结果写入独立的 `data/finetune/`。

## 配置如何生效

- `config.json` 是叉车配置及公共训练参数，类别顺序保持 `Chache1/Chache2/Chache3/Chache4/Qianyinche`。
- `config_diaowu.json` 继承公共参数，覆盖场景、数据路径、输出目录、类别与提示词，唯一类别为 `hanging object`，ID=0。
- `prepare_data.py` 与 `train.py` 使用相同的 `--config`，不再硬编码只支持叉车五类。
- 新增 `dev_images`、`dev_manifest`、`dev_expected_images=100`、`test_manifest`，不用填写空的 `train/chache_dev`。
- 验证图片按manifest固定顺序选择，校验图片和JSON哈希，并要求恰好100张。
- 训练时检查准备产物的场景、类别、shot、验证数量及验证manifest哈希，避免拿错数据版本。
- 训练目录默认包含实际shot、训练seed和val100，不再固定命名为5shot。

三种方法仍为 `yoloe-head`（固定分类头）、`yoloe-vp`（仅SAVPE）、`yolov8`（预训练全参数微调）。
SAVPE适配器支持单类或多类，吊物不会再触发五类数量检查。
两场景输出分别在 `outputs/finetuning/chache_yoloe/` 与 `outputs/finetuning/diaowu_yoloe/`。

## 第一步：只检查数据计划

以下命令供你手动执行；本次没有执行它们。

```bash
cd /home/yuheng/Work/industrial_fewshot_detection
conda activate industrial-fewshot

# 叉车：预计train=50、val=100
python examples/yoloe_finetune_chache/prepare_data.py \
  --config examples/yoloe_finetune_chache/config.json

# 吊物：预计train=10、val=100
python examples/yoloe_finetune_chache/prepare_data.py \
  --config examples/yoloe_finetune_chache/config_diaowu.json
```

不用再传 `--dev-dir`，脚本会从配置读取路径。默认只读检查，不复制、不写入训练数据。

## 第二步：生成独立训练数据

```bash
python examples/yoloe_finetune_chache/prepare_data.py \
  --config examples/yoloe_finetune_chache/config.json --write

python examples/yoloe_finetune_chache/prepare_data.py \
  --config examples/yoloe_finetune_chache/config_diaowu.json --write
```

输出分别为：

```text
data/finetune/chache_10shot_val100/dataset.yaml
data/finetune/diaowu_10shot_val100/dataset.yaml
```

每套包含 `images/{train,val}`、`labels/{train,val}`、`train.txt`、`val.txt`、`preparation.json`。
不把300张测试图写成val。目标目录存在时拒绝覆盖；使用新 `--output` 后还要同步修改配置的 `data`。

## 第三步：查看训练计划

以SAVPE为例：

```bash
python examples/yoloe_finetune_chache/train.py --method yoloe-vp \
  --config examples/yoloe_finetune_chache/config.json

python examples/yoloe_finetune_chache/train.py --method yoloe-vp \
  --config examples/yoloe_finetune_chache/config_diaowu.json
```

换为 `--method yoloe-head` 或 `--method yolov8` 即为另外两组。没有 `--execute` 时不加载模型、不训练。

## 第四步：准备开始时，先做短流程检查

检查依赖与权重，选择空闲GPU与batch后，替换命令中的占位符：

```bash
# 叉车SAVPE：2轮检查
python examples/yoloe_finetune_chache/train.py --method yoloe-vp \
  --config examples/yoloe_finetune_chache/config.json \
  --execute --device GPU编号 --batch 批量大小 \
  --epochs 2 --name yoloe_vp_10shot_val100_smoke_v1

# 吊物SAVPE：batch不得超过10张训练图片
python examples/yoloe_finetune_chache/train.py --method yoloe-vp \
  --config examples/yoloe_finetune_chache/config_diaowu.json \
  --execute --device GPU编号 --batch 批量大小 \
  --epochs 2 --name yoloe_vp_10shot_val100_smoke_v1
```

分别核对训练50/10张、验证100张、类别5/1、实际可训练层、损失与SAVPE参数更新。
通过后去掉 `--epochs 2`，换正式实验名字，从原始预训练权重新开100轮实验，不接着检查权重训练。
方法机制、其他权重与最终评测适配边界见 [COMPARISON_PLAN.md](COMPARISON_PLAN.md)。

## 调整到其他shot

先复制所选场景配置，再一起修改 `shot` 与 `data`，例如5shot指向新的
`data/finetune/<场景>_5shot_val100/dataset.yaml`，数据准备和训练都传入这份新配置。
仅给prepare传 `--shot 5`、训练仍用10shot配置会报错，避免预算不一致。
验证始终用同一100张，测试保留原300张。

## 当前验证范围

本次仅做静态配置/代码检查及输入文件只读核验，不表示模型加载或训练已通过运行验证。
没有执行数据转换、训练、推理、测试或量化，没有改写原始数据或既有实验结果。

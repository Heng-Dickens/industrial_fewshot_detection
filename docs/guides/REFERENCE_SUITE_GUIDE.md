# 参考材料与样本张数对比

当前 `configs/specific/diaowu_ablation_config.json` 已启用此流程，默认每组提供 **5 张正样本原图**。对每种正样本材料组合分别运行不加负样本、增加 5 张负样本两种情况，再加一次仅 prompt 基线，共 **15 组**。

| 符号 | 材料 |
| --- | --- |
| O | 正样本原图，不画框 |
| C | 该正样本的全部目标裁剪图 |
| B | 该正样本带彩色标注框的全图 |
| J | 该正样本的 LabelMe JSON 文本，去除 imageData |

正样本组合为 `O+J`、`O+C`、`O+B`、`O+C+J`、`O+B+J`、`O+C+B`、`O+C+B+J`，不包含仅 O。每组始终提供同一份主 prompt 和当前待识别的全图。负样本仅发送原图，并明确说明该图应返回空结果。

## 配置

在任意独立场景配置内加入以下字段；模型、类别、提示词文件与待测目录继续沿用该场景的其它字段：

```json
{
  "mode": "train",
  "reference_image_dir": "../../data/references/diaowu",
  "reference_mode": "text_only",
  "experiment_suite": {
    "enabled": true,
    "include_prompt_only": true,
    "positive_counts": [5],
    "negative_variants": [false, true],
    "positive_variants": ["O+J", "O+C", "O+B", "O+C+J", "O+B+J", "O+C+B", "O+C+B+J"],
    "selection_seed": 0,
    "exclude_reference_overlap": true
  },
  "ablation": {"enabled": false}
}
```

- `positive_counts: [5]` 是默认值，只测试 5 张正样本这一档。可改为 `[1]`、`[10]`、`[25]`，或 `[1, 5, 10, 25]`；四档全部启用时共 57 组，基线只执行一次。
- `negative_variants: [false, true]` 同时测试不提供、提供负样本；提供时负样本原图张数等于正样本原图张数。只测含负样本时改为 `[true]`。
- `positive_variants` 可选取所需组合；不支持仅原图 O。
- `selection_seed` 固定样本选择顺序。同一档位的所有组合使用相同正样本；加入负样本不会改变正样本。不同档位按同一顺序取前 K 张，1 张集合包含于 5 张，依次到 25 张。
- 此流程按同一原图配对 C/B/J，不受旧 `max_reference_examples`、`max_reference_crops` 的限制。请求材料缺失或正负样本数量不足时，在任何模型调用之前报错，不会静默降低张数。
- `experiment_suite.enabled` 优先于旧 `ablation`；关闭新流程后，旧四组流程和单组 `reference_mode` 仍可使用。

参考目录须为 `tool/prepare_reference_data.py` 输出的数据，包含 `manifest.json`、`positive/images`、`positive/annotations`、`positive/crops`、`positive/boxed_images`、`negative/images`。当前吊物参考集保留 25 正样本、25 负样本，不需要为了默认 5 张缩减磁盘上的样本集。

## 原图张数与实际图片数

5 正样本加 5 负样本表示 **10 个原图参考例子**。C、B 都从选中的 5 张正样本派生，不另算原图例子数，实际发送的图片数会增加。一个正样本含多个标注目标时，发送该图的全部裁剪图。

例如当前固定的 5 张正样本对应 6 张裁剪图：`O+C+B` 含负样本时实际发送 `5 原图 + 6 裁剪图 + 5 带框图 + 5 负样本图 = 21 张参考图`，另加最后一张待测全图。J 是文本，不增加图片数。参考图沿用现有等比例缩放与 JPEG 编码设置。

## 执行与结果

在项目根目录执行：

```powershell
# 只检查分组、样本选择和可复用结果，不调用模型、不写结果
python main.py --config configs/specific/diaowu_ablation_config.json --dry-run

# 正式执行配置中的实验
python main.py --config configs/specific/diaowu_ablation_config.json
```

当前 MLLM YAML 将结果写入 `outputs/runs/<实际模型>_<shot数>shot/`，例如 `qwen3.8-27b_5shot/`。其中 `comparison.csv`、`comparison.json`、`comparison.txt` 汇总所有组；下一层使用 `仅文字提示`、`原图+标注JSON_无负例`、`原图+裁剪图+带框图_含5张负例` 等可读名称。各组目录中直接保存 `run.json`、`summary.json`、`records.jsonl`、`metrics.json`、`evaluation.txt`、`predictions/` 和 `feedback/`，没有额外的时间戳／随机串／task_id 层级。重复运行使用同级 `_第2次` 等后缀保留已有结果。未显式设置 output_dir 的历史配置仍以输入目录同级 `<目录名>_annotations` 为输出根，再使用同样的可读命名。

CSV 新增实验 ID、正/负原图张数、原图例子总数、实际参考图片数、原图/裁剪图/带框图/JSON/负样本计数。每组 `experiment.reference_sets` 记录正负样本文件名，可核对配对关系。`reference_image_count` 不包括最后一张待测图。

指标沿用 Precision、Recall、F1、mAP、整图正确率、负样本错误率及代理 loss，`delta_` 为相对仅 prompt 基线的差值。推理失败计为整图错误；负样本推理失败也计入负样本错误率。`train` 是推理后本地评估，不更新模型权重。

待测图片的真值仅用于本地评估，不进入模型消息。按原始图片内容 SHA256 排除整个参考池与待测集的重叠，各组（包括基线）统一使用排除后的同一批待测图。`exclude_reference_overlap: false` 时，发现重叠会直接报错。当前吊物数据中 `0000090.jpg` 重叠，因此各组使用 99 张待测图片；参考集仍保留完整 50 个例子。

相同有效输入和待测图片自动复用成功推理，改变评估参数无需重跑模型。只有真正发送的参考材料参与缓存指纹，例如修改 J 不会重跑未使用 J 的组。旧基线结果也可自动复用。`--force` 会明确忽略缓存并重新消耗 token。

# 四种参考输入的数值对比

> 本文描述旧的四组 `ablation` 流程。当前 `configs/specific/diaowu_ablation_config.json` 已改用 `experiment_suite`，默认 5 张正样本，共 15 组，详见 [参考材料与样本张数对比](REFERENCE_SUITE_GUIDE.md)。下文四组配置示例用于旧模式；执行当前吊物配置会运行新流程。

所有组都发送同一张当前待识别全图及同一份主检测 prompt。只改变是否额外发送参考图片的标注 JSON 文本、参考目标裁剪小图。**待测图片的真值 JSON 只供本地评估，不进入模型请求。**

| reference_mode | 文字 prompt | 参考标注 JSON 文本 | 参考目标小图 |
| --- | --- | --- | --- |
| `text_only` | 有 | 无 | 无 |
| `text_json` | 有 | 有 | 无 |
| `text_crops` | 有 | 无 | 有 |
| `text_json_crops` | 有 | 有 | 有 |

这四种模式都不发送参考全图或负例。JSON 中的 `imageData` 被移除，JSON 参考即使没有对应原图也可以使用。原有未设置 reference_mode 的配置继续按 `configured` 模式处理带框全图、JSON、裁剪图和负例。

## 吊物场景配置与命令

旧四组实验沿用独立场景配置里的模型和服务设置；吊物图片目录为：

```text
D:\Data\江苏通用\训练数据\模具吊物喷油\模具车间1\00000
```

旧模式可从 `data/references/diaowu` 读取参考 JSON 和裁剪图。按文件名排序选取，各组涉及的参考类型使用相同样例。首次正式运行会为四种不同输入分别推理；再次运行相同配置和图片会自动复用各组成功结果。要运行旧四组，将 experiment_suite.enabled 设为 false、ablation.enabled 设为 true。

当前吊物配置使用 `missing_ground_truth: "negative"`，按这批数据的约定将没有同名 JSON 的待测图片作为负样本参与评估。这些图片若被预测出目标框，会计入 FP，影响 Precision/mAP 和代理 loss，不再从指标中排除。该标记仅在本地评估使用，模型不会获知待测图片是否为负样本。

在项目根目录检查和执行：

```powershell
python main.py --config configs/specific/diaowu_ablation_config.json --dry-run
python main.py --config configs/specific/diaowu_ablation_config.json
```

以上命令供使用者自行执行。开发验证使用模拟模型，没有执行真实对比推理。

## 通过配置切换场景或输入组合

任意独立场景配置均可加入以下字段，无需注册场景或修改入口：

```json
{
  "mode": "train",
  "missing_ground_truth": "negative",
  "reference_mode": "text_json_crops",
  "reference_image_dir": "../../data/references/diaowu",
  "experiment_suite": {"enabled": false},
  "max_reference_examples": 6,
  "max_reference_crops": 6,
  "output_dir": null,
  "output_suffix": "_annotations",
  "ablation": {
    "enabled": true,
    "modes": ["text_only", "text_json", "text_crops", "text_json_crops"]
  }
}
```

`ablation.enabled: true` 表示依次运行 modes 中的所有组。也可设置 false，使用 `--ablation --mode train` 临时开启。要只运行一组，将 enabled 改为 false，再选择对应的 reference_mode。单组支持 train/test；数值对比需要 train 和可用真值。

`max_reference_examples` 是 JSON 数量上限，`max_reference_crops` 是小图数量上限。具备该参考类型的组使用设置值，其他组自动关闭该类型；负例自动关闭。需要 JSON 或小图的组若没有可用样例或上限为 0，会在任何模型调用之前报错，避免实际输入与实验名称不符。

参考 JSON 必须包含 LabelMe 的 shapes、imageWidth、imageHeight；裁剪图沿用本项目文件名规则，例如 `0000010_crop_01_hanging_object.png`。text_only 不要求存在参考目录。

各组共用模型、主 prompt、生成/复核设置、待测图片清单、真值快照和评估阈值。JSON 与小图只是参考信息，不作为当前图片的答案。程序拒绝与当前待测真值文件重合或明确指向待测图片的参考标注；参考目录内的图片会从待测图片扫描中排除。

## 输出与数值

未指定 output_dir 时，输出在输入目录的同级 `<原目录名>_annotations` 下：

```text
00000_annotations/
  .inference_cache/                         # 四组各自按实际输入匹配缓存
  ablation_<UTC时间_随机ID>/
    comparison.csv                         # 四行指标，可直接用 Excel 打开
    comparison.json                        # 完整汇总及各组运行信息
    text_only/<运行时间_ID>/...
    text_json/<运行时间_ID>/...
    text_crops/<运行时间_ID>/...
    text_json_crops/<运行时间_ID>/...
```

各组目录继续包含 metrics.json、records.jsonl 和 LabelMe predictions，格式同普通标注。

CSV 包含 Precision、Recall、F1、mAP50、mAP75、mAP50_95、mean_matched_iou、proxy_loss、TP/FP/FN、样本数、参考数量、失败数、复用/实际推理数量和结果目录。`delta_` 列表示该组减去 text_only 基线组的数值；Precision/Recall/mAP 等越高越好，代理 loss 越低越好。指标是 0～1 比例，差值同样是比例差。

为检查误标，CSV 另含负样本图片数、负样本误报图片数/框数、正确拒识数、失败数和 `negative_false_positive_rate`（误报图片数 / 成功评估的负样本图片数），越低越好；`delta_negative_false_positive_rate` 是相对文字基线的差值。每组 metrics.json 的 `negative_false_positive_images` 列出误报图片路径，方便核查。

CSV 同时包含 `image_accuracy`（整图正确率）、correct_image_count、incorrect_image_count、failed_image_count，以及 `negative_error_rate`（负样本误报或失败的图片数 / 全部负样本图片数）。推理失败始终计为错误并留在这两个指标的分母中。整图正确要求推理成功，且在配置 IoU/置信度阈值下没有误报和漏检；负样本成功返回空结果才正确。标准 Precision 保持 TP/(TP+FP) 定义，与整体正确率分列。

缺少 JSON 的处理由 missing_ground_truth 决定：negative 作为负样本纳入指标，skip 排除出指标，error 立即报错。通用默认仍为 skip；当前吊物对比配置已设为 negative。只有确认为无目标的图片才采用负样本约定，其他场景可在配置中选择相应策略。

推理失败会标记该组和汇总为 incomplete，CSV 保留失败数，避免把未完成的组当成完整结果。失败负样本不会计为正确拒识，所有负样本都失败时误报率为 null。评估异常保留已完成组的比较表；修复后重运行原命令，成功推理自动复用。只改缺少 JSON 的处理策略、评估阈值或真值不会重新消耗推理 token；只修改参考 JSON 会重跑 JSON 两组，只修改裁剪图会重跑小图两组。`--force` 明确重跑所有指定组。

参考目录中的负例用于教模型识别不该标注的情形，待测负样本用于检验模型是否误标；两者分开管理。当前四组依旧只比较文字、参考 JSON 和正例裁剪图，不额外加入负例参考全图。

当前 loss 是本地检测代理损失，不是模型训练损失，train 不更新模型权重。具体定义见 [MAIN_GUIDE.md](MAIN_GUIDE.md)。

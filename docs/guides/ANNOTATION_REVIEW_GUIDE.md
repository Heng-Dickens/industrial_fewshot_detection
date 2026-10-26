# 用 VLM 纠正已有 LabelMe 标注

程序：[review_existing_annotations.py](../../tool/review_existing_annotations.py)

当前配置：[Chache_cannotation_review_config.json](../../configs/specific/Chache_cannotation_review_config.json)
类别参考图：data/references/chache/review/（本地路径 `../../data/references/chache/review/`）

## 1. 当前数据检查结果

数据目录：

```text
D:\Data\江苏通用\标注待检查\calibdata\模具车间1
```

- 5399 张 JPG、5399 个同名 LabelMe 5.1.1 JSON
- 4445 个 rectangle 框
- `Chache1`：2073 框
- `Chache2`：8 框
- `Qianyinche`：2364 框
- 2164 张图片当前没有框

全部 JSON 均可解析。抽查发现现有 8 个 `Chache2` 框高度疑似系统性误标，因此不能用它们反推 `Chache2` 的类别定义。

## 2. 运行前检查类别定义和参考图

当前使用 configs/specific/Chache_cannotation_review_config.json，不需要复制旧的配置。

确认 labels 中的各类别描述与 data/references/chache/review 下对应参考图一致，并保留每个 label 的 reference_image 映射。

确认定义正确后修改：

```json
"label_definitions_confirmed": true
```

在它为 `false` 时，程序只允许 dry-run，正式请求会停止，避免依据错误定义批量删框。

## 3. Prompt 范例

程序会把类别定义和每个现有框的编号动态加入 prompt，核心内容如下：

```text
你是工业场景目标检测标注的质量审核员。
首要任务是删除小模型产生的误标、背景框、重复框和明显错类；
第二任务才是补充非常明确的漏标。

允许的各类目标及现场定义：
- Chache1、Chache2、Chache3、Chache5：分别以配置描述和对应类别参考图为准，必须有叉车门架、立柱或叉臂等结构证据。
- Qianyinche：工业牵引车；前部清晰可见且没有叉车门架和成对货叉时，不能改成 Chache 类。

模型先阅读 labels 中每个类别的参考图，再阅读无标记原图和包含整图编号框、局部裁图的审核图。
对每个已有 object_id 恰好返回一次：
- keep：目标真实且类别正确，不修改原框坐标。
- remove：不属于允许类别、背景框、固定设备、工人、模具、料架或重复框。
- relabel：目标真实但类别错误，给出正确类别。
- uncertain：遮挡、模糊、过小或依据不足，禁止猜测。

已有标签只是待审候选，不是事实。
remove 必须有明确视觉依据；拿不准使用 uncertain。
missing_objects 只填写清晰、明显且未被已有框覆盖的漏标目标。
严格输出 JSON，不要输出 Markdown 或额外解释。
```

期望模型返回：

```json
{
  "existing_reviews": [
    {
      "object_id": "obj_0001",
      "decision": "remove",
      "corrected_label": "",
      "reason": "框内是固定绿色设备边缘，不是车辆"
    },
    {
      "object_id": "obj_0002",
      "decision": "relabel",
      "corrected_label": "Qianyinche",
      "reason": "目标没有货叉和门架，应为牵引车"
    },
    {
      "object_id": "obj_0003",
      "decision": "uncertain",
      "corrected_label": "",
      "reason": "目标被遮挡，无法可靠区分类别"
    }
  ],
  "missing_objects": [
    {
      "label": "Chache1",
      "bbox_2d": [120, 260, 340, 710],
      "reason": "清晰可见的完整叉车，未被已有框覆盖"
    }
  ],
  "image_reason": "已逐框审核，并扫描整图检查漏标"
}
```

## 4. 推荐运行顺序

先只读检查：

```powershell
python tool/review_existing_annotations.py `
  --config configs/specific/Chache_cannotation_review_config.json `
  --dry-run --limit 10
```

第一阶段只纠正已有框，先单并发处理 10 张：

```powershell
python tool/review_existing_annotations.py `
  --config configs/specific/Chache_cannotation_review_config.json `
  --limit 10 --concurrency 1
```

人工检查输出和审核图没有问题后，再运行完整任务：

```powershell
python tool/review_existing_annotations.py `
  --config configs/specific/Chache_cannotation_review_config.json
```

第二阶段才检查空标注图片和漏标。建议先保持 `apply_missing=false`，只收集建议：

```powershell
python tool/review_existing_annotations.py `
  --config configs/specific/Chache_cannotation_review_config.json `
  --include-empty
```

抽检漏标建议可靠后，才把配置中的 `apply_missing` 改为 `true`，并使用新的 output/audit 目录或 `--force` 重新运行。

## 5. 修改规则

| VLM决定 | 默认处理 |
|---|---|
| `keep` | 原框和原类别不变 |
| `remove` | 从新 LabelMe JSON 删除，原 JSON 不变 |
| `relabel` | 只修改新 JSON 中的 label，不重画框 |
| `uncertain` | 保留原框，在审计记录中标为人工复核 |
| `missing_objects` | 默认只记录建议，不写入正式标注 |

漏标框与保留框的 IoU 超过 `missing_max_iou` 时会被当作重复候选拒绝。

## 6. 输出

默认输出为原目录的两个同级目录：


```text
模具车间1_vlm_corrected/   # 图片硬链接和纠正后的逐图 LabelMe JSON
模具车间1_vlm_audit/
  records/                 # 每张图独立的 prompt、原始回复和修改记录
  review_images/           # 编号整图与局部裁图
  review.checkpoint.jsonl  # 断点续跑
  run_summary.json         # 本次统计
```

模型响应失败时，程序向新目录写入原标注副本，并记录 `failed_passthrough`；下次运行不会把它当作成功记录跳过。程序不会修改原始数据目录。

## 7. 类别参考图复核流程

程序不会扫描整个参考目录，而是按 labels 逐项读取：当前配置为 5 张图片——
Chache1.png、Chache2.png、Chache3.png、Chache5.png 和 Qianyinche.png，
均位于 data/references/chache/review/ 下。

每次请求的图片顺序固定为：

1. labels 中每个类别的参考图，每张图前有对应的类别名；
2. 当前图片的无标记原图；
3. 当前图片的整图编号框和局部裁图审核图。

模型必须先比较参考图，提取各类车辆的结构特征，再审核当前候选框。
Chache1 至 Chache5 只有看到对应的叉车门架、立柱或叉臂才可以作为叉车类；
前部不可见、遮挡或无法区分时返回 uncertain。
Qianyinche 前部清晰且没有叉车门架和成对货叉时，优先保留为 Qianyinche。

配置字段包括 reference_image_dir、use_reference_images、review_prompt，
以及 labels 中每项可选的 reference_image。省略 reference_image 时，
程序会根据标签名自动寻找同名图片。

当前配置 apply_relabel 为 true，模型返回的 relabel 建议会直接写入新 JSON
（原目录不受影响）；若只想先收集建议，把它改为 false。
每张审计记录会保存 reference_images 清单（路径、SHA-256、大小、MIME），便于复现。

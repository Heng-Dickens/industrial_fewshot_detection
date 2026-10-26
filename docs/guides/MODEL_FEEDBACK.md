# 保存逐框模型反馈

MLLM 推理默认要求每个 `objects` 元素提供非空 `reason`，用一两句中文描述可见的目标特征、为何符合目标定义，以及框的覆盖范围。整图 `reason` 仍说明总体判断；无目标或无法判断时也会保存对应反馈。

以下仅演示字段格式，不代表真实推理结果：

```json
{
  "label": "hanging object",
  "bbox_2d": [100, 200, 800, 900],
  "confidence": 0.9,
  "reason": "可见吊索连接圆柱状吊物，框覆盖吊具和吊物的完整可见外轮廓。"
}
```

原运行命令无需增加参数。单次运行目录会保存：

```text
RUN_DIR/
  records.jsonl                    每张图的最终框、逐框 reason 和请求记录
  summary.json                     完整记录汇总
  predictions/图片名.json          LabelMe 标注，shape.description 为逐框理由
  feedback/图片名.json             结构化反馈：整图、最终框、首轮、复核及原始响应
  feedback/图片名.txt              可直接阅读的中文说明
```

输入有子目录时，反馈目录保留同样的相对目录结构。`records.jsonl` 中的 `feedback_json`、`feedback_text`、`labelme_json` 指向对应输出文件。

单场景结果直接位于模型／shot 目录（单组运行）或其下的可读实验组目录（对比实验），不再额外套运行时间、随机串或 task_id 目录。旧目录不自动移动。

开启二次复核时，保存首轮 `initial_annotation` 与最终结果各自的框和理由。复核改框后要求同步更新理由；复核失败时保留首轮结果并保存 `review_error`。每次模型请求的原始回答、阶段、尝试编号及校验错误保存在 `model_feedback_attempts`（反馈 JSON 的 `attempts`）中，便于检查格式错误、缺少理由和重试情况。

这些理由是模型自述的可见依据，不是经人工验证的事实或内部推理记录。不需要开启 `enable_thinking`。反馈文件不使用待测真值生成解释。

新增理由要求会改变提示词、响应 schema 和请求签名。**原来不含逐框理由的缓存不会作为新请求的完整结果复用，下一次运行可能产生新的模型请求与费用**；之后相同请求继续自动复用，理由随缓存一起恢复。不会自动重写历史运行或补造旧理由。若需要保留原请求及缓存口径，在场景配置顶层设置 `require_box_reason: false`；此时缺失的逐框理由标为 `not_provided`。

服务端即使不支持 JSON Schema，也会收到提示词要求；缺少逐框理由的正例响应会按现有 `max_attempts` 重试，复核失败则沿用首轮有效结果。全部请求失败仍保存错误及收到的原始回答。YOLOE／GroundingDINO 不输出自然语言理由，反馈会标为 `unsupported`，并保留原有检测框和分数。

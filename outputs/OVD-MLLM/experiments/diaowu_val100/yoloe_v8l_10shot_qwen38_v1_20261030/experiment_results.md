# YOLOE → Qwen3.8-27B 配对实验

数据集：diaowu_val100。YOLOE原始10shot视觉提示，候选下限0.001，seed=0；非新微调权重。

用户指定候选阈值：0.01；全图上限：500。开发集仅核查召回变化，不再搜索阈值。
大模型参考：每类1张固定支持图裁图；temperature=0.1，整图+编号框+上下文裁图，原分数不展示。
B1只筛选；B1D在B1基础上按Qwen确认的同类同实例关系去重，保留原分数最高的框；B2只改B1D框坐标。
AP用全部保存分数，P/R/F2及TP/FP/FN为score≥0.5、IoU≥0.5。AP/P/R/F2以百分数显示。

| 阶段 | 框数 | AP50 | AP75 | AP50:95 | AP85 | AP90 | AP95 | 四项≤20% AP | P | R | F2 | TP/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| source | 9998 | 71.61 | 51.27 | 44.09 | 22.34 | 10.32 | 0.00 | 57.41 | 76.39 | 73.33 | 73.92 | 55/17/20 |
| baseline | 1834 | 71.58 | 51.24 | 44.07 | 22.34 | 10.32 | 0.00 | 57.37 | 76.39 | 73.33 | 73.92 | 55/17/20 |
| filtered | 121 | 75.72 | 53.42 | 45.48 | 22.86 | 10.77 | 0.00 | 59.03 | 82.81 | 70.67 | 72.80 | 53/11/22 |
| deduplicated | 118 | 75.72 | 53.42 | 45.48 | 22.86 | 10.77 | 0.00 | 59.03 | 82.81 | 70.67 | 72.80 | 53/11/22 |
| refined | 118 | 73.42 | 50.47 | 45.83 | 24.39 | 6.17 | 1.01 | 53.50 | 81.25 | 69.33 | 71.43 | 52/12/23 |

与B0比较（百分点）：

- filtered：AP50 +4.14，AP50:95 +1.42，召回 -2.67；FP净减少 6。
- deduplicated：AP50 +4.14，AP50:95 +1.42，召回 -2.67；FP净减少 6。

B2−B1D：AP50:95 +0.35，AP85 +1.53 个百分点。

## 审核审计

```json
{
  "selection": {
    "threshold": 0.01,
    "limit": 500,
    "candidates": 1834,
    "recall50": 0.92,
    "recall85": 0.41333333333333333,
    "box20_coverage": 0.7733333333333333,
    "lost_recall50": 0.02666666666666667,
    "selection_policy": "user_fixed_floor",
    "criterion": "User fixed score>=0.01; no additional global truncation beyond source 100/class (500 total). Dev recall loss is reported, not optimized away.",
    "source_sha256": "1354912c4f3a976fb589dc0401a18a87dbef759ce1f60f0605a586d7a8b218f0"
  },
  "split_audit": {
    "exact_image_overlap": 0,
    "shared_sequences": [
      "s00_00014",
      "s00_00016",
      "s00_00026",
      "s00_00027",
      "s00_00028",
      "s00_00030",
      "s00_00034",
      "s00_00042",
      "s00_00051"
    ],
    "limitation": "Existing cohorts may share video sequences; no claim of sequence-independent generalization."
  },
  "filter_accounting": {
    "requests": 0,
    "cached": 132,
    "recovered_cached": 0,
    "failed": 0,
    "retries": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "seconds": 0.0,
    "wall_seconds": 0.17434856109321117
  },
  "decisions": {
    "keep": 111,
    "remove": 1713,
    "uncertain": 10
  },
  "duplicate_edges": 3,
  "invariant_check": "passed: retained coordinates/classes/scores/order unchanged",
  "refine_accounting": {
    "requests": 0,
    "cached": 82,
    "recovered_cached": 0,
    "failed": 0,
    "retries": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "seconds": 0.0,
    "wall_seconds": 0.04353572800755501
  },
  "refined_boxes": 85
}
```

限制：固定测试集已有历史结果；开发/测试可能共享视频序列，详见split_audit.json，不宣称序列独立泛化。
已保留不确定与请求失败的原框；失败率见审计，存在失败时不能称为完整成功的Qwen审核。
各阶段metrics.json含完整逐类和各置信度指标；原始请求与回答位于outputs/_shared/runtime/ovd_mllm_filtering_20261030。
